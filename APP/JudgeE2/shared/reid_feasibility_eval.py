"""
Cross-session object re-ID feasibility probe (architect_output.md §25).

WHAT THIS MEASURES
===================
Whether MobileSAM_ImageEncoder's `image_embeddings` output ([1,256,64,64],
the frozen fp16-milfix CoreML encoder already shipped in the app) carries
enough discriminative signal to distinguish "two observations of the SAME
physical object" from "two DIFFERENT objects" — without any additional
training. The encoder was trained for class-agnostic segmentation, not
metric/re-ID learning, so this is a repurposing probe, not a validated
capability. Treat the output as a go/no-go signal for whether investing in
a proper re-ID pipeline (real training data, possibly a small projection
head, cosine-similarity threshold tuning) is worthwhile at all.

METHOD
======
1. Load every image under --data_dir/<object_name>/*.{jpg,jpeg,png}.
   Each subdirectory name is one physical-object identity; every image in
   it is a different "observation" (angle / distance / lighting) of that
   object.
2. Preprocess each image exactly like the CoreML encoder's traced input
   contract (see export_encoder_fp16_milfix.py):
     - resize so the longer side = 1024
     - zero-pad (black, pre-normalisation) to 1024x1024, anchored top-left
     - RGB, float32, values in raw [0, 255] range (normalisation by
       pixel_mean/pixel_std is baked into the traced graph — do NOT
       normalise client-side)
     - shape [1, 3, 1024, 1024]
3. Run the CoreML package to get image_embeddings [1, 256, 64, 64].
4. Pool to a single 256-d vector per image, two ways (no mask/decoder is
   run — see CAVEAT below):
     - "global": mean over the full 64x64 grid
     - "center": mean over the center 50% of the grid (32x32 crop),
       a cheap proxy for "the tapped object fills most of the frame"
5. L2-normalize each pooled vector, compute cosine similarity for every
   image pair, split pairs into two buckets by whether the two images
   share an object-identity folder:
     - same-object pairs   (label = 1)
     - different-object pairs (label = 0)
6. Report: per-bucket mean/std of cosine similarity, Cohen's-d-style
   separability index d = (mean_same - mean_diff) / pooled_std, and a
   histogram (PNG via matplotlib if available, always a text histogram
   too) so a human can eyeball overlap.

CAVEAT — proxy pooling, not the real runtime pooling
======================================================
The actual app pools image_embeddings inside the *mask* returned by the
prompt decoder for a specific tap point. This script has no tap point or
mask (it works from plain photos, not from live tap-to-segment sessions),
so it substitutes global/center pooling as a cheap upper-bound-ish proxy.
If "center" pooling already fails to separate same/different object, real
mask-pooling (which is *more* precise, since it excludes background) can
only do as well or better — so a failure here is informative. A pass here
is NOT proof the real masked-pooling pipeline will work; it only justifies
building the masked-pooling version and re-testing.

RUN
===
/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeEverything/.venv_export/bin/python \
    shared/reid_feasibility_eval.py \
    --data_dir shared/reid_capture \
    --model JudgeE2/Segmentation/Models/MobileSAM_ImageEncoder_fp16_milfix.mlpackage \
    --out shared/reid_feasibility_results

Expected --data_dir layout:
    shared/reid_capture/
        mug_A/shot1.jpg  shot2.jpg  shot3.jpg
        mug_B/shot1.jpg  shot2.jpg          <- confusable pair with mug_A
        water_bottle/shot1.jpg  shot2.jpg  shot3.jpg
        ...
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png"}
ENCODER_SIDE = 1024


def letterbox_to_1024(img: Image.Image) -> np.ndarray:
    """Resize longer side to 1024, zero-pad to 1024x1024, top-left anchored.
    Returns float32 array [3, 1024, 1024], RGB, raw [0,255] scale (no
    mean/std normalisation — that is baked into the traced CoreML graph).
    """
    img = img.convert("RGB")
    w, h = img.size
    scale = ENCODER_SIDE / max(w, h)
    nw, nh = round(w * scale), round(h * scale)
    img_r = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (ENCODER_SIDE, ENCODER_SIDE), (0, 0, 0))
    canvas.paste(img_r, (0, 0))
    arr = np.asarray(canvas, dtype=np.float32)  # HWC, RGB, [0,255]
    arr = arr.transpose(2, 0, 1)  # CHW
    return arr


def load_dataset(data_dir: Path) -> dict[str, list[Path]]:
    objects: dict[str, list[Path]] = {}
    for sub in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        imgs = sorted(
            p for p in sub.iterdir() if p.suffix.lower() in IMG_EXTS
        )
        if imgs:
            objects[sub.name] = imgs
    return objects


def run_encoder(mlmodel, arr: np.ndarray) -> np.ndarray:
    """arr: [3,1024,1024] float32 -> returns [256,64,64] float32."""
    out = mlmodel.predict({"image": arr[None, ...]})
    emb = out["image_embeddings"]  # [1,256,64,64]
    return np.asarray(emb, dtype=np.float32)[0]


def pool_global(emb: np.ndarray) -> np.ndarray:
    return emb.reshape(256, -1).mean(axis=1)


def pool_center(emb: np.ndarray, frac: float = 0.5) -> np.ndarray:
    _, gh, gw = emb.shape
    ch, cw = int(gh * frac), int(gw * frac)
    y0, x0 = (gh - ch) // 2, (gw - cw) // 2
    crop = emb[:, y0 : y0 + ch, x0 : x0 + cw]
    return crop.reshape(256, -1).mean(axis=1)


def l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def separability(same: list[float], diff: list[float]) -> float:
    same_a, diff_a = np.array(same), np.array(diff)
    mean_diff_of_means = same_a.mean() - diff_a.mean()
    pooled_std = np.sqrt((same_a.var() + diff_a.var()) / 2)
    if pooled_std == 0:
        return float("nan")
    return float(mean_diff_of_means / pooled_std)


def text_histogram(same: list[float], diff: list[float], bins: int = 20) -> str:
    lo, hi = -1.0, 1.0
    edges = np.linspace(lo, hi, bins + 1)
    same_h, _ = np.histogram(same, bins=edges)
    diff_h, _ = np.histogram(diff, bins=edges)
    max_count = max(same_h.max(initial=0), diff_h.max(initial=0), 1)
    width = 40
    lines = ["cos_sim_range        same-object (S)          different-object (D)"]
    for i in range(bins):
        lo_e, hi_e = edges[i], edges[i + 1]
        s_bar = "S" * int(round(same_h[i] / max_count * width))
        d_bar = "D" * int(round(diff_h[i] / max_count * width))
        lines.append(
            f"[{lo_e:+.2f},{hi_e:+.2f}) {s_bar:<{width}} n={same_h[i]:<4} "
            f"{d_bar:<{width}} n={diff_h[i]:<4}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument(
        "--model",
        type=Path,
        default=Path(
            "JudgeE2/Segmentation/Models/MobileSAM_ImageEncoder_fp16_milfix.mlpackage"
        ),
    )
    ap.add_argument("--out", type=Path, default=Path("shared/reid_feasibility_results"))
    args = ap.parse_args()

    if not args.data_dir.exists():
        raise SystemExit(
            f"--data_dir {args.data_dir} does not exist. See the docstring "
            f"of this file, or shared/reid_feasibility_eval.md, for the "
            f"expected layout and capture protocol."
        )

    objects = load_dataset(args.data_dir)
    n_objects = len(objects)
    n_images = sum(len(v) for v in objects.values())
    print(f"[data] {n_objects} object identities, {n_images} images total")
    for name, imgs in objects.items():
        print(f"  - {name}: {len(imgs)} images")

    if n_objects < 2 or n_images < 4:
        raise SystemExit(
            "Not enough data to form both same-object and different-object "
            "pairs. Need >=2 object folders and >=2 images in at least two "
            "of them. See shared/reid_feasibility_eval.md capture protocol."
        )

    print(f"\n[model] loading {args.model}")
    import coremltools as ct  # local import: slow, only needed once args validated

    mlmodel = ct.models.MLModel(str(args.model))

    print("[encode] running encoder on every image (this is the slow step,"
          " ~0.9-1.3s/image on-device; on a Mac CPU/GPU it may be faster"
          " or slower depending on Neural Engine availability)")
    pooled_global: dict[str, np.ndarray] = {}
    pooled_center: dict[str, np.ndarray] = {}
    obj_of: dict[str, str] = {}
    for obj_name, imgs in objects.items():
        for img_path in imgs:
            key = f"{obj_name}/{img_path.name}"
            arr = letterbox_to_1024(Image.open(img_path))
            emb = run_encoder(mlmodel, arr)
            pooled_global[key] = l2norm(pool_global(emb))
            pooled_center[key] = l2norm(pool_center(emb))
            obj_of[key] = obj_name
            print(f"  encoded {key}")

    args.out.mkdir(parents=True, exist_ok=True)

    for variant_name, pooled in (("global", pooled_global), ("center", pooled_center)):
        same_sims: list[float] = []
        diff_sims: list[float] = []
        keys = list(pooled.keys())
        for k1, k2 in itertools.combinations(keys, 2):
            sim = cosine(pooled[k1], pooled[k2])
            if obj_of[k1] == obj_of[k2]:
                same_sims.append(sim)
            else:
                diff_sims.append(sim)

        d = separability(same_sims, diff_sims)
        summary = {
            "variant": variant_name,
            "n_same_pairs": len(same_sims),
            "n_diff_pairs": len(diff_sims),
            "same_mean": float(np.mean(same_sims)) if same_sims else None,
            "same_std": float(np.std(same_sims)) if same_sims else None,
            "diff_mean": float(np.mean(diff_sims)) if diff_sims else None,
            "diff_std": float(np.std(diff_sims)) if diff_sims else None,
            "separability_d": d,
        }
        print(f"\n=== pooling variant: {variant_name} ===")
        print(json.dumps(summary, indent=2))
        print()
        print(text_histogram(same_sims, diff_sims))

        (args.out / f"summary_{variant_name}.json").write_text(
            json.dumps(summary, indent=2)
        )
        (args.out / f"pairs_{variant_name}.json").write_text(
            json.dumps({"same": same_sims, "diff": diff_sims})
        )

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 4))
            bins = np.linspace(-1, 1, 41)
            ax.hist(diff_sims, bins=bins, alpha=0.6, label="different-object", color="tab:red")
            ax.hist(same_sims, bins=bins, alpha=0.6, label="same-object", color="tab:blue")
            ax.set_xlabel("cosine similarity of pooled image_embeddings")
            ax.set_ylabel("pair count")
            ax.set_title(
                f"re-ID feasibility probe ({variant_name} pooling)  "
                f"d={d:.2f}"
            )
            ax.legend()
            fig.tight_layout()
            fig.savefig(args.out / f"histogram_{variant_name}.png", dpi=150)
            plt.close(fig)
            print(f"[saved] {args.out / f'histogram_{variant_name}.png'}")
        except ImportError:
            print("[skip] matplotlib not available, text histogram above only")

    print(f"\n[done] results written to {args.out}")
    print(
        "\nHow to read separability_d: this is a Cohen's-d-style effect "
        "size = (mean(same) - mean(diff)) / pooled_std. Rough rule of "
        "thumb for THIS go/no-go decision (not a formal statistical test):"
        "\n  d < 0.5              : not separable, re-ID on raw embeddings"
        " is not viable -> do not build this feature on this signal."
        "\n  0.5 <= d < 1.2        : weak/marginal separation -> only viable"
        " with a learned threshold/projection AND generous slack (treat as"
        " a soft re-ranking signal, not a hard match/no-match gate)."
        "\n  d >= 1.2              : reasonably separable -> worth building"
        " a real masked-pooling + thresholded matcher and testing it live."
        "\nAlways eyeball the histogram too — d hides bimodal overlap that"
        " matters for a hard accept/reject UI decision."
    )


if __name__ == "__main__":
    main()
