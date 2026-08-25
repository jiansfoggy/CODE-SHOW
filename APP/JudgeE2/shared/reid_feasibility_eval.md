# Cross-session re-ID feasibility: data inventory + capture protocol

Scope: architect_output.md §25 asks for one piece of data collection —
does MobileSAM_ImageEncoder's `image_embeddings` (pooled) separate "same
physical object, different observation" from "different object"? This is
the only thing this file is about. It does not touch tasks.md,
architect_output.md, builder_progress.md, or debug_report.md.

## 1. Local data inventory — conclusion: not sufficient, do not fabricate a result

Checked (2026-08-24):

| Candidate source | Status | Why it doesn't work for this eval |
|---|---|---|
| `shared/S2-1.MP4`, `S3-1.MP4`, `S5-1.MP4` | **Confirmed deleted.** Not in this repo, not in the sibling `JudgeEverything` repo either (checked). `shared/prep_token0_frames.py`'s docstring still references them but the files themselves are gone. | N/A — gone |
| S1–S5 scenario data in `shared/model_plan.md` (ipad / mouse / chair_back / tumbler / canon_bag / tissue_box / bare marble / bare foam pad) | Only analysis numbers (iou/fill/stab tables) survive in the doc. No raw frames anywhere in the repo. | Analysis artifacts, not images |
| `shared/simu_record_0813.MP4` | **Exists, real device screen recording**, 828×1792, 41.7s @ 60fps, of the app's tap-to-segment UI in live use. Extracted and visually inspected frames (offsets 0s, 12s, 28s, 40s). | It is a **single fixed camera shot** of one static room (a person at a desk, a microwave, a water dispenser, a cabinet, a cubicle divider). The camera never moves and no second object/scene is deliberately captured. There is no way to build both "same object, different viewpoint" pairs *and* "different object" negative pairs from one static shot — everything in frame is either "the same background, unchanged" or "not really a discrete tapped object." Using this would manufacture a fake result, not measure anything. |
| `models/MobileSAM/**/*.jpg,png` (app/assets/picture1-6.jpg, MobileSAMv2/test_images/1.jpg,2.jpg), `models/yolov9/data/images/horses.jpg,zidane.jpg,bus.jpg`, `shared_v1/golden/bus.jpg` | Exist, real images. | Upstream demo/test images from the MobileSAM and YOLOv9 repos and our golden-test fixture. Not phone-camera captures, not this app's objects, not this app's lighting/distance/handheld distortion. Using stock photos of horses/zebras/pedestrians to judge whether the model recognizes *your desk objects across sessions* has no external validity — a pass or fail here says nothing about the real use case. |
| Device container Pin data (`.../scratchpad/p8_after/masks/*.png` etc., from prior device-container inspection sessions) | Exists. | Confirmed to be the segmentation **mask** output (binary/alpha blobs), not the underlying RGB camera frame. No pixel content to run the encoder on. |
| Anywhere else in the repo (`find . -iname "*.png" -o -iname "*.jpg" ...`, excluding vendor/model repos and `.xcassets`) | Checked. | Only the golden-test `bus.jpg` above; nothing else. |

**Bottom line: there is no real-content image set in this repo — or its
sibling `JudgeEverything` repo — that supports a meaningful "same object
vs different object" comparison.** I ran the evaluation script (below)
only as a smoke test against unrelated stock photos to confirm it
executes correctly end-to-end (correct shapes, no crashes, correct
output files). Those smoke-test numbers are **not** a discriminability
finding and must not be read as one — see the script's own output for
the disclaimer.

## 2. Capture protocol (minimal, do this next)

Goal: the smallest photo set that still gives real same-object and
different-object statistics, including hard (confusable) negatives —
not just "different object = obviously different category."

### 2.1 What to shoot

- **6 "easy" everyday objects** (reuse the categories from the original
  S1-S5 walkthroughs since they're already the app's known test set:
  e.g. water bottle/tumbler, mouse, tablet, chair back, tissue box, bag).
  **3 shots each** — different angle and/or distance and/or lighting, the
  way you'd actually re-enter a room and re-point the phone. **= 18
  images.**
- **3 "hard" confusable pairs** — two near-identical instances of the
  same product (two same-model water bottles, two same-color mugs, two
  identical chairs, etc.), **2 shots each instance**. This is the case
  that actually matters for a re-ID gate: category-level similarity is
  cheap, instance-level is the hard part. **= 3 pairs × 2 instances × 2
  shots = 12 images.**
- **Total: ~30 images, 12 distinct object identities.** This gives 12×3
  same-object pairs from the easy set (36) + 3×1 same-object pair per
  hard instance (6) = 42 same-object pairs, and hundreds of
  different-object pairs (the script subsamples nothing — it uses every
  pair, which is fine at this scale). That's enough to eyeball a
  histogram and compute a stable separability number; it is not enough
  for a publishable accuracy claim, and it doesn't need to be — this is
  a go/no-go check.

### 2.2 How to shoot: plain camera photos, not the app's Pin flow — with a caveat

Recommendation: use the iPhone's regular Camera app, held and framed the
way you'd point the phone for tap-to-segment (rear camera, ~0.3-1m from
the object, object filling a good fraction of the frame), rather than
going through the app's tap-to-segment/Pin creation flow.

Why not use the app's own Pin flow, even though it would match the
runtime input distribution more closely (letterbox geometry, ISP
pipeline, exact crop): the device-container inspection in the prior
session established that **Pins currently persist the mask, not the
source RGB frame** — there is nothing today that saves the tapped frame
next to a Pin. Making the app retain frames would be a Swift/storage
change, which is out of scope for this role (no App architecture, no
Swift). Flagging this as a possible follow-up for Architect/Builder if
they want a tighter proxy later; for now, plain photos are the
pragmatic unblock a person can do without any code changes.

Framing guidance to keep the proxy pooling in the script meaningful
(see §3 CAVEAT): fill roughly the center 50-70% of the frame with the
object, since the script's "center" pooling variant approximates masked
pooling by just cropping the middle of the embedding grid.

Save into:
```
shared/reid_capture/<object_name>/<shot_n>.jpg
```
e.g. `shared/reid_capture/mug_A/shot1.jpg`, `.../mug_B/shot1.jpg` for a
confusable pair.

(This directory is deliberately not created yet — it should only exist
once it has real photos in it, so an empty placeholder doesn't get
mistaken for "data is ready.")

## 3. Evaluation script — ready to run, no data yet

`shared/reid_feasibility_eval.py` is complete and was smoke-tested
end-to-end against a throwaway synthetic pair (unrelated stock photos,
just to prove the code path works) — it ran without errors and produced
correct-shaped output (JSON summaries, pairwise similarity dumps, PNG
histograms via matplotlib, plus a text histogram fallback). It is not
run against anything meaningful yet because the data described in §2
doesn't exist yet.

Once `shared/reid_capture/` has real photos:

```bash
/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeEverything/.venv_export/bin/python \
    shared/reid_feasibility_eval.py \
    --data_dir shared/reid_capture \
    --model JudgeE2/Segmentation/Models/MobileSAM_ImageEncoder_fp16_milfix.mlpackage \
    --out shared/reid_feasibility_results
```

What it does: letterboxes each photo to the encoder's exact traced input
contract (1024×1024, RGB, raw [0,255], normalization baked into the
CoreML graph — do not pre-normalize), runs the shipped
`MobileSAM_ImageEncoder_fp16_milfix.mlpackage` via coremltools, pools the
[256,64,64] embedding two ways (global mean, center-50% mean — see the
script's CAVEAT docstring on why there's no masked pooling here), L2
normalizes, computes cosine similarity for every image pair, splits into
same-object vs different-object buckets, and reports:

- per-bucket mean/std of cosine similarity
- a Cohen's-d-style separability index (with the script's own rule-of
  thumb thresholds for the go/no-go call)
- a text histogram always, a PNG histogram if matplotlib is present
  (confirmed present in the conversion venv: matplotlib 3.10.8)

Read the histogram, not just the d number — d hides bimodality that
matters for a hard accept/reject UI gate.

## 4. Honest status

No accuracy or separability number is reported here. The only thing
established in this session is: (a) the data needed to answer the
question does not currently exist locally, with the specific reasons
per source above, and (b) the capture protocol and evaluation script are
ready so that once ~30 photos exist, the next session can get a real
answer in one script run with no further design work.
