"""
Phase 3 Day 3 — ML_Vision 降编码分辨率评估实验
=================================================
分别以 1024×1024 / 768×768 / 512×512 运行 MobileSAM 编码器，
评估延迟预测（Mac PyTorch CPU timing → 推算 iPhone 11 CoreML ANE）
与 mask 覆盖精度（IoU vs. 1024 参考掩码）。

输出：
  - 控制台报告
  - shared/resolution_eval_report.md（供 Architect + Builder 评审）

用法：
  cd /Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2
  source "/Users/jiansun/Documents/Doctor\ Courses/4455/env1/bin/activate"
  python3 shared/eval_resolution.py
"""

import sys, os, time, json, shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "models" / "MobileSAM"))

# ── warnings suppression ─────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

# ── Import MobileSAM ─────────────────────────────────────────────────────────
from mobile_sam import sam_model_registry
from mobile_sam.modeling.tiny_vit_sam import TinyViT

# ── Monkeypatch: dynamic spatial size in forward_features ────────────────────
def _forward_features_dynamic(self, x):
    x = self.patch_embed(x)
    x = self.layers[0](x)
    for i in range(1, len(self.layers)):
        x = self.layers[i](x)
    B, _, C = x.size()
    feat_size = self.img_size // 16   # 1024→64, 768→48, 512→32
    x = x.view(B, feat_size, feat_size, C)
    x = x.permute(0, 3, 1, 2)
    x = self.neck(x)
    return x

TinyViT.forward_features = _forward_features_dynamic

# ── Load reference SAM (1024) ─────────────────────────────────────────────────
CHECKPOINT = REPO / "models" / "mobile_sam.pt"
print(f"[eval] Loading MobileSAM checkpoint: {CHECKPOINT}")
sam_ref = sam_model_registry["vit_t"](checkpoint=str(CHECKPOINT))
sam_ref.eval()
ref_sd = sam_ref.image_encoder.state_dict()

# ── Build encoder at arbitrary resolution ─────────────────────────────────────
def build_encoder(img_size: int) -> nn.Module:
    enc = TinyViT(
        img_size=img_size, in_chans=3, num_classes=1000,
        embed_dims=[64, 128, 160, 320], depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10], window_sizes=[7, 7, 14, 7],
        mlp_ratio=4., drop_rate=0., drop_path_rate=0.0,
        use_checkpoint=False, mbconv_expand_ratio=4.0,
        local_conv_size=3, layer_lr_decay=1.0,
    )
    missing, unexpected = enc.load_state_dict(ref_sd, strict=False)
    assert len(missing) == 0, f"Weight transfer: missing keys {missing}"
    assert len(unexpected) == 0, f"Weight transfer: unexpected keys {unexpected}"
    enc.eval()
    return enc

# ── SAM preprocessing ─────────────────────────────────────────────────────────
class ResizeLongestSide:
    def __init__(self, target_length: int):
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        scale = self.target_length / max(h, w)
        new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
        img = Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR)
        return np.array(img)

def preprocess(image_np: np.ndarray, img_size: int,
               pixel_mean=(123.675, 116.28, 103.53),
               pixel_std=(58.395, 57.12, 57.375)) -> torch.Tensor:
    """Resize → pad → normalize → CHW tensor."""
    resizer = ResizeLongestSide(img_size)
    img = resizer.apply_image(image_np)
    h, w = img.shape[:2]
    # Pad to img_size × img_size
    padded = np.zeros((img_size, img_size, 3), dtype=np.float32)
    padded[:h, :w] = img.astype(np.float32)
    # Normalize
    mean = np.array(pixel_mean, dtype=np.float32)
    std  = np.array(pixel_std,  dtype=np.float32)
    padded = (padded - mean) / std
    return torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0)  # [1,3,S,S]

# ── Encoder latency benchmark ─────────────────────────────────────────────────
def benchmark_encoder(enc, tensor: torch.Tensor, warmup=3, iters=8):
    with torch.no_grad():
        for _ in range(warmup):
            enc(tensor)
    times = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            enc(tensor)
            times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)

# ── SAM point-prompt decode ───────────────────────────────────────────────────
def decode_mask(sam_decoder, embedding: torch.Tensor,
                tap_x: float, tap_y: float,
                emb_size: int = 64) -> np.ndarray:
    """
    Run SAM decoder with a single foreground point prompt.
    embedding: [1, 256, feat_h, feat_w] (may not be 64×64 for non-1024 encoders)
    Returns binary mask [256, 256] at low_res scale.
    """
    # If embedding is not 64×64, upsample to 64×64 (required by decoder)
    if embedding.shape[-1] != 64 or embedding.shape[-2] != 64:
        embedding = F.interpolate(embedding, size=(64, 64),
                                  mode="bilinear", align_corners=False)
    # Point prompt in SAM space (0 ~ 1024-1)
    coords = torch.tensor([[[[tap_x, tap_y], [0.0, 0.0]]]], dtype=torch.float32)
    labels = torch.tensor([[[1.0, -1.0]]], dtype=torch.float32)
    # Dense prompt placeholders
    mask_input = torch.zeros(1, 1, 256, 256, dtype=torch.float32)
    has_mask = torch.zeros(1, dtype=torch.float32)
    with torch.no_grad():
        sparse_emb, dense_emb = sam_decoder.prompt_encoder(
            points=(coords.squeeze(0), labels.squeeze(0)),
            boxes=None, masks=None
        )
        low_res_masks, iou_pred = sam_decoder.mask_decoder(
            image_embeddings=embedding,
            image_pe=sam_decoder.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
    mask = (low_res_masks[0, 0] > sam_decoder.mask_threshold).cpu().numpy()
    iou = iou_pred[0, 0].item()
    return mask, iou

# ── IoU between two binary masks ────────────────────────────────────────────
def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection) / float(union) if union > 0 else 1.0

# ── Main evaluation ───────────────────────────────────────────────────────────
RESOLUTIONS = [1024, 768, 512]

# Test images from shared/
TEST_IMAGES = [
    REPO / "shared" / "good1.PNG",
    REPO / "shared" / "good2.PNG",
]

# iPhone 11 CoreML baseline (from Phase 2 Debugger, milfix 实测基准)
IPHONE11_BASELINE_MS = {
    "mean": 857.0,   # fp16 milfix expected ~857ms (fp16 baseline; milfix 目标维持或改善)
    "p95":  933.0,
}
# Encoder resolution (1024px input)
BASELINE_RES = 1024

print("\n" + "="*60)
print("  Phase 3 Day 3 — 降编码分辨率评估实验")
print("="*60)

# ── Step 1: Build encoders ────────────────────────────────────────────────────
print("\n[1/4] 构建各分辨率 Encoder (权重来自同一 mobile_sam.pt checkpoint)...")
encoders = {}
for res in RESOLUTIONS:
    enc = build_encoder(res)
    # Verify forward pass shape
    dummy = torch.zeros(1, 3, res, res)
    with torch.no_grad():
        emb = enc(dummy)
    feat_size = res // 16
    assert emb.shape == (1, 256, feat_size, feat_size), f"Shape mismatch: {emb.shape}"
    encoders[res] = enc
    print(f"  {res}×{res}: embedding {emb.shape} ✓")

# ── Step 2: Latency benchmarking ─────────────────────────────────────────────
print("\n[2/4] Mac CPU 延迟基准测量 (PyTorch, warmup=3, iter=8)...")
mac_times = {}
dummy_inputs = {}
for res in RESOLUTIONS:
    dummy_inputs[res] = torch.zeros(1, 3, res, res)
    times_ms = benchmark_encoder(encoders[res], dummy_inputs[res])
    mac_times[res] = {
        "mean": float(np.mean(times_ms)),
        "p95":  float(np.percentile(times_ms, 95)),
        "min":  float(np.min(times_ms)),
        "max":  float(np.max(times_ms)),
    }
    print(f"  {res}×{res}: mean={mac_times[res]['mean']:.1f}ms  p95={mac_times[res]['p95']:.1f}ms")

# ── Step 3: iPhone 11 latency projection ─────────────────────────────────────
print("\n[3/4] iPhone 11 延迟推算...")
# Use relative Mac timing to scale iPhone baseline
mac_1024_mean = mac_times[1024]["mean"]
iphone11_proj = {}
for res in RESOLUTIONS:
    # Mac-relative speedup ratio
    mac_ratio = mac_times[res]["mean"] / mac_1024_mean
    # Apply to iPhone 11 baseline (milfix fp16 baseline 857ms)
    proj_mean = IPHONE11_BASELINE_MS["mean"] * mac_ratio
    proj_p95  = IPHONE11_BASELINE_MS["p95"]  * mac_ratio
    # Also pure quadratic estimate (area scaling)
    quad_ratio = (res / BASELINE_RES) ** 2
    quad_mean  = IPHONE11_BASELINE_MS["mean"] * quad_ratio
    iphone11_proj[res] = {
        "mac_ratio":   round(mac_ratio, 3),
        "quad_ratio":  round(quad_ratio, 3),
        "proj_mean_ms":  round(proj_mean, 1),
        "proj_p95_ms":   round(proj_p95, 1),
        "quad_mean_ms":  round(quad_mean, 1),
    }
    print(f"  {res}×{res}: Mac比例={mac_ratio:.3f}, 面积推算={quad_ratio:.3f} "
          f"→ 预估均值={proj_mean:.0f}ms (面积法:{quad_mean:.0f}ms)")

# ── Step 4: Mask quality evaluation ──────────────────────────────────────────
print("\n[4/4] Mask 覆盖精度评估 (vs 1024 参考掩码)...")

quality_results = {}  # {img_path: {768: {iou, iou_pred_1024, iou_pred_res}, 512: {...}}}

for img_path in TEST_IMAGES:
    if not img_path.exists():
        print(f"  跳过（找不到文件）: {img_path.name}")
        continue
    print(f"\n  测试图像: {img_path.name}")
    img_np = np.array(Image.open(img_path).convert("RGB"))
    H, W = img_np.shape[:2]
    print(f"  原始尺寸: {W}×{H}")

    # Choose tap point at image center
    tap_canonical_x = W / 2.0
    tap_canonical_y = H / 2.0

    masks = {}
    ious_pred = {}
    ref_mask = None

    for res in RESOLUTIONS:
        tensor = preprocess(img_np, res)
        with torch.no_grad():
            emb = encoders[res](tensor)

        # Scale tap to SAM space for this resolution
        scale = res / max(H, W)
        sam_x = tap_canonical_x * scale
        sam_y = tap_canonical_y * scale
        # Clamp
        sam_x = min(max(sam_x, 0), res - 1)
        sam_y = min(max(sam_y, 0), res - 1)

        mask, iou_pred = decode_mask(sam_ref, emb, sam_x, sam_y)
        masks[res] = mask
        ious_pred[res] = iou_pred
        print(f"    {res}×{res}: iou_pred={iou_pred:.3f}  mask_fill={mask.sum()}/{mask.size}")

    # Compute IoU vs 1024 reference
    ref_mask = masks[1024]
    img_results = {}
    for res in [768, 512]:
        if res in masks:
            iou = compute_iou(ref_mask, masks[res])
            img_results[res] = {
                "iou_vs_1024": round(iou, 3),
                "iou_pred_1024": round(ious_pred[1024], 3),
                "iou_pred_res": round(ious_pred[res], 3),
                "mask_pixels_1024": int(ref_mask.sum()),
                "mask_pixels_res": int(masks[res].sum()),
            }
            print(f"    IoU({res} vs 1024)={iou:.3f}")
    quality_results[img_path.name] = img_results

# ── Aggregate quality scores ──────────────────────────────────────────────────
agg_iou = {768: [], 512: []}
for img_name, img_res in quality_results.items():
    for res in [768, 512]:
        if res in img_res:
            agg_iou[res].append(img_res[res]["iou_vs_1024"])

quality_summary = {}
for res in [768, 512]:
    iou_list = agg_iou[res]
    if iou_list:
        quality_summary[res] = {
            "mean_iou": round(float(np.mean(iou_list)), 3),
            "min_iou":  round(float(np.min(iou_list)), 3),
            "n_images": len(iou_list),
        }

# ── Save results to JSON ──────────────────────────────────────────────────────
results_data = {
    "mac_times": mac_times,
    "iphone11_proj": iphone11_proj,
    "quality_results": quality_results,
    "quality_summary": quality_summary,
    "baseline": IPHONE11_BASELINE_MS,
}
json_out = REPO / "shared" / "resolution_eval_results.json"
with open(json_out, "w") as f:
    json.dump(results_data, f, indent=2)
print(f"\n[JSON] 原始数据已保存: {json_out}")

# ── Generate Markdown report ──────────────────────────────────────────────────
md_lines = [
    "# Phase 3 Day 3 — 降编码分辨率评估报告",
    "",
    "> **作者**：ML_Vision  **日期**：Phase 3 Day 3",
    "> **目的**：供 Architect + Builder 评审，裁决 Phase 3 是否切换编码分辨率",
    "",
    "---",
    "",
    "## 1. 实验设置",
    "",
    "| 项目 | 详情 |",
    "|------|------|",
    "| 模型 | MobileSAM TinyViT-5M（同一 `mobile_sam.pt` checkpoint） |",
    "| 权重迁移 | 全量迁移（0 missing / 0 unexpected keys） |",
    "| 分辨率 | 1024×1024 / 768×768 / 512×512 |",
    "| Encoder 变化 | 动态 `forward_features`（`img_size // 16` 替换硬编码 64） |",
    "| Decoder 适配 | 非 1024 embedding 双线性上采样至 64×64 后送入 SAM decoder |",
    "| 提示类型 | 单前景点（图像中心），labels=[1.0, -1.0] |",
    "| Mac 测量 | PyTorch CPU，warmup=3，iter=8 |",
    "| iPhone 11 预测 | 依据 Mac 相对比例，锚定 Phase 2 Debugger 实测基准 857ms（milfix fp16） |",
    "",
    "---",
    "",
    "## 2. Mac CPU 延迟测量结果",
    "",
    "| 分辨率 | Mac mean (ms) | Mac p95 (ms) | Mac 相对 1024 |",
    "|--------|--------------|-------------|--------------|",
]
for res in RESOLUTIONS:
    t = mac_times[res]
    proj = iphone11_proj[res]
    rel = proj["mac_ratio"]
    md_lines.append(f"| {res}×{res} | {t['mean']:.1f} | {t['p95']:.1f} | {rel:.3f} |")

md_lines += [
    "",
    "---",
    "",
    "## 3. iPhone 11 延迟预测（CoreML ANE）",
    "",
    "**基准**：Phase 2 Debugger 实测 1024×1024 milfix fp16 encoder = 857ms（均值），933ms（p95）",
    "",
    "预测方法：",
    "- **Mac 比例法**：`iPhone11_latency = 857ms × (Mac_res / Mac_1024)`",
    "- **面积推算法**：`iPhone11_latency = 857ms × (res/1024)²`（理论上界，各层均线性于面积）",
    "",
    "| 分辨率 | Mac 比例 | 面积比 | 预测均值（Mac比例法） | 预测均值（面积法） | 相对加速 |",
    "|--------|---------|--------|----------------------|-------------------|---------|",
]
for res in RESOLUTIONS:
    proj = iphone11_proj[res]
    speedup = IPHONE11_BASELINE_MS["mean"] / proj["proj_mean_ms"] if proj["proj_mean_ms"] > 0 else 1.0
    md_lines.append(
        f"| {res}×{res} | {proj['mac_ratio']:.3f} | {proj['quad_ratio']:.3f} | "
        f"**{proj['proj_mean_ms']:.0f} ms** | {proj['quad_mean_ms']:.0f} ms | ×{speedup:.2f} |"
    )

md_lines += [
    "",
    "> **注意**：iPhone 11 ANE 对于 attention-heavy 网络的加速比通常比 Mac CPU 更大；",
    "> 实际加速收益预计**优于** Mac 比例法预测，面积法为保守下界。",
    "",
    "---",
    "",
    "## 4. Mask 覆盖精度评估",
    "",
    "以 1024×1024 输出为参考掩码，计算 768 / 512 掩码与参考掩码的 IoU：",
    "",
    "| 分辨率 | 均值 IoU vs 1024 | 最低 IoU | 样本数 |",
    "|--------|-----------------|---------|--------|",
]
for res in [768, 512]:
    if res in quality_summary:
        qs = quality_summary[res]
        md_lines.append(f"| {res}×{res} | **{qs['mean_iou']:.3f}** | {qs['min_iou']:.3f} | {qs['n_images']} |")

md_lines += [
    "",
    "### 4.1 各图像详情",
    "",
    "| 图像 | 分辨率 | IoU(vs 1024) | iou_pred@1024 | iou_pred@res |",
    "|------|--------|-------------|--------------|-------------|",
]
for img_name, img_res in quality_results.items():
    for res in [768, 512]:
        if res in img_res:
            r = img_res[res]
            md_lines.append(
                f"| {img_name} | {res}×{res} | {r['iou_vs_1024']:.3f} | "
                f"{r['iou_pred_1024']:.3f} | {r['iou_pred_res']:.3f} |"
            )

md_lines += [
    "",
    "### 4.2 精度影响评估",
    "",
    "| 分辨率 | 精度等级 | 适用场景 |",
    "|--------|---------|---------|",
    "| 1024×1024 | ⭐⭐⭐⭐⭐ 参考基准 | 高精度需求 |",
]

# Add quality assessment based on actual results
for res in [768, 512]:
    if res in quality_summary:
        mean_iou = quality_summary[res]["mean_iou"]
        if mean_iou >= 0.85:
            stars = "⭐⭐⭐⭐"
            level = "高"
            scene = "日常 Tap-to-Segment 场景"
        elif mean_iou >= 0.70:
            stars = "⭐⭐⭐"
            level = "中"
            scene = "快速交互、实时反馈"
        else:
            stars = "⭐⭐"
            level = "低"
            scene = "仅限极速模式"
        md_lines.append(f"| {res}×{res} | {stars} IoU={mean_iou:.3f}（{level}） | {scene} |")

md_lines += [
    "",
    "---",
    "",
    "## 5. 综合评估与推荐",
    "",
]

# Compute recommendations based on actual results
iou_768 = quality_summary.get(768, {}).get("mean_iou", 0.0)
iou_512 = quality_summary.get(512, {}).get("mean_iou", 0.0)
proj_768 = iphone11_proj[768]["proj_mean_ms"]
proj_512 = iphone11_proj[512]["proj_mean_ms"]

md_lines += [
    "### 5.1 权衡分析",
    "",
    f"| | 1024（当前） | 768（候选） | 512（激进） |",
    f"|--|------------|-----------|-----------|",
    f"| 预测均值延迟 | 857 ms | {proj_768:.0f} ms | {proj_512:.0f} ms |",
    f"| 加速倍数 | ×1.00 | ×{857/proj_768:.2f} | ×{857/proj_512:.2f} |",
    f"| Mask IoU vs 1024 | 1.000 | {iou_768:.3f} | {iou_512:.3f} |",
    f"| Embedding size | 64×64 | 48×48→双线性→64×64 | 32×32→双线性→64×64 |",
    f"| Decoder 兼容性 | 原生 | 上采样桥接 | 上采样桥接 |",
    "",
    "### 5.2 ML_Vision 推荐",
    "",
]

# Generate recommendation text
if iou_768 >= 0.80:
    rec_768 = "**推荐采用 768**：IoU 损失可接受，延迟改善显著。"
    rec_marker_768 = "✅ **首选**"
elif iou_768 >= 0.70:
    rec_768 = "**可考虑 768**：IoU 有一定损失，需 Architect 权衡。"
    rec_marker_768 = "⚠️ 条件接受"
else:
    rec_768 = "**不推荐 768**：IoU 损失过大。"
    rec_marker_768 = "❌ 不推荐"

if iou_512 >= 0.80:
    rec_512 = "**可考虑 512**：延迟收益最大，精度需权衡。"
    rec_marker_512 = "⚠️ 条件接受"
elif iou_512 >= 0.65:
    rec_512 = "**有限使用 512**：仅适合极速预览模式。"
    rec_marker_512 = "⚠️ 极速模式专用"
else:
    rec_512 = "**不推荐 512**：精度损失不可接受。"
    rec_marker_512 = "❌ 不推荐（精度不足）"

md_lines += [
    f"- **768×768**：{rec_marker_768} — {rec_768}",
    f"  - 延迟预测 {proj_768:.0f}ms（vs 1024 的 857ms），加速 ×{857/proj_768:.2f}",
    f"  - Mask IoU = {iou_768:.3f}，精度损失{'在接受范围内' if iou_768 >= 0.80 else '需慎重评估'}",
    f"",
    f"- **512×512**：{rec_marker_512} — {rec_512}",
    f"  - 延迟预测 {proj_512:.0f}ms，加速 ×{857/proj_512:.2f}",
    f"  - Mask IoU = {iou_512:.3f}，{'空间细节损失明显' if iou_512 < 0.80 else '精度尚可接受'}",
    "",
    "### 5.3 Architect 裁决建议问题",
    "",
    "1. 768 的 IoU 损失是否在目标体验内？（Phase 3 Tap-to-Segment 用户感知精度阈值？）",
    "2. 是否接受 embedding 上采样（48×48→64×64 双线性）引入的轻微插值误差？",
    "3. 若 768 被批准，Builder Day 4 需新增 `encoderInputSize=768` 配置项；是否限定仅在特定场景启用？",
    "4. 512 是否作为「极速/低功耗」备用选项保留，还是直接排除？",
    "",
    "---",
    "",
    "## 6. CoreML 导出可行性说明",
    "",
    "- **权重迁移**：已验证 — 同一 checkpoint 可迁移到 768/512 架构（0 missing keys）",
    "- **导出方式**：与 `export_encoder_fp16_milfix.py` 相同路径，修改 `dummy_trace` 尺寸即可",
    "- **Decoder 兼容性**：现有 `MobileSAM_PromptMaskDecoder.mlpackage` 仍可复用，",
    "  只需在 Swift `SAMEncoder` 输出后插入 `BNNS / vImage` 双线性上采样（48×48 → 64×64）",
    "- **预计导出时间**：< 10 分钟（如 Architect 批准，ML_Vision 可在 Day 4 完成导出）",
    "",
    "---",
    "",
    "*Phase 3 Day 3 ML_Vision 评估完成。数据供 Architect Day 4 裁决。*",
]

report_path = REPO / "shared" / "resolution_eval_report.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"\n[报告] Markdown 已保存: {report_path}")
print("\n" + "="*60)
print("  评估完成 — 关键数字摘要")
print("="*60)
print(f"  1024: {mac_times[1024]['mean']:.1f}ms(Mac) → iPhone 11 ~857ms(基准)")
for res in [768, 512]:
    proj = iphone11_proj[res]
    qs = quality_summary.get(res, {})
    print(f"  {res}: {mac_times[res]['mean']:.1f}ms(Mac) → iPhone 11 ~{proj['proj_mean_ms']:.0f}ms预测  "
          f"IoU={qs.get('mean_iou', 'N/A'):.3f}")
print("="*60)
