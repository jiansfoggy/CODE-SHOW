# Phase 3 Day 4 — Debugger 剩余两条 checkbox 的真机测试手册

> 本文件：`/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/shared/phase3_day4_test_checklist.md`
>
> 目标：补齐 `tasks.md` Day 4 Debugger 小节仍未勾选的两条
> - ☐ 确认 mask 在 tap 点附近正确显示 → **Session A（评分）**
> - ☐ 确认 Phase 2 YOLO 路径没有因 Tap 分割 pipeline 出现 FPS 回退 → **Session B（性能）**
>
> 编写日期：2026-07-26 ｜ 依据：2026-07-26 真机 session 24 次 tap 日志 + 15 张截图分析
> 设备：iPhone（A13 / h15 ANE），相机 1080×1920，preview bounds 414×896

---

## 路径约定

⚠️ **注意**：本 git 仓库根目录是 `/Users/jiansun/Documents/PostDoc/CODE-SHOW`，
但 Xcode 工程根目录是它下面的 `APP/JudgeE2`。两者不同，写相对路径极易出错。
**本文件中所有路径一律以下面的 `$PROJ` 为准。**

```
PROJ=/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2
```

| 文中简称 | 完整路径 |
|---|---|
| 本 checklist | `$PROJ/shared/phase3_day4_test_checklist.md` |
| `tasks.md` | `$PROJ/shared/tasks.md` |
| `debug_report.md` | `$PROJ/shared/debug_report.md` |
| `architect_output.md` | `$PROJ/shared/architect_output.md` |
| `CameraManager.swift` | `$PROJ/JudgeE2/Detection/CameraManager.swift` |
| `MaskRenderer.swift` | `$PROJ/JudgeE2/Segmentation/MaskRenderer.swift` |
| `SAMDecoder.swift` | `$PROJ/JudgeE2/Segmentation/SAMDecoder.swift` |
| `SAMEncoder.swift` | `$PROJ/JudgeE2/Segmentation/SAMEncoder.swift` |
| `TouchHandler.swift` | `$PROJ/JudgeE2/Interaction/TouchHandler.swift` |
| `PointPromptBuilder.swift` | `$PROJ/JudgeE2/Interaction/PointPromptBuilder.swift` |
| `FrameGeometry.swift` | `$PROJ/JudgeE2/Interaction/FrameGeometry.swift` |
| `CameraPreview.swift` | `$PROJ/JudgeE2/Detection/CameraPreview.swift` |
| `ContentView.swift` | `$PROJ/JudgeE2/UI/ContentView.swift` |
| Xcode 工程 | `$PROJ/JudgeE2/JudgeE2.xcodeproj` |

> ⚠️ 同级还有一个 `$PROJ/JudgeE2 copy/` 目录，**是旧副本，不要改、不要参考**。

**本轮两个 session 的日志请存到**（目录已存在，直接写入即可）：
```
$PROJ/shared/perf_session_YYYYMMDD.log      # Session B 性能
$PROJ/shared/tap_scoring_YYYYMMDD.log       # Session A 评分
```

编译命令（在任意目录下都可用绝对路径执行）：
```bash
xcodebuild -project /Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/JudgeE2/JudgeE2.xcodeproj \
           -scheme JudgeE2 -destination 'generic/platform=iOS Simulator' build
```

---

## 0. 前置：Builder 必须先合入的 3 处日志改动

> ✅ **M1 / M2 / M3 已于 2026-07-26 合入，BUILD SUCCEEDED，未 commit。**
> 涉及文件：`CameraManager.swift`、`MaskRenderer.swift`、`SAMDecoder.swift`、`TouchHandler.swift`、`CameraPreview.swift`、`ContentView.swift`。
> 未触碰任何分割算法逻辑 / 阈值 / 候选选择规则（60%·85% 上限、形状门槛、`iou_pred >= 0.1` 闸门、flood fill 行为均不变）。
> 实现要点见本节末尾「合入后的实际行为」。

这 3 处不合入，Session A 采到的数据**无法与画面对应**，等于白采。Session B 不依赖它们。

| # | 改动 | 位置 | 为什么必须 | 规模 |
|---|---|---|---|---|
| **M1** | 所有 tap 相关 `print` 前缀加 tap 序号，改成 `[TAP#\(gen)] ...` | `CameraManager.swift` — `handleTap` / `tapEncodeAndDecode` / `tapDecodeWithPoint` **函数签名里已经有 `gen: Int` 参数**，只是没打印出来 | 日志既无时间戳也无序号，多次 tap 之间无法区分 | 纯格式化，零新增状态 |
| **M2** | 屏幕角落显示当前 tap 序号 `#N` | `CameraManager` 加 `@Published var lastTapIndex: Int`，在 `handleTap` 发布 UI 状态的 `main.async` 块里赋 `myGen`；`ContentView.swift` 角落显示 | **整个评分方案的关键** — 画面自带序号，录屏与日志的对齐问题彻底消失 | ~5 行 |
| **M3** | `[TAP] mask displayed` 行打印**被选中候选**的通道号 / iou / 面积 / fill | `MaskRenderer.swift` 把 `sel.ch / sel.iou / sel.comp.count / sel.comp.fill` 回传；`CameraManager.swift:730` 停止用 `result.iouPreds.max()` | 当前打印的是三个候选的 **max**，不是选中那个的值 → 评分表「自动列」会填错（本项目已踩过一次） | ~10 行 |

**可选（非必须）**
- **O1**：`setMode` 里打一条 `=== MODE SWITCH → \(mode) ===` 分隔日志。不加也能靠丢弃第一行统计来切分，加了省约 5 分钟人工整理。
- **O2**：把 `CameraManager.swift:761` 的 `e2eMs` 计时终点挪进主线程 mask 发布之后。当前终点在发布**之前**，低估真实"手指离开→看到 mask"延迟。做了的话 Session B 可顺便把延迟数据重采干净。**是否本轮做由 Architect 定。**

**不要做的改动**
- ✗ mask 存盘调试开关 —— 评的就是用户在屏幕上看到的东西，录屏即 ground truth；存盘会在 `decoderQueue` 上引入磁盘 IO，污染 `tap→mask` 延迟。
- ✗ 临时关掉 tap 模式下 YOLO 的 3 帧降频 —— 理由见 §2.6。

### 合入后的实际行为（2026-07-26）

**一次 tap 的完整日志样例**（decode-only 快路径）：

```
[TAP#12] view=(207.0,448.0) normalized=(0.5000,0.5000) canonical=(540.0,960.0) orientation=90 mirrored=false
[TAP#12] reuse cached embedding (ttlValid=Y geoChanged=N) → decode point=(540.0,960.0)
[SEG][TAP#12] decode latency: 118.42 ms iou_preds: 0.956, 0.392, 0.771
[TAP#12] [MASK] adaptive cap: base=-5.612 → p30=-2.104 (large-object guard)
[TAP#12] candidates: [ch0=18422px bbox=180x142 fill=0.72 iou=0.96 | ch1=1204px bbox=44x38 fill=0.72 iou=0.39] → picked ch1 area=1204
[TAP#12] Mask logits range: min=-24.1, max=18.3 | mean=-8.2, std=5.1, thresh=-2.104 | nonzero=1204/65536 | shape=[1, 3, 256, 256]
[TAP#12] cropRect=(...) outRect=(...) drawRect=(...) nonzero=1204
[TAP#12] mask displayed — sel=ch1 iou=0.392 area=1204px fill=0.72 | gate iou_pred(max)=0.956 | tap→mask 143.7 ms (decode-only)
```

慢路径把 reuse 那行换成 `[TAP#N] encode + decode (reason=…)` 与 `[TAP#N] encode done … → decode`。
所有失败出口也都带 `#N`：`no valid candidate at tap` / `best iou_pred=… < 0.1` / `superseded before/during/after …` / `busy timeout`。

**M3 的两个 iou 现在同时打印，不要混淆**
- `sel=chN iou=…` —— **被选中候选**的真实 iou，评分表「选中(自动)」列填这个
- `gate iou_pred(max)=…` —— 三候选最大值，仅用于 `>= 0.1` 的"整次 decode 是否值得解读"闸门
- 合入前只打印后者，导致真机日志被误读过一次（TAP#12 实际选中 iou=0.39，却显示 0.956）

**故意不带序号的日志行**（不属于任何一次 tap，保持裸 `[TAP]` 前缀以免污染 grep 链）：
`[TAP] background embedding refresh …` / `[TAP][AB] encoder stats` / `[TAP] ignored — geometry not ready` / `[TAP] double-tap → clearAll`

**序号会跳号是正常的**：双击清除也消耗一个序号，打印为 `[TAP#N] pipeline received clearAll`。
所以 `#N` 序列中的空缺可归因于清除动作，而不是丢失的 tap。

**一处已知的非确定性**：`[TAP#N] view=…` 这条几何日志在 `handleTap` 返回后才打印，理论上可能排在 videoQueue 的第一行日志之后。两条都带 `#N`，grep 链完整，只是这两行的先后顺序不保证。

---

## 1. 两个 Session 必须分开跑

| Session | 目的 | 屏幕录制 | 依赖改动 | 时长 |
|---|---|---|---|---|
| **B（性能）** | 补 checkbox 5 | **关** | 无 | 25–30 分钟 |
| **A（评分）** | 补 checkbox 2 | **开** | M1 + M2 + M3 | 15–20 分钟 |

**顺序：同一次真机连线，先 B 后 A。**
录屏会占用 GPU 与视频编码器，**会污染延迟和 FPS**。所以：
1. 冷机起测 Session B（不录屏）
2. 跑完冷却 10 分钟
3. 再跑 Session A（开录屏）—— 本 session 的 `tap→mask` 数字**全部作废**，不要用来评估延迟

---

## 2. Session B — YOLO 无回退性能测试

> **结论：不需要 Builder 改代码，纯靠操作即可采到。** 但下面三条纪律缺一组数据就作废。

### 2.1 准备清单（做一次，约 5 分钟）

- [ ] 关闭**低电量模式**（设置 → 电池）—— 它会直接降 CPU/GPU 频率，最常见的污染源
- [ ] 电量充到 **60%–90%**，**拔掉充电线**，全程不插电（充电发热严重）
- [ ] 开**飞行模式**（或至少勿扰），上滑划掉所有后台 App
- [ ] 屏幕亮度固定 50%，**关闭自动亮度**
- [ ] **确认没有开屏幕录制**
- [ ] 手机**架在支架上**（三脚架或靠书堆），对准一个中等复杂度的固定场景（能稳定检出 3–5 个物体即可，如桌面上的杯子 / 键盘 / 鼠标）
- [ ] **四组全程用同一场景、同一机位，中途绝不移动手机**
- [ ] Xcode 连线，**运行目标选真机（不是模拟器）**，按 Run 重新安装，清空 console（Debug area 左下垃圾桶图标）
- [ ] ⚠️ **自检：确认手机上跑的是当前构建**（2026-08-01 曾因此白跑一轮，见下）
- [ ] 打开 Settings 面板里的 **"Perf Quiet Log"** 开关
- [ ] 手机**静置冷却 10 分钟**（不运行 app）—— 但见 §2.3 步骤 0 的预热判据，冷却后仍需跑到平台期才开始采

> **⚠️ 自检为什么必须做**
> `xcodebuild -destination 'generic/platform=iOS Simulator'` 只做**编译验证**，**不会安装到手机上**。
> Builder 报告 "BUILD SUCCEEDED" ≠ 你手机上的 app 已更新。
>
> **判据（两条都要满足）**：
> 1. app 启动时 console 出现 `=== MODE SWITCH → detectionOnly ===`
>    （`ContentView.onAppear` 会调 `setMode`，新构建光启动就该打这条）
> 2. Settings 面板里能看到 **"Perf Quiet Log"** 开关
>
> 任一条不满足 → 手机上是旧构建，**立即停止采样**，回 Xcode 选真机重新 Run。

### 2.2 四组对照

| 组 | 模式 | 说明 |
|---|---|---|
| **A** | detectionOnly | 纯 YOLO 天花板（`setMode` 会清空 embeddingCache，无任何 SAM 活动）。仅作参照 |
| **B** | segmentation | **Phase 2 基线** —— tasks.md 那条要对比的就是它 |
| **C** | tapToSegment，**全程不点** | 只有 background embedding refresh 在跑 |
| **D** | tapToSegment，**每 5 秒点一次** | 最坏情况，tap encode/decode 与 YOLO 争资源 |

**正式判据是 B vs C 和 B vs D。** A 只用来看 SAM 整体开销有多大。

### 2.3 采样步骤（交错顺序，约 25 分钟，中途不重启 app）

#### 步骤 0 — 预热到平台期（判据驱动，不是固定 60 秒）

> **为什么不能用固定 60 秒**：2026-08-01 第一轮实测，全程 detectionOnly 不换模式，
> Infer mean 却单调爬升 **179.40 → 191.77 → 200.83（+11.9%）**。
> 而判定「有回退」的阈值是 +10% —— **光靠预热漂移就能刷出一个假阳性**。

- [ ] 0. detectionOnly 持续跑，盯着 `Inference time stats` 行
- [ ] 0a. **判据：连续两行的 mean 相差 < 3%** 才算进入平台期。之前的全部丢弃
- [ ] 0b. 冷机起测通常需要 **5–8 个窗口（4–6 分钟）**；若设备本来就热，2–3 个窗口即可

> 2026-08-01 第二轮（接着第一轮跑、设备已热）实测达到平台期后的表现，可作对照：
> `213.38 → 210.56 → 209.36 → 206.63 → 207.85`，后四行彼此相差仅 1.9% ✅

#### 步骤 1–8 — 交错采样

> **为什么交错而不是分块**：原先的 A,A,A,B,B,B 顺序会把全部漂移累加到后面的组上。
> 交错后每次比较都对着一个**时间上紧邻**的基线，线性漂移在每一对里自动抵消。
> 这比"正反序两轮"更稳，也更省时间。

每个**采样单元** = 切到该模式 → **丢弃第 1 行统计**（必然被污染，见 §2.4）→ 采 **1 行**有效数据。
即每个单元要等 **2 行统计**，segmentation 系约 85 秒 / 单元。

- [ ] 1. 【B】切到 segmentation，等 warmup 结束，丢 1 行 + 采 1 行
- [ ] 2. 【C】切到 tapToSegment，**不点屏幕**，丢 1 行 + 采 1 行
- [ ] 3. 【B】切回 segmentation，丢 1 行 + 采 1 行
- [ ] 4. 【D】切到 tapToSegment，**每 5 秒点一次屏幕**（点完不用清除），丢 1 行 + 采 1 行
- [ ] 5. 重复步骤 1–4 **再两轮**，最终每组各得 **3 行**有效数据（B 得 6 行）
- [ ] 6. 【A】最后切 detectionOnly，丢 1 行 + 采 3 行（天花板参照，不参与判定）
- [ ] 7. 全程留意机身温感；**烫手就暂停**，切回 detectionOnly 静置 5 分钟再继续
- [ ] 8. Xcode console 全选（⌘A）复制，存成 `$PROJ/shared/perf_session_YYYYMMDD.log`
      （完整路径 `/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/shared/perf_session_YYYYMMDD.log`）

> **组 A 可以复用已有数据**：2026-08-01 第二轮已采到干净的 detectionOnly 平台期基线
> —— Infer mean **208.6 ms** / p95 **216.2 ms** / Post 中位 **189.2 ms** / 内存 **197 MB**（丢弃首行后的 4 行均值）。
> 若本轮热态相近可直接引用，不必重采。

**为什么中途不重启 app**：重启会重新触发约 8.6 秒的 encoder warmup 和 CoreML 冷编译，把第一批帧全部污染。切模式（`setMode`）已经足够干净。

#### ⏱️ 关于"日志好像卡住了"

**每条统计行要等 100 帧。** segmentation / tapToSegment 约 2.4 FPS → **约 42 秒一行**；
detectionOnly 约 2.9 FPS → 约 34 秒一行。中间看不到新统计行是正常的。

2026-08-01 两轮都误判成"日志不再更新"，实际分别只差 **20 秒**和 **9 秒**。
判断是否真的卡住，**数 `Pre=…` 行的条数**（每帧一条），不要盯着统计行等。

### 2.4 ⚠️ 最容易踩的坑：每次切模式后必须丢弃第一行统计

**已在代码中核实**：`inferenceTimesMs` 只在攒满 100 帧时清空（`CameraManager.swift` 的 `removeAll`），**`setMode` 全程没有碰它**。所以切模式瞬间 buffer 里残留的几十帧属于**上一个模式**，会和新模式的帧混进同一行统计。

**2026-08-01 实测到的真实混合窗口**：第 5 行统计打在第 499 帧，之后又跑了 **34 帧 detection** 才切进 segmentation，切换后再跑 **46 帧**。这个窗口是 34 : 46 的混合体 —— 若它凑满 100 帧打出来，数字将毫无意义。

→ 丢弃每次切换后的第一行即可完全规避，**不需要改代码**。
→ **实践含义：每个采样单元要等两行统计**（第 1 行丢、第 2 行才是数据），这已计入 §2.3 的时间估算。

> **切换点的位置是随机的**：混合比例取决于你在窗口的哪一帧动手。
> 有时运气好（2026-07-26 那次切换正好落在 flush 边界后 1 帧，第一行几乎纯净），
> 但**不要指望运气** —— 无论如何都丢第一行。

### 2.5 比什么指标（不要比 FPS）

**为什么不比 FPS**：`FPS` 在两个模式下**口径不同，本质不可比**。
- `PerfLogger` 里的 `FPS:` = captureOutput 被调用次数/秒，受丢帧影响
- `FPS=` = `1000/Total`，是派生量
- tapToSegment 下 YOLO 被 `CameraManager.swift:1150-1156` 降到每 3 帧一次，跳过的帧不打 `Total` 行 → 分布天然不同

**要比的是 `Inference time stats` 这一行**（`CameraManager.swift:1166-1173`）。它统计 `model.prediction(image:)` 的单次纯推理耗时，**与调用频率完全无关** —— 这才是"YOLO 有没有被 SAM 抢走 ANE/GPU"的干净度量。

> **统计口径**：`Inference time stats` 是**每满 100 帧打一行然后清零**的 tumbling window，不是滑窗。每行即一个独立的 100 帧样本，可直接当一个数据点。

| 指标 | 来源日志行 | 判定「有回退」的阈值 | 备注 |
|---|---|---|---|
| **YOLO 推理 mean** | `Inference time stats (n=100): mean=` | 相对基线 **+10%** | 2026-07-26 实测 225–230 ms，阈值约 +23 ms |
| **YOLO 推理 p95** | 同行 `p95=` | 相对基线 **+15%** | 实测 256–284 ms，p95 波动大故放宽 |
| **Post 分量 mean** | `Pre=..Infer=..Post=..` 的 Post | 相对基线 **+20%** | Post 正比于检出框数 → **必须同场景**才有意义 |
| **内存峰值** | `FPS: .. \| Memory: .. MB` | 峰值超基线 **+400 MB**，或呈单调增长（泄漏） | SAM 常驻约 +170 MB |

### 2.6 控制变量三要点

**(1) 热态 —— 靠预热到平台期 + 交错采样抵消**
这是本测试**最大的风险项**：2026-08-01 实测，单一模式内部的漂移（+11.9%）就已经超过判定阈值（+10%）。
- **先预热到平台期**（§2.3 步骤 0，判据：连续两行 mean 相差 <3%），不要用固定时长
- **再交错采样 B,C,B,D,B,C,B,D…**（§2.3），每个 C/D 都对着时间上紧邻的 B 比较，线性漂移在每一对里抵消
- 判定时**用相邻的那对**，不要拿第 1 个 B 去比第 3 个 D
- 记录每组开始时的机身温感；烫手就暂停，切回 detectionOnly 静置 5 分钟再继续
- 若三对 B-C（或 B-D）的结论互相矛盾 → 说明漂移仍未控制住，**判定为「数据无效」，重采**，不要强行下结论

**(2) 场景 —— 必须完全固定**
`Post` 是 `decodeDetections` + `classAwareNMS` 的 CPU 耗时，直接正比于 `raw_in_boxes` 数量。换个场景检出框从 3 变 8，Post 就翻倍，与 SAM 毫无关系。
→ 采完后核对四组日志里的 `det[N]` 类别与数量是否大体一致；差异很大说明画面动过，该组作废重采。

**(3) 3 帧降频 —— 不要关掉**
- `inferMs` 只测单次 `model.prediction` 耗时，跑得频不频与这个数无关
- **每 3 帧一次就是 tapToSegment 的真实产品配置**。要回答的问题是"用户在 tap 模式下 YOLO 有没有变慢"，不是"强行让 YOLO 满帧跑会怎样"。关掉等于测了一个线上不存在的配置
- 唯一受降频影响的是 `FPS`，而我们本来就不用它做判据

> 若 Architect 另外想知道"SAM 对 ANE 的争抢有多严重"（脱离降频这个缓解手段），那是**另一个独立实验**，需要临时开关；对本条 checkbox 不需要。

### 2.7 结果记录表

**按采样顺序逐行记录**（每行 = 一个采样单元的那 1 行有效统计）：

```markdown
| 序 | 组 | 模式 | Infer mean | Infer p95 | Post 中位 | Mem 峰值 | det 数 | 温感 |
|----|----|------|-----------|----------|----------|---------|--------|------|
| 1  | B  | segmentation         |  |  |  |  |  |  |
| 2  | C  | tapToSegment(idle)   |  |  |  |  |  |  |
| 3  | B  | segmentation         |  |  |  |  |  |  |
| 4  | D  | tapToSegment(active) |  |  |  |  |  |  |
| 5  | B  | segmentation         |  |  |  |  |  |  |
| 6  | C  | tapToSegment(idle)   |  |  |  |  |  |  |
| 7  | B  | segmentation         |  |  |  |  |  |  |
| 8  | D  | tapToSegment(active) |  |  |  |  |  |  |
| …  |    | （共 12 个单元）      |  |  |  |  |  |  |
| A  | A  | detectionOnly（参照） |  |  |  |  |  |  |
```

**再按相邻对计算增幅**（这才是判定依据）：

```markdown
| 配对 | 基线 B | 对照 C/D | Infer mean 增幅 | p95 增幅 | Post 增幅 |
|------|-------|---------|----------------|---------|----------|
| #1-#2 (B→C) |  |  |  % |  % |  % |
| #3-#4 (B→D) |  |  |  % |  % |  % |
| #5-#6 (B→C) |  |  |  % |  % |  % |
| #7-#8 (B→D) |  |  |  % |  % |  % |
```

### 2.8 勾选 checkbox 的条件（全部满足）

判定用**相邻对的增幅**，不要用跨越很远的两组相减。

- [ ] 每一对 B→C 的 Infer mean 增幅 ≤ **+10%**
- [ ] 每一对 B→D 的 Infer mean 增幅 ≤ **+10%**
- [ ] 每一对的 Infer p95 增幅 ≤ **+15%**
- [ ] 每一对的 Post 中位增幅 ≤ **+20%**（且各单元 `det` 数接近，证明场景确实一致）
- [ ] D 的内存峰值 ≤ B 峰值 + **400 MB**，且整个 session 的内存峰值不呈单调上升（排除泄漏）
- [ ] **同类配对（三对 B→C、三对 B→D）结论一致**；互相矛盾 → 漂移未控制住，数据无效，重采

任一条不满足就不勾，并把超出的那个数写进 `$PROJ/shared/debug_report.md` 作为新立项依据。

---

## 3. Session A — mask 正确性人工评分

> **前置：M1 + M2 + M3 必须已合入。**

### 3.1 为什么用录屏而不是逐次截图

- **必须有视觉记录** —— 日志里的 `bbox/fill/iou` 判断不了"这块 mask 是不是那个目标"
- **靠文件修改时间对齐不可靠，别用** —— 截图动作发生在 mask 显示后 1–3 秒，而相邻 tap 间隔常常也就 2–3 秒（2026-07-26 那轮 24 次 tap 挤在 4 分钟内），秒级时间戳必然错配
- 加了 **M2** 之后屏幕上直接有 `#N`，**零对齐成本**
- 录屏还能免去每次按侧边键（按键会晃动手机、改变画面），且一次录完全部样本

### 3.2 准备清单

- [ ] M1 / M2 / M3 已合入并跑通
- [ ] ⚠️ **把 Settings 里的 "Perf Quiet Log" 关掉**（与 Session B 相反！）
- [ ] 手机架在支架上

> **⚠️ 两个 session 的静默开关方向相反，别搞反**
>
> | Session | 开关 | 原因 |
> |---|---|---|
> | B（性能） | **开** | 消除日志量随模式变化的混淆项 |
> | A（评分） | **关** | 评分表的「候选(自动)」列依赖 `[TAP#N] candidates: [ch0=…px bbox=… fill=… iou=…]`，它在 `MaskRenderer.swift` 里用的是 `diagLog`，**静默下会被抑制**；标记降级路径的 `degraded pick` 同理 |
>
> 静默模式下仍会打印的只有 `[TAP#N] mask displayed`（perfLog）和 `no valid candidate`（faultLog）。
> 也就是说：**静默开着采出来的日志，评分表有一整列填不了。**
- [ ] 纸笔（记录每个场景的 5 个 tap 目标点编号）
- [ ] Xcode 连线，清空 console
- [ ] **开启屏幕录制**（控制中心）
- [ ] 提醒自己：本 session 的 `tap→mask` 延迟数字全部作废

### 3.3 五类场景（每类 10 次，共 50 次）

| 代号 | 场景 | 目标大小 | 背景 | 次数 | 2026-07-26 对应截图 |
|---|---|---|---|---|---|
| **S1** | 近处大目标 + 干净背景 | 占画面 >25% | 单一色桌面/墙 | 10 | IMG_0680 |
| **S2** | 近处大目标 + 杂乱背景 | >25% | 多物体堆叠 | 10 | IMG_0684 |
| **S3** | **远处小目标 + 干净背景** | 占画面 3–8% | 单一色 | 10 | **上轮缺失，必须补** |
| **S4** | **远处小目标 + 杂乱背景** | 3–8% | 多物体 | 10 | IMG_0685（故障场景，重点） |
| **S5** | 低对比 / 同色系 | 不限 | 白纸压白桌、黑鼠标压黑垫 | 10 | IMG_0677 / IMG_0678 |

> **S3 是本轮最重要的一组。** 它把「分辨率不足」和「背景杂乱」两个因素分开：
> - S3 良品率高、S4 低 → 主因是**背景杂乱导致的渗流**
> - S3 也差 → 主因是**小目标有效分辨率不足**
>
> 这一组数据直接裁决 debug_report 里的根因排序 #1 vs #2。

**样本量依据**：n=50 时良品率的 95% 置信区间宽约 ±14%，足以分辨"62% → 85%"这个量级的改善；再多收益递减。单次 tap 约 5 秒，一轮 5–8 分钟。改动前采一轮、改动后再采一轮，共 100 次。

### 3.4 每个场景的标准流程

1. 手机**架好对准场景后全程不动**（这消除了 embedding 陈旧的干扰变量）
2. 在画面里选 **5 个 tap 目标点**（5 个不同物体，或同一物体的 5 个部位），事先记在纸上编号 T1–T5
3. 依次点 T1…T5，**每点一次之后**：
   - 等 mask 出现（约 1 秒）
   - **停 2 秒**（让录屏拍清楚）
   - **双击屏幕清除 mask**
   - **再停 2 秒**
4. 走完 T1–T5 后，**再走一遍 T1–T5**（第二轮），共 10 次
   → 第二轮用于看重复性：同一个点两次结果不同，说明 embedding / 时序存在随机性
5. 换下一个场景，重新架机

### 3.5 采样纪律（建议写在纸上贴手机旁）

- [ ] 两次 tap 间隔 **≥ 2 秒**（否则 `tapGeneration` 会 supersede，前一次被丢弃）
- [ ] 每次 tap 前**必须双击清除**，否则旧 mask 留在屏上无法判读
- [ ] 手机**全程不动**；只在换场景时才动
- [ ] **全程录屏，不要按截图键**

**日志导出**：Xcode → Debug area console → 全选（⌘A）复制 → 存成
`/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/shared/tap_scoring_YYYYMMDD.log`。
开始前先清空 console，确保日志段落干净。

### 3.6 评分判据

**先估两个数（10% 一档，不要纠结精度）**
- **覆盖 Recall** = 目标主体被青色盖住的比例
- **溢出 Precision** = 青色区域里真正压在目标上的比例

**类型判定（硬规则，先分类再打分）**

| 覆盖 | 溢出 | 类型 |
|---|---|---|
| ≥0.75 | ≥0.75 | **正确** |
| <0.75 | ≥0.75 | **欠分割** |
| ≥0.75 | <0.75 | **过分割** |
| <0.75 | <0.75 | **双错**（记 1–2 分） |
| — | — | 无 mask / mask 落在别的物体上 → **完全失败**，1 分 |

**分数判据**

| 分 | 判据 | 2026-07-26 对应例子 |
|---|---|---|
| **5** | 覆盖 ≥0.90 且 溢出 ≥0.90，边界贴合，无点阵 | IMG_0680（杯子）、IMG_0690（水壶）、IMG_0686（耳机） |
| **4** | 覆盖 ≥0.75 且 溢出 ≥0.75；允许缺一个小部件（如把手）或多带一圈边缘 | IMG_0682（鼠标，多带一点鼠标垫） |
| **3** | 覆盖 0.50–0.75，或溢出 0.50–0.75（多带了相邻一个完整物体 / 一片桌面） | IMG_0681（水壶只盖了左半）、IMG_0684 |
| **2** | 覆盖 <0.50（明显碎片），或溢出 <0.50（多带的面积超过目标本身） | TAP#8 395px、TAP#10 362px、IMG_0679（横条） |
| **1** | mask 与目标基本无关 / 覆盖大片背景 / 点阵状铺满 / 无 mask | IMG_0685、IMG_0689（TAP#19）、IMG_0677 |

**边界情况明确判法** —— 「mask 覆盖了目标但多带了一点背景」：

| 多带的面积 | 溢出率 | 判定 |
|---|---|---|
| ≤ 目标的 10% | ≥0.90 | **5 分，正确** |
| 10%–25% | 0.75–0.90 | **4 分，正确（轻微溢出）**，备注写"轻微溢出" |
| 25%–100% | 0.50–0.75 | **3 分，过分割** |
| > 100% | <0.50 | **2 分或 1 分，过分割** |

**额外必填的二值标记**（放备注列）：`点阵=Y/N`
凡是 mask 呈网格 / 点阵状（如 IMG_0677 / 0678 / 0685 / 0689）就标 **Y**。这一列直接量化**渗流问题**的发生率，是改动前后对比最敏感的指标。

### 3.7 评分表模板

```markdown
| #  | 场景 | 目标物描述 | 帧号/时间 | 候选(自动) | 选中(自动) | e2e ms(自动) | 覆盖 | 溢出 | 类型 | 分 | 点阵 | 备注 |
|----|------|-----------|----------|-----------|-----------|-------------|------|------|------|----|------|------|
| 1  | S1   | 白杯子     | 00:12    | 5261/5887/36658  | ch0 5261px fill0.59 iou0.97  | 632 | 0.95 | 0.90 | 正确   | 5 | N | |
| 2  | S4   | 远处椅子   | 00:19    | 23669/31546/63977| ch0 23669px fill0.91 iou0.93 | 800 | 1.00 | 0.15 | 过分割 | 1 | Y | 降级路径 |
```

| 列 | 来源 | 说明 |
|---|---|---|
| `#` | **自动**（M1） | 与屏幕角标 `#N` 一一对应 |
| `场景` | 人工 | S1–S5 |
| `目标物描述` | 人工 | 一两个词，如"马克杯""远处的椅背" |
| `帧号/时间` | 人工 | 录屏时间码，方便复查 |
| `候选(自动)` | 日志 `[TAP#N] candidates:` 三个面积 | 直接 copy |
| `选中(自动)` | 日志（**需 M3**）ch / 面积 / fill / iou | 没有 M3 这列会填错 |
| `e2e ms(自动)` | 日志 `tap→mask X ms` | **录屏 session 这列作废，留空** |
| `覆盖` | **人工**，0–1，10% 一档 | Recall |
| `溢出` | **人工**，0–1，10% 一档 | Precision |
| `类型` | 人工，由覆盖/溢出推出 | 见 §3.6 |
| `分` | 人工 1–5 | 见 §3.6 |
| `点阵` | 人工 Y/N | 渗流发生率 |
| `备注` | 人工 | 如"降级路径""碎片是滚轮" |

**自动列一键生成骨架**（M1 合入后，对导出的日志跑）：

```bash
grep -E '^\[TAP#|^\[SEG\]\[TAP' tap_scoring_YYYYMMDD.log
```

> ⚠️ 注意末尾是 `\[TAP` 而**不是** `\[TAP\]` —— 解码器行合入 M1 后已变成 `[SEG][TAP#12]`，多写一个 `\]` 会静默漏掉所有解码器行。

单独捞某一次 tap 的完整链路（会一并捞到 `[SEG][TAP#12]`）：

```bash
grep -F '[TAP#12]' tap_scoring_YYYYMMDD.log
```

> ⚠️ 必须带方括号写成 `'[TAP#12]'`。若图省事写成 `'#1'`，会把 `#1` `#10` `#12` 全部匹配上。

按 `#N` 分组后粘进表格前 6 列，其余人工填。

### 3.8 汇总指标（改动前后各算一遍）

- [ ] 总良品率 = (4 分 + 5 分) / 50 —— 2026-07-26 基线约 **62%（15/24）**
- [ ] 分场景良品率 S1–S5 —— **重点看 S3 vs S4 的差值**
- [ ] 过分割率 / 欠分割率 / 完全失败率
- [ ] **点阵率**（点阵=Y 的比例）—— 渗流问题的直接量化
- [ ] 平均分

---

## 4. 已知会影响解读的两个背景事实

1. **embedding 复用 TTL = 8000 ms、主动刷新阈值 5000 ms** —— tap 可能命中最长 8 秒前的 embedding。Session A 要求"手机全程不动"正是为了消除这个干扰变量。若将来要单独评估这一项的影响，需先在 `[TAP] reuse cached embedding` 行补 cache age 埋点。
2. **当前 `e2eMs` 计时终点在主线程 mask 发布之前**（`CameraManager.swift:761`），所以已有的 429–1030 ms 数据**低估**真实"手指离开→看到 mask"延迟。若 O2 合入，Session B 可顺便重采干净的延迟基线。

---

## 5. 完成后要更新的文件

- [ ] `$PROJ/shared/tasks.md` — Day 4 Debugger 的两条 checkbox（勾选或补充"还缺什么"）
      完整路径 `/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/shared/tasks.md`
      ⚠️ 该文件**未被 git 跟踪**，没有版本回滚安全网 —— 改前先 `cp tasks.md tasks.md.bak`
      （上一轮的备份 `$PROJ/shared/tasks.md.bak` 仍在，可作对照）
- [ ] `$PROJ/shared/debug_report.md` — 记录评分表汇总指标与性能对照表
      完整路径 `/Users/jiansun/Documents/PostDoc/CODE-SHOW/APP/JudgeE2/shared/debug_report.md`
- [ ] 若判定有回退或良品率未达标 → 在 `$PROJ/shared/debug_report.md` 立新项，交 Architect 决策
