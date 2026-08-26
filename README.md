# ShotAgent PoolGraph v2

本 README 只说明当前主路径：使用一个视觉语言模型（VL）生成 `pool-graph/v2`，再由本地确定性执行器完成视频调色。

```text
视频 + 调色指令
  → VL 分镜、选择 Hero/Anchor
  → VL 返回稀疏 Pool 参数
  → 参数校验、Anchor 合并、逐帧扩散
  → 本地 CPU/CUDA Pool 执行器渲染
  → grade JSON + 调色后视频
```

当前文档和示例统一使用 `--backend-config` 进入 PoolGraph v2，不覆盖其他运行模式。

## 1. 安装

推荐 Python 3.11。

```bash
git clone https://github.com/ZhisongXu/ShotAgent.git
cd ShotAgent

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

检查安装：

```bash
python retouch_video.py --help
python -m unittest tests.test_grade_pools_v2
```

## 2. 放置 API Key

API Key 只放在环境变量中，不要写入 JSON、源码或提交到 Git。配置文件里的 `api_key_env` 表示程序应该读取哪个环境变量。

### 2.1 OpenAI

复制 PoolGraph 配置：

```bash
cp configs/unified_vl.example.json configs/unified_vl.json
```

Linux/macOS：

```bash
export OPENAI_API_KEY="你的 API Key"
```

Windows PowerShell：

```powershell
$env:OPENAI_API_KEY="你的 API Key"
```

当前 OpenAI 配置的关键部分是：

```json
{
  "backend": {
    "type": "unified_vl_video",
    "grade_schema": "pool-graph/v2",
    "provider": "openai_responses",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-5.6-sol",
    "api_key_env": "OPENAI_API_KEY"
  }
}
```

若要换模型，只修改本地 `configs/unified_vl.json` 中的 `model`。不要修改 `grade_schema`。

### 2.2 Gemini OpenAI-compatible endpoint

低请求量 smoke test：

```bash
cp configs/unified_vl.gemini-free.json configs/unified_vl.json
export GEMINI_API_KEY="你的 API Key"
```

完整 Pool 阶段与 VL review：

```bash
cp configs/unified_vl.gemini-full.json configs/unified_vl.json
export GEMINI_API_KEY="你的 API Key"
```

Windows PowerShell 使用：

```powershell
$env:GEMINI_API_KEY="你的 API Key"
```

Gemini 配置使用：

```json
{
  "provider": "openai_compatible",
  "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
  "api_key_env": "GEMINI_API_KEY"
}
```

`gemini-free` 关闭视觉 review，并降低分镜图片数和尝试次数，适合验证整条链路；`gemini-full` 启用完整五阶段和 review，API 调用更多。

### 2.3 使用本地密钥文件

程序不会自动读取环境变量文件。如果希望本地保存 Key，可放在仓库已经通过 `.gitignore` 排除的 `.secrets/` 目录：

```bash
mkdir -p .secrets
printf 'OPENAI_API_KEY="在这里填写 Key"\n' > .secrets/poolgraph.env
```

运行前手动载入：

```bash
set -a
source .secrets/poolgraph.env
set +a
```

不要提交其他位置的密钥文件，也不要在终端截图、日志或 grade JSON 中暴露 Key。

## 3. 最小运行命令

准备一个输入视频，例如 `input.mp4`，然后运行：

```bash
mkdir -p outputs

python retouch_video.py \
  --input input.mp4 \
  --instruction "自然暖色电影感，保护肤色与高光" \
  --backend-config configs/unified_vl.json \
  --output outputs/input.poolgraph.json \
  --video-output-dir outputs/input_videos
```

主要输出：

```text
outputs/input.poolgraph.json
outputs/input_videos/input.source.mp4
outputs/input_videos/input.graded.mp4
```

- `input.poolgraph.json`：镜头、Hero/Anchor audit、Pool operation graph、动态 parameter track 和 rollback 信息；
- `input.source.mp4`：本次实际解码/缩放后的源视频；
- `input.graded.mp4`：最终 PoolGraph 渲染结果。

只生成 PoolGraph JSON、不渲染视频时，省略 `--video-output-dir`。

## 4. 使用参考视频

参考视频只负责建立 Hero look；最终 Pool 节点和成片应用到目标视频：

```bash
python retouch_video.py \
  --input target.mp4 \
  --reference-video reference.mp4 \
  --instruction "匹配参考视频的柔和电影感，同时保留目标场景真实曝光" \
  --backend-config configs/unified_vl.json \
  --output outputs/target.poolgraph.json \
  --video-output-dir outputs/target_videos
```

输出的 `pool_metadata.hero.source_video` 应为 `reference_video`。

## 5. 常用运行参数

### 5.1 分开控制分析和渲染分辨率

VL 使用较小代理图，最终成片重新按较大尺寸解码：

```bash
python retouch_video.py \
  --input input.mp4 \
  --instruction "自然、克制的商业广告调色" \
  --backend-config configs/unified_vl.json \
  --output outputs/input.poolgraph.json \
  --video-output-dir outputs/input_videos \
  --analysis-max-side 960 \
  --render-max-side 1920
```

`--analysis-max-side` 不会改变帧编号。分析解码与渲染解码的帧数必须一致。

### 5.2 GPU/CPU

默认有 CUDA 时使用 CUDA，否则使用 Torch CPU：

```bash
--render-device cuda
--render-device cuda:1
--render-device cpu
```

完整示例：

```bash
python retouch_video.py \
  --input input.mp4 \
  --instruction "自然电影感" \
  --backend-config configs/unified_vl.json \
  --output outputs/input.poolgraph.json \
  --video-output-dir outputs/input_videos \
  --render-device cuda \
  --render-batch-size 8
```

显存不足时降低 `--render-batch-size` 或 `--render-max-side`。

### 5.3 限制帧数做 API smoke test

先用短片或少量帧验证 Key、模型和 JSON 返回格式：

```bash
python retouch_video.py \
  --input input.mp4 \
  --instruction "轻微提升曝光，保持自然" \
  --backend-config configs/unified_vl.json \
  --output outputs/smoke.poolgraph.json \
  --max-frames 24 \
  --analysis-max-side 640
```

`--max-frames` 会让输出只覆盖前 N 帧，不是只分析前 N 帧后再渲染完整视频。

### 5.4 精简 JSON

```bash
--compact
```

该选项省略逐帧 `parameter_track`。如果下游需要重建动态 Primary、白平衡或降噪轨迹，不要使用它。

## 6. Log、HDR 和 10/12-bit 输出

例如 S-Log3 输入、ACEScg 工作空间、10-bit Rec.2020 PQ 输出：

```bash
python retouch_video.py \
  --input camera-log.mov \
  --instruction "电影感对比，保护肤色、天空和高光" \
  --backend-config configs/unified_vl.json \
  --output outputs/camera-log.poolgraph.json \
  --video-output-dir outputs/camera-log-videos \
  --input-color-space slog3 \
  --working-color-space acescg \
  --output-color-space rec2020_pq \
  --output-bit-depth 10 \
  --pq-reference-white-nits 203 \
  --render-device cuda
```

高位深路径使用 16-bit RGB 解码、float32 渲染和 HEVC 10/12-bit 编码，并在源视频包含音频时复制音频。VL 看到的是 tone-mapped sRGB 代理。

如需工作室 OpenColorIO 配置：

```bash
python -m pip install -r requirements-color.txt
```

然后同时提供：

```bash
--ocio-config /path/to/config.ocio \
--ocio-input-space "Input Space" \
--ocio-working-space "ACEScg" \
--ocio-display-space "sRGB Display" \
--ocio-output-space "Output Space"
```

## 7. PoolGraph 配置

必须保持：

```json
{
  "backend": {
    "type": "unified_vl_video",
    "grade_schema": "pool-graph/v2"
  }
}
```

当前可启用的 9 类 Pool：

```json
{
  "operations": {
    "denoise": true,
    "white_balance": true,
    "primary": true,
    "color_wheels": true,
    "curves": true,
    "hsl8": true,
    "global_color": true,
    "texture": true,
    "optical_effects": true
  }
}
```

VL 分阶段返回稀疏操作：

```text
technical       denoise, white_balance, primary
look            color_wheels, curves, global_color
selective_color hsl8
texture         texture
optical         optical_effects
```

无论 VL 以什么顺序返回，像素始终按以下顺序执行：

```text
denoise
→ white_balance
→ primary
→ color_wheels
→ curves
→ hsl8
→ global_color
→ texture
→ optical_effects
```

每个操作可以使用 `global`、`person`、`skin` 或 `sky` mask。`primary`、`white_balance` 和 `denoise` 可以生成逐帧轨迹；其他 Pool 默认镜头内静态。

完整参数、范围和实现审计见：

- [PoolGraph v2 技术报告](docs/poolgraph_v2_technical_report.zh-CN.pdf)
- [技术报告 Markdown](docs/poolgraph_v2_technical_report.zh-CN.md)

## 8. 检查输出确实是 PoolGraph v2

运行后执行：

```bash
python - outputs/input.poolgraph.json <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["schema_version"] == "pool-grade-graph/v2"
assert payload["operation_graph"]["schema_version"] == "video-edit-operation-graph/v2"
print("schema:", payload["schema_version"])
print("operations:", len(payload["operation_graph"]["operations"]))
print("accepted:", payload["pool_metadata"]["globally_accepted"])
PY
```

如果 `globally_accepted` 为 `false` 且 operations 为空，通常表示 Hero/Anchor、安全指标或 VL review 未通过，事务回滚已经生效；这不是 JSON 写出失败。

## 9. 常见问题

### `Missing API key environment variable`

当前 shell 没有对应变量，或配置中的 `api_key_env` 与导出的变量名不同：

```bash
test -n "$OPENAI_API_KEY" && echo "OPENAI_API_KEY is set"
test -n "$GEMINI_API_KEY" && echo "GEMINI_API_KEY is set"
```

不要使用会把 Key 内容直接写进共享日志的检查命令。

### HTTP 401/403

检查 API Key、项目权限、模型权限，以及 `base_url` 是否与 provider 匹配。

### HTTP 429

达到每分钟或每日额度。客户端会按配置重试临时 429/5xx，但无法绕过每日额度。可先使用短视频、`--max-frames` 或低请求量配置。

### VL 返回 JSON 校验失败

PoolGraph 会拒绝未知 Pool、错误 stage、重复 Pool、未知字段、非法 mask、非有限数值和越界参数。错误会进入重试提示；超过 `maximum_planning_attempts` 后该 Anchor 失败，并可能触发事务回滚。

### 输出视频没有音频

默认 8-bit H.264 预览是静音视频。高位深 HEVC 路径会在源文件有音频时复制音频。

### CUDA out of memory

降低：

```bash
--render-batch-size 2 --render-max-side 1280
```

或使用：

```bash
--render-device cpu
```

## 10. 与 PoolGraph 直接相关的源码

```text
video_retouch/unified_backend.py       单 backend 构建与最终 manifest
video_retouch/pool_pipeline.py         Hero/Anchor Pool 规划、review、rollback
video_retouch/grade_pools.py           Pool 契约、参数校验、固定顺序、CPU executor
video_retouch/pool_propagation.py      动态参数与时序扩散
video_retouch/gpu_pool_executor.py     Torch CPU/CUDA batch executor
video_retouch/semantic_masks.py        person/skin/sky mask 与跟踪
video_retouch/color_managed_render.py  Log/HDR/ACES 高位深渲染
video_retouch/tasks.py                 VL Pool grade/review prompt
retouch_video.py                       命令行入口
```

运行 PoolGraph 回归测试：

```bash
python -m unittest \
  tests.test_grade_pools_v2 \
  tests.test_unified_backend \
  tests.test_color_managed_render
```
