# ShotAgent 安装、下载与使用指南

本文从一台新机器开始，说明 ShotAgent 需要什么、去哪里下载、怎样运行
Anchor/HeroAnchor 分镜调色、怎样接入 MonetGPT，以及怎样把结果交给
DaVinci Resolve。项目主仓库：
[ZhisongXu/ShotAgent](https://github.com/ZhisongXu/ShotAgent)。

## 1. 先选择运行方式

| 目标 | 必需组件 | 是否需要 LLM API | 是否需要 Resolve |
| --- | --- | --- | --- |
| 本地基础图像/视频调色 | Python、ShotAgent 依赖 | 否 | 否 |
| LLM 分镜、Anchor、HeroAnchor 和多 Agent 评审 | 上述组件、OpenAI API key | 是 | 否 |
| MonetGPT 生成参数并由 ShotAgent 评审/回滚 | 上述组件、MonetGPT 模型及其环境 | MonetGPT 使用本地模型；ShotAgent 的 LLM 评审配置决定是否另需 API | 否 |
| 输出 `.cube`/动态 `.dctl` 并自动应用 | 上述组件、DaVinci Resolve 19.1+ | 取决于前面的生成方式 | 是 |

最少安装只需要 Python。`--offline-native` 会使用物理镜头检测、本地参数编辑器、
指标评审和同一套 MCTS/回滚流程，不会调用在线 LLM。

## 2. 基础安装

### 2.1 系统要求

- Git。
- Python 3.10 或 3.11；推荐 3.11，以便同时兼容 MonetGPT。
- 能安装 PyTorch 的 Windows、macOS 或 Linux。
- GPU 不是 ShotAgent 本地基础模式的硬性要求，但长视频和本地 8B 视觉模型建议使用
  NVIDIA GPU。实际显存需求取决于模型服务的量化与上下文配置。
- MP4 编解码由 `imageio-ffmpeg` 安装的 FFmpeg 二进制提供；一般不需要另装系统
  FFmpeg。若输入格式无法解码，再从 [FFmpeg 官方下载页](https://ffmpeg.org/download.html)
  安装系统版本。

### 2.2 下载和创建环境

Linux/macOS：

```bash
git clone https://github.com/ZhisongXu/ShotAgent.git
cd ShotAgent
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
git clone https://github.com/ZhisongXu/ShotAgent.git
cd ShotAgent
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

验证安装：

```bash
python -m unittest discover -s tests
python retouch_video.py --help
```

训练相关依赖不是推理所需。只有运行 `training/` 时才执行：

```bash
python -m pip install -r requirements-training.txt
```

## 3. 不使用 LLM 的快速运行

单图：

```bash
python retouch_image.py \
  --input input.jpg \
  --output outputs/retouched.jpg \
  --instruction "自动修图"
```

单图默认使用方向性质量门：候选相对原图更接近规划的调色目标，并达到可感知变化后
就会提交。高光剪切和暗部压缩仍记录为诊断指标，但不会否决艺术性调色。`retouch`、
`修图`、`自动调色` 等泛化指令会采用克制的默认调色方向，不再把零参数原图作为最优
结果。可用
`--min-perceptual-delta` 调整显著性阈值（默认 RGB RMS `0.01`），用
`--min-improvement` 调整最小方向收益。旁路 JSON 的 `decision` 字段记录每层候选数、
方向收益和感知变化。

视频：

```bash
python retouch_video.py \
  --input input.mp4 \
  --instruction "自然暖色电影感，保护肤色和高光" \
  --offline-native \
  --output outputs/input.native.grade.json \
  --video-output-dir outputs/input_videos \
  --resolve-package-dir outputs/input_resolve
```

如果有参考视频，可增加：

```bash
python retouch_video.py \
  --input target.mp4 \
  --reference-video reference.mp4 \
  --instruction "参考该视频建立自然电影感，并应用到目标视频" \
  --offline-native \
  --output outputs/target.reference.grade.json \
  --video-output-dir outputs/target_videos
```

系统会在参考视频中选择并调色 HeroShot，再把参考 Hero 原帧和获批调色帧共同作为目标
视频所有 Anchor 的视觉参考。只为目标视频生成逐帧参数和成片；JSON 中
`hero_anchor.source_video` 为 `reference_video`，并额外记录 `reference_video` 元数据。

主要输出：

- `*.grade.json`：镜头边界、各镜头 Anchor、HeroAnchor、基础参数、逐帧参数、置信度和
  回滚原因。
- `*.source.mp4` / `*.graded.mp4`：标准化源视频和预览成片；当前导出的 MP4 不含音频。
- Resolve 包：每个镜头的静态 `.cube`、逐帧变化的动态 `.dctl`、manifest 和自动应用脚本。

## 4. 使用 OpenAI LLM

这里的 LLM 不直接修改像素。它负责长视频分镜判断、每镜头 Anchor 排序、全片
HeroAnchor 选择、参数提案和视觉评审；最终像素和 Resolve LUT/DCTL 都由确定性参数
执行器生成。

1. 在 [OpenAI API 控制台](https://platform.openai.com/) 创建 API key。
2. 复制配置，不要把真实 key 写进 JSON 或提交到 Git：

```bash
cp configs/unified_vl.example.json configs/unified_vl.json
export OPENAI_API_KEY="你的_key"
```

Windows PowerShell：

```powershell
Copy-Item configs/unified_vl.example.json configs/unified_vl.json
$env:OPENAI_API_KEY="你的_key"
```

3. 运行：

```bash
python retouch_video.py \
  --input input.mp4 \
  --instruction "natural warm cinematic grade with protected skin tones" \
  --backend-config configs/unified_vl.json \
  --output outputs/input.grade.json \
  --trajectory-output outputs/input.rollouts.jsonl \
  --video-output-dir outputs/input_videos \
  --resolve-package-dir outputs/input_resolve \
  --analysis-max-side 960 \
  --render-max-side 1920
```

新配置只创建一个 `UnifiedVLVideoBackend` 和一个共享 VL client；分镜、Hero/Anchor
参数规划和视觉复核都由这个 client 分阶段完成，确定性执行器和安全指标属于后端内部
算子。输出 JSON 会包含 `operation_graph` 和脱敏的 `backend_runtime`。

operation graph v1 已真实支持 `global_grade`、单调 `tone_curve`、选择性
`hsl_grade` 和登记过的 3D `.cube` LUT。附加操作按镜头规划，在镜头首帧、Anchor、
尾帧上执行预览，再经过确定性安全检查和同一 VL client 复核；任一检查失败会整组回滚。
LUT 只能通过 `lut_catalog` 中的 ID 选择相对配置文件目录的文件，VL 不能提供任意路径。
`masked_grade`、`denoise` 和 `generative_edit` 仍必须为 `false`。
启用 curve/HSL/LUT 后，每个已接受镜头会增加一次操作规划调用；如果提案通过确定性
检查，还会增加一次视觉复核调用。把三者设为 `false` 即恢复只做全局调色的调用成本。
使用 LUT 时，把文件放在配置文件同级或子目录，并登记 ID，例如
`"lut_catalog": {"film-soft": "luts/film-soft.cube"}`。

目前 Resolve/DCTL 只编码全局 12 维参数。如果本次运行接受了 curve、HSL 或 LUT，且同时
请求 `--resolve-package-dir`，CLI 会明确报错，避免导出与预览成片不一致的包。旧的
`--agent-config` 入口仍保留给多 editor/Monet 实验。

示例配置默认使用 OpenAI Responses API 和 `gpt-5.6-sol`。如使用其他兼容服务，
修改 `provider`、`base_url`、`model` 和 `api_key_env`；兼容 Chat Completions 的服务
使用仓库支持的 `openai_compatible` provider。模型必须能接收图像并稳定返回结构化 JSON。

API 连通性最小检查：

```bash
python - <<'PY'
import json, os, urllib.request
req = urllib.request.Request(
    "https://api.openai.com/v1/models",
    headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
)
with urllib.request.urlopen(req, timeout=30) as response:
    print("OpenAI API HTTP", response.status)
PY
```

返回 200 表示 key 和网络可用；401 通常表示 key 无效，429 表示额度或限流问题。

## 5. MonetGPT：参数直接转 Resolve

官方项目与模型：

- 代码：[niladridutt/monetGPT](https://github.com/niladridutt/monetGPT)
- 权重：[niladridutt/monetGPT on Hugging Face](https://huggingface.co/niladridutt/monetGPT)

建议把它安装在 ShotAgent 仓库外：

```bash
git clone https://github.com/niladridutt/monetGPT.git ../monetGPT
cd ../monetGPT
conda create -n monetgpt python=3.11
conda activate monetgpt
cd llm
sh install.sh
mkdir -p models
huggingface-cli download niladridutt/monetGPT \
  --local-dir models/monetGPT
```

MonetGPT 官方完整图像流水线约有一部分操作依赖 GIMP 2.10。按其 README 安装
[GIMP 2.10](https://www.gimp.org/downloads/)，并检查
`configs/pipeline_config.yaml` 的 GIMP 路径。若仅已有 MonetGPT 最终参数 JSON，运行
ShotAgent 的转换器不需要 GIMP，也不需要重新渲染 MonetGPT 图片。

### 5.1 转换已有 MonetGPT JSON

输入示例：

```json
{
  "Exposure": 20,
  "Temperature": 8,
  "Highlights": -15,
  "Saturation": 8
}
```

转换：

```bash
cd /path/to/ShotAgent
source .venv/bin/activate
python monet_to_resolve.py \
  --input outputs/monet_adjustments.json \
  --output-dir outputs/monet_resolve \
  --strict
```

`Exposure`、`Temperature`、`Tint`、`Contrast`、`Highlights`、`Shadows`、
`Saturation`、`Vibrance` 会映射到共享 GradeIR；`Whites`/`Blacks` 会近似折入
高光/阴影。选择性 HSL、去雾、锐化、蒙版等不能由当前全局 LUT 精确表示，manifest
会列出它们；`--strict` 会直接拒绝非零的不支持参数。

### 5.2 让 MonetGPT 加入 Agent、评审和回滚

在 `configs/photoagent_multi.json` 的 `editors` 中加入：

```json
{
  "name": "monet-parameter-editor",
  "type": "monet_parameters",
  "root": "/absolute/path/to/monetGPT",
  "python_executable": "/absolute/path/to/monetgpt/python",
  "style": "balanced",
  "hero_match_strength": 0.35,
  "reject_unsupported": true
}
```

Linux conda 环境的解释器一般可用 `which python` 查询，Windows 可用
`Get-Command python`。`style` 只能是 `balanced`、`vibrant` 或 `retro`。

此路径读取 MonetGPT 最终 JSON，用 ShotAgent 的确定性执行器制作预览，再交给同一组
critics/MCTS。MonetGPT 仍按官方单图接口运行，但只处理 Hero 和各镜头的少量 Anchor；
目标 Anchor 参数按 `hero_match_strength` 向获批 Hero look 对齐，随后由共享贝叶斯扩散器
生成整段逐帧轨迹，不会逐帧调用 MonetGPT。`0` 表示完全采用当前 Anchor 的 Monet
结果，`1` 表示完全采用 Hero 参数，默认 `0.35`。包含被拒绝参数的候选仍可回滚。

## 6. JarvisArt 当前怎样使用

只从官方模型页下载：
[JarvisArt/JarvisArt-Preview](https://huggingface.co/JarvisArt/JarvisArt-Preview)。模型卡标注
为 8B BF16 视觉语言模型，并给出 Transformers、vLLM 和 SGLang 的启动方式。

ShotAgent 当前没有 JarvisArt/Lightroom 参数到 Resolve 的专用字段映射器，因此：

- 若将 JarvisArt 以 OpenAI-compatible 服务启动，可尝试作为 `vision_model` 的
  `openai_compatible` 端点，让它输出 ShotAgent 共享参数；是否能严格服从 JSON schema
  需要按部署版本验证。
- 若它输出编辑后图片，可用 `type: "command"` 的外部编辑器接口；ShotAgent 会从
  before/after 图像反求共享全局参数。这是近似恢复，不等同于原生 Lightroom 参数。
- 不能声称当前已经把 JarvisArt 的 200 多个 Lightroom 工具无损转换为 Resolve。

## 7. DaVinci Resolve 安装与自动应用

从 Blackmagic Design 官方
[DaVinci Resolve 下载页](https://www.blackmagicdesign.com/event/davinciresolvedownload)
安装 Windows、macOS 或 Linux 版本。动态 DCTL 需要 Resolve 19.1 或更高版本；建议使用
当前受支持版本。ShotAgent 不捆绑 Resolve，也不会修改 Resolve 安装目录。

### 7.1 让 Python 找到 Resolve API

先在 Resolve 的 Preferences 中允许本机脚本访问，再启动 Resolve、打开目标 project 和
timeline。常见环境变量如下；若实际安装路径不同，以 Resolve 安装目录中
`Developer/Scripting/README.txt` 为准。

Linux：

```bash
export RESOLVE_SCRIPT_API="/opt/resolve/Developer/Scripting"
export RESOLVE_SCRIPT_LIB="/opt/resolve/libs/Fusion/fusionscript.so"
export PYTHONPATH="$PYTHONPATH:$RESOLVE_SCRIPT_API/Modules"
```

macOS：

```bash
export RESOLVE_SCRIPT_API="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"
export RESOLVE_SCRIPT_LIB="/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so"
export PYTHONPATH="$PYTHONPATH:$RESOLVE_SCRIPT_API/Modules"
```

Windows PowerShell（路径按实际安装位置调整）：

```powershell
$env:RESOLVE_SCRIPT_API="$env:PROGRAMDATA\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting"
$env:RESOLVE_SCRIPT_LIB="C:\Program Files\Blackmagic Design\DaVinci Resolve\fusionscript.dll"
$env:PYTHONPATH="$env:PYTHONPATH;$env:RESOLVE_SCRIPT_API\Modules"
```

验证：

```bash
python -c "import DaVinciResolveScript; print('Resolve API import OK')"
```

### 7.2 导出逐帧变化包

在视频命令中加入：

```bash
--resolve-package-dir outputs/input_resolve \
--resolve-keyframe-error 0.015
```

“逐帧变化”并不是让 LLM 每帧随意改色。ShotAgent 先在 Anchor 上决定参数，然后在镜头内
生成平滑的 12 维参数轨迹；导出时保留 Anchor 和转折点，并压缩为误差受控的关键点。
Resolve 19.1+ 的 DCTL 通过 `TIMELINE_FRAME_INDEX` 在渲染时插值这些参数。

### 7.3 在 Resolve 中应用

1. 时间线必须按 `resolve_manifest.json` 分镜，且一个 manifest shot 对应一个 clip；帧数
   必须一致。
2. 每个 clip 在 Color 页同一个位置创建一个专用空节点，例如第 2 个节点。
3. 在 Resolve Preferences 的 Color Management / Lookup Tables 中确认一个可写的 LUT
   搜索目录，作为下面的 `--lut-dir`。
4. 保持 Resolve、project 和 timeline 打开，然后运行：

```bash
python outputs/input_resolve/apply_dynamic_grade.py \
  --lut-dir /absolute/path/to/resolve/LUT \
  --video-track 1 \
  --node-index 2
```

脚本会核对 clip 数量、顺序、时长和节点，复制 DCTL、刷新 LUT 列表、应用到指定节点并
保存 project。它不会猜测节点，也不会覆盖不存在的节点。先复制项目或建立 Resolve 项目
备份，是正式素材的推荐做法。

静态结果也可以手工使用 `LUT/shot-XXXX.cube`；动态变化则使用
`DCTL/shot-XXXX.dctl`。

## 8. Anchor、HeroAnchor、传播和回滚

- LLM 模式先做全片稀疏总览，再按重叠窗口判断边界，并在每个已确认镜头内排序 Anchor。
- 全片 Anchor 经过分批比较形成 HeroAnchor 候选；获批的 Hero grade 是全片视觉参考。
- 其他镜头不会原样复制 Hero LUT，而是同时查看 Hero 原图和获批成片，估计适合本镜头的
  参数；随后在该镜头内从 Anchor 向各帧平滑传播。
- 静态 `.cube` 表示镜头基础 grade；动态 `.dctl` 表示同一镜头里的逐帧轨迹。
- 每个候选都经过视觉 critics 和确定性时序安全检查。全部失败、低于阈值或没有改善时，
  回滚为 identity 参数；`grade.json` 中记录 `rolled_back` 和 `rollback_reason`。

## 9. 常见故障

| 现象 | 检查 |
| --- | --- |
| `OPENAI_API_KEY` 缺失 | 重新导出环境变量，并确认启动 ShotAgent 的终端是同一个终端 |
| HTTP 401 | key 无效、被撤销或复制时带了多余字符 |
| HTTP 429 | API 额度、并发或速率限制 |
| `inference_cli.py` not found | Monet 配置的 `root` 必须是包含该文件的官方仓库根目录 |
| Monet 未生成 `.json` | 检查其模型路径、服务是否启动、`python_executable` 和 Monet stdout/stderr |
| 不支持的 Monet 参数导致回滚 | 查看 conversion audit；需要近似时显式设 `reject_unsupported: false` |
| `No module named DaVinciResolveScript` | 设置 `RESOLVE_SCRIPT_API`、`RESOLVE_SCRIPT_LIB`、`PYTHONPATH`，并使用 Resolve 支持的 Python |
| `Resolve is not running` | 启动 Resolve，打开项目与时间线，并启用脚本访问 |
| clip 数量/时长不匹配 | 按 manifest 精确切镜，一个 shot 对应一个 clip |
| `RefreshLUTList` 或 `SetLUT` 失败 | 确认 `--lut-dir` 已加入 Resolve LUT 搜索路径且可写，并检查指定节点存在 |
| 视频内存不足或太慢 | 减小 `--analysis-max-side`、`--render-max-side`、`--render-batch-size` 或 `--max-frames` |

## 10. 当前边界

- Resolve 输出面向 display-referred RGB `[0,1]`，推荐时间线为 Rec.709 Gamma 2.4；若使用
  Log、ACES 或其他色彩管理，应在正确的输入/输出变换之间放置节点并自行校准。
- 全局 3D LUT/DCTL 不能携带局部蒙版参数，导出会拒绝非零局部参数。
- `--resolve-keyframe-error` 控制的是共享参数空间的最大压缩误差，不是 Delta E。
- Resolve 公共脚本 API 不负责创建任意 Color 页参数关键帧；因此本项目用时间感知 DCTL
  表达动态参数。
- 请勿提交 API key、模型权重、输入素材或 `outputs/`。仓库 `.gitignore` 已忽略常见权重
  和输出目录，但密钥仍应只放在环境变量或本机秘密管理器中。
