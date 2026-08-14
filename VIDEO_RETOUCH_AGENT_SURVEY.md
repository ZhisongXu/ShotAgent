# Video Retouching Agent Survey

更新日期：2026-08-07

## 1. 调研范围与结论

本文调研四个相邻方向：

1. 可解释的单图修图 Agent；
2. 自动视频调色与 LUT 生成；
3. 关键帧驱动的视频编辑和时序传播；
4. 视频质量评价、数据集与用户偏好建模。

核心结论：

- “VLM 分析图片并输出 Lightroom 参数”已经不是空白；
- “规划—执行—评价—反思”也已经被多篇工作覆盖；
- “选一张关键帧并对全视频应用同一个 LUT”已有 ICCV 2025 和 2026 年的直接相关工作；
- 单个全局 LUT 能保证无闪烁，但难以处理动态曝光、混合光照、人物/天空局部调整和多镜头；
- 生成式关键帧传播可以进行局部编辑，但容易带来内容漂移、闪烁和较高计算成本；
- 更有潜力的研究问题是：**学习可解释、可编辑、空间变化且时间连续的参数场，并通过主动 Anchor 选择控制精修成本和传播风险。**

推荐课题暂定为：

> **DynamicGradeAgent: Active Anchor Planning and Spatio-Temporal Parameter Fields for Explainable Video Retouching**

## 2. 单图修图与图像编辑 Agent

| 工作 | 任务 | 核心方法 | 输出空间 | 主要局限 |
|---|---|---|---|---|
| [MonetGPT, SIGGRAPH 2025](https://arxiv.org/abs/2505.06176) | 自动照片润色 | 用操作识别、参数排序、完整规划三类视觉谜题训练 MLLM | 程序化修图操作 | 主要面向单图；倾向学习固定专家风格；没有视频时序建模 |
| [PhotoArtAgent, 2025](https://arxiv.org/abs/2505.23130) | 艺术化照片润色 | VLM 分析、规划 Lightroom 参数、执行后迭代评价 | Lightroom 参数 | 规划—反思框架本身已不新；评价主观且单图化 |
| [JarvisArt, NeurIPS 2025](https://arxiv.org/abs/2506.17612) | 专业修图 Agent | 200+ Lightroom 工具；CoT SFT + GRPO-R | Lightroom 工具轨迹 | 工具调用和 RL 路线已经覆盖；视频扩展缺少时序约束 |
| [4KAgent, NeurIPS 2025](https://arxiv.org/abs/2507.07105) | 通用恢复与 4K 超分 | Perception Agent + Restoration Agent + Q-MoE + rollback | 多个恢复专家工具 | 偏恢复而非审美调色；逐图执行；固定指标容易被利用 |
| [RetouchIQ, CVPR 2026](https://arxiv.org/abs/2602.17558) | 指令式可执行修图 | generalist reward MLLM 为工具策略提供 RL 奖励；190K 指令—推理对 | 可执行调整参数 | generalist reward 和 RL 已有人做；未解决时间一致性 |
| [JarvisEvo, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Lin_JarvisEvo_Towards_a_Self-Evolving_Photo_Editing_Agent_with_Synergistic_Editor-Evaluator_CVPR_2026_paper.html) | 自进化修图 Agent | interleaved multimodal CoT；Editor–Evaluator 协同优化 | Lightroom + 生成式工具 | 自进化与 evaluator 共训已被覆盖；仍以单图为中心 |
| [IEA, CVPR 2026 Findings](https://openaccess.thecvf.com/content/CVPR2026F/html/Zhu_IEA_Amateur-Friendly_Conversational_Image_Editing_Agent_via_Three_Stages_of_CVPRF_2026_paper.html) | 对话式修图 | 16 个参数化工具；SFT、GRPO、合成多任务训练 | 可解释工具轨迹 | 工具较少；视频和长期风格一致性未涉及 |
| [PhotoAgent, 2026](https://arxiv.org/abs/2602.22809) | 自主照片编辑 | VLM 提议动作，MCTS 长期搜索，执行后由 UGC reward 评价 | 程序化与生成式工具 | MCTS + aesthetic evaluator 已被覆盖；搜索成本较高 |
| [MIRA, CVPR 2026 Findings](https://openaccess.thecvf.com/content/CVPR2026F/html/Zeng_MIRA_Multimodal_Iterative_Reasoning_Agent_for_Image_Editing_CVPRF_2026_paper.html) | 复杂指令编辑 | 逐步 perception–reasoning–action；150K 工具数据；SFT + GRPO | 原子生成式编辑指令 | 偏复杂语义编辑，不强调保真程序化调色 |
| [InstantRetouch, CVPR 2026](https://arxiv.org/abs/2606.05071) | 高保真指令修图 | 将 diffusion prior 蒸馏到单步 bilateral grid | 空间变化的仿射颜色变换 | 不是 Agent；没有视频参数轨迹，但非常适合作为我们的执行器基线 |

### 2.1 单图方向已经拥挤的点

以下内容单独作为贡献已经偏弱：

- 使用 VLM 输出曝光、对比度、HSL 参数；
- 多 Agent 分成 perception、planning、execution、reflection；
- 使用 SFT + GRPO 学工具调用；
- 用 aesthetic reward 选择最佳结果；
- 用 MCTS 搜索多步修图轨迹；
- 设计操作识别和参数预测视觉谜题。

单图工作仍然能为视频课题提供三个组件：

1. MonetGPT 的 operation-aware 数据构造；
2. RetouchIQ/JarvisEvo 的 learned evaluator；
3. InstantRetouch 的高保真 spatially varying executor。

## 3. 视频调色与视频编辑

| 工作 | 设置 | 核心方法 | 一致性来源 | 对我们的启示/局限 |
|---|---|---|---|---|
| [Learning Blind Video Temporal Consistency, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Wei-Sheng_Lai_Real-Time_Blind_Video_ECCV_2018_paper.html) | 任意逐帧图像处理后处理 | 原视频与逐帧结果输入 recurrent network | 短期、长期 temporal loss | 可作为通用去闪烁 baseline，但会修正结果而不是产生可编辑参数 |
| [StableVideo, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Chai_StableVideo_Text-driven_Consistency-aware_Diffusion_Video_Editing_ICCV_2023_paper.html) | 文本驱动生成式视频编辑 | 分层表示与跨帧传播 | temporal dependency | 适合语义编辑，不适合严格保真的专业调色 |
| [Video Color Grading via LUT, ICCV 2025](https://arxiv.org/abs/2508.00548) | 参考图/视频驱动调色 | CLIP 选择匹配关键帧，diffusion 生成显式 LUT | 全视频使用相同 LUT | 与“选最优帧再调色”高度重合；作者明确指出 scene change 是局限 |
| [SA-LUT, 2025](https://arxiv.org/abs/2506.13465) | 照片真实感风格迁移 | spatial-adaptive 4D LUT | 非生成式颜色映射 | 说明局部自适应 LUT 已存在；尚未系统建模视频时间轨迹 |
| [Align-A-Video, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Align-A-Video_Deterministic_Reward_Tuning_of_Image_Diffusion_Models_for_Consistent_CVPR_2025_paper.html) | reward-guided 生成式视频编辑 | 优化 Anchor，再传播特征 | Anchor feature propagation | “优化关键帧后传播”已经存在于生成式编辑领域 |
| [V-RGBX, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Fang_V-RGBX_Video_Editing_with_Accurate_Controls_over_Intrinsic_Properties_CVPR_2026_paper.html) | 物理属性视频编辑 | 视频 intrinsic decomposition + keyframe editing | intrinsic-conditioned video synthesis | 物理可控但系统复杂；不是程序化专业调色 Agent |
| [FFP-300K, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_FFP-300K_Scaling_First-Frame_Propagation_for_Generalizable_Video_Editing_CVPR_2026_paper.html) | first-frame propagation | 300K 视频对；appearance/motion 解耦 | 视频 diffusion temporal prior | 第一帧生成式传播已有强基线；数据规模门槛高 |
| [Occlusion-Aware Keyframe Selection, 2026](https://arxiv.org/abs/2605.23192) | 鲁棒视频编辑 | 完整度、跟踪稳定性、属性可见性联合选 Anchor | 双向 tracking 与 mask propagation | “选一个更可靠 Anchor”也已有直接工作；我们应做主动、多 Anchor、任务反馈驱动选择 |
| [MA-VFR, CVPR 2026 Findings](https://openaccess.thecvf.com/content/CVPR2026F/html/Tang_Evolutionary_Multi-Agent_Collaboration_for_Real-World_Video_Face_Restoration_CVPRF_2026_paper.html) | 视频人脸恢复 | facial agent、temporal agent、evolving coordinator | temporal agent + memory | 多 Agent 视频恢复已有工作；我们不能只把 4KAgent 改为视频版本 |
| [LumiVideo, 2026](https://arxiv.org/abs/2604.02409) | 自动 Log 视频调色 | VLM/物理双流感知，RAG + Tree-of-Thought，输出 ASC-CDL 与 3D LUT | 同一 LUT 数学上无闪烁 | 最直接竞争者；公开描述的主要限制是缺少 spatially varying adjustment/power windows |

### 3.1 LumiVideo 对课题定义的影响

LumiVideo 已覆盖：

- Agentic video color grading；
- 自动分析 Log 视频；
- RAG 与 Tree-of-Thought 搜索；
- ASC-CDL/3D LUT 可解释输出；
- 自然语言 reflection；
- LumiGrade：100+ Log 视频 benchmark。

所以以下课题已经不够新：

> “让 VLM 看视频，选一帧，输出一个全局 LUT，并支持用户文字修改。”

但全局 LUT 的“绝对时间一致性”来自对所有帧执行相同映射，它没有真正解决：

- 镜头内自动曝光变化；
- 从室内走到室外；
- 混合色温；
- 人物、天空、背景的局部调色；
- 主体进入、离开与遮挡；
- 多镜头间相同叙事风格、不同技术校正；
- 在何处需要增加新关键帧。

这些正是我们的候选空白。

## 4. 研究空白地图

| 能力 | 单图 Agent | LUT 视频方法 | 生成式视频编辑 | 建议研究目标 |
|---|---:|---:|---:|---:|
| 可解释操作参数 | 强 | 强 | 弱 | 强 |
| 高分辨率内容保真 | 强 | 强 | 中/弱 | 强 |
| 全局时间一致性 | 不适用 | 强 | 中 | 强 |
| 动态曝光适应 | 无 | 弱 | 中 | 强 |
| 空间局部调整 | 强 | 弱 | 强 | 强 |
| 参数时间轨迹 | 无 | 通常固定 | 不可解释 | 强 |
| 主动多 Anchor | 无 | 弱 | 部分覆盖 | 强，且由任务误差反馈驱动 |
| 不确定性与风险 | 少量 | 弱 | 少量 | 强 |
| 个体审美适应 | 部分 | 文本反馈 | 文本条件 | 可选增强点 |
| 专业 NLE 可编辑输出 | Lightroom | LUT/CDL | 弱 | CDL/LUT + keyframe timeline + masks |

## 5. 推荐主课题：DynamicGradeAgent

### 5.1 问题定义

输入：视频 `V`、可选用户要求 `u`、可选参考图/视频 `r`。

输出不是重新生成的视频，而是一个可以在专业软件中继续编辑的 grade graph：

```text
shot boundaries
+ shared base grade per shot
+ temporal parameter keyframes
+ spatial masks / power windows
+ local parameter tracks
+ confidence and rollback points
```

每帧参数分解为：

```text
theta_t(x, y) = theta_shot + delta_theta_t + theta_local_t(x, y)
```

- `theta_shot`：镜头共享的 base look；
- `delta_theta_t`：低频、时间连续的曝光/白平衡补偿；
- `theta_local_t(x,y)`：人物、天空、背景等局部参数场。

### 5.2 方法模块

1. **Shot-aware perception**
   - 颜色空间与相机 Log profile 归一化；
   - 镜头切分；
   - 估计曝光、白平衡、动态范围、人物和语义区域；
   - 生成 shot-level technical correction 与 creative intent。

2. **Active multi-anchor selection**
   - 初始 Anchor 优化覆盖度、清晰度、主体可见性和光照代表性；
   - 传播后根据不确定性、参数预测误差、局部 mask 漂移和 temporal critic 分数主动插入 Anchor；
   - 与固定间隔、单 Anchor、聚类 Anchor 和 occlusion-aware 单 Anchor 比较。

3. **Anchor retouching**
   - VLM 先输出问题、意图和参数范围；
   - 程序化 executor 产生多个候选；
   - evaluator 选择 Pareto-optimal 结果，而非仅最大化单一 aesthetic score。

4. **Spatio-temporal parameter propagation**
   - 传播参数和区域，而不是复制 Anchor RGB；
   - 双向传播和遮挡感知；
   - 使用低频 spline/state-space model 预测 `delta_theta_t`；
   - 新出现区域触发重新分割或局部 Anchor。

5. **Temporal counterfactual critic**
   - 比较编辑残差而非直接比较相邻输出帧：`E_t = O_t - I_t`；
   - 使用 motion-compensated residual consistency 区分真实光照变化和算法闪烁；
   - 输出质量、指令、内容保持、时序、风险和不确定性六个维度。

6. **Professional export**
   - ASC-CDL / `.cube` LUT；
   - 时间变化参数以 keyframe timeline 导出；
   - 局部区域以 mask/power-window track 导出；
   - 保存可回滚的 edit graph。

### 5.3 可能的论文贡献

1. 第一个同时建模 shot-level look、frame-level correction 和 region-level adjustment 的可解释视频修图 Agent；
2. 基于执行反馈和不确定性的主动多 Anchor 策略，而不是固定采样或一次性选最佳帧；
3. 以 motion-compensated edit residual 为核心的 temporal critic，避免把真实光照变化误判为闪烁；
4. 一个包含专业参数时间线、局部区域轨迹和人类偏好的 Video Retouch Benchmark。

## 6. 数据集

### 6.1 可用公开数据

| 数据集 | 内容 | 用途 | 局限 |
|---|---|---|---|
| [MIT-Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/) | 5,000 RAW 图片、5 位专家版本 | 单图技术/审美预训练，多解偏好 | 无视频轨迹 |
| [PPR10K](https://openaccess.thecvf.com/content/CVPR2021/html/Liang_PPR10K_A_Large-Scale_Portrait_Photo_Retouching_Dataset_With_Human-Region_Mask_CVPR_2021_paper.html) | 10K 级人像、3 位专家、人像 mask、组一致性 | 人像区域、肤色、专家风格 | 仍是静态图片 |
| MMArt-Bench | 真实用户修图操作 | 工具计划与意图评价 | 获取和许可需核实 |
| UGC-Edit | 7,000 UGC 图片和审美 reward | evaluator 预训练 | 不包含视频时序 |
| iRetouch | 指令修图 benchmark | 高保真和指令评价 | 主要为单图 |
| [Video Color Grading 数据](https://arxiv.org/abs/2508.00548) | 电影片段 + LUT 合成训练对 | LUT baseline 与参考风格迁移 | 专业电影帧不等于原始 Log—专家 grade 对 |
| LumiGrade | 100+ 多相机 Log 视频 | 自动基础调色评价 | 规模较小；可用性需要确认 |
| [FFP-300K](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_FFP-300K_Scaling_First-Frame_Propagation_for_Generalizable_Video_Editing_CVPR_2026_paper.html) | 300K、720p、81 帧编辑视频对 | 生成式传播 baseline | 任务偏生成式编辑 |
| [VE-Bench](https://arxiv.org/abs/2408.11481) | 视频编辑结果和主观 MOS | 视频编辑 evaluator | 偏文本生成式编辑 |
| [DIVIDE-3k / DOVER](https://arxiv.org/abs/2211.04894) | UGC 视频审美/技术质量 | 视频质量评价预训练 | 不是修图偏好数据 |

### 6.2 建议自建数据

建议三层数据：

1. **Synthetic temporal interventions**
   - 对视频施加平滑曝光、白平衡、曲线、HSL 和局部区域参数轨迹；
   - 保存真实 `state-action-next_state`；
   - 用于 operation world model、inverse action 和 temporal critic。

2. **Professional grade trajectories**
   - 邀请调色师使用 DaVinci Resolve；
   - 保存原始 Log、最终输出、node graph、关键帧、power windows 和操作时间线；
   - 这是最稀缺、最有论文价值的数据。

3. **Preference pairs**
   - 同一视频生成 natural、warm、cinematic、vivid 等多个可接受版本；
   - 收集普通用户、摄影爱好者和专业调色师的 pairwise preference。

## 7. 评价指标

### 7.1 单帧质量与保真

- MUSIQ、MANIQA、NIQE、DOVER technical/aesthetic；
- SSIM、LPIPS、DISTS、DINO similarity；
- ArcFace 与肤色偏移（人像）；
- highlight/shadow clipping、色域越界、banding。

### 7.2 时间一致性

- temporal LPIPS（tLPIPS）；
- temporal optical-flow error（tOF）与 warping error；
- luminance/chroma flicker；
- mask boundary jitter；
- 长期 Anchor drift；
- motion-compensated edit-residual consistency。

注意：warping error 会受到光流误差和视频真实变化影响，不能单独作为结论；已有视频恢复工作也观察到低 warping error 不一定对应更高用户偏好。

### 7.3 Agent 与系统指标

- Anchor 数量；
- 每分钟视频的 VLM/高级 executor 调用数；
- planning regret：相对于全帧/穷举 oracle；
- risk–coverage curve 与不确定性 ECE；
- 用户反馈 0/1/3 次后的 preference regret；
- 专业调色师继续修改所需的操作次数和时间。

## 8. 基线与消融

### 8.1 必要基线

1. Frame-wise image retouching；
2. Single global LUT；
3. LumiVideo 风格的 CDL + LUT Agent；
4. ICCV 2025 LUT generation；
5. Single Anchor + fixed parameters；
6. Uniform multi-anchor；
7. Clustering-based anchors；
8. Occlusion-aware single anchor；
9. Blind temporal consistency post-processing；
10. InstantRetouch frame-wise + temporal smoothing；
11. 全帧高级处理 oracle。

### 8.2 关键消融

- 去掉 shot segmentation；
- 去掉 active anchor insertion；
- 固定 LUT 替代 temporal parameter field；
- 去掉 local spatial field；
- 去掉 bidirectional propagation；
- 直接 output consistency 替代 edit-residual consistency；
- 去掉 uncertainty；
- 单一 reward 替代多目标/Pareto evaluator；
- 去掉真实执行验证。

## 9. 候选课题优先级

| 方向 | 新颖性 | 实现风险 | 数据成本 | 建议 |
|---|---:|---:|---:|---|
| 全局 LUT 视频 Agent | 低 | 低 | 低 | 不建议，LumiVideo 已直接覆盖 |
| 单 Anchor 修图后传播 | 中低 | 中 | 中 | 只能作为 baseline；相关关键帧传播工作很多 |
| 主动多 Anchor + temporal parameter field | 高 | 中 | 中 | 推荐主线 |
| 空间—时间局部参数场 + power windows | 高 | 中高 | 高 | 推荐作为核心增强点 |
| Video retouch world model | 高 | 高 | 高 | 可做第二阶段或完整大论文 |
| 个性化视频审美适应 | 中高 | 中 | 高 | 可作为附加贡献，不宜单独撑主线 |
| 生成式万能视频编辑 Agent | 中 | 极高 | 极高 | 竞争最激烈，不适合当前仓库起步 |

## 10. 最小研究闭环

第一版不需要训练大型视频生成模型：

```text
视频解码
→ shot detection
→ 初始 Anchor 选择
→ Anchor 上运行程序化修图 Agent
→ 拟合 shared grade + temporal residual parameters
→ 光流/跟踪传播局部 mask
→ temporal critic 评分
→ 主动插入新 Anchor
→ 输出视频与参数时间线
```

建议先限制操作空间：

- exposure；
- white balance；
- contrast / tone curve；
- highlights / shadows；
- saturation / vibrance；
- 8-color HSL；
- portrait/sky 两类局部 mask。

现有 4KAgent 可以复用：

- 工具注册与 executor；
- 候选结果树；
- reflection 与 rollback；
- profile 系统；
- 图像质量指标和恢复工具。

需要新增：

- 视频/shot 数据结构；
- 颜色管理和 Log/CST；
- 参数化调色 executor；
- Anchor selector；
- 参数轨迹模型；
- mask tracking；
- temporal critic；
- 视频 benchmark/evaluation。

## 11. 推荐的一句话论文定位

> Existing video grading agents achieve flicker-free results by applying a single global LUT, but cannot adapt to spatially and temporally varying content. We formulate professional video retouching as active inference over an editable spatio-temporal parameter field, where uncertainty-driven anchors provide sparse expert decisions and a temporal counterfactual critic controls propagation and rollback.

