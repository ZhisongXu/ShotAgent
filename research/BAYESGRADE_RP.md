# BayesGrade: Annotation-Free Active Anchor Acquisition for Video Retouching

## Research Proposal

**暂定中文题目：** BayesGrade：面向视频修图的无标注主动 Anchor 获取与贝叶斯参数场  
**目标投稿：** ICCV / CVPR  
**版本：** v0.2，2026-08-08

## 1. 摘要

现有视频调色方法通常从一个关键帧预测全局 LUT，或将单图编辑结果传播到整段视频。这类方法隐含两个限制：第一，单一全局映射无法表达镜头内曝光、白平衡和局部主体状态的动态变化；第二，关键帧通常通过一次性质量排序或语义匹配选出，系统无法在传播失败后主动获取新的视觉证据。本文拟研究 **BayesGrade**：一种数据高效、参数可解释的视频修图 Agent。BayesGrade 将曝光、色温、曲线和局部区域调整表示为带后验不确定性的时空参数场，并把 Anchor 精修视为有成本的信息获取操作。给定少量已精修 Anchor，系统首先利用视频条件 Gaussian Process 推断整段视频的参数均值与方差；进一步在低维 spline 控制点空间运行 GP-preconditioned Langevin dynamics，以融合 Anchor、时序、编辑残差和指令约束并采样非退化参数后验。解析 GP 方差降低和 Langevin 样本分歧共同决定最有信息价值的新 Anchor。系统无需大规模专家视频轨迹、端到端视频生成模型或强化学习策略数据，可直接复用冻结的单图修图器、分割器和跟踪器。我们计划以质量—Anchor 预算曲线为核心评价，验证主动获取能否以更少 Anchor 达到固定采样和一次性关键帧方法的质量，并研究 motion-compensated edit residual 对闪烁、局部 mask 漂移和真实光照变化的区分能力。

## 2. 研究背景与问题

### 2.1 已有研究覆盖

- MonetGPT、JarvisArt 已覆盖操作感知训练、专业工具调用和参数预测；
- RetouchIQ、JarvisEvo 已覆盖通用奖励模型、编辑—评价闭环和反思；
- PhotoAgent 已将单图编辑建模为长时域规划；
- InstantRetouch 已提供基于 bilateral grid 的高保真单图执行器；
- Video Color Grading via LUT 已覆盖 CLIP 关键帧匹配、diffusion LUT 生成及全片共享 LUT；
- LumiVideo 已覆盖 Log 视频分析、RAG、Tree-of-Thought、ASC-CDL/3D LUT 输出和反思；
- SA-LUT 已实现空间自适应 4D LUT；
- Occlusion-Aware Keyframe Selection 已通过结构完整度、循环跟踪稳定性和语义可见性选择单个可靠 Anchor。

因此，下列方案不足以构成主要贡献：

```text
VLM 分析视频 → 选择一个关键帧 → 输出全局 LUT → 反思重试
```

### 2.2 核心研究问题

本项目关注以下问题：

> 在没有大规模专家视频调色轨迹的条件下，如何让 Agent 以尽可能少的 Anchor，主动构建可解释、时间连续且空间自适应的视频修图参数场？

该问题包含三个子问题：

1. **参数表示：** 如何表示镜头共享、帧级动态和对象级局部修图参数？
2. **不确定性：** 如何判断当前 Anchor 是否足以解释整段视频？
3. **主动获取：** 下一次精修哪一帧，能够最大程度降低全视频预测风险？

## 3. 核心假设

### H1：视频修图参数具有低维、分段平滑结构

专业调色轨迹通常由少量稳定区间和变化节点构成，而非逐帧自由变化。曝光、色温和局部调整可以用稀疏 Anchor 加连续参数场近似。

### H2：传播失败可以由后验不确定性预测

光照状态变化、主体遮挡、镜头运动和区域跟踪失败会降低 Anchor 与目标帧的相关性，因此应在参数后验中表现为方差增大。

### H3：信息价值优于一次性关键帧质量

清晰度最高或语义最匹配的帧不一定最能解释整段视频。基于全局后验方差降低的 Anchor acquisition，在相同预算下应优于首帧、固定间隔、聚类和一次性质量排序。

## 4. 方法概述

### 4.1 输入与输出

输入：

```text
视频 V = {I_t}_{t=1}^T
+ 用户指令 u
+ 可选参考图或参考视频 r
```

输出为可编辑 grade graph，而非重新生成的视频：

```text
shot boundaries
+ selected anchors
+ shot-level base grade
+ frame-level parameter curves
+ object masks and local parameter curves
+ posterior uncertainty timeline
+ rollback / reinitialization events
```

参数分解：

\[
\theta_t(x,y)
=
\theta_t^{\mathrm{global}}
+
\sum_r M_t^r(x,y)\theta_t^r,
\]

其中 \(M_t^r\) 为区域或对象 \(r\) 的时空 mask。

### 4.2 冻结工具

第一阶段不训练大型端到端 Agent，直接复用：

- 冻结 VLM：分析场景、用户意图和动态事件；
- 冻结单图修图器：生成 Anchor 的可执行参数；
- 冻结分割/跟踪模型：建立人物、天空和背景轨迹；
- 程序化 executor：应用曝光、白平衡、曲线、HSL 和局部参数。

首版动作空间限制为：

```text
Exposure, Temperature, Tint, Contrast,
Highlights, Shadows, Saturation, Vibrance,
Tone Curve, Local Exposure, Local Temperature, Local Saturation
```

### 4.3 Video-conditioned Bayesian Grade Field

对每个参数维度建模：

\[
\theta(t)\sim\mathcal{GP}(m(t), k(z_i,z_j)),
\]

其中帧状态 \(z_t\) 包含：

```text
normalized timestamp
global luminance/color statistics
semantic/visual embedding
motion magnitude
tracking confidence
shot/event indicators
```

组合 kernel：

\[
k(i,j)=
k_{\mathrm{time}}(i,j)
\cdot
k_{\mathrm{appearance}}(z_i,z_j)
\cdot
k_{\mathrm{track}}(i,j).
\]

给定 Anchor 集合

\[
\mathcal A=\{(t_i,\theta_{t_i})\}_{i=1}^N,
\]

推断得到：

\[
p(\theta_t\mid V,\mathcal A)
=
\mathcal N(\mu_t,\sigma_t^2).
\]

均值作为当前参数预测，方差作为传播风险。

### 4.4 Training-free Active Anchor Acquisition

对未选择候选帧 \(t\)，计算将其加入 Anchor 集合后的预期全局方差降低：

\[
a(t)=
\sum_{\tau=1}^{T}
\left[
\sigma_\tau^2-sigma_{\tau\mid t}^2
\right]
-\lambda_c C(t)
-\lambda_r R(t).
\]

其中：

- \(C(t)\)：单帧精修和局部重新初始化成本；
- \(R(t)\)：模糊、遮挡和跟踪失败风险；
- 第一项可由 GP 后验协方差解析计算，不需要知道候选帧真实参数。

选择：

\[
t^*=\arg\max_t a(t).
\]

停止条件：

```text
integrated posterior variance < threshold
AND temporal critic risk < threshold
OR anchor budget exhausted
```

### 4.5 Motion-compensated Edit Residual

定义编辑残差：

\[
E_t=O_t-I_t,
\]

其中 \(I_t\) 和 \(O_t\) 分别为原始帧和编辑后帧。通过光流对齐比较：

\[
\mathcal L_{\mathrm{edit-res}}
=
\left\|
E_t-W(E_{t-1},F_{t-1\rightarrow t})
\right\|_1.
\]

与直接比较相邻输出帧相比，该指标更有机会区分原视频中的真实光照变化和算法引入的编辑闪烁。

### 4.6 Agent 闭环

```text
ANALYZE_VIDEO
→ INITIALIZE_ANCHOR
→ RETOUCH_ANCHOR
→ FIT_BAYESIAN_FIELD
→ PROPAGATE_PARAMETERS
→ VALIDATE_TEMPORAL_RESULT
→ SELECT_NEXT_ANCHOR / REINITIALIZE_MASK / STOP
→ EXPORT_GRADE_GRAPH
```

Agent 接收结构化工具反馈，例如：

```json
{
  "mean_uncertainty": 0.12,
  "high_risk_intervals": [[43, 57]],
  "recommended_anchor": 48,
  "expected_variance_reduction": 0.37,
  "failure_type": "illumination_regime_change"
}
```

### 4.7 GP-preconditioned Langevin Posterior

解析 GP 提供稳定的线性—高斯基线，但难以表达多模态审美解、参数边界和非线性时序约束。我们以少量 spline 控制点 \(C\) 表示整段轨迹：

\[
\theta_t=B(t)C,
\]

并定义参数空间能量：

\[
E(C)=
\lambda_AE_{\mathrm{anchor}}
+\lambda_PE_{\mathrm{GP}}
+\lambda_TE_{\mathrm{event\text{-}smooth}}
+\lambda_RE_{\mathrm{edit\text{-}residual}}
+\lambda_IE_{\mathrm{instruction}}
+\lambda_BE_{\mathrm{bounds}}.
\]

使用 GP 后验方差构造对角预条件器 \(M\)，运行：

\[
C^{k+1}=C^k-
\frac{\eta}{2}M\nabla_CE(C^k)
+\sqrt{\eta\tau M}\,\epsilon^k,
\quad \epsilon^k\sim\mathcal N(0,I).
\]

这里 \(k\) 是推断迭代，不是视频帧时间。采样得到多条完整参数轨迹 \(\{\theta^{(m)}_{1:T}\}_{m=1}^M\)，其跨样本协方差提供非高斯约束下的传播分歧。首版 acquisition 使用稳健的混合形式：

\[
a_{\mathrm{hybrid}}(t)=
(1-\alpha)\,\bar a_{\mathrm{GP}}(t)
+\alpha\,\bar a_{\mathrm{LD}}(t),
\]

其中 \(a_{\mathrm{LD}}\) 由样本跨帧协方差估计。GP 保留校准良好的基础信息价值，Langevin 项用于暴露 GP kernel 未表达的非线性约束和多模态分歧。

## 5. 数据高效训练策略

### 5.1 不需要训练的模块

| 模块 | 首版策略 |
|---|---|
| VLM | 冻结，使用结构化工具提示 |
| Anchor 修图器 | 冻结现有单图方法 |
| 分割器/跟踪器 | 冻结 |
| Bayesian parameter field | 解析推断 |
| Anchor acquisition | 解析计算 |
| Agent policy | 规则状态机 + VLM 工具调用 |

### 5.2 可选训练模块

1. **Kernel calibration：** 在无标注视频上最大化 marginal likelihood，或通过少量合成参数轨迹选择超参数；
2. **Temporal critic：** 在自然视频上自动注入闪烁、参数跳变、mask 漂移和过度平滑，无需人工标签；
3. **Langevin inference：** 无需训练，仅校准能量权重、温度、步长和采样预算；
4. **VLM LoRA：** 仅在 Agent 工具调用不稳定时，使用少量自动生成轨迹进行微调。

### 5.3 现有数据使用

- 单图修图：MIT-Adobe FiveK、PPR10K 或现有公开修图器；
- 无标注视频：DAVIS、Vimeo-90K、REDS 或已有视频片段；
- 测试：已有视频调色基准、可获得的 Log 视频以及少量人工主观评价。

合成参数轨迹和错误在训练时在线生成，不构建新的大规模离线数据集。

## 6. 实验设计

### 6.1 核心任务

在 Anchor 预算 \(B\in\{1,2,3,5\}\) 下完成视频修图，比较质量—成本关系。

### 6.2 Anchor Baselines

- First frame；
- Uniform sampling；
- K-medoids/feature clustering；
- CLIP reference matching；
- Quality-only ranking；
- Occlusion-aware ranking；
- Maximum current error/uncertainty；
- BayesGrade integrated variance reduction。

### 6.3 Retouching Baselines

- 单一全局 LUT；
- 每镜头全局 LUT；
- Anchor 参数线性插值；
- 时间 spline；
- SA-LUT 逐帧执行；
- BayesGrade global field；
- BayesGrade global + local object fields。

### 6.4 指标

**质量与保真：** LPIPS、DISTS、结构/身份保持、用户偏好。  
**时序：** warping error、motion-compensated edit residual、参数二阶变化。  
**主动获取：** 达到目标质量所需 Anchor 数、quality–budget AUC。  
**不确定性：** NLL、ECE、risk–coverage curve。  
**可编辑性：** 参数数量、关键节点数量、导出成功率。

### 6.5 关键消融

- 时间 kernel vs. 视频条件 kernel；
- 无 appearance feature / 无 tracking confidence；
- 后验方差 vs. 启发式质量分数；
- GP posterior mean vs. MAP vs. Langevin posterior sampling；
- 解析 GP acquisition vs. Langevin disagreement vs. hybrid acquisition；
- 单一 global field vs. global + local fields；
- 直接帧一致性 vs. edit-residual critic；
- 不同冻结 Anchor executor 的跨模型泛化。

### 6.6 预期关键结果

理想主结论：

> BayesGrade 使用 2 个主动 Anchor，达到固定采样或聚类方法使用 5 个 Anchor 的质量，同时产生校准的不确定性和可导出的参数轨迹。

## 7. 预期贡献

1. 将视频修图形式化为预算约束的主动信息获取问题；
2. 提出 video-conditioned Bayesian grade field，统一表示参数预测和传播不确定性；
3. 提出在可编辑参数空间中的 GP-preconditioned Langevin posterior，以融合非高斯约束并表达多解；
4. 提出无需选帧监督的解析/样本混合 Anchor acquisition；
5. 提出基于 motion-compensated edit residual 的无标注时序验证；
6. 建立以质量—Anchor 预算曲线为核心的实验协议。

## 8. 风险与备选方案

### 风险 1：GP 在长视频上为 \(O(T^3)\)

首版只在候选 Anchor/事件节点上建模；后续使用 inducing points、分段 GP 或 state-space GP。

### 风险 2：不确定性与真实误差不相关

加入 appearance/motion/tracking features，并使用在线合成参数轨迹校准；报告 risk–coverage 而非仅可视化方差。

### 风险 3：单图修图器输出不稳定参数

限制为可执行参数空间；对同一 Anchor 多次采样并估计 observation noise；将生成式 executor 仅作为对比。

### 风险 4：局部 mask 传播成为主要误差来源

第一阶段先完成全局参数场；局部区域作为第二阶段扩展，Agent 可以执行 `REINITIALIZE_MASK` 而非强制继续传播。

### 风险 5：Agent 成分被认为只是包装

主要算法贡献放在 posterior belief、active acquisition 和停止策略；VLM 只负责语义状态与工具编排，不宣称为核心算法。

## 9. 里程碑

### M0：最小数学原型（第 1 周）

- Bayesian parameter field；
- 后验均值/方差；
- integrated variance reduction acquisition；
- 合成一维参数轨迹实验。

### M1：真实视频全局参数（第 2–3 周）

- 视频抽帧与颜色统计；
- video-conditioned kernel；
- 曝光/色温参数传播；
- First/Uniform/K-medoids/BayesGrade 对比。

### M2：时序评价（第 4–5 周）

- 程序化参数应用；
- edit residual；
- 自动注入闪烁和参数跳变；
- uncertainty calibration。

### M3：局部对象参数场（第 6–8 周）

- mask/track 接口；
- 人物与天空局部参数；
- mask 失败检测与重新初始化动作。

### M4：Agent 与完整实验（第 9–12 周）

- 结构化工具调用；
- 闭环停止策略；
- 质量—预算曲线；
- 消融、用户研究与论文写作。

## 10. 当前最小实现目标

本仓库第一阶段只实现：

```text
输入：T 帧的时间和低维视觉特征
+ 少量 Anchor 参数

输出：每帧参数均值、后验方差
+ 下一推荐 Anchor
+ 预计全局方差降低
```

通过合成 piecewise-smooth 参数轨迹验证：

1. 加入 Anchor 后后验方差单调下降；
2. 主动 acquisition 不重复选择已有 Anchor；
3. 在相同 Anchor 预算下，主动选帧优于固定均匀采样；
4. appearance feature 能使光照状态突变附近的不确定性上升。

## 参考工作

- [MonetGPT](https://arxiv.org/abs/2505.06176)
- [JarvisArt](https://arxiv.org/abs/2506.17612)
- [RetouchIQ](https://arxiv.org/abs/2602.17558)
- [JarvisEvo](https://arxiv.org/abs/2511.23002)
- [PhotoAgent](https://arxiv.org/abs/2602.22809)
- [InstantRetouch](https://arxiv.org/abs/2606.05071)
- [Video Color Grading via LUT](https://arxiv.org/abs/2508.00548)
- [LumiVideo](https://arxiv.org/abs/2604.02409)
- [SA-LUT](https://arxiv.org/abs/2506.13465)
- [Occlusion-Aware Keyframe Selection](https://arxiv.org/abs/2605.23192)
