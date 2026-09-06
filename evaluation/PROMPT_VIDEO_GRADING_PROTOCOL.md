# Pure-Prompt Video Grading Protocol

## Task

Given an ungraded source video and a frozen natural-language instruction,
produce a graded video. No reference image, reference video, palette statistic,
LUT, target frame, or pixel ground truth is available to any method.

The first benchmark contains eight prompt-video pairs: two official NLUT source
videos, each graded with four deliberately different instructions (modern
neo-noir, bleach bypass, late-1970s 35 mm, and luxury pastel commercial). Each
sequence uses the first 72 frames at native cadence with the long side capped at
512 pixels. The same source under four prompts tests whether the method follows
language instead of returning one attractive default look.

A separate three-video challenge split uses the official NLUT `night`,
`pedestrian`, and `stream2` sequences with one content-specific complex prompt
per video. Its manifest disables the style-ID-driven preset control so every
ranked editor receives only the source video and prompt.

## Compared methods

| Display name | Paper / implementation | Video adaptation |
|---|---|---|
| ShotAgent Prompt Pool | Ours | Plans the whole source sequence, proposes grades with three API agents, and selects a temporally coherent result. |
| CLIPtone | *CLIPtone: Unsupervised Learning for Text-based Image Tone Adjustment*, Lee et al., CVPR 2024 | Official image model and checkpoints; exact frozen prompt is supplied to every frame, subject to the official CLIP RN50 77-token limit. |
| T2ONet | *Learning by Planning: Language-Guided Global Image Editing*, Shi et al., CVPR 2021 | Official image model and checkpoint; exact frozen prompt is supplied to every frame, subject to the official 15-word vocabulary input. |
| Text2Preset | Frozen non-learned sanity baseline | Prompt category selects one fixed split-tone/contrast/saturation transform for the whole sequence. |
| Identity | No-edit calibration | Copies the source and exposes judges that reward preservation while ignoring the requested style. |

CLIPtone and T2ONet are reported as framewise video adaptations rather than
native video methods. Reference-conditioned LUT/color-transfer methods from the
reference-video benchmark are ineligible because their required conditioning
input is absent.

## Metrics

The primary style metric is an anonymous MLLM review of eight ordered frames,
conditioned on the exact frozen prompt. Candidate identities are deterministically
shuffled. Six separate 1–5 judgments cover black/tonal hierarchy,
shadow/midtone palette, warm-cool relation, saturation hierarchy, highlight
rolloff, and genre mood. Their mean is converted to `LLM Prompt Style` on 0–1
with `(rating - 1) / 4`.

`LLM Quality` is the mean of content preservation, temporal consistency, and
artifact-free judgments, converted to 0–1. `LLM Balanced Overall` gives 50%
weight to the six-field style mean and 50% to the three-field quality mean. The
full judge prompt and per-sample rationales are stored in the evaluator and
report.

`CLIP Prompt Similarity` is the unscaled cosine similarity between the frozen
prompt and eight uniformly sampled output frames using CLIP RN50, averaged per
sequence. It is an auxiliary style signal rather than the sole rank criterion:
it has limited sensitivity to the prompt's detailed tonal and regional color
relationships, and CLIPtone directly optimizes a CLIP-based objective.

Independent objective diagnostics are also reported:

| Axis | Metric | Direction |
|---|---|---|
| Content | DINOv2 content similarity | Higher |
| Content | Local-structure correlation | Higher |
| Content | Edge-SSIM | Higher |
| Temporal | Source-flow output warping error | Lower |
| Temporal | Edit-field warping error | Lower |
| Temporal | Fitted transform drift | Lower |
| Quality | MUSIQ no-reference quality | Higher |
| Artifact | Newly introduced shadow clipping | Lower |
| Artifact | Newly introduced highlight clipping | Lower |

Mean normalized Lab edit magnitude is retained only as a diagnostic. It is not
a quality metric and does not enter any ranking because the correct strength is
prompt- and content-dependent.

## Reproduction

Generate ShotAgent outputs:

```bash
.venv/bin/python -m evaluation.video_benchmark \
  --manifest evaluation/manifests/prompt_video_grading_v1.json \
  --agent-config configs/prompt_video_pool.json \
  --output outputs/prompt_video_eval/v1/shotagent/report.json \
  --grade-output-dir outputs/prompt_video_eval/v1/shotagent/grades \
  --video-output-dir outputs/prompt_video_eval/v1/shotagent/videos \
  --maximum-evaluations 4 --fail-fast
```

Run official image baselines with the repository adapters, then evaluate all
methods together:

```bash
.venv/bin/python -m evaluation.prompt_video_benchmark \
  --manifest evaluation/manifests/prompt_video_grading_v1.json \
  --output-dir outputs/prompt_video_eval/v1/benchmark \
  --external 'ShotAgent Prompt Pool=outputs/prompt_video_eval/v1/shotagent/videos' \
  --external 'CLIPtone (CVPR 2024)=outputs/prompt_video_eval/v1/baseline_outputs/cliptone' \
  --external 'T2ONet (CVPR 2021)=outputs/prompt_video_eval/v1/baseline_outputs/t2onet' \
  --learned-metrics --judge-model gpt-5.6-sol
```

The benchmark writes `report.json`, flat and aggregate CSV tables, anonymization
keys, per-sample videos, and source/method comparison videos.

For the smaller official-video challenge, substitute
`prompt_video_grading_nlut3_v1.json` in the same commands and use a separate
output directory such as `outputs/prompt_video_eval/nlut3_v1`.
