# Pure-prompt video color-grading pilot results

This no-GT pilot applies four frozen prompts to two official NLUT source videos
(`girl` and `city`), for eight prompt-video sequences and 576 evaluated frames
per method. The methods receive the source video and text only. No reference
image, reference video, LUT, palette statistics, or hidden visual prior is used.

`Identity` and `Text2Preset` are sanity controls. CLIPtone and T2ONet use their
official published repositories and checkpoints in a fixed-prompt framewise
video adaptation. ShotAgent uses the three-editor prompt pool in direct API
mode, with the pool critic selecting among proposed grades.

## Prompt adherence and perceived quality

All values are macro means over eight sequences. LLM candidates were anonymous
and deterministically shuffled. LLM Style is the mean of six explicit grading
dimensions; LLM Quality is the mean of preservation, temporal consistency, and
artifact freedom. Balanced Overall weights these two category means equally.
CLIP Prompt Sim is the unscaled mean cosine from CLIP RN50 over eight uniformly
sampled frames; it is auxiliary because CLIPtone is itself optimized with CLIP.

| Method | CLIP Prompt Sim ↑ | LLM Style ↑ | LLM Quality ↑ | Balanced Overall ↑ | Overall Preference ↑ |
|---|---:|---:|---:|---:|---:|
| Identity (sanity) | 0.1231 | 0.6005 | **0.9510** | 0.7758 | 0.6313 |
| Text2Preset (sanity) | 0.1268 | 0.6661 | 0.8615 | 0.7638 | 0.6813 |
| **ShotAgent Prompt Pool** | 0.1311 | **0.7323** | 0.8823 | **0.8073** | **0.7563** |
| CLIPtone (CVPR 2024) | **0.1388** | 0.3505 | 0.7031 | 0.5268 | 0.3531 |
| T2ONet (CVPR 2021) | 0.1309 | 0.6052 | 0.8448 | 0.7250 | 0.6156 |

ShotAgent ranks first on literal prompt style, the equal-weight balanced score,
and anonymous overall preference. CLIPtone ranks first on CLIP similarity, but
its low six-dimensional style judgment shows why CLIP cosine alone cannot
measure detailed color relationships, tonal hierarchy, or artifact control.

## Content, temporal stability, and technical quality

| Method | DINO ↑ | Structure Corr. ↑ | Edge-SSIM ↑ | Flow Warp ↓ | Edit-field Warp ↓ | Transform Drift ↓ | MUSIQ ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Identity (sanity) | **1.0000** | **1.0000** | **1.0000** | 0.01141 | **0.00000** | **0.00000** | **70.05** |
| Text2Preset (sanity) | 0.9921 | 0.9953 | 0.9283 | **0.01130** | 0.00260 | 0.00274 | 69.38 |
| **ShotAgent Prompt Pool** | 0.9799 | 0.9811 | 0.8853 | 0.01197 | 0.00507 | 0.01270 | 68.82 |
| CLIPtone (CVPR 2024) | 0.9657 | 0.9809 | 0.8617 | 0.01373 | 0.00609 | 0.01748 | 67.67 |
| T2ONet (CVPR 2021) | 0.9778 | 0.9830 | 0.8992 | 0.01260 | 0.00534 | 0.01697 | 68.59 |

The identity control gives trivial upper bounds for preservation metrics because
it performs no requested edit. Among the three learned methods, ShotAgent is
best on DINO, flow warping, edit-field warping, transform drift, and MUSIQ;
T2ONet is slightly higher on structure correlation and Edge-SSIM. ShotAgent's
new-shadow clipping fraction is 0.02658 and new-highlight clipping fraction is
0.00002; edit magnitude is reported only as a diagnostic and never rewarded.

The machine-readable report is written to
`outputs/prompt_video_eval/v1/benchmark/report.json`, the flat tables to
`results.csv` and `aggregate.csv`, and each sample directory contains a labeled
`comparison.mp4`.
