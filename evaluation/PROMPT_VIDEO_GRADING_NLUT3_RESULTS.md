# NLUT three-video pure-prompt challenge

This small challenge split uses three additional real videos from the official
NLUT repository: `night`, `pedestrian`, and `stream2`. Each method receives only
the source video and one frozen, content-specific grading instruction. Every
sequence contains the first 72 frames at native cadence, resized to a maximum
long side of 512 pixels. No reference media, LUT, palette statistics, ground
truth, or manifest style label is available to a competing method.

Identity is retained only as a no-edit calibration. The style-ID-driven
Text2Preset control is disabled for this split because reading the manifest's
category label would violate the source-video-plus-prompt condition.

## Prompt adherence and perceived quality

All scores are macro means over three videos. LLM candidates are anonymous and
deterministically shuffled. Balanced Overall gives 50% weight to six-field LLM
Style and 50% to three-field LLM Quality.

| Method | CLIP Prompt Sim ↑ | LLM Style ↑ | LLM Quality ↑ | Balanced Overall ↑ | Overall Preference ↑ |
|---|---:|---:|---:|---:|---:|
| Identity (calibration) | 0.1729 | 0.4528 | **0.9500** | 0.7014 | 0.4917 |
| **ShotAgent Prompt Pool** | 0.1805 | **0.6875** | 0.8639 | **0.7757** | **0.7333** |
| CLIPtone (CVPR 2024) | **0.1947** | 0.4000 | 0.6889 | 0.5444 | 0.4250 |
| T2ONet (CVPR 2021) | 0.1779 | 0.3611 | 0.6778 | 0.5194 | 0.3833 |

## Content, temporal stability, and technical quality

| Method | DINO ↑ | Structure Corr. ↑ | Edge-SSIM ↑ | Flow Warp ↓ | Edit-field Warp ↓ | Transform Drift ↓ | MUSIQ ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Identity (calibration) | **1.0000** | **1.0000** | **1.0000** | 0.01292 | **0.00000** | **0.00000** | **65.98** |
| **ShotAgent Prompt Pool** | **0.9799** | 0.9837 | 0.7748 | **0.01248** | **0.00571** | **0.00862** | 64.64 |
| CLIPtone (CVPR 2024) | 0.9727 | **0.9850** | **0.8148** | 0.01423 | 0.00588 | 0.00981 | **65.43** |
| T2ONet (CVPR 2021) | 0.9609 | 0.9649 | 0.6648 | 0.01532 | 0.01033 | 0.02691 | 63.74 |

Bold objective values compare the three edited methods; Identity is a trivial
preservation ceiling. ShotAgent leads four of seven objective columns among the
edited methods. CLIPtone is slightly higher on structure, Edge-SSIM, and MUSIQ;
its stronger yellow-green casts are visible in the comparison videos and lower
its anonymous prompt-style and artifact judgments.

The complete report is at
`outputs/prompt_video_eval/nlut3_v1/benchmark/report.json`. Each sample directory
contains the source, every method output, and a labeled `comparison.mp4`.
