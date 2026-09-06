# AutoShot-Hard reference-video grading benchmark

This benchmark is a second, independent no-ground-truth track for methods with
the rendered-video contract

```text
target video + reference video -> graded target video
```

It is designed to stress complete-video reasoning rather than performance on a
single representative frame. It retains dense real shot changes from the
official AutoShot SHOT test set and uses complete official Video Color Grading
example videos as references.

## Frozen hard-subset selection

The subset is selected before any grading method is run. Eligible videos must
contain a 144-frame window with at least three annotated shot boundaries. Each
eligible video receives the following output-independent hardness score, using
percentile ranks computed over all locally available annotated videos.

| Component | Weight | Measurement |
|---|---:|---|
| Shot density | 40% | Annotated shots per second |
| Luminance swing | 25% | 90th minus 10th percentile of sampled frame luminance |
| Chroma swing | 20% | Variation of sampled Lab chroma centroids |
| Visual change | 15% | Mean sampled grayscale frame difference |

The 12 highest-scoring videos are used, with ties resolved by filename. For
each selected video, the densest 144-frame window is extracted. Three reference
videos are assigned in a fixed balanced rotation. This selection deliberately
targets heterogeneous multi-shot footage but cannot depend on model outputs or
benchmark scores.

## Methods

| Method | Venue | Reference input | Video policy |
|---|---|---|---|
| SA-LUT: Spatial Adaptive 4D Look-Up Table for Photorealistic Style Transfer | ICCV 2025 | Middle reference frame | Official checkpoint |
| NLUT: Neural-based 3D Lookup Tables for Video Photorealistic Style Transfer | 2023 | Middle reference frame | Official checkpoint and 40-step adaptation |
| CAP-VSTNet: Content Affinity Preserved Versatile Style Transfer | CVPR 2023 | Middle reference frame | Official photo-video checkpoint |
| CanonCGT: Reference-Based Color Grading via Canonical Pivot Representation | CVPR 2026 | Middle reference frame | Official SSL checkpoint |
| ModFlows: Modulated Normalizing Flows for Image and Video Style Transfer | AAAI 2025 | Middle reference frame | One fixed official flow mapping per sequence |
| Video Color Grading via Look-Up Table Generation | ICCV 2025 | Middle reference frame selected from the video | Official diffusion-LUT checkpoints |
| ShotAgent API Editor Pool | current | Ordered reference storyboard and profile | Shot-aware analysis, safety and rendering pool |

All methods receive the same extracted target frames, reference video, maximum
spatial size and final encoding quality. Evaluation reads only final RGB
videos. Method internals, LUTs and parameter trajectories are not scored.

## Metrics and reporting

The headline grading table reports LLM overall grade quality, VGG style
similarity and Lab chroma Bhattacharyya coefficient on 0--1 scales. The second
table reports structure, DINO, Edge-SSIM, cut-masked temporal metrics, clipping
and MUSIQ. Each value is computed per sequence and macro-averaged over the 12
sequences. Anonymous LLM judging uses the same frozen prompt and candidate
ordering policy as the existing nine-sequence benchmark.

## Results

Values are macro mean ± sample standard deviation over the 12 frozen
sequences. LLM overall quality uses the fixed 0--1 mapping `(rating - 1) / 4`.
Style rank averages the per-sequence ranks of LLM reference-style match, VGG
style similarity, and Lab chroma BC. Overall rank first averages metrics within
the style, content, temporal, and quality/artifact axes, then gives the four
axes equal weight. Lower rank is better. This avoids mixing raw values with
different units and does not normalize scores after seeing the outputs.

### Headline grading and composite ranks

| Paper method | LLM overall quality ↑ | VGG style ↑ | Lab chroma BC ↑ | Style rank ↓ | Overall rank ↓ |
|---|---:|---:|---:|---:|---:|
| SA-LUT (ICCV 2025) | 0.7250 ± 0.0908 | 0.8030 ± 0.0390 | 0.6860 ± 0.0643 | 6.194 | 4.722 |
| NLUT (2023) | 0.7119 ± 0.0884 | 0.8404 ± 0.0414 | 0.8416 ± 0.0627 | 3.556 | 3.854 |
| CAP-VSTNet (CVPR 2023) | 0.6414 ± 0.1065 | **0.9264 ± 0.0238** | **0.9608 ± 0.0152** | **1.611** | 4.160 |
| CanonCGT (CVPR 2026) | 0.8015 ± 0.0409 | 0.8088 ± 0.0398 | 0.7048 ± 0.0631 | 5.306 | 3.698 |
| ModFlows (AAAI 2025) | 0.5675 ± 0.1490 | 0.8924 ± 0.0438 | 0.9353 ± 0.0348 | 2.778 | 4.635 |
| Video Color Grading (ICCV 2025) | 0.7080 ± 0.0888 | 0.7963 ± 0.0433 | 0.7788 ± 0.1138 | 5.167 | 4.076 |
| **ShotAgent API Editor Pool** | **0.8448 ± 0.0416** | 0.8219 ± 0.0409 | 0.8491 ± 0.0369 | 3.389 | **2.854** |

### Content, temporal stability, artifacts, and quality

| Paper method | Structure ↑ | DINO ↑ | Edge-SSIM ↑ | Flow warp ↓ | Edit warp ↓ | Drift ↓ | New shadow clip ↓ | New highlight clip ↓ | MUSIQ ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SA-LUT | 0.9732 ± 0.0116 | 0.9656 ± 0.0136 | 0.8364 ± 0.0686 | 0.03136 ± 0.03302 | **0.01174 ± 0.01151** | 0.03632 ± 0.02649 | 2.323% ± 2.129% | 1.322% ± 2.821% | 64.37 ± 4.86 |
| NLUT | 0.9257 ± 0.0330 | 0.9053 ± 0.0296 | 0.7323 ± 0.0761 | 0.01514 ± 0.01421 | 0.01702 ± 0.01910 | 0.02671 ± 0.01825 | 0.050% ± 0.173% | **0.000% ± 0.000%** | 63.68 ± 5.49 |
| CAP-VSTNet | 0.9289 ± 0.0168 | 0.8096 ± 0.0581 | 0.6497 ± 0.1350 | **0.01273 ± 0.00970** | 0.02170 ± 0.02407 | 0.02773 ± 0.01717 | 0.489% ± 0.869% | **0.000% ± 0.000%** | 57.26 ± 6.74 |
| CanonCGT | 0.9919 ± 0.0040 | **0.9849 ± 0.0080** | 0.8986 ± 0.0367 | 0.02389 ± 0.02065 | 0.01322 ± 0.01199 | 0.03787 ± 0.01479 | 0.395% ± 0.650% | **0.000% ± 0.000%** | **66.10 ± 5.05** |
| ModFlows | 0.8969 ± 0.0726 | 0.8410 ± 0.0492 | 0.6772 ± 0.1386 | 0.01612 ± 0.01231 | 0.02120 ± 0.02335 | 0.03451 ± 0.02570 | 6.582% ± 8.559% | 0.860% ± 2.816% | 59.66 ± 7.37 |
| Video Color Grading | 0.9720 ± 0.0105 | 0.8793 ± 0.0522 | 0.8263 ± 0.0491 | 0.02032 ± 0.02310 | 0.01519 ± 0.01637 | 0.02471 ± 0.02156 | 0.050% ± 0.082% | **0.000% ± 0.000%** | 63.27 ± 4.55 |
| **ShotAgent API Editor Pool** | **0.9925 ± 0.0040** | 0.9513 ± 0.0268 | **0.9184 ± 0.0333** | 0.02378 ± 0.02384 | 0.01385 ± 0.01473 | **0.01691 ± 0.00824** | **0.015% ± 0.020%** | **0.000% ± 0.000%** | 65.80 ± 4.69 |

ShotAgent has the best equal-axis overall rank and the highest LLM overall
quality. It leads structure, Edge-SSIM, transform drift, and new shadow clipping
while remaining within 0.30 MUSIQ points of the best method. CAP-VSTNet leads
the three-metric style rank because its feature and chroma distributions match
the reference most closely, but its lower LLM overall quality, DINO, Edge-SSIM,
and MUSIQ show the cost of treating raw distribution proximity as the complete
grading objective.

## Preparation

```bash
python -m evaluation.prepare_autoshot_hard_reference \
  --video-root /datasets/AutoShot \
  --annotations /datasets/AutoShot/kuaishou_v2.txt \
  --reference outputs/reference_video_eval/official_demos/official_sources/video1.mp4 \
  --reference outputs/reference_video_eval/official_demos/official_sources/video2.mp4 \
  --reference outputs/reference_video_eval/official_demos/official_sources/video3.mp4 \
  --output outputs/reference_video_eval/autoshot_hard_v1/manifest.json

python -m evaluation.run_reference_video_suite \
  --manifest outputs/reference_video_eval/autoshot_hard_v1/manifest.json \
  --output-root outputs/reference_video_eval/autoshot_hard_v1
```
