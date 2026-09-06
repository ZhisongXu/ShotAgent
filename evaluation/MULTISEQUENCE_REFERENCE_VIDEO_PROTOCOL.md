# Multi-sequence reference-video color-grading benchmark

This protocol evaluates the rendered-video contract

```text
target video + reference video -> graded target video
```

without pixel-aligned ground truth. It follows the common video photorealistic
style-transfer practice of evaluating multiple independent target sequences,
using one fixed reference-conditioned transform over each complete sequence,
and computing every metric per sequence before macro-averaging.

## Data and pairing

- Nine 150-frame, 1080p target videos come from the official
  [NLUT repository](https://github.com/semchan/NLUT): `city`, `city2`, `girl`,
  `kelly`, `monkey`, `night`, `pedestrian`, `stream2`, and `sunset`.
- Three reference videos come from the official
  [Video Color Grading repository](https://github.com/seunghyuns98/VideoColorGrading/tree/main/examples).
- A balanced fixed rotation assigns each reference to three targets. This
  produces nine target/reference trials without weighting one reference more
  heavily.
- Each trial uses the first 96 target frames at the native 25 fps. The long
  side is limited to 512 pixels for all methods.
- Methods whose published interface accepts one style image use the temporal
  middle frame of the reference video. ShotAgent consumes an ordered reference
  storyboard and a profile measured over sampled reference frames.

The executable manifest is
`evaluation/manifests/reference_video_multisequence_v1.json`.

## Methods

1. **SA-LUT: Spatial Adaptive 4D Look-Up Table for Photorealistic Style
   Transfer** (ICCV 2025), official checkpoint.
2. **NLUT: Neural-based 3D Lookup Tables for Video Photorealistic Style
   Transfer** (2023), official checkpoint and 40-step test-time adaptation for
   every target/reference pair.
3. **CAP-VSTNet: Content Affinity Preserved Versatile Style Transfer**
   (CVPR 2023), official photorealistic-video checkpoint.
4. **CanonCGT: Reference-Based Color Grading via Canonical Pivot
   Representation** (CVPR 2026), official SSL checkpoint.
5. **ModFlows: Modulated Normalizing Flows for Image and Video Style
   Transfer** (AAAI 2025), official encoder checkpoint and one fixed mapping
   per sequence.
6. **Video Color Grading via Look-Up Table Generation** (ICCV 2025), official
   GS-Extractor and L-Diffuser checkpoints.
7. **ShotAgent API Editor Pool**, three API editor roles plus deterministic
   safety and reference-affinity tools. It does not consume a baseline's output
   or weights.

All video outputs use the same frame count, cadence, spatial limit, and
high-quality H.264 encoding. Evaluation reads final RGB videos only; LUTs,
parameter trajectories, and method internals are not scored.

## Aggregation and ranking

Every objective metric is computed separately for each of the nine sequences.
The report contains the unweighted sequence mean, sample standard deviation,
normal 95% interval, and mean per-sequence rank. Lower average rank is better.
There is no cross-axis composite score.

The LLM style score is a development diagnostic. For each sequence, one
  independent judge sees the target, reference, and seven anonymous candidate
storyboards. Reference-style similarity is the equal-weight mean of nine
fields: deep-shadow black level, shadow chroma, midtone luminance, midtone
palette, highlight roll-off, neutral-axis temperature, palette hierarchy,
saturation hierarchy, and local contrast/depth. A formal claim still requires
the supplied blinded A/B human-review forms with at least three raters.

## Results on the nine-sequence protocol

Values are macro mean ± sample standard deviation. LLM overall grade quality
uses the fixed 0--1 mapping `(rating - 1) / 4`. The original 1--5 mean, its four
component ratings, and the style-only LLM score remain in the machine-readable
report.

### Headline grading evaluation

| Paper method | LLM overall grade quality ↑ (0--1) | VGG style ↑ | Lab chroma BC ↑ |
|---|---:|---:|---:|
| SA-LUT | 0.5975 ± 0.1286 | 0.8966 ± 0.0322 | 0.7744 ± 0.1477 |
| NLUT | 0.7975 ± 0.0700 | 0.9472 ± 0.0233 | 0.9341 ± 0.0317 |
| CAP-VSTNet | 0.6146 ± 0.1543 | **0.9757 ± 0.0107** | **0.9654 ± 0.0150** |
| CanonCGT | 0.8409 ± 0.0302 | 0.9063 ± 0.0383 | 0.7782 ± 0.1865 |
| ModFlows | 0.6709 ± 0.1551 | 0.9372 ± 0.0353 | 0.9462 ± 0.0215 |
| Video Color Grading | 0.6762 ± 0.0963 | 0.8720 ± 0.0495 | 0.7112 ± 0.1202 |
| **ShotAgent API Editor Pool** | **0.8505 ± 0.0378** | 0.9114 ± 0.0384 | 0.9095 ± 0.0609 |

### Content, temporal stability, artifacts, and quality

| Paper method | Structure ↑ | DINO ↑ | Edge-SSIM ↑ | Flow warp ↓ | Edit warp ↓ | Drift ↓ | New shadow clip ↓ | New highlight clip ↓ | MUSIQ ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SA-LUT | 0.9694 ± 0.0124 | 0.9551 ± 0.0246 | 0.6635 ± 0.1624 | 0.01542 ± 0.00780 | 0.00784 ± 0.00408 | 0.02043 ± 0.00995 | 8.875% ± 10.056% | 0.030% ± 0.077% | 63.81 ± 2.46 |
| NLUT | 0.9623 ± 0.0124 | 0.9545 ± 0.0178 | 0.6553 ± 0.1475 | 0.00906 ± 0.00497 | 0.00596 ± 0.00193 | 0.01147 ± 0.00622 | 4.207% ± 5.915% | 0.018% ± 0.041% | 62.08 ± 4.62 |
| CAP-VSTNet | 0.9197 ± 0.0535 | 0.8764 ± 0.0660 | 0.6240 ± 0.2042 | **0.00750 ± 0.00437** | 0.00688 ± 0.00246 | 0.01033 ± 0.00443 | 3.270% ± 4.510% | 0.019% ± 0.037% | 57.95 ± 6.64 |
| CanonCGT | 0.9918 ± 0.0056 | **0.9944 ± 0.0015** | 0.8122 ± 0.1048 | 0.01028 ± 0.00380 | 0.00627 ± 0.00282 | 0.02974 ± 0.02122 | 3.352% ± 6.959% | **0.000% ± 0.000%** | **65.10 ± 4.46** |
| ModFlows | 0.9508 ± 0.0272 | 0.9072 ± 0.0873 | 0.5812 ± 0.2222 | 0.01288 ± 0.01231 | 0.00889 ± 0.00893 | 0.01183 ± 0.00841 | 6.652% ± 8.098% | 0.216% ± 0.478% | 59.17 ± 4.36 |
| Video Color Grading | 0.9616 ± 0.0169 | 0.9243 ± 0.0397 | 0.7330 ± 0.0761 | 0.01190 ± 0.00437 | 0.00694 ± 0.00195 | 0.01641 ± 0.00852 | 2.210% ± 3.737% | **0.000% ± 0.000%** | **65.61 ± 3.24** |
| **ShotAgent API Editor Pool** | **0.9920 ± 0.0031** | 0.9686 ± 0.0255 | **0.8823 ± 0.0580** | 0.01148 ± 0.00434 | **0.00389 ± 0.00114** | **0.00672 ± 0.00314** | **0.197% ± 0.426%** | **0.000% ± 0.000%** | 64.74 ± 2.64 |

### Mean per-sequence rank by axis

| Method | Overall ↓ | Style (3 metrics) ↓ | Content (3) ↓ | Temporal (3) ↓ | Quality/artifact (3) ↓ |
|---|---:|---:|---:|---:|---:|
| SA-LUT | 5.037 | 5.889 | 3.926 | 5.593 | 4.741 |
| NLUT | 3.648 | 2.685 | 4.593 | 3.074 | 4.241 |
| CAP-VSTNet | 3.903 | **1.667** | 5.741 | 3.296 | 4.907 |
| CanonCGT | 3.537 | 4.741 | 1.667 | 4.481 | 3.259 |
| ModFlows | 4.375 | 3.130 | 5.630 | 3.741 | 5.000 |
| Video Color Grading | 4.810 | 6.148 | 4.852 | 5.296 | 2.944 |
| **ShotAgent API Editor Pool** | **2.690** | 3.741 | **1.593** | **2.519** | **2.907** |

ShotAgent leads LLM overall grade quality and the content and temporal axis
ranks. CAP-VSTNet leads pure reference-style similarity, while its content and
quality losses keep it behind on the equal-axis overall rank. Equal weighting
of the four axis ranks gives ShotAgent the best overall result. The remaining
weakness is reference feature/distribution matching rather than content
preservation or temporal stability.

## Reproduction

Render all methods:

```bash
python -m evaluation.run_reference_video_suite \
  --manifest evaluation/manifests/reference_video_multisequence_v1.json \
  --output-root outputs/reference_video_eval/multisequence_v1
```

Run the no-GT evaluator with the seven rendered output directories as
`--external METHOD=DIR`. The generated benchmark directory contains
`results.csv`, `aggregate.csv`, `report.json`, nine comparison videos, and
anonymous individual/pairwise review sheets.
