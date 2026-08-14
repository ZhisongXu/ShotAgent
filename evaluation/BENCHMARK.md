# VideoGradeBench-v1 benchmark card

## Scope

VideoGradeBench evaluates one product contract:

```text
input video + natural-language intent -> editable per-shot parameter graph
```

It is not an image-retouching benchmark. A still-image path is rejected. Frame
directories are accepted only as decoded consecutive frames from a video.

The benchmark is training-regime neutral: a training-free Agent and a trained
Agent run exactly the same manifests and emit the same report schema.

## Tracks

| Track | Dataset/protocol | Primary capability | Primary score |
|---|---|---|---|
| Intent/Parameter | standard bilingual parameter probes on real videos | text to executable parameters | intent score |
| Paired Quality | SDSD LQ/GT video pairs | natural enhancement and temporal consistency | paired-quality score |
| Storyboard/Anchor | AutoShot SHOT plus controlled cuts | shot boundaries and Anchor coverage | storyboard score |
| Safety/Rollback | injected failures on real motion videos | reject unsafe branches and preserve safe ones | balanced rollback accuracy |
| Critic/MOS | VDPVE source/enhancement/MOS | agreement with human video-quality judgement | mapped Spearman correlation |

The full overall score is the geometric mean of available track scores. A
partial run must report track coverage and must not be compared directly with a
five-track result.

## Frozen v1 score definitions

All component values are clipped to `[0, 1]` before weighting.

Intent/Parameter:

```text
0.35 * active parameter sign accuracy
+ 0.25 * (1 - normalized parameter MAE / 0.10)
+ 0.25 * relative improvement over identity
+ 0.15 * (1 - motion-compensated edit residual / 0.02)
```

Paired Quality:

```text
0.35 * reference SSIM
+ 0.35 * relative improvement over identity
+ 0.15 * (1 - motion-compensated temporal reference residual / 0.02)
+ 0.15 * (1 - normalized parameter jerk / 0.02)
```

Storyboard/Anchor:

```text
0.80 * shot-boundary F1
+ 0.20 * (1 - normalized Anchor coverage error / 2)
```

Safety/Rollback is balanced accuracy over unsafe recall and safe specificity.
Critic/MOS is `(Spearman + 1) / 2`. The constants are engineering thresholds,
not learned weights; changing them requires a benchmark version bump.

## Standard intent cases

`prepare_intent_benchmark.py` defines ten fixed cases: brighter, warm, cool,
vivid, and cinematic, each in English and Chinese. Every case runs on real
motion video. The reference is rendered from the declared 12-D target
parameters at evaluation time, so this track measures the parameter contract,
not subjective cinematic taste.

Reports include language and intent slices. Video IDs—not clips or frames—must
be the unit of any train/validation/test split.

## Natural paired-video protocol

[SDSD](https://github.com/dvlab-research/SDSD) provides aligned low-light and
normal-light dynamic video pairs: 70 indoor and 80 outdoor pairs. Follow the
official held-out scene lists and evaluate the first 30 frames of each scene.
The benchmark reports reference MAE/PSNR/SSIM, improvement over returning the
input unchanged, optical-flow-compensated temporal error, parameter velocity,
jerk, and rollback rate.

SDSD covers exposure recovery, not open-ended artistic styles. Its score must
not be presented as a complete grading score.

## Storyboard protocol

[AutoShot SHOT](https://github.com/wentaozhu/AutoShot) contains 853 short videos
and 11,606 shot annotations, including 2,716 high-quality boundaries in 200 test
videos. Gradual-transition annotations are converted to the first stable frame
after the transition, with a two-frame matching tolerance. The controlled-cut
probe uses one-frame tolerance.

Anchor coverage is the mean nearest-Anchor distance in normalized luminance,
contrast, saturation, and warmth feature space. It does not claim to be a human
Anchor-preference label; it measures whether selected Anchors cover the visual
states of their shots.

## Safety and rollback protocol

Every real video receives five deterministic candidate trajectories:

1. safe mild constant grade;
2. extreme overexposure;
3. crushed shadows;
4. alternating exposure/color flicker;
5. a single-frame parameter spike.

Unsafe cases must be rejected; the safe case must be retained. The benchmark
tests the Critic decision used by rollback. Transactional restoration to the
ancestor parameter graph remains covered by the pipeline integration tests.

## Video Critic protocol

[VDPVE](https://arxiv.org/abs/2303.09290) contains 1,211 human-rated enhanced
videos, including 600 color/brightness/contrast samples. The track computes
Pearson and Spearman correlation between Critic score and MOS. Source and
candidate videos are both required; evaluating candidate frames without their
source is a different task and is rejected by the manifest converter.

## Reproducibility

Every report records the source manifest SHA-256, sanitized model/runtime IDs,
and suite reports additionally record Python, NumPy, OpenCV, PyTorch, and
platform versions. API keys are never written. Report JSON and the exact
manifest must be retained together.

## Running the suite

Prepare the individual manifests, copy
`evaluation/configs/videogradebench_v1.example.json`, fix its paths, then run:

```bash
python -m evaluation.suite \
  --suite path/to/videogradebench_v1.json \
  --agent-config configs/photoagent_multi.json \
  --output outputs/benchmark/report.json \
  --markdown outputs/benchmark/report.md \
  --artifact-dir outputs/benchmark/report_artifacts \
  --fail-fast
```

Omit `--agent-config` for the deterministic offline training-free baseline. A
model-endpoint failure is a failed sample; the benchmark does not silently
replace a VL model with the native baseline. The artifact directory contains a
report for every track and, for agent tracks, one executable `.grade.json`
parameter graph plus input/result MP4 artifacts per input video. MP4 artifacts
contain the exact evaluated frames and are silent. If `--artifact-dir` is
omitted, the suite uses `<output-stem>_artifacts` next to the JSON scorecard.
