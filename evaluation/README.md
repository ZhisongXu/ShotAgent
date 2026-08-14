# Training-free evaluation

The formal benchmark definition is [`BENCHMARK.md`](BENCHMARK.md). The unified
runner is `python -m evaluation.suite`; the commands below prepare individual
tracks.

The benchmark evaluates the product contract directly: a video and an
instruction go in, and a dense editable parameter trajectory comes out. It does
not score a generated Anchor image in isolation.

## Metrics

For paired input/reference media the report contains:

- `reference_mae`, `reference_psnr`, and `reference_ssim` for fidelity to the
  target rendition;
- `relative_reference_improvement`, measured against returning the input
  unchanged (positive is better);
- `motion_compensated_temporal_reference_residual`, measuring frame-to-frame
  changes in the reference error after optical-flow alignment (lower is better);
- motion-compensated edit residual plus normalized parameter velocity and jerk;
- accepted-shot rate and rollback count.

If a sample provides `target_parameters`, the report also contains normalized
parameter MAE and active-parameter sign accuracy. If it provides
`expect_rollback`, rollback accuracy is measured as well. Optional
`shot_boundaries` enables tolerance-aware shot-boundary precision, recall, and
F1.

## Fast controlled probe

This takes a real video (or a directory of genuine consecutive frames) that you
are licensed to use, applies known target parameters to every frame, and builds
a manifest. It tests text-to-parameter mapping and temporal parameter stability
independently of a particular retoucher's taste. Still images are deliberately
not accepted by this benchmark.

For the frozen ten-case English/Chinese intent track over a directory of real
videos, use:

```bash
python -m evaluation.prepare_intent_benchmark \
  --video-root /datasets/licensed-videos \
  --limit-videos 100 \
  --max-frames 48 \
  --output outputs/benchmark/intent/manifest.json
```

```bash
python -m evaluation.prepare_parameter_probe \
  --input path/to/source.mp4 \
  --output-root outputs/parameter_probe \
  --instruction "make it brighter and warmer" \
  --parameters '{"exposure": 0.35, "temperature": 0.30}'

python -m evaluation.video_benchmark \
  --manifest outputs/parameter_probe/manifest.json \
  --output outputs/parameter_probe/report.json \
  --grade-output-dir outputs/parameter_probe/grades \
  --video-output-dir outputs/parameter_probe/videos \
  --maximum-evaluations 3
```

Omitting `--agent-config` evaluates the fully local native baseline. Supplying
the normal multi-agent config evaluates the VL training-free system with its
Storyboard, Editor, and Critic roles:

```bash
python -m evaluation.video_benchmark \
  --manifest outputs/parameter_probe/manifest.json \
  --agent-config configs/photoagent_multi.json \
  --output outputs/parameter_probe/vl-report.json
```

## Natural paired-video benchmark: SDSD

[SDSD](https://github.com/dvlab-research/SDSD) contains 70 indoor and 80
outdoor aligned low-light/normal-light dynamic video pairs. The official test
protocol evaluates the first 30 frames of held-out scenes. After downloading
and extracting the PNG data, create the manifest and run:

```bash
python -m evaluation.prepare_sdsd \
  --root /datasets/SDSD \
  --subset all \
  --max-frames 30 \
  --output outputs/sdsd/manifest.json

python -m evaluation.video_benchmark \
  --manifest outputs/sdsd/manifest.json \
  --output outputs/sdsd/offline-native.json \
  --grade-output-dir outputs/sdsd/grades
```

Use the official repository's `testing_dir` scene lists for a paper-comparable
split. `--limit 5` is useful for a quick engineering check. SDSD measures
natural exposure recovery and temporal consistency; it does not measure
open-ended cinematic style or language preference.

## Video Critic and rollback data

[VDPVE](https://arxiv.org/abs/2303.09290) contains 1,211 enhanced videos with
human mean-opinion scores, including 600 color/brightness/contrast examples. It
is appropriate for Spearman/Pearson correlation of the video Critic, but it
does not provide text instructions or ground-truth editable parameters.

There is no public video dataset with branch-level accept/reject decisions and
rollback targets. That track must be constructed on real SDSD/VDPVE or licensed
videos by injecting clipping, crushed shadows, color casts, parameter jumps,
wrong Anchors, and frame flicker, then recording whether the safety Critic
rejects the branch and restores the correct ancestor. Single-image preference
sets are not included in the main Video Agent benchmark.

Create and run the standard video safety track:

```bash
python -m evaluation.prepare_safety_manifest \
  --video-root /datasets/licensed-videos \
  --output outputs/benchmark/safety/manifest.json

python -m evaluation.safety_benchmark \
  --manifest outputs/benchmark/safety/manifest.json \
  --output outputs/benchmark/safety/report.json
```

For AutoShot, convert the official text annotation after downloading the
videos:

```bash
python -m evaluation.prepare_autoshot \
  --video-root /datasets/AutoShot \
  --annotations /datasets/AutoShot/kuaishou_v2.txt \
  --output outputs/benchmark/autoshot/manifest.json
```

No public dataset currently covers video, free-form intent, shot/Anchor labels,
editable grading parameters, multi-step search, and rollback in one package.
The benchmark therefore keeps reference quality, parameter recovery, Critic
calibration, and rollback as separate measurable tracks.

## Manifest format

`input` and `reference` may each be a video file or a directory of ordered image
frames. Relative paths are resolved from the manifest directory.

```json
{
  "dataset": "my paired video set",
  "samples": [
    {
      "id": "clip-001",
      "input": "LQ/clip-001",
      "reference": "GT/clip-001",
      "fps": 24,
      "max_frames": 30,
      "instruction": "recover natural exposure",
      "target_parameters": {"exposure": 0.4},
      "expect_rollback": false
    }
  ]
}
```
