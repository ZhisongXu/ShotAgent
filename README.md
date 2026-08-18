# BayesGrade / AnchorRetouchAgent

This repository is being refactored from the original **4KAgent** restoration
system into a data-efficient, interpretable image/video retouching Agent.

The primary video stack is:

```text
Video Perceiver: full-video scan + hierarchical VL shot/Anchor planning
  → rank per-shot Anchors into a global HeroAnchor shortlist
  → develop and approve one HeroAnchor as the master look
  → match every other shot Anchor to the accepted Hero visual reference
  → UCT-MCTS trajectory planner with action memory
  → multiple independent editing Agents / MonetGPT / external tools
  → deterministic RetouchExecutor + Bayesian temporal parameter field
  → multiple visual critics + deterministic temporal safety veto
  → reward back-propagation, Anchor replacement, Hero reselection, and rollback
```

The deployed pipeline follows PhotoAgent's agent-in-the-loop structure while
remaining model-family neutral. Perception, editing, and evaluation can use
different OpenAI-compatible multimodal endpoints or external editing tools.

## What is implemented

- 12-parameter global/local photometric executor;
- mask-local exposure, temperature, and saturation;
- hierarchical long-video VL planner with overlapping windows, boundary
  adjudication, and per-shot Anchor ranking;
- tournament-style HeroAnchor selection across all shot Anchors;
- visual shot matching against the accepted graded HeroAnchor;
- training-free heuristic planner retained only as an explicit ablation/fallback;
- provider-neutral multi-model Agent configuration;
- real UCT-MCTS selection, expansion, simulation, and reward back-propagation;
- competing editing-Agent proposals and independent critic ensemble;
- per-shot action memory, trajectory export, and transactional rollback;
- Anchor parameter covariance output;
- Bayesian temporal parameter field;
- analytic and Langevin-disagreement Anchor acquisition.

The original NeurIPS 2025 4KAgent restoration code is retained under
[`legacy/4kagent`](legacy/4kagent/) for reproducibility, but is no longer part of
the main project import path.

For a clean-machine setup, required downloads, OpenAI/MonetGPT configuration,
and DaVinci Resolve application steps, see the Chinese
[`INSTALLATION_AND_USAGE.zh-CN.md`](INSTALLATION_AND_USAGE.zh-CN.md) guide.

## Video to grading parameters

The primary video entry point accepts a video plus a text instruction and emits
an editable grade graph. The product output is JSON containing shot boundaries,
Anchor parameter keyframes, the shot base grade, a dense 12-D parameter
trajectory, confidence, and rollback history. A normalized source video and the
final parameter-rendered result can also be exported for visual inspection.

```bash
python retouch_video.py \
  --input input.mp4 \
  --instruction "natural warm cinematic grade with protected skin tones" \
  --agent-config configs/photoagent_multi.json \
  --output outputs/input.grade.json \
  --trajectory-output outputs/input.rollouts.jsonl \
  --video-output-dir outputs/input_videos \
  --resolve-package-dir outputs/input_resolve \
  --analysis-max-side 960 \
  --render-max-side 1920
```

The video directory contains `<input>.source.mp4` and `<input>.graded.mp4`.
The Agent can reason over a lightweight proxy while the accepted dense
trajectory is rendered from a separate high-resolution decode. MP4 delivery
uses high-quality H.264; artifacts are currently silent because audio muxing is
outside the grading benchmark contract.

The optional Resolve package contains a static 33³ `.cube` LUT and a time-aware
`.dctl` for every shot, plus a frame-accurate conform manifest and
`apply_dynamic_grade.py`. Anchor frames and trajectory turns are retained while
redundant dense samples are compressed; tune the maximum per-parameter error
with `--resolve-keyframe-error` (default `0.015`).

Resolve's public API can apply LUTs and keyframed DRX grades but cannot create
arbitrary Color-page parameter keyframes. The generated Resolve 19.1+ DCTL uses
`TIMELINE_FRAME_INDEX` to interpolate the exported controls at render time.
After conforming the timeline to exactly one clip per manifest shot, create a
dedicated empty Color node at the same index on every clip, then run:

```bash
python outputs/input_resolve/apply_dynamic_grade.py \
  --lut-dir /path/already/configured/in/resolve \
  --video-track 1 \
  --node-index 2
```

The node index is mandatory so the script cannot silently choose which existing
grade to replace. Direct scripting requires DaVinci Resolve to be running and
its bundled scripting API to be available.

MonetGPT's final adjustment JSON can also be converted without running its
GIMP/NumPy image executor:

```bash
python monet_to_resolve.py \
  --input outputs/monet_adjustments.json \
  --output-dir outputs/monet_resolve
```

The input may be one native MonetGPT adjustment object such as
`{"Exposure": 20, "Highlights": -15, "Saturation": 8}`, or a `shots` array
whose entries contain `shot_id`, optional frame bounds, and `adjustments`.
Global controls are mapped to the shared GradeIR and baked into per-shot
Resolve LUTs. The manifest records approximated and unsupported controls;
use `--strict` to reject selective HSL, dehaze, spatial, or other controls that
the current GradeIR cannot reproduce exactly.

To put this parameter path inside the normal critic/MCTS rollback transaction,
configure it as an editor Agent:

```json
{
  "name": "monet-parameter-editor",
  "type": "monet_parameters",
  "root": "/absolute/path/to/monetGPT",
  "style": "balanced",
  "reject_unsupported": true
}
```

The backend reads MonetGPT's final JSON rather than its rendered TIFF, creates
the exact preview represented by the future Resolve LUT, and submits that
preview to the existing critics. A rejected proposal is discarded by MCTS; if
no Hero/shot trajectory passes, the video grade rolls back to identity.

Copy `configs/photoagent_multi.example.json` to `configs/photoagent_multi.json`
and export `OPENAI_API_KEY`. The supplied runtime uses OpenAI's native Responses
API and one `gpt-5.6-sol` model for the storyboard, editor, and critic roles.
The storyboard path performs a sparse whole-video overview, a full-frame
physical scan, overlapping 20-second VL windows, dense boundary adjudication,
task-aware Anchor ranking inside every verified shot, then selects a global
HeroAnchor. The editor first develops and approves the Hero look; all remaining
Anchors receive both the Hero source and accepted Hero grade as their visual
matching reference. Add
`--allow-storyboard-fallback` only for physical shot-detection ablations.
MonetGPT and any command-line image editor can still be members of the editor
pool.

A fully local training-free baseline is available before configuring any model
endpoint. It uses deterministic shot detection, the native parameter-search
Editor, metric critics, and the same MCTS/rollback path:

```bash
python retouch_video.py \
  --input input.mp4 \
  --instruction "natural brighter grade" \
  --offline-native \
  --output outputs/input.native.grade.json
```

See [`VIDEO_PIPELINE.md`](VIDEO_PIPELINE.md) for the data contract, rollback
semantics, and the exact boundary between integrated and optional components.
See [`evaluation/README.md`](evaluation/README.md) for controlled parameter
probes and the paired SDSD video benchmark.
The frozen five-track benchmark card is
[`evaluation/BENCHMARK.md`](evaluation/BENCHMARK.md).

## Single-image retouching

```bash
python retouch_image.py \
  --input input.jpg \
  --output outputs/retouched.jpg \
  --instruction "make it brighter, warm, and cinematic"
```

Add a local grayscale mask:

```bash
python retouch_image.py \
  --input input.jpg \
  --mask person_mask.png \
  --output outputs/retouched.jpg \
  --instruction "brighten the person and use a restrained cinematic look"
```

## Directional quality gate and rollback

Rollback is enabled by default. The input image is the transaction checkpoint,
but identity is not treated as a preferred result. A candidate is committed
when it produces a perceptible RGB change and moves closer to the planner's
grading target than the input. Clipping and crushing metrics remain in the
metadata for diagnosis, but they do not veto an artistic grade. Generic
`retouch`, `修图`, and automatic-enhancement instructions receive a balanced
default grade with highlight recovery, opened shadows, tonal separation, and
restrained vibrance. An executor or third-party evaluator can still mark a
numerically broken candidate invalid.

The sidecar metadata contains `rolled_back`, `rollback_reason`, and a `decision`
object with the perceptual delta, directional improvement, thresholds, and
candidate counts for every gate.

Require a larger score improvement before committing:

```bash
python retouch_image.py \
  --input input.jpg \
  --output outputs/retouched.jpg \
  --instruction "warm cinematic portrait" \
  --min-improvement 0.1
```

For ablations only, rollback can be disabled with `--no-rollback`.
The visibility threshold can be tuned with `--min-perceptual-delta` (RGB RMS,
default `0.01`). `--min-improvement` controls the minimum target-alignment gain.

Each output image is accompanied by a JSON file containing the selected
parameters, parameter covariance, plan, constraints, and evaluation metrics.

## Bayesian/Langevin prototype

```bash
python -m bayesgrade.demo --frames 120 --budget 5
python -m bayesgrade.demo_langevin --frames 120 --samples 16
python -m bayesgrade.benchmark_synthetic --max-budget 5 --seeds 20
```

Python integration:

```python
from bayesgrade import BayesGradeRetouchPipeline

result = BayesGradeRetouchPipeline().run(
    frames,
    instruction="warm cinematic portrait",
    anchor_indices=[0],
)

print(result.parameter_mean.shape)  # [T, 12]
print(result.next_anchor)
```

## Tests

```bash
python -m unittest discover -s tests
```

## Research documents

- [`research/BAYESGRADE_RP.md`](research/BAYESGRADE_RP.md)
- [`research/BAYESGRADE_EXPERIMENT_LOG.md`](research/BAYESGRADE_EXPERIMENT_LOG.md)
- [`VIDEO_RETOUCH_AGENT_SURVEY.md`](VIDEO_RETOUCH_AGENT_SURVEY.md)
- [`Video_Retouch_Agent_Survey.pptx`](Video_Retouch_Agent_Survey.pptx)
