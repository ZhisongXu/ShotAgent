# BayesGrade / AnchorRetouchAgent

This repository is being refactored from the original **4KAgent** restoration
system into a data-efficient, interpretable image/video retouching Agent.

The primary video stack is exposed through one `UnifiedVLVideoBackend`:

```text
Video Perceiver: full-video scan + hierarchical VL shot/Anchor planning
  → rank per-shot Anchors into a global HeroAnchor shortlist
  → develop and approve one HeroAnchor as the master look
  → match every other shot Anchor to the accepted Hero visual reference
  → one VL editor producing sparse typed Grade Pool operations
  → deterministic Pool executor + type-aware Bayesian temporal fields
  → the same VL client reviews previews + deterministic temporal safety veto
  → Anchor replacement, Hero reselection, and transactional rollback
```

The public runtime uses one model endpoint for perception, editing decisions,
and visual review. Deterministic pixel operators and safety metrics are internal
capabilities rather than separately configured backends. The previous
multi-agent manifest remains available as a compatibility path.

## What is implemented

- Pool graph v2 with Primary, white balance, global color, HSL8, three-way
  color wheels, four-channel curves, texture, optical effects, and denoise;
- fixed operation order, strict model-output validation, and deterministic
  per-frame grain;
- Torch `BCHW` Pool rendering on CUDA with a complete Torch CPU fallback;
- 16-bit RGB FFmpeg decode and HEVC 10/12-bit Rec.709, PQ, or HLG delivery;
- built-in sRGB/Rec.709/LogC3/S-Log3/V-Log/PQ/HLG transforms through ACEScg,
  plus optional studio OpenColorIO configs;
- person, skin, and sky Pool masks with optical-flow tracking and periodic
  semantic refresh;
- hierarchical long-video VL planner with overlapping windows, boundary
  adjudication, and per-shot Anchor ranking;
- tournament-style HeroAnchor selection across all shot Anchors;
- visual shot matching against the accepted graded HeroAnchor;
- training-free heuristic planner retained only as an explicit ablation/fallback;
- provider-neutral single-backend configuration with one shared VL client;
- versioned sparse operation graph with per-Pool temporal policies;
- legacy multi-editor UCT-MCTS selection and reward back-propagation;
- legacy competing editing-Agent proposals and independent critic ensemble;
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
an editable Pool graph. The JSON contains shot boundaries, Hero/Anchor audits,
typed operation nodes, dense tracks only for controls that are allowed to vary
over time, confidence, and rollback history. The old 12-D trajectory is not
used or exposed by `pool-graph/v2`.

```bash
python retouch_video.py \
  --input input.mp4 \
  --reference-video reference.mp4 \
  --instruction "natural warm cinematic grade with protected skin tones" \
  --backend-config configs/unified_vl.gemini-full.json \
  --output outputs/input.grade.json \
  --trajectory-output outputs/input.rollouts.jsonl \
  --video-output-dir outputs/input_videos \
  --analysis-max-side 960 \
  --render-max-side 1920
```

`--reference-video` is optional. When supplied, the storyboard Agent selects
the HeroShot from that video, the editor develops its grade, and all target
video Anchors match the pair of reference source frame and accepted graded
reference frame. Only the target video receives the resulting Pool tracks and
rendered output. `pool_metadata.hero.source_video` records the reference source.

The video directory contains `<input>.source.mp4` and `<input>.graded.mp4`.
The Agent can reason over a lightweight proxy while the accepted dense
trajectory is rendered from a separate high-resolution decode. The default is
high-quality 8-bit H.264. A 10/12-bit render uses HEVC and copies source audio
when present.

For example, grade S-Log3 in ACEScg and deliver 10-bit HDR PQ:

```bash
python retouch_video.py \
  --input camera-log.mov \
  --instruction "natural cinematic contrast; protect skin and sky" \
  --backend-config configs/unified_vl.json \
  --output outputs/camera-log.grade.json \
  --video-output-dir outputs/camera-log-videos \
  --input-color-space slog3 \
  --working-color-space acescg \
  --output-color-space rec2020_pq \
  --output-bit-depth 10 \
  --render-device cuda
```

Analysis uses a tone-mapped sRGB proxy, but delivery is decoded as `rgb48le`,
graded in float32 batches, and encoded without an 8-bit intermediate. For a
studio OCIO config, install `requirements-color.txt` and additionally pass
`--ocio-config` plus the four explicit `--ocio-*-space` names. Masked Pool nodes
use `"mask": "person"`, `"skin"`, or `"sky"`; omitted masks are global.

The Resolve DCTL exporter remains available for the legacy 12-D compatibility
runtime. Pool v2 effects such as denoise, clarity, bloom, vignette, chromatic
aberration, and grain are spatial operators and cannot be represented faithfully
by a global LUT/DCTL package; use `--video-output-dir` for a complete Pool v2
render.

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
  "hero_match_strength": 0.35,
  "reject_unsupported": true
}
```

The backend reads MonetGPT's final JSON rather than its rendered TIFF, creates
the exact preview represented by the future Resolve LUT, and submits that
preview to the existing critics. MonetGPT remains a single-image call: it runs
only on selected Hero/shot Anchors, blends target-Anchor parameters toward the
accepted Hero look, and lets the shared Bayesian diffuser create the full video
trajectory. `hero_match_strength` controls that blend from `0` (independent
Monet result) to `1` (Hero parameters), with `0.35` as the default. A rejected
proposal is discarded by MCTS; if no Hero/shot trajectory passes, the video
grade rolls back to identity.

### Model providers

The unified backend supports both OpenAI's native Responses API and
OpenAI-compatible multimodal endpoints. In either mode, one provider/model
client is shared by storyboard perception, Hero/shot editing, and visual
review; switching providers does not create a legacy editor/evaluator pool.

For OpenAI, copy the primary configuration and export the key locally:

```bash
cp configs/unified_vl.example.json configs/unified_vl.json
export OPENAI_API_KEY="..."
```

The configuration uses `provider: "openai_responses"`, `/v1/responses`, ordered
`input_image` items, native JSON output mode, and automatic retry for 429/5xx
responses. It defaults to one shared `gpt-5.6-sol` client; change `model` in the
JSON when using another image-capable Responses model. See the official
[OpenAI vision guide](https://developers.openai.com/api/docs/guides/images-vision/)
and [Responses API reference](https://developers.openai.com/api/reference/resources/responses/methods/create).

For Gemini's OpenAI-compatible endpoint, choose either the quota-conscious
smoke-test configuration or the complete quality configuration:

```bash
cp configs/unified_vl.gemini-free.json configs/unified_vl.json
# or: cp configs/unified_vl.gemini-full.json configs/unified_vl.json
export GEMINI_API_KEY="..."
```

`gemini-free` keeps one editor stage, deterministic review, and one search
evaluation to validate the full pipeline within a small request budget.
`gemini-full` enables the three editor stages, MKL matching, visual review,
tone-curve/HSL planning, and multi-round search. On multi-shot long videos the
full configuration can exceed a free provider's per-minute or daily request
quota; transport retries cannot bypass a daily quota.

Run either provider with the same CLI contract:

```bash
python retouch_video.py \
  --input input.mp4 \
  --instruction "visible but controlled warm cinematic grade" \
  --backend-config configs/unified_vl.json \
  --output outputs/input.grade.json
```

The supplied runtime uses one shared client for storyboard, editor, and review
roles.

The storyboard path performs a sparse whole-video overview, a full-frame
physical scan, overlapping 20-second VL windows, dense boundary adjudication,
task-aware Anchor ranking inside every verified shot, then selects a global
HeroAnchor. The editor first develops and approves the Hero look; all remaining
Anchors receive both the Hero source and accepted Hero grade as their visual
matching reference. Add
`--allow-storyboard-fallback` only for physical shot-detection ablations.
The output includes `operation_graph/v2` and a sanitized `backend_runtime`
manifest. The VL colorist runs technical, look, selective-color, texture, and
optical stages. Each response is sparse, validated against its Pool contract,
rendered locally, reviewed, and committed transactionally. Primary, white
balance, and denoise controls receive frame tracks from grading Anchors;
HSL8, wheels, curves, global color, texture, and optical effects are normally
shot-static. Grain parameters are static but the seeded grain realization
changes deterministically with the absolute frame number.

The older `--agent-config configs/photoagent_multi.json` path is retained for
experiments that explicitly require MonetGPT, command tools, or competing
editor pools.

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
