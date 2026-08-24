# DynamicGrade Pool video pipeline

## Contract

Input:

```text
target video file + natural-language grading instruction
+ optional reference video file
```

Output:

```text
pool-grade-graph/v2 JSON + optional rendered video
```

The main output is a sparse typed operation graph. Primary, white-balance, and
denoise nodes may contain dense frame tracks; creative and spatial nodes remain
shot-static. The legacy `frame_parameters[T,12]` object is not used by v2.

## Pipeline

```text
16-bit RGB video decode + input transfer/gamut transform
  -> ACEScg (or explicit OCIO working space)
  -> tone-mapped sRGB analysis proxy for VL only
  -> full-frame physical discontinuity scan
  -> GPT-5.6 Sol sparse global overview
  -> overlapping VL windows propose hard/soft/semantic boundaries
  -> dense before/after VL adjudication merges and verifies boundaries
  -> per-shot diverse candidate generation + VL Anchor ranking
  -> tournament ranking selects a global HeroAnchor shortlist
       -> when a reference video is supplied, rank and develop HeroShots there
  -> five restricted VL colorist stages develop and approve the Hero look
       -> technical: denoise, white balance, Primary
       -> look: color wheels, four-channel curves, global color
       -> selective_color: HSL8
       -> texture: clarity, texture, dehaze, sharpening
       -> optical: vignette, grain, bloom, halation, diffusion, aberration
  -> every target Anchor sees Hero source + accepted Hero grade
  -> strict Pool canonicalization rejects unknown fields and invalid ranges
  -> deterministic preview rendering in a fixed operation order
       -> optional person / skin / sky mask per Pool node
       -> semantic refresh + optical-flow mask tracking
  -> Bayesian diffusion for Primary, white balance, and denoise tracks
  -> shot-static consensus for creative/spatial Pools
  -> source-guided exposure-flicker compensation with hard Anchors
  -> start/Anchor/end deterministic and VL review
       -> commit the complete typed stack
       -> or add/replace an Anchor and review again
       -> or retry Hero development
       -> or transactionally roll the complete video back
  -> float32 Torch BCHW Pool batches on CUDA (Torch/NumPy CPU fallback)
  -> output transform + 8/10/12-bit SDR/PQ/HLG encode
  -> Pool graph JSON and optional full render
```

The propagated objects are typed numeric controls, never generated Anchor RGB.

The analysis image is intentionally not the delivery image. Log/HDR sources are
decoded to 16-bit RGB, transformed to a scene-linear working space, and only
then tone-mapped for VL inspection. The accepted display-proxy grade is
transferred back to scene-linear luminance/chroma before the output transform,
so the high-bit path has no 8-bit pixel intermediate. Built-in transforms cover
sRGB, Rec.709, LogC3, S-Log3, V-Log, PQ, HLG, Rec.2020, ACEScg, and ACES2065-1;
an optional OCIO config can replace all named transforms.

## Legacy 12-D research path

The sections below describe the retained `legacy-12d/v1` research and
compatibility path. It is not used by the main Pool configurations.

## Hybrid color-science layer

The professional matching path combines semantic and numerical roles instead
of asking either one to solve the whole problem:

- The vision model decides *what may correspond*: skin-to-skin, sky-to-sky,
  neutrals-to-neutrals, and which memory or narrative colors must be protected.
- A CIELAB linear Monge--Kantorovich map proposes *how* to align first- and
  second-order color statistics. It is a conservative proposal, not a final
  pixel operation. The vision model must accept, attenuate, or reject it.
- A fixed shot-level Lab palette supplies per-frame color-cluster occupancy and
  means to the Bayesian field. This is a lightweight long-video approximation
  to a spatial-temporal geometric palette; it does not claim to reproduce the
  paper's 4-D skew-polytope extraction.
- Source-video exposure, temperature, and tint residuals guide a sparse
  parameter-trajectory solve. It regularizes velocity and curvature while
  preserving selected Anchor parameter vectors exactly. This is a
  parameter-domain adaptation of tonal stabilization and blind input-guided
  temporal consistency, not a reproduction of either pixel-domain method.

The original and accepted Hero images, the MKL proposal, transform matrix,
projection error, semantic correspondences, protected regions, decision, and
weight are retained in each proposal's search memory. Stabilization diagnostics
are retained for every accepted or rejected trajectory, so a rollback does not
erase the evidence that caused it.

The mask-aware MKL primitive already accepts source/reference masks. The
current vision-model path records semantic region names but does not pretend
that text labels are pixel-accurate masks. Region-wise OT should only be enabled
after a segmentation-and-tracking component supplies reliable masks.

Method references:

- [Tonal Stabilization of Video](https://pages.cs.huji.ac.il/danix-lab/cglab/projects/tonestab/)
- [Blind Video Temporal Consistency](https://perso.liris.cnrs.fr/nicolas.bonneel/consistency/)
- [Video Recoloring via Spatial-Temporal Geometric Palettes](https://cragl.cs.gmu.edu/videopalette/)
- [Linear Monge-Kantorovich Colour Mapping](https://www.tara.tcd.ie/bitstreams/ee98044d-2ef3-46cf-9e56-ee567085cd24/download)

## Output schema

```json
{
  "schema_version": "dynamic-grade-graph/v1",
  "orchestrator": "photoagent-uct-mcts",
  "instruction": "warm cinematic grade",
  "parameter_schema": {
    "names": ["exposure", "temperature", "..."],
    "lower_bounds": [-3.0, -1.0],
    "upper_bounds": [3.0, 1.0],
    "units": {"exposure": "stops", "others": "normalized"}
  },
  "storyboard": {
    "planner": "hierarchical-vision-storyboard/v2",
    "hero_anchor": {
      "frame": 42,
      "ranked_candidates": [42, 315, 811]
    },
    "shots": []
  },
  "hero_anchor": {
    "frame": 42,
    "shot_id": 1,
    "source_video": "target_video",
    "parameters": [],
    "backend": "lighting-editor",
    "attempts": []
  },
  "shots": [
    {
      "start_frame": 0,
      "end_frame": 95,
      "base_parameters": [],
      "parameter_keyframes": {"42": []},
      "confidence": 0.87,
      "rolled_back": false,
      "attempts": [],
      "search_memory": {
        "algorithm": "photoagent-uct-mcts",
        "editor_agents": [],
        "rounds": [],
        "proposals": {},
        "evaluated_trajectories": 0,
        "selected": {}
      }
    }
  ],
  "frame_parameters": []
}
```

With `--reference-video`, `hero_anchor.frame` and its ranked candidates use the
reference video's frame coordinate system, `hero_anchor.source_video` is
`reference_video`, and the top-level `reference_video` object records its path,
dimensions, frame rate, and frame count. Target-shot keyframes always remain in
the target video's coordinate system.

`--compact` omits the dense trajectory while retaining sparse keyframes.

## Unified single-backend runtime

`configs/unified_vl.example.json` is the primary runtime contract. It creates
one `UnifiedVLVideoBackend`, one vision client, and one VL editor. The same
client performs storyboard perception, editable parameter planning, and visual
review; deterministic rendering, safety metrics, diffusion, and rollback are
internal operators rather than public backends.

Run it with `--backend-config`. The main configurations select
`pool-graph/v2`. The result contains a sanitized runtime manifest and typed
operations for Primary, white balance, global color, HSL8, three-way wheels,
four-channel curves, texture, optical effects, and denoise. The shared VL
colorist plans five restricted stages; deterministic rendering and VL review
close the loop before transactional commit.

Primary, white balance, and denoise have Anchor-conditioned frame tracks.
Global color, HSL8, wheels, curves, texture, and optical effects use shot-static
policies. Grain is shot-static in control space but seeded by absolute frame.
The old `DynamicGradePipeline` and 12-D Resolve exporter remain only as the
legacy compatibility runtime; Pool v2 full renders are delivered as video
because its spatial/frequency effects cannot be represented by a global LUT.

## Legacy multi-model runtime

`configs/photoagent_multi.example.json` defines the Video Perceiver, editing
roles, evaluator roles, safety-veto weights, and UCT search settings. The
provided configuration uses one `gpt-5.6-sol` model through OpenAI's native
Responses API for every semantic role and reads `OPENAI_API_KEY`. The roles are
separate calls and contracts, not separate model families. The long-video
window duration, overlap, image budgets, boundary confidence, and Anchor
candidate/Hero tournament budgets live under `storyboard.long_video`.

Rollback is transactional at two levels. A rejected shot round discards its
temporary parameter trajectory and replaces one Anchor using the critic's
recommended frame or the shot's ranked backup pool. If the full pass still has
rejected shots, the pass is not globally committed and the next ranked
HeroAnchor is developed and matched. The JSON records every Hero attempt,
every Anchor set, replacement source, rejection reason, and the finally
committed reference.

The output records a sanitized `agent_runtime` manifest containing model IDs but
never API keys.

For an endpoint-free engineering baseline, pass `--offline-native` instead of
`--agent-config`. This keeps the same grade graph, parameter propagation,
metric-Critic, MCTS, and rollback code paths, while replacing the VL Storyboard
and VL Editor with deterministic local components. It is an ablation baseline,
not a substitute for evaluating the intended VL system.

The normal configured path is strict: a missing key, endpoint failure, or
malformed storyboard response stops the run. `--allow-storyboard-fallback` is
the only opt-in path that permits a heuristic storyboard.

## Anchor backends

### MonetGPT

The adapter calls the official command:

```text
python inference_cli.py single INPUT --output OUTPUT
```

MonetGPT's model server, weights, Python environment, and optional GIMP runtime
remain external dependencies. The adapter does not silently replace MonetGPT if
it fails. A failed Anchor is recorded, and the shot is retried or rolled back.
The official single-image command is not run frame by frame. All built-in
backends now share `grade` and `grade_with_reference`; MonetGPT grades sparse
Anchors, its editable parameters are blended toward the accepted Hero look, and
the common parameter diffuser produces the dense video trajectory. Configure
the blend with `hero_match_strength` in `[0, 1]` (default `0.35`). Both the
pre-match parameters and matching method are retained in proposal metadata.

### JarvisArt and other agents

JarvisArt's official basic mode emits a Lightroom preset and its end-to-end mode
requires Lightroom Classic. It can be connected with an Anchor config of type
`command` once the command creates the requested `{output}` preview image. A
dedicated Lightroom preset parser is not yet implemented.

### Native baseline

An Anchor config of type `native` uses the repository's 12-parameter single-image
baseline. This is for tests and ablations, not a claim that MonetGPT or JarvisArt
was executed.

## UCT-MCTS, memory, and rollback

For every Anchor, all configured editors propose a candidate grade. A tree node
is a partial assignment of an editor proposal to each Anchor. UCT selection
balances the mean evaluator reward with an exploration bonus; expansion adds an
unvisited proposal, simulation completes the remaining Anchor assignments, and
the evaluator reward is back-propagated to every visited node.

Every unique terminal trajectory is rendered and evaluated once. Candidate
parameters, failures, scores, critic reasons, tree-round statistics, and the
selected branch are retained under `search_memory`. Use
`--trajectory-output FILE.jsonl` to export one reusable rollout record per shot.

## Evaluator ensemble

The evaluator follows PhotoAgent's published closed-loop pattern and combines
multiple independent visual specialists with explicit video checks:

- content fidelity;
- highlight and shadow clipping;
- motion-compensated edit-residual consistency;
- parameter trajectory jerk;
- reconstruction of the single-image Agent's Anchor result.

PhotoAgent's official repository currently states that its code and pretrained
models are forthcoming. Therefore this implementation is deliberately called
`PhotoAgentStyleCritic`; it does not claim to contain the unreleased UGC reward
model.

Evaluator weights control ranking, while a member configured with `veto: true`
can reject a branch regardless of its aesthetic score. When a shot fails, the
ensemble recommends the highest-risk unobserved frame. The pipeline adds it as
an Anchor and starts a deeper search round within the evaluation budget. If no
trajectory passes, every parameter in that shot is reset to zero; proposals,
failure reasons, and the complete search history remain in the JSON.
