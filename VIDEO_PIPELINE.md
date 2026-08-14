# DynamicGrade video-to-parameter pipeline

## Contract

Input:

```text
video file + natural-language grading instruction
```

Output:

```text
dynamic-grade-graph/v1 JSON
```

The output is a grading parameter graph, not a rendered video. Its dense form is
`frame_parameters[T, 12]`; its editable sparse form contains one base grade and
Anchor parameter keyframes per shot. Exposure is measured in stops. The other
dimensions use the normalized ranges recorded in `parameter_schema`.

## Pipeline

```text
video decode
  -> full-frame physical discontinuity scan
  -> GPT-5.6 Sol sparse global overview
  -> overlapping VL windows propose hard/soft/semantic boundaries
  -> dense before/after VL adjudication merges and verifies boundaries
  -> per-shot diverse candidate generation + VL Anchor ranking
  -> tournament ranking selects a global HeroAnchor shortlist
  -> editor/critic look development approves the HeroAnchor grade
  -> every other shot Anchor is matched to Hero source + accepted Hero grade
  -> Lab-space MKL proposes a conservative distribution match
       -> MLLM names semantic correspondences and protected memory colors
       -> accept / attenuate / reject the distribution prior
       -> project the accepted proposal back into the editable 12-D parameters
  -> UCT-MCTS selects and expands editing trajectories
  -> editor pool: independent vision models / MonetGPT / external tools
  -> recover the shared 12-D parameters from RGB-only editors
  -> shot-local Bayesian parameter diffusion using time-varying palette traces
  -> source-guided tonal stabilization in parameter space, with hard Anchors
  -> temporary preview rendering
  -> evaluator ensemble: visual specialists + deterministic temporal safety
  -> back-propagate reward into the search tree
       -> commit the highest-scoring safe trajectory
       -> or roll back that round, replace the Anchor, and search again
       -> or retry the whole matching pass with the next HeroAnchor
       -> or roll the complete shot/video attempt back to identity parameters
  -> grade graph JSON
```

The propagated object is always the parameter trajectory, never Anchor RGB.

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

`--compact` omits the dense trajectory while retaining sparse keyframes.

## Multi-model runtime

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
