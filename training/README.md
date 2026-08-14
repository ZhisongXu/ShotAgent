# PhotoAgent-style multi-Agent learning

The deployed system is not a single checkpoint. It has one Perceiver, multiple
editing Agents, multiple evaluator Agents, deterministic tools, and a UCT-MCTS
controller:

```text
Video Perceiver    shot segmentation and Anchor selection
Editor pool        diverse operation-aware parameter proposals
Evaluator ensemble independent instruction/aesthetic/safety judgments
UCT-MCTS            trajectory search and reward back-propagation
Action memory       successful and failed execution history
```

The vision components may use different architectures, providers, or frozen API
models. Weight training is optional: the first learning loop is inference-time
tree search plus trajectory memory. The executor, optical-flow metrics,
Bayesian parameter field, and search coordinator remain deterministic.

## Learning loop

1. Every editor proposes parameters for the same Anchor.
2. The executor renders each explored trajectory.
3. Independent evaluators score instruction adherence, aesthetics, preservation,
   and temporal safety.
4. MCTS back-propagates the score and explores another branch.
5. Accepted and rejected paths are exported with `--trajectory-output`.
6. Human preference labels can correct evaluator scores.
7. Only then, optionally distill selected paths into specialized editors or
   critics. The runtime does not require this distillation.

## Training methods reused

### Editing Agents

Use MonetGPT-style operation grounding:

- Puzzle A identifies an operation and magnitude from before/after images;
- Puzzle B ranks parameter strengths and identifies the best edit;
- Puzzle C generates staged `parameter_updates` with grounded explanations.

Then add Jarvis-style professional traces:

- map Lightroom/tool actions into the shared 12-D parameter schema;
- supervise interleaved current-image observation and next action;
- keep global/local operation provenance even when the initial executor supports
  only global video trajectories.

### Storyboard Agent

Train on ordered video storyboards with:

- true camera-edit boundaries;
- shot descriptions;
- subject visibility and occlusion;
- Anchor coverage of lighting and appearance states;
- hard negatives consisting of object motion without a shot change.

### Evaluator Agents

Use PhotoAgent/JarvisEvo-style executed trajectories:

- accept/reject and multi-dimensional scores;
- failure reasons and high-risk next Anchor;
- unsuccessful branches, rollback events, and successful re-plans;
- human video preferences plus deterministic clipping, fidelity, and temporal
  residual checks.

Evaluators are kept independent from editing Agents. If weights are trained,
freeze evaluators while optimizing editors to reduce same-model reward hacking.

## Canonical manifest

All roles use one validated record format while remaining separate datasets:

```json
{
  "example_id": "monet-c-0001",
  "role": "anchor_grade",
  "method": "monet_puzzle_c",
  "images": ["images/source.png", "images/current.png"],
  "prompt": "Stage lighting. Instruction: natural warm portrait",
  "response": {
    "diagnosis": {"issues": ["underexposed"]},
    "parameter_updates": {"exposure": 0.35},
    "constraints": ["protect_highlights"],
    "confidence": 0.9
  }
}
```

`training/schema.py` validates role outputs and preserves whether a sample came
from Monet puzzles, Jarvis traces, PhotoAgent preferences, synthetic temporal
interventions, or human grades.

## Build role datasets

```bash
python -m training.build_role_datasets \
  --input manifests/monet.jsonl \
  --input manifests/jarvis.jsonl \
  --input manifests/video_storyboard.jsonl \
  --input manifests/critique.jsonl \
  --output-directory training/data
```

This creates the three optional distillation datasets and copies the required
`dataset_info.json` into the same directory:

```text
dynamicgrade_storyboard.json
dynamicgrade_anchor_grade.json
dynamicgrade_critique.json
```

## SFT

The three example LLaMA-Factory configurations contain generic `MODEL_PATH` and
`MODEL_TEMPLATE` values. They are optional distillation recipes; replace the
placeholders only when training a component whose weights you control.

```bash
llamafactory-cli train training/configs/storyboard_agent_lora_sft.yaml
llamafactory-cli train training/configs/anchor_agent_lora_sft.yaml
llamafactory-cli train training/configs/critic_agent_lora_sft.yaml
```

After SFT, the Anchor Agent can be preference-optimized against a frozen Critic
using GRPO/DPO, while the temporal safety metrics remain non-learned constraints.
