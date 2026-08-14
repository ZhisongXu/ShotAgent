"""Canonical task tokens and contracts shared across specialized Agents."""

from __future__ import annotations

import json

from retouch_agent.parameters import (
    PARAMETER_LOWER_BOUNDS,
    PARAMETER_NAMES,
    PARAMETER_UPPER_BOUNDS,
)


STORYBOARD_TASK = "<TASK_STORYBOARD>"
LONG_VIDEO_OVERVIEW_TASK = "<TASK_LONG_VIDEO_OVERVIEW>"
SHOT_WINDOW_TASK = "<TASK_SHOT_WINDOW>"
BOUNDARY_VERIFY_TASK = "<TASK_BOUNDARY_VERIFY>"
ANCHOR_SELECTION_TASK = "<TASK_ANCHOR_SELECTION>"
HERO_ANCHOR_SELECTION_TASK = "<TASK_HERO_ANCHOR_SELECTION>"
ANCHOR_GRADE_TASK = "<TASK_ANCHOR_GRADE>"
ANCHOR_MATCH_TASK = "<TASK_ANCHOR_MATCH>"
CRITIQUE_TASK = "<TASK_CRITIQUE>"

AGENT_SYSTEM_PROMPT = """
You are one specialized component in a professional multi-model video color
grading system. Follow the task token and return JSON matching that role's
schema. Never synthesize pixels; plan editable operations or judge previews
produced by deterministic tools.
""".strip()


def parameter_contract() -> str:
    schema = {
        name: [float(lower), float(upper)]
        for name, lower, upper in zip(
            PARAMETER_NAMES, PARAMETER_LOWER_BOUNDS, PARAMETER_UPPER_BOUNDS
        )
    }
    return json.dumps(schema, ensure_ascii=False)


def storyboard_prompt(
    instruction: str,
    frame_count: int,
    anchors_per_shot: int,
) -> str:
    return f"""
{STORYBOARD_TASK}
Inspect the ordered storyboard frames. Split only on true camera edits or scene
changes, not object motion. For every shot choose {anchors_per_shot}
representative Anchor frame(s) with clear subjects, low occlusion, useful tonal
range, and coverage of lighting states. The downstream grade role predicts
editable parameters at these Anchors.

Instruction: {instruction}
Return JSON only:
{{"shots":[{{"start_frame":0,"end_frame":10,"anchor_frames":[5],
"description":"...","selection_reason":"..."}}]}}
Use the original frame_id labels. Cover frame 0 through frame
{frame_count - 1} without gaps or overlaps.
""".strip()


def long_video_overview_prompt(
    instruction: str,
    frame_count: int,
    fps: float,
) -> str:
    duration = frame_count / fps
    return f"""
{LONG_VIDEO_OVERVIEW_TASK}
Build a global editorial map from sparse frames sampled across an entire long
video. This is context for later local shot-boundary decisions; do not invent
cuts from this sparse overview. Identify recurring subjects and locations,
large visual regimes, chronology, lighting changes, and transitions that a
local window should inspect carefully.

Video: {frame_count} frames, {fps:.6f} fps, {duration:.3f} seconds.
Color-grading instruction: {instruction}
Every image label contains an absolute frame_id and timestamp.

Return JSON only:
{{"summary":"...","recurring_elements":["..."],
"visual_regimes":[{{"start_frame":0,"end_frame":100,"description":"..."}}],
"continuity_notes":["..."]}}
""".strip()


def shot_window_prompt(
    instruction: str,
    frame_count: int,
    fps: float,
    window_start: int,
    window_end: int,
    overview: dict[str, object],
    technical_candidates: list[dict[str, object]],
) -> str:
    return f"""
{SHOT_WINDOW_TASK}
Inspect this ordered, overlapping window from a long video. Locate genuine
shot boundaries: hard cuts, dissolves, fades, or a semantic scene/camera setup
change. Do not split for subject motion, camera motion inside one take,
exposure flicker, flashes, occlusion, or a color shift alone.

Video: {frame_count} frames at {fps:.6f} fps.
Window: absolute frames {window_start} through {window_end}, inclusive.
Global context: {json.dumps(overview, ensure_ascii=False)}
Low-level candidate peaks (hints, never ground truth):
{json.dumps(technical_candidates, ensure_ascii=False)}
Color-grading instruction: {instruction}

Use only absolute frame_id values. A boundary_frame is the first frame of the
new shot. It may be estimated between supplied samples when the evidence is a
gradual transition. Overlap with adjacent windows is intentional, so report
all boundaries visible in this window even if they may be duplicated later.

Return JSON only:
{{"boundaries":[{{"boundary_frame":120,"transition_type":"hard_cut",
"confidence":0.95,"evidence":"..."}}],
"window_description":"...","uncertain_ranges":[{{"start_frame":0,
"end_frame":0,"reason":"..."}}]}}
""".strip()


def boundary_verification_prompt(
    frame_count: int,
    fps: float,
    candidates: list[dict[str, object]],
) -> str:
    return f"""
{BOUNDARY_VERIFY_TASK}
Adjudicate possible shot boundaries using dense before/after context. Each
candidate_id may contain evidence from several overlapping windows and from a
full-video physical discontinuity scan. Reject motion, flashes, occlusion,
brief black/damaged frames, and within-shot lighting changes. Accept only a
true editorial transition. If accepted, boundary_frame is the first frame of
the new shot and should be refined using the supplied absolute frame labels.

Video: {frame_count} frames at {fps:.6f} fps.
Candidates: {json.dumps(candidates, ensure_ascii=False)}

Return exactly one decision per candidate_id as JSON only:
{{"decisions":[{{"candidate_id":0,"accept":true,
"boundary_frame":120,"transition_type":"hard_cut","confidence":0.95,
"reason":"..."}}]}}
""".strip()


def anchor_selection_prompt(
    instruction: str,
    fps: float,
    anchors_per_shot: int,
    shot_candidates: list[dict[str, object]],
    overview: dict[str, object],
) -> str:
    return f"""
{ANCHOR_SELECTION_TASK}
Select professional color-grading Anchor frames independently for each shot.
An Anchor must be inside its shot, outside an edit/transition, sharp and
unoccluded, and representative of the shot's subjects, illumination, exposure,
white balance, highlight/shadow range, and important skin tones. When more
than one Anchor is requested, choose complementary lighting or composition
states rather than near-duplicates. Do not choose merely the temporal middle.

Video fps: {fps:.6f}
Color-grading instruction: {instruction}
Global context: {json.dumps(overview, ensure_ascii=False)}
Shot candidate contract: {json.dumps(shot_candidates, ensure_ascii=False)}

Return JSON only and exactly {anchors_per_shot} ranked Anchor(s) per shot when
the shot contains that many candidates:
{{"shots":[{{"shot_id":0,"description":"...","anchors":[
{{"frame":12,"rank":1,"confidence":0.9,"reason":"..."}}]}}]}}
Use only candidate frame IDs supplied for that shot.
""".strip()


def hero_anchor_selection_prompt(
    instruction: str,
    candidates: list[dict[str, object]],
    overview: dict[str, object],
    requested_candidates: int,
    final_round: bool,
) -> str:
    stage = "final ranking" if final_round else "regional nomination"
    return f"""
{HERO_ANCHOR_SELECTION_TASK}
Perform {stage} for a long video's HeroAnchor. All inputs are already-selected
per-shot grading Anchors. Choose the frame that can serve as the master visual
reference for matching every other shot: narratively central, technically
clean, sharp, stable, with useful tonal and chromatic range, important subjects
or skin tones, and lighting that expresses the requested look without being an
unrepresentative transition, effect, extreme exposure, or mixed-light outlier.
Judge reference suitability, not simply beauty in isolation.

Color-grading instruction: {instruction}
Global video context: {json.dumps(overview, ensure_ascii=False)}
Candidate contract: {json.dumps(candidates, ensure_ascii=False)}

Return up to {requested_candidates} candidates in best-first order as JSON only:
{{"ranked_candidates":[{{"frame":12,"shot_id":2,"confidence":0.92,
"reason":"best master reference because ..."}}]}}
Use only supplied candidate frame IDs.
""".strip()


def anchor_grade_prompt(
    instruction: str,
    stage: str,
    current_parameters: dict[str, float],
) -> str:
    return f"""
{ANCHOR_GRADE_TASK}
Act as the operation-aware Anchor grading role. Compare the source and current
preview. At stage "{stage}", propose conservative parameter_updates that move
the preview toward the instruction while preserving identity, texture, skin,
highlights, and shadows. Updates are absolute adjustments added to the current
parameter state; omit unrelated parameters.

Instruction: {instruction}
Current parameters: {json.dumps(current_parameters, ensure_ascii=False)}
Valid parameter ranges: {parameter_contract()}
No local mask is available in the video Anchor path. Do not update
local_exposure, local_temperature, or local_saturation.

Return JSON only:
{{"diagnosis":{{"issues":["..."]}},"parameter_updates":{{"exposure":0.1}},
"constraints":["..."],"confidence":0.0}}
""".strip()


def anchor_match_prompt(
    instruction: str,
    stage: str,
    current_parameters: dict[str, float],
    hero_frame: int,
    hero_shot_id: int,
    mkl_prior: dict[str, object] | None = None,
) -> str:
    mkl_contract = (
        "No distribution-transfer proposal is supplied."
        if mkl_prior is None
        else f"""
A fifth image is a conservative linear Monge--Kantorovich (MKL) proposal in
CIELAB. It only aligns global first/second-order color statistics and has no
semantic knowledge. Treat it as a hypothesis, never as ground truth. Decide
whether corresponding content really supports it: match skin to skin, sky to
sky, foliage to foliage, neutral objects to neutrals, and explicitly protect
memory colors, mixed lighting, practical lights, product colors, and narrative
exceptions. Do not let a large background region recolor an unrelated subject.
MKL diagnostics: {json.dumps(mkl_prior, ensure_ascii=False)}
Return mkl_decision as accept, attenuate, or reject. For attenuate, return an
mkl_weight in [0,1]. Also return semantic_correspondences and protected_regions.
""".strip()
    )
    return f"""
{ANCHOR_MATCH_TASK}
Act as the operation-aware shot-matching role. The inputs contain the original
HeroAnchor, its accepted graded version, the current shot Anchor source, and
the current shot Anchor preview. At stage "{stage}", propose conservative
parameter_updates for the current shot Anchor so it belongs to the HeroAnchor's
accepted visual world while preserving the current shot's physically different
exposure, time of day, skin, local contrast, and narrative intent. Match look
characteristics (neutral axis, contrast curve, saturation hierarchy, highlight
roll-off, shadow color, and subject treatment); do not force raw pixel or
histogram equality and do not copy content.

HeroAnchor: frame {hero_frame}, shot {hero_shot_id}
Instruction: {instruction}
Current target parameters: {json.dumps(current_parameters, ensure_ascii=False)}
Valid parameter ranges: {parameter_contract()}
No local mask is available. Do not update local_exposure, local_temperature,
or local_saturation.
Distribution-prior contract: {mkl_contract}

Return JSON only:
{{"diagnosis":{{"match_gaps":["..."]}},
"parameter_updates":{{"temperature":0.1}},"constraints":["..."],
"semantic_correspondences":[{{"hero":"sky","target":"sky"}}],
"protected_regions":["skin"],"mkl_decision":"attenuate",
"mkl_weight":0.5,"confidence":0.0}}
""".strip()


def critique_prompt(
    instruction: str,
    focus: str = "overall professional quality",
    hero_frame: int | None = None,
) -> str:
    hero_contract = (
        "No HeroAnchor reference is supplied for this look-development review."
        if hero_frame is None
        else (
            f"The first labeled pair is HeroAnchor frame {hero_frame} source and "
            "its accepted grade. Judge whether the current shot belongs to that "
            "look while retaining its legitimate scene-specific lighting."
        )
    )
    return f"""
{CRITIQUE_TASK}
Act as a conservative professional video color-grading critic. Compare every
source/graded pair against instruction fulfillment, content preservation,
skin/highlight safety, coherent look, and cross-frame consistency.

Instruction: {instruction}
Evaluator specialty: {focus}
HeroAnchor contract: {hero_contract}
Return JSON only:
{{"accept":true,"score":0.0,"instruction_score":0.0,
"content_score":0.0,"consistency_score":0.0,"hero_match_score":0.0,
"recommended_anchor":null,"reasons":["..."]}}
All scores are in [0,1]. Reject visible artifacts, content changes, clipping,
inconsistent grading, or a shot that fails the HeroAnchor look contract. When
rejecting because the current Anchor is unrepresentative, recommended_anchor
may be an absolute frame ID visible in the supplied labels; otherwise use null.
""".strip()
