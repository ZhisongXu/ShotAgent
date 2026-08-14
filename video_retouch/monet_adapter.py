"""Convert MonetGPT adjustment JSON into Resolve-compatible LUT packages."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from retouch_agent import RetouchParameters

from .resolve_export import write_cube_lut


_DIRECT_FIELDS = {
    "Exposure": "exposure",
    "Temperature": "temperature",
    "Tint": "tint",
    "Contrast": "contrast",
    "Highlights": "highlights",
    "Shadows": "shadows",
    "Saturation": "saturation",
    "Vibrance": "vibrance",
}

# The current GradeIR has broad highlight/shadow controls but no separate
# whites/blacks wheels. These weights preserve the direction of the Monet edit
# without pretending that the two implementations are numerically identical.
_FOLDED_FIELDS = {
    "Whites": ("highlights", 0.5),
    "Blacks": ("shadows", 0.5),
}


@dataclass(frozen=True)
class MonetConversion:
    """Converted grade plus a machine-readable fidelity audit."""

    parameters: RetouchParameters
    consumed_fields: tuple[str, ...]
    approximated_fields: Mapping[str, str]
    unsupported_fields: Mapping[str, float]
    clipped_fields: Mapping[str, Mapping[str, float]]

    def to_dict(self) -> dict[str, object]:
        return {
            "parameters": self.parameters.to_dict(),
            "consumed_fields": list(self.consumed_fields),
            "approximated_fields": dict(self.approximated_fields),
            "unsupported_fields": dict(self.unsupported_fields),
            "clipped_fields": {
                key: dict(value) for key, value in self.clipped_fields.items()
            },
        }


def _numeric_adjustments(payload: Mapping[str, object]) -> dict[str, float]:
    adjustments: dict[str, float] = {}
    for key, raw_value in payload.items():
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            continue
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"MonetGPT adjustment {key!r} must be finite.")
        adjustments[str(key)] = value
    if not adjustments:
        raise ValueError("MonetGPT payload contains no numeric adjustments.")
    return adjustments


def convert_monet_adjustments(
    payload: Mapping[str, object],
    *,
    strict: bool = False,
) -> MonetConversion:
    """Map MonetGPT's native -100..100 sliders to the shared GradeIR.

    This expects MonetGPT's final adjustment JSON, before its image executor
    divides slider values by 100. Unknown or spatial/selective controls are
    retained in the audit instead of being silently discarded.
    """

    nested = payload.get("adjustments")
    source = nested if isinstance(nested, Mapping) else payload
    adjustments = _numeric_adjustments(source)
    values: dict[str, float] = {}
    consumed: list[str] = []
    approximated: dict[str, str] = {}
    unsupported: dict[str, float] = {}
    clipped: dict[str, dict[str, float]] = {}

    for source_name, value in adjustments.items():
        normalized = value / 100.0
        if source_name in _DIRECT_FIELDS:
            target = _DIRECT_FIELDS[source_name]
            values[target] = values.get(target, 0.0) + normalized
            consumed.append(source_name)
        elif source_name in _FOLDED_FIELDS:
            target, weight = _FOLDED_FIELDS[source_name]
            values[target] = values.get(target, 0.0) + normalized * weight
            consumed.append(source_name)
            approximated[source_name] = (
                f"folded into {target} at weight {weight:g}; Resolve result is approximate"
            )
        elif value != 0.0:
            unsupported[source_name] = value

    if strict and unsupported:
        raise ValueError(
            "Unsupported non-zero MonetGPT adjustments: "
            + ", ".join(sorted(unsupported))
        )

    # All mapped global controls except exposure use [-1, 1]. Exposure uses
    # [-3, 3], but Monet's documented slider still maps to [-1, 1] EV here.
    for name, requested in tuple(values.items()):
        applied = min(max(requested, -1.0), 1.0)
        if applied != requested:
            clipped[name] = {"requested": requested, "applied": applied}
            values[name] = applied

    return MonetConversion(
        parameters=RetouchParameters.from_mapping(values),
        consumed_fields=tuple(sorted(consumed)),
        approximated_fields=approximated,
        unsupported_fields=unsupported,
        clipped_fields=clipped,
    )


def _shot_payloads(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, Mapping):
        raise ValueError("MonetGPT input must be a JSON object.")
    raw_shots = payload.get("shots")
    if raw_shots is None:
        return [{"shot_id": 0, "adjustments": payload}]
    if not isinstance(raw_shots, Sequence) or isinstance(raw_shots, (str, bytes)):
        raise ValueError("MonetGPT shots must be a JSON array.")
    shots: list[dict[str, object]] = []
    for index, raw_shot in enumerate(raw_shots):
        if not isinstance(raw_shot, Mapping):
            raise ValueError(f"MonetGPT shot {index} must be an object.")
        adjustments = raw_shot.get("adjustments", raw_shot.get("parameters"))
        if not isinstance(adjustments, Mapping):
            raise ValueError(f"MonetGPT shot {index} is missing adjustments.")
        shots.append(
            {
                "shot_id": int(raw_shot.get("shot_id", index)),
                "start_frame": raw_shot.get("start_frame"),
                "end_frame": raw_shot.get("end_frame"),
                "adjustments": adjustments,
            }
        )
    if not shots:
        raise ValueError("MonetGPT shots cannot be empty.")
    return shots


def export_monet_resolve_package(
    payload: object,
    output_dir: Path,
    *,
    lut_size: int = 33,
    strict: bool = False,
) -> Path:
    """Export MonetGPT JSON as per-shot .cube files and an audit manifest."""

    output_dir = Path(output_dir).resolve()
    lut_dir = output_dir / "LUT"
    lut_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, object]] = []
    seen_ids: set[int] = set()
    for shot in _shot_payloads(payload):
        shot_id = int(shot["shot_id"])
        if shot_id in seen_ids:
            raise ValueError(f"Duplicate MonetGPT shot_id: {shot_id}")
        seen_ids.add(shot_id)
        adjustments = shot["adjustments"]
        assert isinstance(adjustments, Mapping)
        conversion = convert_monet_adjustments(adjustments, strict=strict)
        lut_path = write_cube_lut(
            conversion.parameters,
            lut_dir / f"shot-{shot_id:04d}.cube",
            size=lut_size,
            title=f"MonetGPT adapted shot {shot_id}",
        )
        exported.append(
            {
                "shot_id": shot_id,
                "start_frame": shot.get("start_frame"),
                "end_frame": shot.get("end_frame"),
                "lut": str(lut_path),
                "source_adjustments": dict(adjustments),
                "conversion": conversion.to_dict(),
            }
        )

    manifest = {
        "schema_version": "monetgpt-resolve-package/v1",
        "source_parameter_scale": "MonetGPT final sliders [-100, 100]",
        "color_pipeline": {
            "lut_domain": "display-referred RGB [0,1]",
            "recommended_timeline": "Rec.709 Gamma 2.4",
            "lut_size": lut_size,
        },
        "shots": exported,
        "fidelity": (
            "Global controls are converted automatically. Whites/Blacks are "
            "approximated; selective HSL, dehaze, clarity, sharpening, masks, "
            "and other spatial controls are reported but not baked."
        ),
    }
    manifest_path = output_dir / "resolve_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path
