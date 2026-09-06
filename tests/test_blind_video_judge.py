import json
from pathlib import Path

from evaluation.blind_video_judge import _validate, attach_reference_style_similarity


def test_validate_derives_style_similarity_from_style_only_dimensions() -> None:
    result = _validate(
        {
            "candidate_scores": {
                "C01": {
                    "deep_shadow_black_level_match": 5.0,
                    "shadow_chroma_match": 4.0,
                    "midtone_luminance_match": 3.0,
                    "midtone_palette_match": 2.0,
                    "highlight_rolloff_match": 1.0,
                    "neutral_axis_temperature_match": 2.0,
                    "palette_hierarchy_match": 3.0,
                    "saturation_hierarchy_match": 4.0,
                    "local_contrast_depth_match": 5.0,
                    "content_preservation": 1.0,
                    "temporal_consistency": 1.0,
                    "artifact_free": 1.0,
                    "overall_preference": 1.0,
                    "rationale": "style and preservation are scored independently",
                }
            }
        },
        ["C01"],
    )

    assert result["candidate_scores"]["C01"]["reference_style_match"] == 29 / 9
    assert (
        result["candidate_scores"]["C01"]["overall_grade_quality"]
        == (29 / 9 + 1.0 + 1.0 + 1.0) / 4
    )


def test_attach_reference_style_similarity(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    key_path = tmp_path / "blind_review_key.json"
    report_path.write_text(
        json.dumps(
            {
                "rows": [
                    {"sample": "sample-1", "method": "method-a", "edge_ssim": 0.9}
                ],
                "aggregate": {"method-a": {"edge_ssim": 0.9}},
            }
        ),
        encoding="utf-8",
    )
    key_path.write_text(
        json.dumps(
            [
                {
                    "sample": "sample-1",
                    "candidate_code": "C01",
                    "method": "method-a",
                }
            ]
        ),
        encoding="utf-8",
    )
    review = {
        "sample": "sample-1",
        "judge_model": "judge-model",
        "evidence": "ordered frames",
        "candidate_scores": {
            "C01": {
                "reference_style_match": 3.0,
                "overall_grade_quality": 3.5,
            }
        },
    }

    result = attach_reference_style_similarity(report_path, key_path, review)

    assert result["rows"][0]["llm_reference_style_similarity"] == 0.5
    assert result["rows"][0]["llm_reference_style_rating"] == 3.0
    assert result["rows"][0]["llm_overall_grade_quality"] == 3.5
    assert result["aggregate"]["method-a"]["llm_reference_style_similarity"] == 0.5
    assert result["aggregate"]["method-a"]["llm_overall_grade_quality"] == 3.5
    assert "llm_reference_style_similarity" in (tmp_path / "results.csv").read_text()
