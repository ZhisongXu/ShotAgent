import json
from pathlib import Path

from evaluation.blind_video_judge import (
    _validate,
    attach_axis_rank_summary,
    attach_reference_style_similarity,
)


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


def test_attach_axis_rank_summary_uses_equal_axis_weights() -> None:
    report = {
        "average_ranks": {
            "method-a": {
                "llm_reference_style_rating": 1.0,
                "vgg_style_similarity": 2.0,
                "lab_chroma_histogram_bhattacharyya": 3.0,
                "content_structure_correlation": 2.0,
                "dino_content_similarity": 2.0,
                "edge_ssim": 2.0,
                "temporal_flow_warp_error": 3.0,
                "temporal_edit_warp_error": 3.0,
                "temporal_transform_drift": 3.0,
                "musiq_score": 4.0,
                "new_shadow_clip_fraction": 4.0,
                "new_highlight_clip_fraction": 4.0,
            }
        }
    }

    result = attach_axis_rank_summary(report)

    assert result["axis_average_ranks"]["method-a"] == {
        "style": 2.0,
        "content": 2.0,
        "temporal": 3.0,
        "quality_artifact": 4.0,
    }
    assert result["overall_axis_average_rank"]["method-a"] == 2.75
