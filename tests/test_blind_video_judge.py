import json
from pathlib import Path

from evaluation.blind_video_judge import attach_reference_style_similarity


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
        "candidate_scores": {"C01": {"reference_style_match": 3.0}},
    }

    result = attach_reference_style_similarity(report_path, key_path, review)

    assert result["rows"][0]["llm_reference_style_similarity"] == 0.5
    assert result["aggregate"]["method-a"]["llm_reference_style_similarity"] == 0.5
    assert "llm_reference_style_similarity" in (tmp_path / "results.csv").read_text()
