from pathlib import Path
import json
from tqdm import tqdm

from utils.scorer import scorer, METRIC_NAME_LST
from utils.misc import sorted_glob


def get_task_scores(task_dir: Path,) -> dict[str, float]:
    scores = {}
    for img_path in tqdm(sorted_glob(task_dir), desc=task_dir.name):
        hq_path = Path("dataset/HQ") / img_path.name
        score_lst = scorer(img_path, hq_path)
        scores[img_path.stem] = {
            metric: score for metric, _, score in score_lst
        }
    return scores


def average(method_name: str, mask=None, mask_name=""):
    with open(Path("performance/scores")/"detail"/f"{method_name}.json", "r") as f:
        all_scores = json.load(f)
    average_scores = {}
    img_cnt = 0
    for task, scores in all_scores.items():
        average_scores[task] = {
            metric: [] for metric in METRIC_NAME_LST
        }
        for img_idx, score_dict in scores.items():
            if mask is not None and img_idx in mask[task]:
                continue
            img_cnt += 1
            for metric, score in score_dict.items():
                average_scores[task][metric].append(score)
        for metric, score_lst in average_scores[task].items():
            average_scores[task][metric] = sum(score_lst) / len(score_lst)
    print(f"Total images: {img_cnt}")
    print(mask_name)
    if mask_name:
        mask_name = '_' + mask_name
    with open(Path("performance/scores")/"average"/f"{method_name}{mask_name}.json", "w") as f:
        json.dump(average_scores, f, indent=2)


def filter_rb(method_name: str) -> dict[str, set[str]]:
    method_dir = Path("output/final") / method_name
    mask = {}
    mask_cnt = 0
    for task_dir in sorted_glob(method_dir, "*/*"):
        mask[task_dir.name] = set()
        for img_dir in sorted_glob(task_dir, "*/agent/*"):
            with open(img_dir / "logs" / "summary.json", "r") as f:
                summary = json.load(f)
            if not summary["plan"]["adjusted"]:
                idx = img_dir.name
                idx = idx[:3]
                mask[task_dir.name].add(idx)
                mask_cnt += 1
    print(f"Total masked: {mask_cnt}")
    return mask

if __name__ == "__main__":
    method_name = "default"

    output_dir = Path("methods") / method_name
    all_scores = {}
    for task_dir in output_dir.glob("d[23]/*"):
        if task_dir.is_dir():
            all_scores[task_dir.name] = get_task_scores(task_dir)

    with open(Path("performance/scores")/"detail"/f"{method_name}.json", "w") as f:
        json.dump(all_scores, f, indent=2)

    average(method_name)

    # mask = filter_rb("default")
    # average("default", mask=mask, mask_name="rb")
    # average("worb", mask=mask, mask_name="rb")
