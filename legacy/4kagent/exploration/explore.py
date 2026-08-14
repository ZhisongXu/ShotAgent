from pathlib import Path
from tqdm import tqdm
import json

from llm import DepictQA
from utils.img_tree import ImgTree
from utils.misc import sorted_glob


train_dir = [
    "rain+haze", 
    "motion blur+low resolution", 
    "dark+noise", 
    "defocus blur+jpeg compression artifact", 
    "noise+jpeg compression artifact", 
    "rain+low resolution", 
    "motion blur+dark", 
    "defocus blur+haze", 
]


def explore_order():
    experience = {}
    for task_dir in sorted_glob(root_exh_dir, "d2/*"):
        if task_dir.stem not in train_dir:
            continue
        degradations = task_dir.stem.split('+')
        n_d = int(task_dir.parent.stem[1])
        leave_pat = "*/output.png"  # relative to tree
        for _ in range(n_d):
            leave_pat = "*/*/" + leave_pat

        task = task_dir.stem
        experience[task] = {}
        img_dir_lst = sorted_glob(task_dir)
        img_dir_lst = [img_dir for img_dir in img_dir_lst if 1 <= int(img_dir.stem) % 10 <= 2]
        for img_dir in tqdm(img_dir_lst, desc=task):
            tree_dir = img_dir / "tree"
            img_tree = ImgTree(tree_dir)
            for leave_path in sorted_glob(tree_dir, leave_pat):
                exe_path = img_tree.get_execution_path(leave_path)
                assert len(exe_path) == n_d
                plan = '+'.join([subtask for subtask, _ in exe_path])
                if plan not in experience[task]:
                    experience[task][plan] = {
                        "total": 0
                    }
                    for degra in degradations:
                        experience[task][plan][degra] = 0
                experience[task][plan]["total"] += 1
                for degra in degradations:
                    level = eval(dqa(img_path=leave_path, 
                                task="eval_degradation",
                                degradation=degra))[0][1]
                    # experience[task][plan][degra][level] = experience[task][plan][degra].get(level, 0)+1
                    if level in levels_to_address:
                        experience[task][plan][degra] += 1
        
        for plan in experience[task].keys():
            experience[task][plan]["fail rate"] = {
                degra: n_fail / experience[task][plan]["total"]
                for degra, n_fail in experience[task][plan].items()
                if degra != "total"
            }
            experience[task][plan]["fail rate"]["total"] = sum(experience[task][plan]["fail rate"].values()) / len(experience[task][plan]["fail rate"])
        # sort plans by total fail rate
        experience[task] = dict(sorted(experience[task].items(), key=lambda x: x[1]["fail rate"]["total"]))
        with open(mem_dir / "fail_rate.json", "w") as f:
            json.dump(experience, f, indent=2)

mem_dir = Path("memory")
root_exh_dir = Path("exhaustive_sequences")
dqa = DepictQA()
levels_to_address = {"medium", "high", "very high"}
explore_order()