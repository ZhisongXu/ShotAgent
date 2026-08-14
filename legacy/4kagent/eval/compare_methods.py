from pathlib import Path
import json


def order_task(data: dict) -> dict:
    if len(list(data.keys())[0]) == 1:
        task_order = ["A", "B", "C"]
    else:
        task_order = [
            "rain+haze",
            "motion blur+low resolution",
            "dark+noise",
            "defocus blur+jpeg compression artifact",
            "noise+jpeg compression artifact",
            "rain+low resolution",
            "motion blur+dark",
            "defocus blur+haze",
            "motion blur+jpeg compression artifact",
            "haze+noise",
            "defocus blur+low resolution",
            "rain+dark",
            "haze+motion blur+low resolution",
            "rain+noise+low resolution",
            "dark+defocus blur+jpeg compression artifact",
            "motion blur+defocus blur+noise",
        ]
    data = {task: data[task] for task in task_order if task in data}
    return data


def pad_4eff(num: float) -> str:
    """Pads zeros to make the number have 4 effective digits."""    
    num = f"{num:.4}"
    assert '.' in num
    eff_flag = False
    eff_cnt = 0
    for d in num:
        if d != '.':
            if d != '0':
                eff_flag = True
            if eff_flag:
                eff_cnt += 1
    assert eff_cnt <= 4
    return num + (4 - eff_cnt) * '0'


def gen_md(data: dict, file_path: Path, ours: str = "ours") -> None:
    """Dumps comparison table to markdown file.

    Args:
        data (dict): Data in the format of
        ```
        {
            "task1": {
                "method1": {
                    "metric1": xxx,
                    "metric2": xxx,
                    ...
                },
                "method2": {
                    "metric1": xxx,
                    "metric2": xxx,
                    ...
                },
                ...
            },
            ...
        }
        ```
        file_path (Path): Path to the file.
    """    

    with open(file_path, 'w') as f:
        f.write("| Degradations | Method |")
        metrics = list(list(list(data.values())[0].values())[0].keys())
        n_metrics = len(metrics)
        f.write(" ".join([f" {m.upper()} |" for m in metrics]))
        f.write('\n')
        f.write("|:---|:---|")
        f.write(" ".join([":---:|" for _ in range(n_metrics)]))
        f.write('\n')

        scoreboard = {
            "all": {"win": 0, "total": 0}
        }
        scoreboard.update({
            metric: {"win": 0, "total": 0} for metric in metrics
        })

        data = order_task(data)
        for task, method_vals in data.items():
            task_flag = False

            # find the winner
            order_dict = {metric: [] for metric in metrics}
            for metric in metrics:
                mtd_val_lst = [
                    (method, metric_vals[metric]) for method, metric_vals in method_vals.items()
                ]
                sign = -1 if metric.lower() == "lpips" else 1
                mtd_val_lst.sort(key=lambda x: sign*x[1], reverse=True)
                order_dict[metric] = mtd_val_lst
                if mtd_val_lst[0][0] == ours:
                    scoreboard[metric]["win"] += 1
                    scoreboard["all"]["win"] += 1
                scoreboard[metric]["total"] += 1
                scoreboard["all"]["total"] += 1

            # write the table
            for method, metric_vals in method_vals.items():
                t = task if not task_flag else ""
                t = t.replace(" compression artifact", "")
                t = t.replace("low resolution", "lr")
                task_flag = True
                f.write(f"| {t} | {method} |")
                for m in metrics:
                    val_str = f"{metric_vals[m]:.4}"
                    if method == order_dict[m][0][0]:
                        val_str = f"<font color=red>{val_str}</font>"
                    elif len(order_dict[m]) > 2 and method == order_dict[m][1][0]:
                        val_str = f"<font color=blue><u>{val_str}</u></font>"
                    f.write(f" {val_str} |")
                f.write('\n')

        f.write("\n| Metric | Win count | Win rate |\n")
        f.write("|:---:|:---:|:---:|\n")
        for metric, cnts in scoreboard.items():
            win_rate = cnts["win"]/cnts["total"]
            if win_rate < 0.5:
                win_rate_str = f"<font color=blue>{win_rate:.2%}</font>"
            else:
                win_rate_str = f"{win_rate:.2%}"
            f.write(f"| {metric} | {cnts['win']} | {win_rate_str} |\n")


def gen_latex(data: dict, file_path: Path, ablation: bool, avg_over_group: bool) -> None:
    def multi(dim, n, content):
        sarg = '*' if dim == 'row' else 'c'
        return f"\\multi{dim}{{{n}}}{{{sarg}}}{{{content}}}"
    
    method_to_disp_dict = {
        "default": "\\method", "random_deggt": "Random_deggt", "random_degpred": "Random_degpred",
        "transweather": "TransWeather", "airnet": "AirNet", "promptir": "PromptIR",
        "mioir": "MiOIR", "daclip": "DA-CLIP", "instructir": "InstructIR", "autodir": "AutoDIR",
        "woretr": " \\xmark \\cmark & \\cmark",
        "woretr_woref_worb": " \\xmark & \\xmark & \\xmark", 
        "woretr_worb": " \\xmark & \\cmark & \\xmark ",
        "worb": " \\cmark & \\xmark ", 
        "woref_worb": " \\xmark & \\xmark ",
        "fixedplan": "\\cmark", "fixedplan_r": "\\xmark",
        "default_rb": "\\method", "worb_rb": " \\cmark & \\xmark ",
    }
    if ablation:
        method_to_disp_dict['default'] = " \\cmark & \\cmark "
        method_to_disp_dict['default_rb'] = " \\cmark & \\cmark "
    with open(file_path, 'w') as f:
        metrics = list(list(list(data.values())[0].values())[0].keys())
        metrics_to_disp = [m.upper() for m in metrics]
        for i, metric in enumerate(metrics_to_disp):
            if metric == 'CLIPIQA':
                metrics_to_disp[i] = 'CLIP-IQA'
            elif metric == 'LPIPS':
                metrics_to_disp[i] = 'LPIPS$\downarrow$'
        n_cols = len(metrics) + 2
        if ablation:
            n_cols += 1
        n_methods = len(list(data.values())[0].keys())
        if avg_over_group:
            align = 'c'*n_cols
        else:
            align = 'l' + 'c'*(n_cols-1)

        f.write(f"\\begin{{tabular}}{{{align}}}\n")
        f.write("\\toprule\n")
        if ablation and avg_over_group:
            f.write(f'{multi("row", 2, "Degradations")} & {multi("column", 2, "Method")} &')
            f.write(" & ".join([multi("row", 2, m) for m in metrics_to_disp]))
            f.write(f"\\\\\n")
            f.write(f" & Ref. & Rb. {'&'*6}")
        else:
            method_head = "Method" if not ablation else "As planned"
            f.write(f"Degradations & {method_head} &")
            f.write(" & ".join(metrics_to_disp))
        f.write("\\\\\n")

        data = order_task(data)
        for task, method_vals in data.items():
            if avg_over_group:
                task = f"Group {task}"
            else:
                degs = task.split('+')
                if len(degs) == 2:
                    task = f"\\makecell[l]{{{degs[0]},\\\\{degs[1]}}}"
                else:
                    task = f"\\makecell[l]{{{degs[0]}, {degs[1]},\\\\{degs[2]}}}"
            f.write("\\midrule\n")
            task_flag = False

            # find the winner
            order_dict = {metric: [] for metric in metrics}
            for metric in metrics:
                mtd_val_lst = [
                    (method, metric_vals[metric]) for method, metric_vals in method_vals.items()
                ]
                sign = -1 if metric.lower() == "lpips" else 1
                mtd_val_lst.sort(key=lambda x: sign*x[1], reverse=True)
                order_dict[metric] = mtd_val_lst

            # write the table
            for method, metric_vals in method_vals.items():
                t = multi("row", n_methods, task) if not task_flag else ""
                task_flag = True
                mtd_to_disp = method_to_disp_dict.get(method, method)
                f.write(f" {t} & {mtd_to_disp} ")
                for m in metrics:
                    val_str = pad_4eff(metric_vals[m])
                    if method == order_dict[m][0][0]:
                        val_str = f"\\best{{{val_str}}}"
                    elif len(order_dict[m]) > 2 and method == order_dict[m][1][0]:
                        val_str = f"\\sbest{{{val_str}}}"
                    f.write(f"& {val_str}")
                f.write('\\\\\n')
            
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

groups = {
    "A": [
        "rain+haze",
        "motion blur+low resolution",
        "dark+noise",
        "defocus blur+jpeg compression artifact",
        "noise+jpeg compression artifact",
        "rain+low resolution",
        "motion blur+dark",
        "defocus blur+haze",
    ],
    "B": [
        "motion blur+jpeg compression artifact",
        "haze+noise",
        "defocus blur+low resolution",
        "rain+dark",
    ],
    "C": [
        "haze+motion blur+low resolution",
        "rain+noise+low resolution",
        "dark+defocus blur+jpeg compression artifact",
        "motion blur+defocus blur+noise",
    ]
}

comp_dir = Path("performance/comparison")
score_dir = Path("performance/scores")

def fill_data(data, method, tasks=None, avg_over_group=True):
    score_path = score_dir / "average" / f"{method}.json"
    with open(score_path) as f:
        scores = json.load(f)

    if avg_over_group:
        group_scores = {}
        for task, task_scores in scores.items():
            for group, group_tasks in groups.items():
                if task in group_tasks:
                    if group not in group_scores:
                        group_scores[group] = {}
                    for metric, score in task_scores.items():
                        if metric not in group_scores[group]:
                            group_scores[group][metric] = []
                        group_scores[group][metric].append(score)
                    break
            else:
                raise ValueError(f"{task} not in any group.")
        for group, this_group_scores in group_scores.items():
            for metric, m_scores in this_group_scores.items():
                group_scores[group][metric] = sum(m_scores) / len(m_scores)
        scores = group_scores

    for task, task_scores in scores.items():
        if tasks is not None and task not in tasks:
            continue
        if task not in data:
            data[task] = {}
        data[task][method] = task_scores

def gen_comp_table(method_lst, avg_over_group=True, ablated=None):
    data = {}
    if avg_over_group:
        tasks = None
    else:
        with open(score_dir / "average" / f"{method_lst[-1]}.json") as f:
            scores = json.load(f)
        tasks = list(scores.keys())  # avoid non-alignment of tasks
    for method in method_lst:
        fill_data(data, method, tasks, avg_over_group)
    suffix = "_detail" if not avg_over_group else ""
    if ablated is not None:
        save_dir = comp_dir / "ablation study" / ablated
    else:
        save_dir = comp_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir/f"{'+'.join(method_lst)}{suffix}.md"
    gen_md(data, save_path, ours=method_lst[0])
    gen_latex(data, save_path.with_suffix(".tex"), 
              ablation=bool(ablated), avg_over_group=avg_over_group)


gen_comp_table(["default", "random_deggt"])
gen_comp_table(["default", "random_deggt"], avg_over_group=False)
gen_comp_table(["default", "airnet", "promptir", "mioir", "daclip", "instructir", "autodir"])

gen_comp_table(["fixedplan", "random_deggt"])
gen_comp_table(["default", "random_degpred"])
gen_comp_table(["default", "random_degpred"], avg_over_group=False)

# ablation study
# rollback ? | w/ retrieval, w/ reflection
gen_comp_table(["default", "worb"], ablated="rollback")
gen_comp_table(["default", "worb"], ablated="rollback", avg_over_group=False)
gen_comp_table(["default_rb", "worb_rb"], ablated="rollback")
# rollback ? | w/o retrieval, w/ reflection
gen_comp_table(["woretr", "woretr_worb"], ablated="rollback")

# reflection ? | w/ retrieval, w/o rollback
gen_comp_table(["worb", "woref_worb"], ablated="reflection")
# reflection ? | w/o retrieval, w/o rollback
gen_comp_table(["woretr_worb", "woretr_woref_worb"], ablated="reflection")

# retrieval ? | w/ reflection, w/ rollback
# gen_comp_table(["default", "woretr"], ablated="retrieval")
gen_comp_table(["default", "woretr"], ablated="retrieval", avg_over_group=False)
# retrieval ? | w/ reflection, w/o rollback
gen_comp_table(["worb", "woretr_worb"], ablated="retrieval")
# retrieval ? | w/o reflection, w/o rollback
gen_comp_table(["woref_worb", "woretr_woref_worb"], ablated="retrieval")


gen_comp_table(["default", "woretr_worb"])

gen_comp_table(["fixedplan", "fixedplan_r"])
gen_comp_table(["fixedplan", "fixedplan_r"], avg_over_group=False)
