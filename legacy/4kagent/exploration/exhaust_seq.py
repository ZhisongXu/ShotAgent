from pathlib import Path
import argparse
import shutil
from tqdm import tqdm

from executor import executor
from utils.misc import sorted_glob, sorted_rglob
from utils.img_tree import ImgTree


distortion_subtask_dict = {
    "low resolution": "super-resolution",
    "noise": "denoising",
    "jpeg compression artifact": "jpeg compression artifact removal",
    "dark": "brightening",
    "haze": "dehazing",
    "motion blur": "motion deblurring",
    "defocus blur": "defocus deblurring",
    "rain": "deraining",
}


def get_n_leaves(subtask_idx_lst: list[int]) -> int:
    """If #tools for each subtask is n_1, ..., n_d, then #leaves is
       f(n_1, ..., n_d) 
     = sum_{i=1}^d n_i * f(n_1, ..., n_{i-1}, n_{i+1}, ..., n_d) 
     = d! * prod_{i=1}^d n_i
    """    
    if not subtask_idx_lst:
        return 1
    n_leaves = 0
    for i, subtask_idx in enumerate(subtask_idx_lst):
        rem_subtask_idx_lst = subtask_idx_lst[:i] + subtask_idx_lst[i+1:]
        n_leaves += n_tool_lst[subtask_idx] * get_n_leaves(rem_subtask_idx_lst)
    return n_leaves


def get_n_nodes(subtask_idx_lst: list[int]) -> int:
    """If #tools for each subtask is n_1, ..., n_d, then #leaves is
       f(n_1, ..., n_d) 
     = 1 + sum_{i=1}^d n_i * f(n_1, ..., n_{i-1}, n_{i+1}, ..., n_d) 
    """  
    n_nodes = 1
    for i, subtask_idx in enumerate(subtask_idx_lst):
        rem_subtask_idx_lst = subtask_idx_lst[:i] + subtask_idx_lst[i+1:]
        n_nodes += n_tool_lst[subtask_idx] * get_n_nodes(rem_subtask_idx_lst)
    return n_nodes


def generate_tree(subtask_idx_lst: list[int], root_dir: Path, virtual: bool = False):
    if not subtask_idx_lst:
        return
    for i, subtask_idx in enumerate(subtask_idx_lst):
        rem_subtask_idx_lst = subtask_idx_lst[:i] + subtask_idx_lst[i+1:]
        subtask = subtasks[subtask_idx]
        toolbox = toolboxes[subtask_idx]

        subtask_dir = root_dir / f"subtask-{subtask}"
        subtask_dir.mkdir()
        input_dir = root_dir / '0-img'
        for tool in toolbox:
            tool_dir = subtask_dir / f"tool-{tool.tool_name}"
            tool_dir.mkdir()
            output_dir = tool_dir / '0-img'
            output_dir.mkdir()
            if not virtual:
                tool(input_dir, output_dir, silent=True)
            generate_tree(rem_subtask_idx_lst, tool_dir, virtual=virtual)


parser = argparse.ArgumentParser()
parser.add_argument("--n_d", type=int, default=2)
parser.add_argument("--idx", type=int, default=0)
parser.add_argument("--range", type=int, nargs=2, default=[1, 100])
args = parser.parse_args()

input_dir = Path("dataset/train").resolve()
root_output_dir = Path("exhaustive_sequences").resolve()

n_d: int = args.n_d
idx: int = args.idx
start, end = args.range
start -= 1

deg_dirs = sorted_glob(input_dir, f"d{n_d}/*")
deg_dir = deg_dirs[idx]
degs = deg_dir.stem.split('+')
subtasks = [distortion_subtask_dict[deg] for deg in degs]
toolboxes = [executor.toolbox_router[subtask] for subtask in subtasks]
n_tool_lst = [len(toolbox) for toolbox in toolboxes]

nd_output_dir = root_output_dir / f"d{n_d}" / deg_dir.stem
nd_output_dir.mkdir(exist_ok=True, parents=True)

for i in range(n_d):
    print(f"{i+1}/{n_d}")
    print(f"degradation: {degs[i]}")
    print(f"subtask: {subtasks[i]}")
    print(f"toolbox: {[tool.tool_name for tool in toolboxes[i]]}")
    print(f"#tools: {n_tool_lst[i]}")
print()

all_subtask_idx_lst = list(range(n_d))

expected_n_leaves = get_n_leaves(all_subtask_idx_lst)
expected_n_nodes = get_n_nodes(all_subtask_idx_lst) - 1
print(f"Expected #leaves: {expected_n_leaves}")
print(f"Expected #nodes (except root): {expected_n_nodes}")

leave_pat = "0-img"
for i in range(n_d):
    leave_pat = "*/*/" + leave_pat


def generate_imgs(virtual=False):
    input_img_path_lst = sorted_glob(deg_dir, "*")[start:end]
    for input_img_path in tqdm(input_img_path_lst):
        img_tree_dir = nd_output_dir / input_img_path.stem / "tree"
        img_tree_dir.mkdir(parents=True)
        input_dir = img_tree_dir / '0-img'
        input_dir.mkdir()
        shutil.copy(input_img_path, input_dir/"input.png")
        generate_tree(all_subtask_idx_lst, img_tree_dir, virtual=virtual)


def generate_html():
    for img_tree_dir in sorted_rglob(nd_output_dir, "tree"):
        ImgTree(img_tree_dir, img_tree_dir.parent).to_html()
        

def check_number():
    for img_tree_dir in sorted_rglob(nd_output_dir, "tree"):
        n_leaves = len(sorted_glob(img_tree_dir, leave_pat))
        assert n_leaves == expected_n_leaves, \
            f"Expected {expected_n_leaves} images, but got {n_leaves} images."
        n_nodes = len(sorted_rglob(img_tree_dir, "0-img")) - 1
        assert n_nodes == expected_n_nodes, \
            f"Expected {expected_n_nodes} nodes, but got {n_nodes} nodes."
        

if __name__ == "__main__":
    # virtual = True
    virtual = False
    generate_imgs(virtual=virtual)

    # all
    if not virtual:
        generate_html()
    check_number()
