# One can manually invoke multiple IR tools to restore a complexly degraded image

from flask import Flask, render_template, request
from pathlib import Path
import shutil
from typing import Optional

from utils.img_tree import ImgTree, ImgNode
from executor import executor


BASE_DIR: Path = Path('manual_exp')


class ExpManager:
    """Three modes
    + batch: requires dataset_path, start
    + single: requires input_path
    + resume: requires task_dir
    """

    def __init__(self,
                 dataset_path: Optional[Path] = None,
                 start: Optional[int] = 0,
                 input_path: Optional[Path] = None,
                 task_dir: Optional[Path] = None
                 ):
        if dataset_path is not None:  # mode: batch
            self._iter: list[Path] = sorted(list(dataset_path.glob('*')))
            self._idx: int = start
            self._exp_dir: Path = BASE_DIR / dataset_path.relative_to('.')

            self._exp_dir.mkdir(parents=True, exist_ok=True)

            # initialize the first task
            self._cur_task: TaskManager = TaskManager(
                self._iter[start], self._exp_dir)

        elif input_path is not None:  # mode: single
            self._cur_task: TaskManager = TaskManager(input_path, BASE_DIR)

        elif task_dir is not None:  # mode: resume
            self._cur_task: TaskManager = TaskManager(task_dir=task_dir)

    def init_next_task(self):
        """For mode 'batch', move to the next task."""
        self._idx += 1
        self._cur_task = TaskManager(self._iter[self._idx], self._exp_dir)

    @property
    def img_dom(self) -> str:
        return self._cur_task.img_dom

    @property
    def task_dir(self) -> Path:
        return self._cur_task.task_dir

    @property
    def comment(self) -> str:
        return self._cur_task.comment

    def update_comment(self, comment: str) -> None:
        self._cur_task.comment = comment


class TaskManager:
    def __init__(self,
                 input_path: Optional[Path] = None,
                 exp_dir: Optional[Path] = None,
                 task_dir: Optional[Path] = None) -> None:
        """Prepare the directory for the task."""
        if input_path is not None:
            self._input_path: Path = input_path
            self._task_id: str = input_path.stem
            self.task_dir: Path = exp_dir / self._task_id

            # if self.task_dir.exists():
            #     shutil.rmtree(self.task_dir)

            root_input_dir: Path = self.task_dir / '0-img'
            self._root_input_path: Path = root_input_dir / 'input.png'

            root_input_dir.mkdir(parents=True)
            shutil.copy(self._input_path, self._root_input_path)

        else:
            assert task_dir is not None
            self.task_dir = task_dir

        self.comment: str = ""

    @property
    def img_dom(self) -> str:
        img_tree = ImgTree(self.task_dir)
        return self._get_img_dom(img_tree.root)

    def _get_img_dom(self, node: ImgNode):
        dom = """\
<details open>
<summary>{name}</summary>
<img src='{img_path}' onclick='execute("{img_path}")' />
{subtasks}
</details>""".format(
            name=node.name, img_path=node.img_path,
            subtasks="\n".join(
                self._get_subtask_dom(subtask, children)
                for subtask, children in node.children_dict.items()
            ))
        return dom

    def _get_subtask_dom(self, subtask: str, children: list[ImgNode]):
        dom = """\
<details open>
<summary>{subtask}</summary>
{descendants}
</details>""".format(
            subtask=subtask,
            descendants="\n".join(
                self._get_img_dom(child)
                for child in children
            ))
        return dom


app = Flask(__name__, static_folder=BASE_DIR, template_folder='')


@app.route('/')
def render_page():
    """Return the HTML page of the task tree."""

    dom = exp_manager.img_dom
    task_dir = exp_manager.task_dir
    comment = exp_manager.comment

    page = render_template(
        'playground.html',
        dom=dom, comment=comment)
    with open(task_dir / 'archive.html', 'w') as f:
        page_archive = page.replace(task_dir.as_posix()+'/', '')
        page_archive = page_archive.replace(
            '<button onclick="next()">Next Image</button>', '')
        page_archive = page_archive.replace(
            '<button onclick="saveComment()">Save Comment</button>', '')
        f.write(page_archive)
    return page


@app.route('/execute')
def execute():
    img_path: Path = Path().resolve() / request.args.get('img_path')
    subtask: str = request.args.get('subtask')
    executor.execute_subtask(subtask, input_path=img_path)
    return {}


@app.route('/next')
def make_next_task():
    render_page()  # save the current page with comment
    executor._executed_subtask_cnt = 0
    exp_manager.init_next_task()
    return {}


@app.route('/set_comment')
def set_comment():
    exp_manager.update_comment(request.args.get('comment'))
    return {}


@app.route('/mark_as_best')
def mark_as_best():
    img_path: Path = Path().resolve() / request.args.get('img_path')
    task_dir = exp_manager.task_dir

    # get name
    def get_stem(name): return name[name.find('-')+1:]
    name = f"{get_stem(img_path.parents[2].name)}_{get_stem(img_path.parents[1].name)}.png"
    name = 'lq.png'
    cur_path = img_path.parents[1].relative_to(Path().resolve())

    while cur_path != task_dir:
        name = f"{get_stem(cur_path.parents[0].name)}_{get_stem(cur_path.name)} {name}"
        cur_path = cur_path.parents[1]

    name = 'best ' + name

    shutil.copy(img_path, task_dir / name)
    return {}


if __name__ == '__main__':
    # # mode: batch
    # dataset_path: Path = None
    # exp_manager = ExpManager(dataset_path=dataset_path, start=22)

    # mode: single
    input_path: Path = Path("dataset/example.png").resolve()
    exp_manager = ExpManager(input_path=input_path)

    # # mode: resume
    # task_dir: Path = BASE_DIR / 'dataset' / 'LQ' / "d3" / \
        # "dark+defocus blur+jpeg compression artifact" / "001"
    # exp_manager = ExpManager(task_dir=task_dir)

    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
