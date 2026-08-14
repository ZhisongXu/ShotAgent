import os
from pathlib import Path
from typing import Optional

from .custom_types import Subtask, ToolName
from .misc import sorted_glob, sorted_rglob


class ImgNode:
    """Attributes:
    - img_dir (Path)
    - img_path (Path | None)
    - is_root (bool)
    - If not is_root:
        - parent_img_dir (Path)
        - subtask (Subtask)
        - tool (Tool)
    - name (str)
    - children_dict (dict[Subtask, list[ImgNode]])
    """

    def __init__(self, img_dir: Path, is_root: bool = False):
        self.img_dir = img_dir

        try:
            img_name: str = next(img_dir.glob('*.png')).name
            self.img_path: Path = img_dir / img_name
        except:
            self.img_path = None

        self.is_root: bool = is_root
        if not is_root:
            tool_dir = img_dir.parent
            subtask_dir = tool_dir.parent

            self.parent_img_dir = subtask_dir.parent / '0-img'
            self.subtask: Subtask = subtask_dir.name[
                subtask_dir.name.find('-')+1:]
            self.tool: ToolName = tool_dir.name[tool_dir.name.find('-')+1:]

        self.name = "input" if is_root else self.tool

        children: list[ImgNode] = [
            ImgNode(child_dir)
            for child_dir in sorted_glob(img_dir.parent, '*/*/0-img')
        ]
        # group by subtask
        self.children_dict: dict[Subtask, list[ImgNode]] = {}
        for child in children:
            if child.subtask not in self.children_dict:
                self.children_dict[child.subtask] = []
            self.children_dict[child.subtask].append(child)


class ImgTree:
    """Attributes:
    tree_dir (Path)
    root (ImgNode)
    node_dict (dict[Path, ImgNode])
    html_dir (Path)
    html_page (str)

    Structure of the directory is like:
    ```
    {tree_dir}
    ├── 0-img
    │   └── {input_path}
    ├── {subtask_dir} 1
    |   ├── {tool_dir} 1
    |   │   ├── 0-img
    |   │   |   └── output.png
    |   │   └── {subtask_dir}
    |   │       └── ...
    |   ├── ...
    |   └── {tool_dir} 1_n
    |       └── 0-img
    |           └── output.png
    ├── ...
    └── {subtask_dir} m
        ├── {tool_dir} 1
        │   └── 0-img
        │       └── output.png
        ├── ...
        └── {tool_dir} m_n
            └── 0-img
                └── output.png
    ```
    """

    def __init__(self, tree_dir: Path, html_dir: Optional[Path] = None):
        self.tree_dir: Path = tree_dir
        self.root: ImgNode = ImgNode(
            self.tree_dir / '0-img', is_root=True)

        self.node_dict: dict[Path, ImgNode] = {self.root.img_dir: self.root}
        self.node_dict.update({
            img_dir: ImgNode(img_dir)
            for img_dir in sorted_rglob(self.tree_dir, '0-img')
            if img_dir != self.root.img_dir
        })

        self.n_nodes = len(self.node_dict)
        self.n_leaves = 0
        for node in self.node_dict.values():
            if not node.children_dict:
                self.n_leaves += 1

        if html_dir is None:
            html_dir = tree_dir.parent
        self.html_dir = html_dir  # to enable using relative path in html

        self._set_html_templates()

    def get_execution_path(self, img_path: Path
                           ) -> list[tuple[Subtask, ToolName]]:
        """Returns the execution path of the restored image, which is a list of tuples (subtask, tool)."""
        execution_path: list[tuple[Subtask, ToolName]] = []

        def get_stem(name: str):
            return name[name.find('-')+1:]
        cur_img_path = img_path
        while cur_img_path != self.root.img_path:
            tool_dir = cur_img_path.parents[1]
            tool_name = get_stem(tool_dir.name)
            subtask_dir = tool_dir.parent
            subtask_name = get_stem(subtask_dir.name)
            execution_path.append((subtask_name, tool_name))
            cur_img_path = sorted_glob(subtask_dir.parent / '0-img')[0]
        return execution_path[::-1]

    def to_html(self) -> None:
        with open(self.html_dir/"img_tree.html", 'w') as f:
            f.write(self.html_page)

    @property
    def html_page(self) -> str:
        return self._page_html_template.format(
            img_tree=self._get_img_html(self.root))

    def _get_img_html(self, node: ImgNode):
        img_html = self._img_html_template.format(
            name=node.name,
            img_path=os.path.relpath(node.img_path, self.html_dir),
            subtasks="\n".join(
                self._get_subtask_html(subtask, children)
                for subtask, children in node.children_dict.items()
            ))
        return img_html

    def _get_subtask_html(self, subtask: str, children: list[ImgNode]):
        subtask_html = self._subtask_html_template.format(
            subtask=subtask,
            descendants="\n".join(
                self._get_img_html(child)
                for child in children
            ))
        return subtask_html

    def __str__(self):
        def _subtree_str(node: ImgNode, indent: int = 0):
            res = '  ' * indent + node.name + '\n'
            for subtask, children in node.children_dict.items():
                res += '  ' * (indent + 1) + f"subtask:{subtask}\n"
                for child in children:
                    res += _subtree_str(child, indent + 2)
            return res
        return _subtree_str(self.root)

    def _set_html_templates(self):
        self._page_html_template = (
            """<!DOCTYPE html><html lang="en">"""
            """  <head>"""
            """  <meta charset="UTF-8"><title>Image Tree</title>"""
            """  <style>"""
            """    summary {{font-size: 30px}}"""
            """    details {{margin-left: 30px}}"""
            """  </style>"""
            """  </head>"""
            """  <body>{img_tree}</body>"""
            """</html>"""
        )
        self._img_html_template = (
            """<details open>"""
            """  <summary>{name}</summary>"""
            """  <img src='{img_path}'/>"""
            """  {subtasks}"""
            """</details>"""
        )
        self._subtask_html_template = (
            """<details open>"""
            """  <summary>{subtask}</summary>"""
            """  {descendants}"""
            """</details>"""
        )


if __name__ == "__main__":
    img_dir = Path("exhaustive_sequences/d2/dark+noise/001").resolve()
    tree_dir = img_dir / "tree"
    tree = ImgTree(tree_dir)
    print(tree.root.img_dir)
    print(tree)
    print(tree.root.children_dict)
    print(tree.n_leaves)
    print(tree.n_nodes)
    tree.to_html()
