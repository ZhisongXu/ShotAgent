import os

from ..tool import Tool
from ..multitask_tools import *


__all__ = ['deraining_toolbox']


subtask = 'deraining'
deraining_toolbox = [
    MPRNet(subtask=subtask),
    MAXIM(subtask=subtask),
    XRestormer(subtask=subtask),
    Restormer(subtask=subtask),
    DiffPlugin(subtask=subtask),
    # AutoDIR(subtask=subtask),
]