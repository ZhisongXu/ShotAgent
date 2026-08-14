import os
from shutil import rmtree

from ..tool import Tool
from ..multitask_tools import *


__all__ = ['denoising_toolbox']


subtask = 'denoising'
denoising_toolbox = [
    XRestormer(subtask=subtask),
    SwinIR(subtask=subtask, pretrained_on='15'),
    SwinIR(subtask=subtask, pretrained_on='50'),
    MPRNet(subtask=subtask),
    MAXIM(subtask=subtask),
    Restormer(subtask=subtask),
    NAFNet(subtask=subtask),
    # AutoDIR(subtask=subtask),
]
