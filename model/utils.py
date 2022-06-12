import copy

import torch
from torch import nn


def clones(module, N):
    """
    复制 N 个相同的模型层
    :param module: 要被复制的模型
    :param N: 复制次数
    :return:
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



