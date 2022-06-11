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


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 这里之所以使用两组参数，是因为强制的将隐藏层输入拉到标准正态分布会限制模型的表达能力
        # 这样经过课学习参数的 scale 和 shift 可以恢复其表达能力
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
