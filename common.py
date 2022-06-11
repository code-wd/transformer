"""
这个文件用于存储 Encoder 和 decoder 中通用的组件
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from utils import clones


def attention(query, key, value, mask=None, dropout=None):
    """
    计算 Scaled Dot Product Attention
    :param query: query 向量
    :param key: key 向量
    :param value: value 向量
    :param mask: 如果使用 mask 机制，对应的 mask 向量
    :param dropout: 如果使用 dropout，对应的值
    :return:
    """
    d_k = query.size(-1)  # query 和 key 的维度相同，都是 d_k
    scores = torch.matmul(query, key.transpose(-2, -1))  # 注意这里的转置操作【query 与 key 维度相同，所以转置】
    scores = scores / math.sqrt(d_k)  # Scale 缩放操作，使得满足方差为 1，均值为 0 的操作

    # TODO: 这里细节还需要商榷，但是可以肯定的是，让 mask=0 的位置替换为一个非常小的负数
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)  # 使用softmax计算 attention

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        :param h: 多头的数量
        :param d_model: attention 的维度（多头维度叠加）
        :param dropout: dropout 系数
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0  # d_model 维度必须是 head 数的整数倍
        self.d_k = d_model // h  # 这里假设 d_k == d_q == d_v
        self.h = h
        # 4个全连接层，其中三个用于 Q，K，V 向量，最后一个用于多头 Attention 输出部分
        self.linear_list = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None  # 这个是得到的 attention 值
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # 这里增加一个维度，因为下面在计算 Attention 的时候也增加了一个维度
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_num = query.size(0)

        # 1. linear projection 操作（相当于 weights）
        query, key, value = [
            linear_layer(x).view(batch_num, -1, self.h, self.d_k)
            for linear_layer, x in zip(self.linear_list, (query, key, value))
        ]

        # 2. Attention 操作
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3. 多头之间 concat 操作，并通过最后的 linear 层
        x = (
            x.transpose(1, 2).contiguous().view(batch_num, -1, self.h*self.d_k)
        )
        del query
        del key
        del value
        return self.linear_list[-1](x)


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


class PositionwiseFeedForward(nn.Module):
    def __int__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__int__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
