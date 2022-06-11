import torch
from torch import nn

from utils import clones
from common import SublayerConnection, LayerNorm


class EncoderLayer(nn.Module):
    """
    这是一个 Encoder 层，Encoder 是多个相同的 Encoder 层堆叠的
    一个 Encoder 层包含两个部分 MultiHeadAttention 层和 PositionwiseFeedForward 层
    但是这两个子层是要 SubLayerConnection 层包裹起来，即增加了残差和 LayerNorm
    """

    def __int__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__int__()
        self.attn = self_attn
        self.feed_forward = feed_forward
        # encoder 中包含两个子层，因此，这里复制了两个连接部分
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # 这个匿名函数中，传入的形参只有 t，这个 t 的大小将会在 sublayer 中确定
        # 在这个例子中，t 对应的实参是： self.norm(x) 【根据传入 sublayer 中 x 参数确定】
        # 而最后一个参数 mask 则是直接传入的
        x = self.sublayer[0](x, lambda t: self.attn(t, t, t, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """
    这里的 Encoder 是 N 个相同的 EncoderLayer 的堆叠
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
