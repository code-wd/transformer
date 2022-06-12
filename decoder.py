from torch import nn

from utils import clones
from common import SublayerConnection, LayerNorm


class DecoderLayer(nn.Module):
    """
    这是一个 DecoderLayer 类的定义，一个 Decoder 是 N 个 DecoderLayer 的堆叠
    一个 DecoderLayer 包含三个子层：
        1. Masked MultiHeadAttention Layer
        2. Cross MultiHeadAttention Layer
        3. PositionwiseFeedForward Layer
    但是这三个子层是要 SubLayerConnection 层包裹起来，即增加了残差和 LayerNorm
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, target_mask):
        m = memory
        x = self.sublayers[0](x, lambda t: self.self_attn(t, t, t, target_mask))
        x = self.sublayers[1](x, lambda t: self.src_attn(x, m, m, src_mask))
        return self.sublayers[2](x, self.feed_forward)


class Decoder(nn.Module):
    """
    Decoder 是 N 个 DecoderLayer 堆叠而成
    """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, target_mask)
        return self.norm