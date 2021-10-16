"""Transformer implementation."""

# pylint: disable = too-many-arguments, arguments-differ, invalid-name, too-many-instance-attributes

import mindspore.common.dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore import Tensor
from mindspore.common.initializer import initializer
from mindspore.nn import SequentialCell, Dense, Dropout, Cell, CellList
from mindspore.ops import operations as P

from nn.cells import ResidualCell
from utils import dynamic_call

import numpy as np


class BatchDense(Cell):
    """BatchDense module."""

    def __init__(self, in_features, out_features, initialization, has_bias=True):
        super().__init__()
        self.out_features = out_features
        self.dense = Dense(in_features, out_features, has_bias=has_bias)
        self.dense.weight.set_data(initializer(
            initialization, [out_features, in_features]))
        self.reshape = P.Reshape()

    def construct(self, x):
        bs, seq_len, d_model = x.shape
        out = self.reshape(x, (bs * seq_len, d_model))
        out = self.dense(out)
        out = self.reshape(out, (bs, seq_len, self.out_features))
        return out


class Attention(Cell):
    """Attention layer implementation."""

    def __init__(self, size_cfg, initialization, activation, dropout_rate=0.0):
        super().__init__()
        print('Attention----in----')

        d_model = size_cfg["d_model"]
        dim_head = size_cfg["dim_head"]
        heads = size_cfg["heads"]

        inner_dim = heads * dim_head
        self.dim_head = dim_head
        self.heads = heads
        self.scale = Tensor([dim_head ** -0.5])

        self.to_q = Dense(d_model, inner_dim, has_bias=True)
        self.to_q.weight.set_data(initializer(
            initialization, [inner_dim, d_model]))
        self.to_k = Dense(d_model, inner_dim, has_bias=True)
        self.to_k.weight.set_data(initializer(
            initialization, [inner_dim, d_model]))
        self.to_v = Dense(d_model, inner_dim, has_bias=True)
        self.to_v.weight.set_data(initializer(
            initialization, [inner_dim, d_model]))

        self.to_out = Dense(inner_dim, d_model, has_bias=True)
        self.to_out.weight.set_data(initializer(
            initialization, [inner_dim, d_model]))
        self.dropout = Dropout(1 - dropout_rate)

        self.activation = activation

        # auxiliary functions
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.cast = P.Cast()
        self.mul = P.Mul()
        self.q_matmul_k = P.BatchMatMul(transpose_b=True)
        self.attn_matmul_v = P.BatchMatMul()

    def construct(self, x):
        '''x size - BxNxd_model'''
        bs, seq_len, d_model, h, d = x.shape[0], x.shape[1], x.shape[2], self.heads, self.dim_head

        x_2d = self.reshape(x, (-1, d_model))
        q, k, v = self.to_q(x_2d), self.to_k(x_2d), self.to_v(x_2d)

        q = self.reshape(q, (bs, seq_len, h, d))
        q = self.transpose(q, (0, 2, 1, 3))
        k = self.reshape(k, (bs, seq_len, h, d))
        k = self.transpose(k, (0, 2, 1, 3))
        v = self.reshape(v, (bs, seq_len, h, d))
        v = self.transpose(v, (0, 2, 1, 3))

        attn_scores = self.q_matmul_k(q, k)  # bs x h x seq_len x seq_len
        attn_scores = self.cast(attn_scores, mstype.float32)
        attn_scores = self.mul(attn_scores, self.scale)
        attn_scores = self.cast(attn_scores, x.dtype)
        attn_scores = self.activation(attn_scores)

        out = self.attn_matmul_v(attn_scores, v)  # bs x h x seq_len x dim_head
        out = self.transpose(out, (0, 2, 1, 3))
        out = self.reshape(out, (bs*seq_len, h*d))
        out = self.to_out(out)
        out = self.reshape(out, (bs, seq_len, d_model))
        #out = self.dropout(out)
        y = self.cast(out, mstype.float32)
        y = self.dropout(y)
        out = self.cast(y, out.dtype)
        #out = self.reshape(out, (bs, seq_len, d_model))
        return out


class FeedForward(Cell):
    """FeedForward layer implementation."""

    def __init__(self, size_cfg, initialization, activation, dropout_rate=0.1, in_d_model=-1):
        super().__init__()
        print('FeedForward----in----')

        d_model = size_cfg["d_model"]

        if in_d_model == -1:
            in_d_model = d_model
        else:
            in_d_model = in_d_model

        hidden_dim = size_cfg["mlp_dim"]

        print('d_model:', d_model)
        print('in_d_model:', in_d_model)
        print('hidden_dim:', hidden_dim)

        self.ff1 = BatchDense(in_d_model, hidden_dim, initialization)
        self.activation = activation
        self.dropout = Dropout(keep_prob=1.-dropout_rate)
        self.ff2 = BatchDense(hidden_dim, d_model, initialization)
        self.cast = P.Cast()

    def construct(self, x):
        # y = self.ff1(x)
        # y = self.cast(y, mstype.float32)
        # y = self.activation(y)
        # y = self.cast(y, x.dtype)
        # y = self.dropout(y)
        # y = self.ff2(y)
        # y = self.dropout(y)
        y = self.ff1(x)
        y = self.cast(y, mstype.float32)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.cast(y, x.dtype)
        y = self.ff2(y)
        y = self.cast(y, mstype.float32)
        y = self.dropout(y)
        y = self.cast(y, x.dtype)
        return y


class Transformer(Cell):
    """Transformer implementation."""

    def __init__(self, depth, normalization, attention, feedforward, drop_path_rate=0.):
        super().__init__()
        print('Transformer----in----')

        #drop_path_rate = 0.
        dpr = [x.item() for x in np.linspace(0, drop_path_rate, depth)]
        att_seeds = [np.random.randint(1024) for _ in range(depth)]
        mlp_seeds = [np.random.randint(1024) for _ in range(depth)]

        layers = []
        for i in range(depth):
            if drop_path_rate > 0:
                layers.append(
                    SequentialCell([
                        ResidualCell(SequentialCell([dynamic_call(normalization, turn_on_func=True),
                                                     dynamic_call(
                                                         attention, turn_on_func=True),
                                                     DropPath(dpr[i], att_seeds[i])])),
                        ResidualCell(SequentialCell([dynamic_call(normalization, turn_on_func=True),
                                                     dynamic_call(
                                                         feedforward, turn_on_func=True),
                                                     DropPath(dpr[i], mlp_seeds[i])]))
                    ])
                )
            else:
                layers.append(
                    SequentialCell([
                        ResidualCell(SequentialCell([dynamic_call(normalization, turn_on_func=True),
                                                     dynamic_call(attention, turn_on_func=True)])),
                        ResidualCell(SequentialCell([dynamic_call(normalization, turn_on_func=True),
                                                     dynamic_call(feedforward, turn_on_func=True)]))
                    ])
                )

        self.layers = SequentialCell(layers)

    def construct(self, x):
        return self.layers(x)


class DropPath(Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, seed=0):
        super(DropPath, self).__init__()
        #self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob
        self.rand = P.UniformReal()  # (seed=seed)
        self.shape = P.Shape()
        self.floor = P.Floor()
        self.print = P.Print()

    def construct(self, x):
        if self.training:
            x_shape = self.shape(x)  # B N C
            random_tensor = self.rand((x_shape[0], 1, 1))
            #self.print("random_tensor=", random_tensor)
            random_tensor = random_tensor + self.keep_prob
            random_tensor = self.floor(random_tensor)
            x = x / self.keep_prob
            x = x * random_tensor
        return x


# def drop_path(x, drop_prob: float = 0., training: bool = False):

#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     # work with diff dim tensors, not just 2D ConvNets
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)
#     random_tensor = keep_prob + \
#         torch.rand(shape, dtype=x.dtype, device=x.device)
#     random_tensor.floor_()  # binarize
#     output = x.div(keep_prob) * random_tensor
#     return output
