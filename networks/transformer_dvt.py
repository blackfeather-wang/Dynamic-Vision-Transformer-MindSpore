"""Transformer implementation."""

# pylint: disable = too-many-arguments, arguments-differ, invalid-name, too-many-instance-attributes

import mindspore.common.dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore import Tensor
from mindspore.common.initializer import initializer
from mindspore.nn import SequentialCell, Dense, Dropout, Cell, CellList, ResizeBilinear
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from nn.cells import ResidualCell
from utils import dynamic_call
from .transformer import BatchDense

import numpy as np


class DvtAttention(Cell):
    """DvtAttention layer implementation."""

    def __init__(self, size_cfg, initialization, activation, dropout_rate=0.0):
        super().__init__()
        print('DvtAttention----in----')

        d_model = size_cfg["d_model"]
        dim_head = size_cfg["dim_head"]
        heads = size_cfg["heads"]

        inner_dim = heads * dim_head
        self.dim_head = dim_head
        self.heads = heads
        self.scale = Tensor([dim_head ** -0.5], mstype.float32)

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

    def construct(self, x, relations_to_be_reused):
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

        if relations_to_be_reused is not None:
            attn_scores = attn_scores + relations_to_be_reused

        relation = attn_scores

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
        return out, relation



class DvtFeatureReuseFeedForward(Cell):
    """FeedForward layer implementation."""

    def __init__(self, size_cfg, normalization, initialization, activation, dropout_rate=0.1, dvt_feature_reuse_ff_hidden_dim=128, dvt_feature_reuse_ff_output_dim=48):
        super().__init__()
        print('DvtFeatureReuseFeedForward----in----')

        self.norm1 = dynamic_call(normalization, turn_on_func=True)

        d_model = size_cfg["d_model"]
        # hidden_dim = size_cfg["mlp_dim"]
        hidden_dim = dvt_feature_reuse_ff_hidden_dim

        self.ff1 = BatchDense(d_model, hidden_dim, initialization)
        self.activation = activation
        self.dropout = Dropout(keep_prob=1.-dropout_rate)
        self.ff2 = BatchDense(hidden_dim, dvt_feature_reuse_ff_output_dim, initialization)
        self.cast = P.Cast()

    def construct(self, x):
        # y = self.ff1(x)
        # y = self.cast(y, mstype.float32)
        # y = self.activation(y)
        # y = self.cast(y, x.dtype)
        # y = self.dropout(y)
        # y = self.ff2(y)
        # y = self.dropout(y)
        y = self.norm1(x)
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


class DvtTransformerBlock(Cell):

    def __init__(self, depth, normalization1, normalization2, attention, feedforward, feature_resuse_feedforward, drop_path_rate=0., feature_reuse=False, relation_reuse=False, patch_size=16, batch_size=256, dvt_feature_reuse_ff_output_dim=48):
        super().__init__()
        print('DvtTransformerBlock----in----')

        # #drop_path_rate = 0.
        # dpr = [x.item() for x in np.linspace(0, drop_path_rate, depth)]
        # att_seeds = [np.random.randint(1024) for _ in range(depth)]
        # mlp_seeds = [np.random.randint(1024) for _ in range(depth)]

        self.feature_reuse = feature_reuse
        self.relation_reuse = relation_reuse

        self.norm1 = dynamic_call(normalization1, turn_on_func=True)
        self.attn = dynamic_call(attention, turn_on_func=True)
        # self.drop_path1 = DropPath(dpr[i], att_seeds[i])

        # dim should add 48 when feature reuse
        self.norm2 = dynamic_call(normalization2, turn_on_func=True)
        self.mlp = dynamic_call(feedforward, turn_on_func=True)

        self.feature_resuse_mlp = dynamic_call(
            feature_resuse_feedforward, turn_on_func=True)  # should be DvtFeatureReuseFeedForward
        # self.drop_path2 = DropPath(dpr[i], att_seeds[i])
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        # self.stop_grad = F.stop_gradient()
        self.cat_1 = P.Concat(axis=1)
        self.cat_2 = P.Concat(axis=2)
        self.patch_size = patch_size
        self.patch_len = 224 // self.patch_size
        self.less_patch_len = self.patch_len // 2
        self.patch_len_pow_2 = self.patch_len * self.patch_len
        print('self.patch_len_pow_2:', self.patch_len_pow_2)
        self.interpolate = P.ResizeNearestNeighbor((self.patch_len, self.patch_len))
        # self.interpolate = ResizeBilinear()
        # self.interpolate = P.ResizeBilinear((self.patch_len, self.patch_len), align_corners=True)
        # self.sqrt = P.Sqrt()
        # self.cast = P.Cast()
        # self.H = 7
        # self.W = 7
        self.batch_size = batch_size
        print('batch_size:', self.batch_size)
        # self.print = P.Print()
        self.pad_zero_feature_map = Tensor(
            np.zeros((self.batch_size, 1, dvt_feature_reuse_ff_output_dim)), mstype.float32)  # !!! B 256 !!!
        # self.fake_zero_feature_map = Tensor(
        #     np.zeros((self.batch_size, 1+self.patch_len_pow_2, 48)), mstype.float32)  # !!! B 256 !!!

    def construct(self, x, features_to_be_reused=None, relations_to_be_reused=None):
        identity = x
        x, relation = self.attn(self.norm1(x), relations_to_be_reused)
        x = identity + x

        identity = x

        # if features_to_be_reused is not None:
        if self.feature_reuse:

            features_to_be_reused = F.stop_gradient(features_to_be_reused)  # !!! stop gradient
            features_to_be_reused = self.feature_resuse_mlp(
                features_to_be_reused)

            feature_temp = features_to_be_reused[:, 1:, :]

            B, new_HW, C = feature_temp.shape

            feature_temp = self.transpose(feature_temp, (0, 2, 1))
            # now just support 7*7->14*14
            feature_temp = self.reshape(feature_temp, (B, C, self.less_patch_len, self.less_patch_len))
            feature_temp = self.interpolate(feature_temp)
            # feature_temp = self.interpolate(feature_temp, size=(self.patch_len, self.patch_len))
            feature_temp = self.reshape(feature_temp, (B, C, self.patch_len_pow_2))
            feature_temp = self.transpose(feature_temp, (0, 2, 1))
            # self.print('Print feature_temp:', feature_temp)

            feature_temp = self.cat_1(
                (self.pad_zero_feature_map, feature_temp))
            x = self.cat_2((x, feature_temp))
            # x = self.cat_2((x, self.fake_zero_feature_map))

        # x = identity + self.drop_path(self.mlp(self.norm2(x)))
        x = identity + self.mlp(self.norm2(x))
        return x, relation

