"""Vision Transformer implementation."""

# pylint: disable = arguments-differ, invalid-name, too-many-instance-attributes, too-many-locals
# pylint: disable = too-many-arguments

from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.nn import Cell, Dense, Dropout, SequentialCell, LayerNorm, CellList, Conv2d, GELU
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import mindspore.common.dtype as mstype
import math

#from nn.test_framework.utils.debug_util import PrintShapeTypeCell

from .transformer import BatchDense
from .transformer_dvt import DvtTransformerBlock
from utils import dynamic_call

MIN_NUM_PATCHES = 4


def pretrain_head(size_cfg, dropout_rate, num_classes, initialization, normalization, activation):
    """Head for ViT pretraining."""
    d_model = size_cfg["d_model"]
    mlp_dim = size_cfg["mlp_dim"]
    dense1 = Dense(d_model, mlp_dim)
    dense1.weight.set_data(initializer(initialization, [mlp_dim, d_model]))
    dense2 = Dense(mlp_dim, num_classes)
    dense2.weight.set_data(initializer(initialization, [num_classes, mlp_dim]))

    return SequentialCell([
        normalization,
        dense1,
        activation,
        Dropout(keep_prob=(1. - dropout_rate)),
        dense2
    ])


def origin_head(size_cfg, dropout_rate, num_classes, initialization, normalization, activation):
    """Head for ViT pretraining."""
    d_model = size_cfg["d_model"]
    mlp_dim = size_cfg["mlp_dim"]
    # dense1 = Dense(d_model, mlp_dim)
    # dense1.weight.set_data(initializer(initialization, [mlp_dim, d_model]))
    # dense2 = Dense(mlp_dim, num_classes)
    # dense2.weight.set_data(initializer(initialization, [num_classes, mlp_dim]))

    dense = Dense(d_model, num_classes)
    dense.weight.set_data(initializer(initialization, [num_classes, d_model]))

    return SequentialCell([
        dense
    ])


class VitStem(Cell):
    """Stem layer for ViT."""

    def __init__(self, d_model, patch_size, image_size, initialization, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches {num_patches} is too small'
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.patch_to_embedding = BatchDense(
            patch_dim, d_model, initialization, has_bias=True)

    def construct(self, img):
        p = self.patch_size
        bs, channels, H, W = img.shape
        x = self.reshape(img, (bs, channels, H // p, p, W // p, p))
        x = self.transpose(x, (0, 2, 4, 1, 3, 5))
        x = self.reshape(x, (bs, (H//p)*(W//p), channels*p*p))
        x = self.patch_to_embedding(x)
        return x


class ViT(Cell):
    """Vision Transformer implementation."""

    def __init__(self, d_model, image_size, patch_size, initialization, stem, head, 
    # block,
    dvt_block_depth, dvt_block_normalization1, dvt_block_normalization2, dvt_block_attention, dvt_block_feedforward, dvt_block_feature_resuse_feedforward, dvt_block_feature_reuse, dvt_block_relation_reuse, dvt_block_batch_size, dvt_block_patch_size=16, dvt_block_drop_path_rate=0.,
                 pool='cls', dropout_rate=0.1, norm=None, depth=12, num_heads=12, feature_reuse=True, relation_reuse=True, dvt_feature_reuse_ff_output_dim=48):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls or mean'
        self.patch_size = patch_size
        self.num_patches = (image_size // self.patch_size) ** 2

        self.feature_reuse = feature_reuse
        self.relation_reuse = relation_reuse

        self.less_patch_len = image_size // self.patch_size // 2

        if pool == "cls":
            self.cls_token = Parameter(initializer(initialization, (1, 1, d_model)),
                                       name='cls', requires_grad=True)
            self.pos_embedding = Parameter(initializer(initialization,
                                                       (1, self.num_patches + 1, d_model)),
                                           name='pos_embedding', requires_grad=True)
            self.tile = P.Tile()
            self.cat_1 = P.Concat(axis=1)
        else:
            self.pos_embedding = Parameter(initializer(initialization, (1, self.num_patches, d_model)),
                                           name='pos_embedding', requires_grad=True)
            self.mean = P.ReduceMean(keep_dims=False)
        self.pool = pool

        self.cast = P.Cast()
        self.dropout = Dropout(keep_prob=(1. - dropout_rate))
        self.stem = stem

        # self.body = body
        self.depth = depth
        self.num_heads = num_heads
        self.layers = CellList()
        # print('dvt_block_normalization1:', dvt_block_normalization1)
        for i in range(depth):
            print('i:', i)
            # # block_i = dynamic_call(block)
            # dvt_block_normalization1 = dynamic_call(dvt_block_normalization1)
            # dvt_block_normalization2 = dynamic_call(dvt_block_normalization2)
            # dvt_block_attention = dynamic_call(dvt_block_attention)
            # dvt_block_feedforward = dynamic_call(dvt_block_feedforward)
            # dvt_block_feature_resuse_feedforward = dynamic_call(dvt_block_feature_resuse_feedforward)
            block_i = DvtTransformerBlock(depth=dvt_block_depth, normalization1=dvt_block_normalization1, normalization2=dvt_block_normalization2, attention=dvt_block_attention, feedforward=dvt_block_feedforward, feature_resuse_feedforward=dvt_block_feature_resuse_feedforward, drop_path_rate=dvt_block_drop_path_rate, feature_reuse=dvt_block_feature_reuse, relation_reuse=dvt_block_relation_reuse, patch_size=dvt_block_patch_size, batch_size=dvt_block_batch_size, dvt_feature_reuse_ff_output_dim=dvt_feature_reuse_ff_output_dim)
            self.layers.append(block_i)  # DvtTransformerBlock
        # print('layers:', self.layers)
        # exit(0)

        self.head = head

        self.relation_reuse_upsample = P.ResizeNearestNeighbor(
            (int(math.sqrt(self.num_patches)), int(math.sqrt(self.num_patches))))
        # self.relation_reuse_upsample = P.ResizeBilinear(
        #     (int(math.sqrt(self.num_patches)), int(math.sqrt(self.num_patches))))

        if norm is not None:
            self.norm = dynamic_call(norm, turn_on_func=True)
        else:
            self.norm = None
        self.cat_2 = P.Concat(axis=2)
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        #self.print = PrintShapeTypeCell()
        self.chunk = P.Split(1, self.depth)

        if self.relation_reuse:
            print('self.relation_reuse:', self.relation_reuse)
            self.relation_reuse_conv = SequentialCell([
                Conv2d(self.num_heads * self.depth, self.num_heads *
                       self.depth * 3, kernel_size=1, stride=1, padding=0, has_bias=True),
                GELU(),
                Conv2d(self.num_heads * self.depth * 3, self.num_heads *
                       self.depth, kernel_size=1, stride=1, padding=0, has_bias=True)])

    def construct(self, img, features_to_be_reused_list, relations_to_be_reused_list):
        x = self.stem(img)
        bs, seq_len, _ = x.shape

        if self.pool == "cls":
            cls_tokens = self.tile(self.cls_token, (bs, 1, 1))
            # now x has shape = (bs, seq_len+1, d)
            x = self.cat_1((cls_tokens, x))
            x += self.pos_embedding[:, :(seq_len + 1)]
        else:
            x += self.pos_embedding[:, :seq_len]

        y = self.cast(x, mstype.float32)
        y = self.dropout(y)
        x = self.cast(y, x.dtype)
        #x = self.dropout(x)

        if relations_to_be_reused_list is not None:

            # new_hw = x.size(1) - 1
            # new_hw = 196
            new_hw = self.num_patches 
            relations_to_be_reused = self.cat_1(relations_to_be_reused_list)
            relations_to_be_reused = self.relation_reuse_conv(
                relations_to_be_reused)
            relation_temp = relations_to_be_reused[:, :, :, 1:]
            B, h, n, hw = relation_temp.shape
            Bh = B*h
            relation_temp = self.reshape(relation_temp, (Bh, n, self.less_patch_len, self.less_patch_len))

            # if True:
            #     split_index = int(relation_temp.size(0) / 2)
            #     relation_temp = torch.cat(
            #         (
            #             self.relation_reuse_upsample(
            #                 relation_temp[:split_index]),
            #             self.relation_reuse_upsample(
            #                 relation_temp[split_index:]),
            #         ), 0
            #     )
            relation_temp = self.relation_reuse_upsample(relation_temp)

            relation_temp = self.reshape(relation_temp, (Bh, n, new_hw))

            relations_to_be_reused_0 = relations_to_be_reused[:, :, :, 0]
            relations_to_be_reused_0 = self.reshape(
                relations_to_be_reused_0, (Bh, n, 1))
            relation_temp = self.cat_2(
                (relations_to_be_reused_0, relation_temp))
            relation_temp = self.transpose(relation_temp, (0, 2, 1))

            relation_cls_token_temp = relation_temp[:, :, 0:1]
            relation_temp = relation_temp[:, :, 1:]
            relation_temp = self.reshape(relation_temp, (Bh, (new_hw+1), self.less_patch_len, self.less_patch_len))

            # if True:
            #     split_index = int(relation_temp.size(0) / 2)
            #     relation_temp = torch.cat(
            #         (
            #             self.relation_reuse_upsample(
            #                 relation_temp[:split_index]),
            #             self.relation_reuse_upsample(
            #                 relation_temp[split_index:]),
            #         ), 0
            #     )
            relation_temp = self.relation_reuse_upsample(relation_temp)

            relation_temp = self.reshape(
                relation_temp, (Bh, (new_hw + 1), new_hw))
            relation_temp = self.cat_2(
                (relation_cls_token_temp, relation_temp))
            relation_temp = self.transpose(relation_temp, (0, 2, 1))
            relation_temp = self.reshape(
                relation_temp, (B, h, (new_hw+1), (new_hw+1)))

            # relations_to_be_reused_list = relation_temp.chunk(self.depth, 1)
            relations_to_be_reused_list = self.chunk(relation_temp)

        feature_list = ()
        relation_list = ()

        # x = self.body(x)
        for i in range(self.depth):
            if features_to_be_reused_list is not None:
                features_to_be_reused = features_to_be_reused_list[0]
            else:
                features_to_be_reused = None

            if relations_to_be_reused_list is not None:
                relations_to_be_reused = relations_to_be_reused_list[i]
            else:
                relations_to_be_reused = None

            x, relation = self.layers[i](
                x, features_to_be_reused, relations_to_be_reused)
            relation_list = relation_list + (relation,)

        feature_list = feature_list + (x,)

        if self.norm is not None:
            x = self.norm(x)

        if self.pool == "cls":
            x = x[:, 0]
        else:
            x = self.mean(x, (-2,))

        x = self.head(x)
        return x, feature_list, relation_list


class Vit_Dvt(Cell):
    """Vision Transformer implementation."""

    def __init__(self, less_token_backbone, normal_token_backbone, feature_reuse=True, relation_reuse=True):
        super().__init__()
        self.less_token = less_token_backbone
        self.normal_token = normal_token_backbone
        self.feature_reuse = feature_reuse
        self.relation_reuse = relation_reuse

    def construct(self, x):
        if self.feature_reuse == True and self.relation_reuse == True:
            less_token_output, features_to_be_reused_list, relations_to_be_reused_list = self.less_token(
                x, features_to_be_reused_list=None, relations_to_be_reused_list=None)
            output, _, _ = self.normal_token(
                x, features_to_be_reused_list=features_to_be_reused_list, relations_to_be_reused_list=relations_to_be_reused_list)

        elif self.feature_reuse == False and self.relation_reuse == True:
            less_token_output, features_to_be_reused_list, relations_to_be_reused_list = self.less_token(
                x, features_to_be_reused_list=None, relations_to_be_reused_list=None)
            output, _, _ = self.normal_token(
                x, features_to_be_reused_list=None, relations_to_be_reused_list=relations_to_be_reused_list)

        elif self.feature_reuse == True and self.relation_reuse == False:
            less_token_output, features_to_be_reused_list, relations_to_be_reused_list = self.less_token(
                x, features_to_be_reused_list=None, relations_to_be_reused_list=None)
            output, _, _ = self.normal_token(
                x, features_to_be_reused_list=features_to_be_reused_list, relations_to_be_reused_list=None)

        else:
            less_token_output, features_to_be_reused_list, relations_to_be_reused_list = self.less_token(
                x, features_to_be_reused_list=None, relations_to_be_reused_list=None)
            output, _, _ = self.normal_token(
                x, features_to_be_reused_list=None, relations_to_be_reused_list=None)
        return less_token_output, output
