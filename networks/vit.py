"""Vision Transformer implementation."""

# pylint: disable = arguments-differ, invalid-name, too-many-instance-attributes, too-many-locals
# pylint: disable = too-many-arguments

from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.nn import Cell, Dense, Dropout, SequentialCell, LayerNorm
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype

#from nn.test_framework.utils.debug_util import PrintShapeTypeCell

from .transformer import BatchDense
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

    def __init__(self, d_model, image_size, patch_size, initialization, stem, body, head,
                 pool='cls', dropout_rate=0.1, norm=None):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls or mean'
        num_patches = (image_size // patch_size) ** 2

        if pool == "cls":
            self.cls_token = Parameter(initializer(initialization, (1, 1, d_model)),
                                       name='cls', requires_grad=True)
            self.pos_embedding = Parameter(initializer(initialization,
                                                       (1, num_patches + 1, d_model)),
                                           name='pos_embedding', requires_grad=True)
            self.tile = P.Tile()
            self.cat_1 = P.Concat(axis=1)
        else:
            self.pos_embedding = Parameter(initializer(initialization, (1, num_patches, d_model)),
                                           name='pos_embedding', requires_grad=True)
            self.mean = P.ReduceMean(keep_dims=False)
        self.pool = pool

        self.cast = P.Cast()
        self.dropout = Dropout(keep_prob=(1. - dropout_rate))
        self.stem = stem
        self.body = body
        self.head = head

        if norm is not None:
            self.norm = dynamic_call(norm, turn_on_func=True)
        else:
            self.norm = None
        #self.print = PrintShapeTypeCell()
        # exit(0)

    def construct(self, img):
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

        x = self.body(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.pool == "cls":
            x = x[:, 0]
        else:
            x = self.mean(x, (-2,))

        return self.head(x)
