"""Affine layer."""

# pylint: disable = arguments-differ

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.common import Parameter
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P

class Affine(nn.Cell):
    """Affine layer implementation."""
    def __init__(self, n_features):
        super().__init__()
        self.gamma = Parameter(initializer('ones', (n_features,)), requires_grad=True, name="gamma")
        self.beta = Parameter(initializer('zeros', (n_features,)), requires_grad=True, name="beta")

    def construct(self, x):
        x = self.gamma * x + self.beta
        return x

class Affine32(nn.Cell):
    """Affine layer implementation with forced float32 cast."""
    def __init__(self, n_features):
        super().__init__()
        self.gamma = Parameter(initializer('ones', (n_features,)), requires_grad=True, name="gamma")
        self.beta = Parameter(initializer('zeros', (n_features,)), requires_grad=True, name="beta")
        self.cast = P.Cast()

    def construct(self, x):
        dtype = x.dtype
        x = self.cast(x, mstype.float32)
        x = self.gamma * x + self.beta
        x = self.cast(x, dtype)
        return x
