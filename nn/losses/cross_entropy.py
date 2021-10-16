"""CrossEntropy loss for label smoothing."""

# pylint: disable = arguments-differ

import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.nn.loss.loss import _Loss
from mindspore.ops import functional as F
from mindspore.ops import operations as P

__all__ = ['CrossEntropySmooth', 'CrossEntropySmoothMixup']

class CrossEntropySmooth(_Loss):
    """CrossEntropy"""
    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super().__init__()
        self.onehot = P.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)
        self.print = P.Print()
        self.reshape = P.Reshape()

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)
            #self.print("F.shape(logit)=", F.shape(logit))
            #self.print("F.shape(logit)[1]=", F.shape(logit)[1])
            #label = self.reshape(logit, (1, 1))
        loss = self.cross_entropy(logit, label)
        return loss


class CrossEntropySmoothMixup(_Loss):
    """CrossEntropy"""
    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super().__init__()
        self.onehot = P.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        #self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 2), mstype.float32)
        self.off_value = 1.0 * smooth_factor / (num_classes - 2)
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):

        off_label = P.Select()(P.Equal()(label, 0.0), \
           P.Fill()(mstype.float32, P.Shape()(label), self.off_value), \
           P.Fill()(mstype.float32, P.Shape()(label), 0.0))

        label = self.on_value * label + off_label
        loss = self.cross_entropy(logit, label)
        return loss


class CrossEntropySmoothMixup2(_Loss):
    """CrossEntropy"""
    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super().__init__()
        self.onehot = P.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        #self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 2), mstype.float32)
        self.off_value = 1.0 * smooth_factor / (num_classes - 2)
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logits, label):

        logit1, logit2 = logits

        off_label = P.Select()(P.Equal()(label, 0.0), \
           P.Fill()(mstype.float32, P.Shape()(label), self.off_value), \
           P.Fill()(mstype.float32, P.Shape()(label), 0.0))

        label = self.on_value * label + off_label
        loss = self.cross_entropy(logit1, label) + self.cross_entropy(logit2, label)
        return loss

