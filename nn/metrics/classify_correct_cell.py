"""Cell that returns correct count of the prediction in classification network."""

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.communication.management import GlobalComm
from mindspore.ops import operations as P

# pylint: disable = arguments-differ

class ClassifyCorrectCell(nn.Cell):
    r"""
    Cell that returns correct count of the prediction in classification network.

    This Cell accepts a network as arguments.
    It returns correct count of the prediction to calculate the metrics.

    Args:
        network (Cell): The network Cell.

    Inputs:
        - **data** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
        - **label** (Tensor) - Tensor of shape :math:`(N, \ldots)`.

    Outputs:
        Tuple, containing a scalar correct count of the prediction

    Examples:
        >>> # For a defined network Net without loss function
        >>> net = Net()
        >>> eval_net = nn.ClassifyCorrectCell(net)
    """

    def __init__(self, network):
        super().__init__(auto_prefix=False)
        self._network = network
        self.argmax = P.Argmax()
        self.equal = P.Equal()
        self.cast = P.Cast()
        self.reduce_sum = P.ReduceSum()
        self.allreduce = P.AllReduce(P.ReduceOp.SUM, GlobalComm.WORLD_COMM_GROUP)

    def construct(self, data, label):
        outputs = self._network(data)
        y_pred = self.argmax(outputs)
        y_pred = self.cast(y_pred, mstype.int32)
        y_correct = self.equal(y_pred, label)
        y_correct = self.cast(y_correct, mstype.float32)
        y_correct = self.reduce_sum(y_correct)
        total_correct = self.allreduce(y_correct)
        return (total_correct,)


class ClassifyCorrectCell2(nn.Cell):
    r"""
    Cell that returns correct count of the prediction in classification network.

    This Cell accepts a network as arguments.
    It returns correct count of the prediction to calculate the metrics.

    Args:
        network (Cell): The network Cell.

    Inputs:
        - **data** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
        - **label** (Tensor) - Tensor of shape :math:`(N, \ldots)`.

    Outputs:
        Tuple, containing a scalar correct count of the prediction

    Examples:
        >>> # For a defined network Net without loss function
        >>> net = Net()
        >>> eval_net = nn.ClassifyCorrectCell(net)
    """

    def __init__(self, network):
        super().__init__(auto_prefix=False)
        self._network = network
        self.argmax = P.Argmax()
        self.equal = P.Equal()
        self.cast = P.Cast()
        self.reduce_sum = P.ReduceSum()
        self.allreduce = P.AllReduce(P.ReduceOp.SUM, GlobalComm.WORLD_COMM_GROUP)

    def construct(self, data, label):
        outputs = self._network(data)
        outputs_1, outputs_2 = outputs

        y_pred_1 = self.argmax(outputs_1)
        y_pred_1 = self.cast(y_pred_1, mstype.int32)
        y_correct_1 = self.equal(y_pred_1, label)
        y_correct_1 = self.cast(y_correct_1, mstype.float32)
        y_correct_1 = self.reduce_sum(y_correct_1)
        total_correct_1 = self.allreduce(y_correct_1)

        y_pred_2 = self.argmax(outputs_2)
        y_pred_2 = self.cast(y_pred_2, mstype.int32)
        y_correct_2 = self.equal(y_pred_2, label)
        y_correct_2 = self.cast(y_correct_2, mstype.float32)
        y_correct_2 = self.reduce_sum(y_correct_2)
        total_correct_2 = self.allreduce(y_correct_2)

        return (total_correct_1, total_correct_2)
