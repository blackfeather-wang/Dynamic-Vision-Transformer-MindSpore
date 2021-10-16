"""Calculates the accuracy for classification data in distributed mode."""

# pylint: disable = line-too-long

import mindspore.nn as nn


class DistAccuracy(nn.Metric):
    r"""
    Calculates the accuracy for classification data in distributed mode.

    The accuracy class creates two local variables, correct number and total number that are used
    to compute the frequency with which predictions matches labels. This frequency is ultimately
    returned as the accuracy: an idempotent operation that simply divides correct number by total
    number.

    .. math::
        \text{accuracy} =\frac{\text{true_positive} + \text{true_negative}}
        {\text{true_positive} + \text{true_negative} + \text{false_positive} + \text{false_negative}}

    Args:
        eval_type (str): Metric to calculate the accuracy over a dataset, for
            classification (single-label).

    Examples:
        >>> y_correct = Tensor(np.array([20]))
        >>> metric = nn.DistAccuracy(batch_size=3, device_num=8)
        >>> metric.clear()
        >>> metric.update(y_correct)
        >>> accuracy = metric.eval()
    """

    def __init__(self, batch_size, device_num, val_len):
        super().__init__()
        self.clear()
        self.batch_size = batch_size
        self.device_num = device_num
        self.val_len = val_len

    def clear(self):
        """Clears the internal evaluation result."""
        self._correct_num = 0
        self._total_num = 0

    def update(self, *inputs):
        """
        Updates the internal evaluation result :math:`y_{pred}` and :math:`y`.

        Args:
            inputs: Input `y_correct`. `y_correct` is a `scalar Tensor`.
                `y_correct` is the right prediction count that gathered from all devices
                it's a scalar in float type

        Raises:
            ValueError: If the number of the input is not 1.
        """
        if len(inputs) != 1:
            raise ValueError('Distribute accuracy needs 1 input (y_correct), but got {}'.\
                format(len(inputs)))
        y_correct = self._convert_data(inputs[0])
        self._correct_num += y_correct
        self._total_num += self.batch_size * self.device_num

    def eval(self):
        """
        Computes the accuracy.

        Returns:
            Float, the computed result.

        Raises:
            RuntimeError: If the sample size is 0.
        """
        if self._total_num == 0:
            raise RuntimeError("Accuracy can't be calculated, because the number of samples is 0.")
        return self._correct_num * 100 / self.val_len


class DistAccuracy2(nn.Metric):
    r"""
    Calculates the accuracy for classification data in distributed mode.

    The accuracy class creates two local variables, correct number and total number that are used
    to compute the frequency with which predictions matches labels. This frequency is ultimately
    returned as the accuracy: an idempotent operation that simply divides correct number by total
    number.

    .. math::
        \text{accuracy} =\frac{\text{true_positive} + \text{true_negative}}
        {\text{true_positive} + \text{true_negative} + \text{false_positive} + \text{false_negative}}

    Args:
        eval_type (str): Metric to calculate the accuracy over a dataset, for
            classification (single-label).

    Examples:
        >>> y_correct = Tensor(np.array([20]))
        >>> metric = nn.DistAccuracy(batch_size=3, device_num=8)
        >>> metric.clear()
        >>> metric.update(y_correct)
        >>> accuracy = metric.eval()
    """

    def __init__(self, batch_size, device_num, val_len):
        super().__init__()
        self.clear()
        self.batch_size = batch_size
        self.device_num = device_num
        self.val_len = val_len

    def clear(self):
        """Clears the internal evaluation result."""
        self._correct_num_1 = 0
        self._correct_num_2 = 0
        self._total_num = 0

    def update(self, *inputs):
        """
        Updates the internal evaluation result :math:`y_{pred}` and :math:`y`.

        Args:
            inputs: Input `y_correct`. `y_correct` is a `scalar Tensor`.
                `y_correct` is the right prediction count that gathered from all devices
                it's a scalar in float type

        Raises:
            ValueError: If the number of the input is not 1.
        """
        # if len(inputs) != 1:
        #     raise ValueError('Distribute accuracy needs 1 input (y_correct), but got {}'.\
        #         format(len(inputs)))
        y_correct_1 = self._convert_data(inputs[0])
        y_correct_2 = self._convert_data(inputs[1])
        self._correct_num_1 += y_correct_1
        self._correct_num_2 += y_correct_2
        self._total_num += self.batch_size * self.device_num

    def eval(self):
        """
        Computes the accuracy.

        Returns:
            Float, the computed result.

        Raises:
            RuntimeError: If the sample size is 0.
        """
        if self._total_num == 0:
            raise RuntimeError("Accuracy can't be calculated, because the number of samples is 0.")
        return (self._correct_num_1 * 100 / self.val_len, self._correct_num_2 * 100 / self.val_len)
