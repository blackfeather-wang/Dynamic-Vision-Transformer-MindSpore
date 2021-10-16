"""weight initilization"""

import math
import numpy as np
import torch
from mindspore.common import initializer as init
import mindspore as ms

try:
    from mindspore.common.initializer import _Initializer as MeInitializer
except:
    from mindspore.common.initializer import Initializer as MeInitializer

import mindspore.nn as nn
from mindspore import Tensor


def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    ================= ====================================================

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def _calculate_correct_fan(array, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(array)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_uniform_(arr, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `Tensor`
        a: the negative slope of the rectifier used after this layer (only
        used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = np.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    """
    fan = _calculate_correct_fan(arr, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return np.random.uniform(-bound, bound, arr.shape)


def kaiming_normal_(arr, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    normal distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `Tensor`
        a: the negative slope of the rectifier used after this layer (only
        used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = np.empty(3, 5)
        >>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    """
    fan = _calculate_correct_fan(arr, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return np.random.normal(0, std, arr.shape)


def _calculate_fan_in_and_fan_out(arr):
    dimensions = len(arr.shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for array with fewer than 2 dimensions")

    num_input_fmaps = arr.shape[1]
    num_output_fmaps = arr.shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        receptive_field_size = arr[0][0].size
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def xavier_uniform_(arr, gain=1.):
    # type: (Tensor, float) -> Tensor
    r"""Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = np.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(arr)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return np.random.uniform(-a, a, arr.shape)


class XavierUniform(MeInitializer):
    def __init__(self, gain=1.):
        super(XavierUniform, self).__init__()
        self.gain = gain

    def _initialize(self, arr):
        tmp = xavier_uniform_(arr, self.gain)
        init._assignment(arr, tmp)


class KaimingUniform(MeInitializer):
    def __init__(self, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        super(KaimingUniform, self).__init__()
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity

    def _initialize(self, arr):
        tmp = kaiming_uniform_(arr, self.a, self.mode, self.nonlinearity)
        init._assignment(arr, tmp)


class KaimingNormal(MeInitializer):
    def __init__(self, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        super(KaimingNormal, self).__init__()
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity

    def _initialize(self, arr):
        tmp = kaiming_normal_(arr, self.a, self.mode, self.nonlinearity)
        init._assignment(arr, tmp)

class RandomNormal(MeInitializer):
    def __init__(self, std=0.001):
        super(RandomNormal, self).__init__()
        self.std = std

    def _initialize(self, arr):
        std = self.std
        tmp = np.random.normal(0, std, arr.shape)
        init._assignment(arr, tmp)

def default_recurisive_init(custom_cell):
    np.random.seed(123)
    for name, cell in custom_cell.cells_and_names():
        if 'hm' in name or 'wh' in name or 'off' in name or 'kps' in name:
            if isinstance(cell, (nn.Conv2d)):
                cell.weight.set_data(init.initializer(RandomNormal(),
                                                         cell.weight.data.shape,
                                                         cell.weight.data.dtype).to_tensor())
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer('zeros', cell.bias.data.shape,
                                                            cell.bias.data.dtype).to_tensor())
                continue

        if isinstance(cell, (nn.Conv2d)):
            cell.weight.set_data(init.initializer(KaimingNormal(mode='fan_out'), #KaimingUniform(a=math.sqrt(5)),
                                                         cell.weight.data.shape,
                                                         cell.weight.data.dtype).to_tensor())
            if cell.bias is not None:
                cell.bias.set_data(init.initializer('zeros', cell.bias.data.shape,
                                                           cell.bias.data.dtype).to_tensor())
        elif isinstance(cell, nn.Dense):
            cell.weight.set_data(init.initializer(KaimingNormal(mode='fan_out'), #KaimingUniform(a=math.sqrt(5)),
                                                         cell.weight.data.shape,
                                                         cell.weight.data.dtype).to_tensor())
            if cell.bias is not None:
                cell.bias.set_data(init.initializer('zeros', cell.bias.data.shape,
                                                           cell.bias.data.dtype).to_tensor())
        elif isinstance(cell, nn.BatchNorm2d) or isinstance(cell, nn.BatchNorm1d):
            cell.gamma.set_data(init.initializer('ones', cell.gamma.data.shape).to_tensor())
            cell.beta.set_data(init.initializer('zeros', cell.beta.data.shape).to_tensor())
        elif isinstance(cell, nn.Conv2dTranspose):
            cell.weight.set_data(init.initializer(KaimingUniform(a=math.sqrt(5), mode='fan_out'), #KaimingUniform(a=math.sqrt(5)),
                                                         cell.weight.data.shape,
                                                         cell.weight.data.dtype).to_tensor())
            if cell.bias is not None:
                cell.bias.set_data(init.initializer('zeros', cell.bias.data.shape,
                                                           cell.bias.data.dtype).to_tensor())

class trunc_normal_(MeInitializer):
    def __init__(self, mean=0., std=.02, a=-2., b=2.):
        super(trunc_normal_, self).__init__()
        self.std = std
        self.mean = mean
        self.a = a
        self.b = b

    def _initialize(self, arr):
        std = self.std
        mean = self.mean
        a = self.a
        b = self.b

        tensor = np.random.normal(0, std, arr.shape)
        tensor = _numpy_trunc_normal_(tensor, mean, std, a, b)
        init._assignment(arr, tensor)


# class trunc_normal_(MeInitializer):
#     def __init__(self, mean=0., std=.02, a=-2., b=2.):
#         super(trunc_normal_, self).__init__()
#         self.std = std
#         self.mean = mean
#         self.a = a
#         self.b = b

#     def _initialize(self, arr):
#         std = self.std
#         mean = self.mean
#         a = self.a
#         b = self.b

#         tensor = torch.zeros(arr.shape)
#         tensor = _no_grad_trunc_normal_(tensor, mean, std, a, b)
#         tmp = tensor.numpy()
#         init._assignment(arr, tmp)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def _numpy_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor = np.random.uniform(2 * l - 1, 2 * u - 1, tensor.shape)
    #tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    from scipy import special
    tensor = special.erfinv(tensor)
    #tensor.erfinv_()

    # Transform to proper mean, std
    tensor = tensor * (std * math.sqrt(2.))
    tensor += mean

    # Clamp to ensure it's in the proper range
    tensor = np.clip(tensor, a, b)

    return tensor
