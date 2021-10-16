"""Gradient clipping wrapper for optimizers."""

# pylint: disable = arguments-differ, too-many-instance-attributes, too-many-arguments

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore._checkparam import Validator as validator
from mindspore.nn import AdamWeightDecay
from mindspore.ops import functional as F
from mindspore.ops import operations as P

import numpy as np

from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
#from mindspore.ops import operations as P
from mindspore.ops import composite as C
#from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
#from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from mindspore.nn import AdamWeightDecay
from mindspore.nn.optim import Optimizer

#from ..test_framework.utils.debug_util import PrintShapeTypeCell


def _check_param_value(norm, prim_name):
    validator.check_value_type("norm", norm, [float], prim_name)

class AdamWGCNorm(AdamWeightDecay):
    """
    Implements the gradient clipping by norm for a AdamWeightDecay optimizer.
    """

    def __init__(self, parameters, learning_rate, norm=1., gradient_centralization=False, stat=0):
        super().__init__(parameters, learning_rate)
        _check_param_value(norm, self.cls_name)
        self.norm = norm
        self.clip_by_norm = nn.ClipByNorm()
        self.cast = P.Cast()
        self.dtype = P.DType()
        self.mean = P.ReduceMean(keep_dims=True)
        self.sum = P.ReduceSum()
        self.gradient_centralization = gradient_centralization
        self.abs = P.Abs()
        self.max = P.ReduceMax()
        self.mul = P.Mul()
        self.sqrt = P.Sqrt()

        #self.print = PrintShapeTypeCell()

        self.stat = stat
        if stat == 1:
            self.scalar_summary = ops.ScalarSummary()
        self.weight_names = [param.name for param in self.parameters]

    def construct(self, gradients):
        if self.stat == 1:
            self.scalar_summary("lr", self.get_lr())

            for i, param in enumerate(self.parameters):
                amax_w = self.max(self.abs(param))
                self.scalar_summary(self.weight_names[i] + ".amax", amax_w)
                norm_w = self.sqrt(self.sum(self.mul(param, param)))
                self.scalar_summary(self.weight_names[i] + ".norm", norm_w)

        new_grads = ()
        for i, grad in enumerate(gradients):
            #self.print(self.weight_names[i], grad)
            dtype = self.dtype(grad)
            if self.gradient_centralization:
                if len(grad.shape) == 2:
                    grad = grad - self.mean(grad, (1))
            new_grad = self.clip_by_norm(grad, self.cast(F.tuple_to_array((self.norm,)), dtype))

            if self.stat == 1:
                amax_g = self.max(self.abs(grad))
                self.scalar_summary(self.weight_names[i] + ".grad.amax", amax_g)
                norm_g = self.sqrt(self.sum(self.mul(grad, grad)))
                self.scalar_summary(self.weight_names[i] + ".grad.norm", norm_g)

                amax_g_clip = self.max(self.abs(new_grad))
                self.scalar_summary(self.weight_names[i] + ".grad.amax_clip", amax_g_clip)
                norm_g_clip = self.sqrt(self.sum(self.mul(new_grad, new_grad)))
                self.scalar_summary(self.weight_names[i] + ".grad.norm_clip", norm_g_clip)

            new_grads = new_grads + (new_grad,)
        return super().construct(new_grads)



def _check_param_value_4(beta1, beta2, eps, prim_name):
    """Check the type of inputs."""
    validator.check_value_type("beta1", beta1, [float], prim_name)
    validator.check_value_type("beta2", beta2, [float], prim_name)
    validator.check_value_type("eps", eps, [float], prim_name)
    validator.check_float_range(beta1, 0.0, 1.0, Rel.INC_NEITHER, "beta1", prim_name)
    validator.check_float_range(beta2, 0.0, 1.0, Rel.INC_NEITHER, "beta2", prim_name)
    validator.check_positive_float(eps, "eps", prim_name)

_grad_scale = C.MultitypeFuncGraph("grad_scale")
_indices_deduplicate = C.MultitypeFuncGraph("indices_deduplicate")
op_mul = P.Mul()
map_ = C.Map()

@_grad_scale.register("Number", "Tensor")
def tensor_grad_scale(scale, grad):
    """Get grad with scale."""
    if scale == 1.0:
        return grad
    return op_mul(grad, F.cast(scale, F.dtype(grad)))

@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale_with_tensor(scale, grad):
    """Get grad with scale."""
    return op_mul(grad, F.cast(scale, F.dtype(grad)))

# @_grad_scale.register("Tensor", "RowTensor")
# def tensor_grad_scale_with_sparse(scale, grad):
#     """Get grad with scale."""
#     return RowTensor(grad.indices, grad.values * F.cast(scale, F.dtype(grad.values)), grad.dense_shape)

def scale_grad(gradients, reciprocal_scale):
    gradients = map_(F.partial(_grad_scale, reciprocal_scale), gradients)
    return gradients

_adam_opt = C.MultitypeFuncGraph("adam_opt")
_scaler_one = Tensor(1, mstype.int32)
_scaler_ten = Tensor(10, mstype.float32)


@_adam_opt.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Number", "Tensor", "Tensor", "Tensor",
                    "Tensor", "Bool", "Bool")
def _update_run_op(beta1_power, beta2_power, beta1, beta2, eps, lr, weight_decay, param, m, v, gradient, decay_flag, optim_filter):
    """
    Update parameters.

    Args:
        beta1 (Tensor): The exponential decay rate for the 1st moment estimations. Should be in range (0.0, 1.0).
        beta2 (Tensor): The exponential decay rate for the 2nd moment estimations. Should be in range (0.0, 1.0).
        eps (Tensor): Term added to the denominator to improve numerical stability. Should be greater than 0.
        lr (Tensor): Learning rate.
        weight_decay (Number): Weight decay. Should be equal to or greater than 0.
        param (Tensor): Parameters.
        m (Tensor): m value of parameters.
        v (Tensor): v value of parameters.
        gradient (Tensor): Gradient of parameters.
        decay_flag (bool): Applies weight decay or not.
        optim_filter (bool): Applies parameter update or not.

    Returns:
        Tensor, the new value of v after updating.
    """
    if optim_filter:
        op_mul = P.Mul()
        op_square = P.Square()
        op_sqrt = P.Sqrt()
        op_cast = P.Cast()
        op_reshape = P.Reshape()
        op_shape = P.Shape()

        param_fp32 = op_cast(param, mstype.float32)
        m_fp32 = op_cast(m, mstype.float32)
        v_fp32 = op_cast(v, mstype.float32)
        gradient_fp32 = op_cast(gradient, mstype.float32)

        next_m = op_mul(beta1, m_fp32) + op_mul(op_cast(F.tuple_to_array((1.0,)), mstype.float32)
                                                - beta1, gradient_fp32)

        next_v = op_mul(beta2, v_fp32) + op_mul(op_cast(F.tuple_to_array((1.0,)), mstype.float32)
                                                - beta2, op_square(gradient_fp32))

        regulate_m = next_m / (_scaler_one - beta1_power)
        regulate_v = next_v / (_scaler_one - beta2_power)
        #update = next_m / (eps + op_sqrt(next_v))
        update = regulate_m / (eps + op_sqrt(regulate_v))
        if decay_flag:
            update = op_mul(weight_decay, param_fp32) + update

        update_with_lr = op_mul(lr, update)
        next_param = param_fp32 - op_reshape(update_with_lr, op_shape(param_fp32))

        next_param = F.depend(next_param, F.assign(param, op_cast(next_param, F.dtype(param))))
        next_param = F.depend(next_param, F.assign(m, op_cast(next_m, F.dtype(m))))
        next_param = F.depend(next_param, F.assign(v, op_cast(next_v, F.dtype(v))))

        return op_cast(next_param, F.dtype(param))
    return gradient

class AdamW(Optimizer):
    """
    Implements the gradient clipping by norm for a AdamWeightDecay optimizer.
    """

    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, loss_scale=1.0):
        super(AdamW, self).__init__(learning_rate, params, weight_decay)
        _check_param_value_4(beta1, beta2, eps, self.cls_name)
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.moments1 = self.parameters.clone(prefix="adam_m", init='zeros')
        self.moments2 = self.parameters.clone(prefix="adam_v", init='zeros')
        self.hyper_map = C.HyperMap()
        self.beta1_power = Parameter(initializer(1, [1], mstype.float32), name="beta1_power")
        self.beta2_power = Parameter(initializer(1, [1], mstype.float32), name="beta2_power")

        self.reciprocal_scale = Tensor(1.0 / loss_scale, mstype.float32)

    def construct(self, gradients):
        lr = self.get_lr()
        #params = self.parameters
        #moment1 = self.moment1
        #moment2 = self.moment2
        #gradients = self.decay_weight(gradients)
        #gradients = self.gradients_centralization(gradients)
        gradients = scale_grad(gradients, self.reciprocal_scale)
        #gradients = self._grad_sparse_indices_deduplicate(gradients)
        #lr = self.get_lr()

        beta1_power = self.beta1_power * self.beta1
        self.beta1_power = beta1_power
        beta2_power = self.beta2_power * self.beta2
        self.beta2_power = beta2_power

        if self.is_group:
            if self.is_group_lr:
                optim_result = self.hyper_map(F.partial(_adam_opt, beta1_power, beta2_power, self.beta1, self.beta2, self.eps),
                                              lr, self.weight_decay, self.parameters, self.moments1, self.moments2,
                                              gradients, self.decay_flags, self.optim_filter)
            else:
                optim_result = self.hyper_map(F.partial(_adam_opt, beta1_power, beta2_power, self.beta1, self.beta2, self.eps, lr),
                                              self.weight_decay, self.parameters, self.moments1, self.moments2,
                                              gradients, self.decay_flags, self.optim_filter)
        else:
            optim_result = self.hyper_map(F.partial(_adam_opt, beta1_power, beta2_power, self.beta1, self.beta2, self.eps, lr, self.weight_decay),
                                          self.parameters, self.moments1, self.moments2,
                                          gradients, self.decay_flags, self.optim_filter)
        if self.use_parallel:
            self.broadcast_params(optim_result)
        return optim_result

