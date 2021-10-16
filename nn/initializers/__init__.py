"""Initialization routines."""

import math

import numpy as np
from mindspore import Tensor

def calc_fanin_and_fanout(shape):
    """Calculate fan_in and fan_out values for"""
    num_input_fmaps = shape[1]
    num_output_fmaps = shape[0]
    receptive_field_size = 1
    if len(shape) > 2:
        receptive_field_size = np.prod(shape[2:])
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out

def lecun_normal(shape):
    fan_in, _ = calc_fanin_and_fanout(shape)
    std = math.sqrt(1. / fan_in)
    values = np.random.normal(0., std, shape)
    return Tensor(values)
