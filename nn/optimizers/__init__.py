"""Optimizers and some related functions."""

from .adamw_gcnorm import AdamWGCNorm

def gamma_beta_bias_wd_filter(params, weight_decay):
    """Turn off weight decay for BN beta/gamma and biases."""
    decayed_params = []
    no_decayed_params = []
    for param in params:
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': params}]
    return group_params

def beta_bias_wd_filter(params, weight_decay):
    """Turn off weight decay for BN beta/gamma and biases."""
    decayed_params = []
    no_decayed_params = []
    for param in params:
        if 'beta' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': params}]
    return group_params
