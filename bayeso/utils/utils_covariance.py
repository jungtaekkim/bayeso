# utils_covariance
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: May 30, 2018

import numpy as np


def get_hyps(str_cov, num_dim):
    assert isinstance(str_cov, str)
    assert isinstance(num_dim, int)

    hyps = dict()
    hyps['noise'] = 0.1
    if str_cov == 'se':
        hyps['signal'] = 1.0
        hyps['lengthscales'] = np.zeros(num_dim) + 1.0
    else:
        raise ValueError('get_hyps: str_cov is not defined.')
    return hyps

def convert_hyps(str_cov, hyps):
    assert isinstance(str_cov, str)
    assert isinstance(hyps, dict)

    list_hyps = []
    list_hyps.append(hyps['noise'])
    if str_cov == 'se':
        list_hyps.append(hyps['signal'])
        for elem_lengthscale in hyps['lengthscales']:
            list_hyps.append(elem_lengthscale)
    else:
        raise ValueError('convert_hyps: str_cov is not defined.')
    return np.array(list_hyps)

def restore_hyps(str_cov, hyps):
    assert isinstance(str_cov, str)
    assert isinstance(hyps, np.ndarray)
    assert len(hyps.shape) == 1

    dict_hyps = dict()
    dict_hyps['noise'] = hyps[0]
    if str_cov == 'se':
        dict_hyps['signal'] = hyps[1]
        list_lengthscales = []
        for ind_elem in range(2, len(hyps)):
            list_lengthscales.append(hyps[ind_elem])
        dict_hyps['lengthscales'] = np.array(list_lengthscales)
    else:
        raise ValueError('restore_hyps: str_cov is not defined.')
    return dict_hyps
