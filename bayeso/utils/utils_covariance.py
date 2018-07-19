# utils_covariance
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 18, 2018

import numpy as np

from bayeso import constants


def get_hyps(str_cov, num_dim, is_ard=True):
    assert isinstance(str_cov, str)
    assert isinstance(num_dim, int)
    assert isinstance(is_ard, bool)
    assert str_cov in constants.ALLOWED_GP_COV

    hyps = dict()
    hyps['noise'] = constants.GP_NOISE
    if str_cov == 'se' or str_cov == 'matern32' or str_cov == 'matern52':
        hyps['signal'] = 1.0
        if is_ard:
            hyps['lengthscales'] = np.ones(num_dim)
        else:
            hyps['lengthscales'] = 1.0
    else:
        raise NotImplementedError('get_hyps: allowed str_cov, but it is not implemented.')
    return hyps

def convert_hyps(str_cov, hyps, is_fixed_noise=False):
    assert isinstance(str_cov, str)
    assert isinstance(hyps, dict)
    assert isinstance(is_fixed_noise, bool)
    assert str_cov in constants.ALLOWED_GP_COV

    list_hyps = []
    if not is_fixed_noise:
        list_hyps.append(hyps['noise'])
    if str_cov == 'se' or str_cov == 'matern32' or str_cov == 'matern52':
        list_hyps.append(hyps['signal'])
        for elem_lengthscale in hyps['lengthscales']:
            list_hyps.append(elem_lengthscale)
    else:
        raise NotImplementedError('convert_hyps: allowed str_cov, but it is not implemented.')
    return np.array(list_hyps)

def restore_hyps(str_cov, hyps, is_fixed_noise=False, fixed_noise=constants.GP_NOISE):
    assert isinstance(str_cov, str)
    assert isinstance(hyps, np.ndarray)
    assert isinstance(is_fixed_noise, bool)
    assert isinstance(fixed_noise, float)
    assert len(hyps.shape) == 1
    assert str_cov in constants.ALLOWED_GP_COV

    dict_hyps = dict()
    if not is_fixed_noise:
        dict_hyps['noise'] = hyps[0]
        ind_start = 1
    else:
        dict_hyps['noise'] = fixed_noise
        ind_start = 0

    if str_cov == 'se' or str_cov == 'matern32' or str_cov == 'matern52':
        dict_hyps['signal'] = hyps[ind_start]
        list_lengthscales = []
        for ind_elem in range(ind_start+1, len(hyps)):
            list_lengthscales.append(hyps[ind_elem])
        dict_hyps['lengthscales'] = np.array(list_lengthscales)
    else:
        raise NotImplementedError('restore_hyps: allowed str_cov, but it is not implemented.')
    return dict_hyps

def validate_hyps_dict(dict_hyps, str_cov, int_dim):
    is_valid = True
    if 'noise' not in dict_hyps:
        is_valid = False
    else:
        if not isinstance(dict_hyps['noise'], float):
            is_valid = False
        else:
            if np.abs(dict_hyps['noise']) >= constants.BOUND_UPPER_GP_NOISE:
                dict_hyps['noise'] = constants.BOUND_UPPER_GP_NOISE
    if str_cov == 'se' or str_cov == 'matern32' or str_cov == 'matern52':
        if 'lengthscales' not in dict_hyps:
            is_valid = False
        else:
            if isinstance(dict_hyps['lengthscales'], np.ndarray) and dict_hyps['lengthscales'].shape[0] != int_dim:
                is_valid = False
            if not isinstance(dict_hyps['lengthscales'], np.ndarray) and not isinstance(dict_hyps['lengthscales'], float):
                is_valid = False
        if 'signal' not in dict_hyps:
            is_valid = False
        else:
            if not isinstance(dict_hyps['signal'], float):
                is_valid = False
    else:
        is_valid = False
    return dict_hyps, is_valid

def validate_hyps_arr(arr_hyps, str_cov, int_dim):
    raise NotImplementedError('validate_hyps_arr in utils_covariance.py')
