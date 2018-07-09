# utils_benchmarks
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 09, 2018

import numpy as np

from bayeso import benchmarks
from bayeso import constants


def validate_info(dict_info):
    assert isinstance(dict_info, dict)

    is_valid = True
    for elem_key in constants.KEYS_INFO_BENCHMARK:
        if not elem_key in dict_info.keys():
            is_valid = False

    print(is_valid)
    if is_valid:
        dim_fun = dict_info['dim_fun']
        bounds = dict_info['bounds']
        global_minimum_X = dict_info['global_minimum_X']
        global_minimum_y = dict_info['global_minimum_y']

        is_valid = (isinstance(dim_fun, int) or dim_fun == np.inf) &\
            isinstance(bounds, np.ndarray) &\
            isinstance(global_minimum_X, np.ndarray) &\
            isinstance(global_minimum_y, float)

    print(is_valid)
    if is_valid:
        bounds = dict_info['bounds']
        global_minimum_X = dict_info['global_minimum_X']

        is_valid = len(bounds.shape) == 2 &\
            len(global_minimum_X.shape) == 2

    print(is_valid)
    if is_valid:
        dim_fun = dict_info['dim_fun']
        bounds = dict_info['bounds']
        global_minimum_X = dict_info['global_minimum_X']

        # TODO: if dim_fun is counterfeited from np.inf to 1, then it would return True as is_valid, because dim_fun == bounds.shape[0] == global_minimum_X.shape[1] == 1.
        if dim_fun < np.inf:
            is_valid = dim_fun == bounds.shape[0] == global_minimum_X.shape[1]
        else:
            is_valid = bounds.shape[0] == global_minimum_X.shape[1] == 1

    print(is_valid)
    if is_valid:
        bounds = dict_info['bounds']
        is_valid = bounds.shape[1] == 2
    return is_valid

def get_bounds(dict_info, int_dim):
    assert isinstance(dict_info, dict)
    assert isinstance(int_dim, int)
    assert int_dim > 0 and int_dim < np.inf

    if dict_info.get('dim_fun') == int_dim:
        bounds = dict_info.get('bounds')
    elif dict_info.get('dim_fun') == np.inf:
        bounds = np.repeat(dict_info.get('bounds'), int_dim, axis=0)
    else:
        raise ValueError('get_bounds: invalid int_dim')
    return bounds

def get_covariate(dict_info, cur_covariate, int_dim):
    assert isinstance(dict_info, dict)
    assert isinstance(cur_covariate, np.ndarray)
    assert isinstance(int_dim, int)
    assert len(cur_covariate.shape) == 1
    assert int_dim > 0 and int_dim < np.inf

    if dict_info.get('dim_fun') == cur_covariate.shape[0] == int_dim:
        covariate = cur_covariate
    elif dict_info.get('dim_fun') == np.inf and cur_covariate.shape[0] == 1:
        covariate = np.repeat(cur_covariate, int_dim)
    else:
        raise ValueError('get_covariate: invalid cur_covariate and int_dim')
    return covariate
