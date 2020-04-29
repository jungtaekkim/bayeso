import logging
import numpy as np


def get_logger(str_name):
    assert isinstance(str_name, str)

    logger = logging.getLogger(str_name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(levelname)s-%(name)s-%(asctime)s] %(message)s', datefmt='%m/%d/%Y-%H:%M:%S')
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    return logger

def get_str_array_1d(arr):
    assert isinstance(arr, np.ndarray)
    assert len(arr.shape) == 1

    list_str = []

    for elem in arr:
        if isinstance(elem, float):
            elem_ = '{:.3f}'.format(elem)
        else:
            elem_ = '{}'.format(elem)

        list_str.append(elem_)

    str_ = ', '.join(list_str)
    str_ = '[' + str_ + ']'
    return str_

def get_str_array_2d(arr):
    assert isinstance(arr, np.ndarray)
    assert len(arr.shape) == 2

    list_str = [get_str_array_1d(elem) for elem in arr]

    str_ = ',\n'.join(list_str)
    str_ = '[' + str_ + ']'
    return str_

def get_str_array_3d(arr):
    assert isinstance(arr, np.ndarray)
    assert len(arr.shape) == 3

    list_str = [get_str_array_2d(elem) for elem in arr]

    str_ = ',\n'.join(list_str)
    str_ = '[' + str_ + ']'
    return str_

def get_str_array(arr):
    assert isinstance(arr, np.ndarray)
    len_arr = len(arr.shape)

    if len_arr == 1:
        str_ = get_str_array_1d(arr)
    elif len_arr == 2:
        str_ = get_str_array_2d(arr)
    elif len_arr == 3:
        str_ = get_str_array_3d(arr)
    else:
        raise NotImplementedError('invalid len_arr.')

    return str_

def get_str_hyps(hyps):
    assert isinstance(hyps, dict)

    list_str = []

    for key, val in hyps.items():
        if isinstance(val, np.ndarray):
            str_val = get_str_array(val)
        elif isinstance(val, float):
            str_val = '{:.3f}'.format(val)
        else:
            str_val = '{}'.format(val)

        list_str.append("'{}'".format(key) + ': ' + str_val)

    str_ = ', '.join(list_str)
    str_ = '{' + str_ + '}'
    return str_
