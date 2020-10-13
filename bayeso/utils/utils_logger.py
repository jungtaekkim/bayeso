#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: September 24, 2020
#
"""It is utilities for loggers."""

import logging
import numpy as np

from bayeso.utils import utils_common


@utils_common.validate_types
def get_logger(str_name: str) -> logging.Logger:
    """
    It returns a logger to record the messages generated in our package.

    :param str_name: a logger name.
    :type str_name: str.

    :returns: a logger.
    :rtype: logging.Logger

    :raises: AssertionError

    """

    assert isinstance(str_name, str)

    logger = logging.getLogger(str_name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(levelname)s-%(name)s-%(asctime)s] %(message)s',
        datefmt='%m/%d/%Y-%H:%M:%S')
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    return logger

@utils_common.validate_types
def get_str_array_1d(arr: np.ndarray) -> str:
    """
    It converts a one-dimensional array into string.

    :param arr: an array to be converted.
    :type arr: numpy.ndarray

    :returns: a string.
    :rtype: str.

    :raises: AssertionError

    """

    assert isinstance(arr, np.ndarray)
    assert len(arr.shape) == 1

    list_str = []

    for elem in arr:
        if isinstance(elem, float):
            elem_ = '{:.3f}'.format(elem)
        else:
            elem_ = '{}'.format(elem)

        list_str.append(elem_)

    str_arr = ', '.join(list_str)
    str_arr = '[' + str_arr + ']'
    return str_arr

@utils_common.validate_types
def get_str_array_2d(arr: np.ndarray) -> str:
    """
    It converts a two-dimensional array into string.

    :param arr: an array to be converted.
    :type arr: numpy.ndarray

    :returns: a string.
    :rtype: str.

    :raises: AssertionError

    """

    assert isinstance(arr, np.ndarray)
    assert len(arr.shape) == 2

    list_str = [get_str_array_1d(elem) for elem in arr]

    str_arr = ',\n'.join(list_str)
    str_arr = '[' + str_arr + ']'
    return str_arr

@utils_common.validate_types
def get_str_array_3d(arr: np.ndarray) -> str:
    """
    It converts a three-dimensional array into string.

    :param arr: an array to be converted.
    :type arr: numpy.ndarray

    :returns: a string.
    :rtype: str.

    :raises: AssertionError

    """

    assert isinstance(arr, np.ndarray)
    assert len(arr.shape) == 3

    list_str = [get_str_array_2d(elem) for elem in arr]

    str_arr = ',\n'.join(list_str)
    str_arr = '[' + str_arr + ']'
    return str_arr

@utils_common.validate_types
def get_str_array(arr: np.ndarray) -> str:
    """
    It converts an array into string. It can take one-dimensional,
    two-dimensional, and three-dimensional arrays.

    :param arr: an array to be converted.
    :type arr: numpy.ndarray

    :returns: a string.
    :rtype: str.

    :raises: AssertionError

    """

    assert isinstance(arr, np.ndarray)
    len_arr = len(arr.shape)

    if len_arr == 1:
        str_arr = get_str_array_1d(arr)
    elif len_arr == 2:
        str_arr = get_str_array_2d(arr)
    elif len_arr == 3:
        str_arr = get_str_array_3d(arr)
    else:
        raise NotImplementedError('invalid len_arr.')

    return str_arr

@utils_common.validate_types
def get_str_hyps(hyps: dict) -> str:
    """
    It converts a dictionary of hyperparameters into string.

    :param hyps: a hyperparameter dictionary to be converted.
    :type hyps: dict.

    :returns: a string.
    :rtype: str.

    :raises: AssertionError

    """

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

    str_hyps = ', '.join(list_str)
    str_hyps = '{' + str_hyps + '}'
    return str_hyps
