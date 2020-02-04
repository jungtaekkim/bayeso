# utils_common
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 04, 2020

import numpy as np


def get_minimum(data_all, int_init):
    """
    It returns accumulated minima at each iteration, their arithmetic means over rounds, and their standard deviations over rounds, which is widely used in Bayesian optimization community.

    :param data_all: historical function values. Shape: (r, t) where r is the number of Bayesian optimization rounds and t is the number of iterations including initial points for each round. For example, if we run 50 iterations with 5 initial examples and repeat this procedure 3 times, r would be 3 and t would be 55 (= 50 + 5).
    :type data_all: numpy.ndarray
    :param int_init: the number of initial points.
    :type int_init: int.

    :returns: tuple of accumulated minima, their arithmetic means over rounds, and their standard deviations over rounds. Shape: ((r, t - `int_init` + 1), (t - `int_init` + 1, ), (t - `int_init` + 1, )).
    :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray)

    :raises: AssertionError

    """

    assert isinstance(data_all, np.ndarray)
    assert isinstance(int_init, int)
    assert len(data_all.shape) == 2
    assert data_all.shape[1] > int_init

    list_minimum = []
    for cur_data in data_all:
        cur_minimum = np.inf
        cur_list = []
        for cur_elem in cur_data[:int_init]:
            if cur_minimum > cur_elem:
                cur_minimum = cur_elem
        cur_list.append(cur_minimum)
        for cur_elem in cur_data[int_init:]:
            if cur_minimum > cur_elem:
                cur_minimum = cur_elem
            cur_list.append(cur_minimum)
        list_minimum.append(cur_list)
    arr_minimum = np.array(list_minimum)
    mean_minimum = np.mean(arr_minimum, axis=0)
    std_minimum = np.std(arr_minimum, axis=0)
    return arr_minimum, mean_minimum, std_minimum

def get_time(arr_time, int_init, is_initial):
    """
    It returns the means of accumulated execution times over rounds.

    :param arr_time: execution times for all Bayesian optimization rounds. Shape: (r, t) where r is the number of Bayesian optimization rounds and t is the number of iterations (including initial points if `is_initial` is True, or excluding them if `is_initial` is False) for each round.

    :type arr_time: numpy.ndarray
    :param int_init: the number of initial points. If `is_initial` is False, it is ignored even if it is provided.
    :type int_init: int.
    :param is_initial: flag for describing whether execution times to observe initial examples have been included or not.
    :type is_initial: bool.

    :returns: arithmetic means of accumulated execution times over rounds. Shape: (t - `int_init`, ) if `is_initial` is True. (t, ), otherwise.
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(arr_time, np.ndarray)
    assert isinstance(int_init, int)
    assert isinstance(is_initial, bool)
    assert len(arr_time.shape) == 2
    if is_initial:
        assert arr_time.shape[1] > int_init

    list_time = []
    for elem_time in arr_time:
        cur_list = np.array([0.0])
        cur_time = 0.0

        if is_initial:
            elem_time = elem_time[int_init:]
        cur_list = np.concatenate((cur_list, np.cumsum(elem_time)))
        list_time.append(cur_list)
    list_time = np.array(list_time)
    return np.mean(list_time, axis=0)
