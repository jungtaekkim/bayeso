# utils_common
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 06, 2018

import numpy as np


def get_minimum(all_data, int_init):
    assert isinstance(all_data, np.ndarray)
    assert isinstance(int_init, int)
    assert len(all_data.shape) == 2
    assert all_data.shape[1] > int_init

    list_minimum = []
    for cur_data in all_data:
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
