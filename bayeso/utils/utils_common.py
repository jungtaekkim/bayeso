# utils_common
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: June 24, 2018

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
