# utils_common
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: June 01, 2018

import numpy as np


def get_minimum(all_data, num_init):
    assert isinstance(all_data, np.ndarray)
    assert len(all_data.shape) == 2
    assert (all_data.dtype == np.float64) or (all_data.dtype == float32)
    assert isinstance(num_init, int)
    assert all_data.shape[1] > num_init

    list_minimum = []
    for cur_data in all_data:
        cur_minimum = np.inf
        cur_list = []
        for cur_elem in cur_data[:num_init]:
            if cur_minimum > cur_elem:
                cur_minimum = cur_elem
        cur_list.append(cur_minimum)
        for cur_elem in cur_data[num_init:]:
            if cur_minimum > cur_elem:
                cur_minimum = cur_elem
            cur_list.append(cur_minimum)
        list_minimum.append(cur_list)
    list_minimum = np.array(list_minimum)
    mean_minimum = np.mean(list_minimum, axis=0)
    std_minimum = np.std(list_minimum, axis=0)
    return list_minimum, mean_minimum, std_minimum
