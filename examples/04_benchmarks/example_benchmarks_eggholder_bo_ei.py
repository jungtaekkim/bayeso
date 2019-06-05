# example_benchmarks_eggholder_ei
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 12, 2018

import numpy as np
import os

from bayeso import bo
from bayeso import benchmarks
from bayeso.utils import utils_bo
from bayeso.utils import utils_plotting
from bayeso.utils import utils_benchmarks

INFO_TARGET = benchmarks.INFO_EGGHOLDER
STR_FUN_TARGET = 'eggholder'
PATH_SAVE = '../figures/benchmarks/'


def fun_target(X):
    return benchmarks.eggholder(X)

def main():
    int_bo = 5
    int_iter = 100
    int_init = 3
    
    int_dim = 2

    bounds = utils_benchmarks.get_bounds(INFO_TARGET, int_dim)
    model_bo = bo.BO(bounds, debug=True)
    list_Y = []
    list_time = []
    for ind_bo in range(0, int_bo):
        print('BO Iteration', ind_bo)
        X_final, Y_final, time_final = utils_bo.optimize_many_with_random_init(model_bo, fun_target, int_init, int_iter, str_initial_method_bo='uniform', str_initial_method_ao='uniform', int_samples_ao=100)
        print(X_final)
        print(Y_final)
        print(time_final)
        list_Y.append(Y_final)
        list_time.append(time_final)

    arr_Y = np.array(list_Y)
    arr_Y = np.expand_dims(np.squeeze(arr_Y), axis=0)
    arr_time = np.array(list_time)
    arr_time = np.expand_dims(arr_time, axis=0)
    utils_plotting.plot_minimum(arr_Y, [STR_FUN_TARGET], int_init, True, path_save=PATH_SAVE, str_postfix=STR_FUN_TARGET)
    utils_plotting.plot_minimum_time(arr_time, arr_Y, [STR_FUN_TARGET], int_init, True, path_save=PATH_SAVE, str_postfix=STR_FUN_TARGET)

if __name__ == '__main__':
    if not os.path.isdir(PATH_SAVE):
        os.makedirs(PATH_SAVE)
    main()

