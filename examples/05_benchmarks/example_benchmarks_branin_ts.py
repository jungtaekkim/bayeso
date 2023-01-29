# example_benchmarks_branin_ts
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: December 19, 2021

import numpy as np
import os

from bayeso import thompson_sampling as ts
from bayeso_benchmarks import Branin
from bayeso.utils import utils_bo
from bayeso.utils import utils_plotting


STR_FUN_TARGET = 'branin'
PATH_SAVE = '../figures/benchmarks/'

obj_fun = Branin()


def fun_target(X):
    return obj_fun.output(X)

def main():
    num_bo = 5
    num_init = 1
    num_iter = 50

    debug = True

    bounds = obj_fun.get_bounds()

    list_Y = []
    for ind_bo in range(0, num_bo):
        print('BO Round', ind_bo + 1)
        X, Y = ts.thompson_sampling_gp(bounds, fun_target, num_init, num_iter, debug=debug)

        print(X)
        print(Y)

        list_Y.append(Y)

        bx_best, y_best = utils_bo.get_best_acquisition_by_history(X, Y)
        print(bx_best, y_best)

    arr_Y = np.array(list_Y)
    arr_Y = np.expand_dims(np.squeeze(arr_Y), axis=0)
    utils_plotting.plot_minimum_vs_iter(arr_Y, [STR_FUN_TARGET], num_init, True, path_save=PATH_SAVE, str_postfix=STR_FUN_TARGET)


if __name__ == '__main__':
    if not os.path.isdir(PATH_SAVE):
        os.makedirs(PATH_SAVE)
    main()
