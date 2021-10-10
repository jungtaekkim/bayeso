# example_benchmarks_bohachevsky_bo_ei
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 8, 2021

import numpy as np
import os

from bayeso import bo
from bayeso_benchmarks import Bohachevsky
import bayeso.wrappers as wrappers
from bayeso.utils import utils_bo
from bayeso.utils import utils_plotting


STR_FUN_TARGET = 'bohachevksy'
PATH_SAVE = '../figures/benchmarks/'

obj_fun = Bohachevsky()


def fun_target(X):
    return obj_fun.output(X)

def main():
    num_bo = 5
    num_iter = 10
    num_init = 5

    bounds = obj_fun.get_bounds()

    model_bo = wrappers.BayesianOptimization(
        bounds,
        fun_target,
        num_iter,
        str_surrogate='gp',
        str_cov='matern52',
        str_acq='ei',
        str_initial_method_bo='sobol',
        str_sampling_method_ao='sobol',
        str_optimizer_method_gp='BFGS',
        str_optimizer_method_bo='L-BFGS-B',
        num_samples_ao=100,
        debug=True,
    )

    list_Y = []
    list_time = []
    for ind_bo in range(0, num_bo):
        print('BO Round', ind_bo + 1)
        X, Y, time_all, _, _ = model_bo.optimize(num_init, seed=42 * ind_bo)
        list_Y.append(Y)
        list_time.append(time_all)

        bx_best, y_best = utils_bo.get_best_acquisition_by_history(X, Y)
        print(bx_best, y_best)

    Ys = np.array(list_Y)
    Ys = np.expand_dims(np.squeeze(Ys), axis=0)
    times = np.array(list_time)
    times = np.expand_dims(times, axis=0)

    utils_plotting.plot_minimum_vs_iter(Ys, [STR_FUN_TARGET], num_init, True, path_save=PATH_SAVE, str_postfix=STR_FUN_TARGET)
    utils_plotting.plot_minimum_vs_time(times, Ys, [STR_FUN_TARGET], num_init, True, path_save=PATH_SAVE, str_postfix=STR_FUN_TARGET)


if __name__ == '__main__':
    if not os.path.isdir(PATH_SAVE):
        os.makedirs(PATH_SAVE)
    main()
