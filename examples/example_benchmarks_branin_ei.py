# example_benchmarks_branin_ei
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 04, 2018

import numpy as np
import sklearn
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model

from bayeso import gp
from bayeso import bo
from bayeso import acquisition
from bayeso import benchmarks
from bayeso.utils import utils_bo
from bayeso.utils import utils_plotting

INFO_TARGET = benchmarks.INFO_BRANIN
STR_FUN_TARGET = 'branin'


def fun_target(X):
    return benchmarks.branin(X)

def main():
    int_bo = 3
    int_iter = 40
    int_init = 3

    model_bo = bo.BO(INFO_TARGET.get('bounds'), debug=True)
    list_Y = []
    for _ in range(0, int_bo):
        X_final, Y_final = utils_bo.optimize_many_with_random_init(model_bo, fun_target, int_init, int_iter, str_initial_method_bo='uniform', str_initial_method_ao='uniform', int_samples_ao=100)
        print(X_final)
        print(Y_final)
        list_Y.append(Y_final)
    arr_Y = np.array(list_Y)
    arr_Y = np.expand_dims(np.squeeze(arr_Y), axis=0)
    utils_plotting.plot_minimum(arr_Y, [STR_FUN_TARGET], int_init, True, path_save='../results/benchmarks/', str_postfix=STR_FUN_TARGET)

if __name__ == '__main__':
    main()

