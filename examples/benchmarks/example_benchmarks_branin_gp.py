# example_gp_benchmarks_branin
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 12, 2018

import numpy as np

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
    num_points = 100
    is_fixed_noise = False
    model_bo = bo.BO(INFO_TARGET.get('bounds'), debug=True)
    X_init = model_bo.get_initial('uniform', fun_objective=fun_target, int_samples=num_points)
    X_test = utils_bo.get_grid(INFO_TARGET.get('bounds'), 50)
    mu, sigma = gp.predict_optimized(X_init, np.expand_dims(fun_target(X_init), axis=1), X_test, is_fixed_noise=is_fixed_noise, debug=True)

if __name__ == '__main__':
    main()

