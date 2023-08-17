#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 17, 2023
#

import numpy as np

from bayeso import bo
from bayeso import acquisition
from bayeso.gp import gp
from bayeso.utils import utils_common
from bayeso.utils import utils_plotting
from bayeso_benchmarks.two_dim_branin import Branin


STR_FUN_TARGET = 'branin'

obj_fun = Branin()


def fun_target(X):
    return obj_fun.output(X)

def main():
    num_points = 100
    fix_noise = False
    bounds = obj_fun.get_bounds()

    model_bo = bo.BO(bounds, debug=True)
    X_init = model_bo.get_initials('uniform', num_points)
    X_test = utils_common.get_grids(bounds, 50)
    mu, sigma, Sigma = gp.predict_with_optimized_hyps(X_init, fun_target(X_init), X_test, str_optimizer_method='Nelder-Mead', fix_noise=fix_noise, debug=True)


if __name__ == '__main__':
    main()
