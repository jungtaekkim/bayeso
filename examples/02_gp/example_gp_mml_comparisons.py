# example_gp_mml_comparisons
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: June 24, 2021

import numpy as np
import os

from bayeso.gp import gp_kernel
from bayeso_benchmarks import Branin
from bayeso_benchmarks import Eggholder
from bayeso_benchmarks import SixHumpCamel


def main(obj_fun, num_train, str_cov, str_optimizer_method):
    fix_noise = False

    X_train = obj_fun.sample_uniform(num_train)
    Y_train = obj_fun.output(X_train)

    gp_kernel.get_optimized_kernel(X_train, Y_train, None, str_cov, str_optimizer_method=str_optimizer_method, fix_noise=fix_noise, debug=True, use_ard=False)


if __name__ == '__main__':
    str_cov = 'se'
    print('str_cov', str_cov)

    obj_fun = Branin()
    main(obj_fun, 20, str_cov, 'BFGS')
    main(obj_fun, 20, str_cov, 'Nelder-Mead')

    obj_fun = Eggholder()
    main(obj_fun, 20, str_cov, 'BFGS')
    main(obj_fun, 20, str_cov, 'Nelder-Mead')

    obj_fun = SixHumpCamel()
    main(obj_fun, 20, str_cov, 'BFGS')
    main(obj_fun, 20, str_cov, 'Nelder-Mead')
