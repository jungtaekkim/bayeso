#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 16, 2023
#

import numpy as np
import os

from bayeso.gp import gp
from bayeso.utils import utils_plotting


def main(scale, path_save, str_postfix):
    X_train = np.array([
        [-3.0],
        [-2.0],
        [-1.0],
        [2.0],
        [1.2],
        [1.1],
    ])
    Y_train = np.cos(X_train) * scale
    num_test = 200
    X_test = np.linspace(-3, 3, num_test)
    X_test = X_test.reshape((num_test, 1))
    Y_test = np.cos(X_test) * scale

    mu, sigma, Sigma = gp.predict_with_optimized_hyps(X_train, Y_train, X_test, fix_noise=False, debug=True)
    utils_plotting.plot_gp_via_distribution(X_train, Y_train, X_test, mu, sigma, Y_test, path_save, 'test_optimized_{}_y'.format(str_postfix))


if __name__ == '__main__':
    path_save = None

    if path_save is not None and not os.path.isdir(path_save):
        os.makedirs(path_save)
    main(0.01, path_save, 'small')
    main(100000.0, path_save, 'large')
