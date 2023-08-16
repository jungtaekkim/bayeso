#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 16, 2023
#

import numpy as np
import os

from bayeso.gp import gp
from bayeso.utils import utils_plotting


def main(path_save):
    num_train = 200
    num_test = 1000
    X_train = np.random.randn(num_train, 1) * 5.0
    Y_train = np.cos(X_train) + 10.0
    X_test = np.linspace(-10, 10, num_test)
    X_test = X_test.reshape((num_test, 1))
    Y_test = np.cos(X_test) + 10.0

    mu, sigma, Sigma = gp.predict_with_optimized_hyps(X_train, Y_train, X_test, debug=True)
    utils_plotting.plot_gp_via_distribution(X_train, Y_train, X_test, mu, sigma,
        Y_test=Y_test, path_save=path_save, str_postfix='test_optimized_many_points')


if __name__ == '__main__':
    path_save = None

    if path_save is not None and not os.path.isdir(path_save):
        os.makedirs(path_save)
    main(path_save)
