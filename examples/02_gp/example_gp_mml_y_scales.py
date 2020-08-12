# example_gp_mml_large_y
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 07, 2020

import numpy as np
import os

from bayeso.gp import gp
from bayeso.utils import utils_plotting


PATH_SAVE = '../figures/gp/'

def main(scale, str_postfix):
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
    Y_test_truth = np.cos(X_test) * scale
    mu, sigma, Sigma = gp.predict_optimized(X_train, Y_train, X_test, is_fixed_noise=False)
    utils_plotting.plot_gp(X_train, Y_train, X_test, mu, sigma, Y_test_truth, PATH_SAVE, 'test_optimized_{}_y'.format(str_postfix))


if __name__ == '__main__':
    if not os.path.isdir(PATH_SAVE):
        os.makedirs(PATH_SAVE)
    main(0.01, 'small')
    main(100000.0, 'large')
