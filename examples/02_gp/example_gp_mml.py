# example_gp_mml
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: April 28, 2020

import numpy as np
import os

from bayeso import gp
from bayeso.utils import utils_plotting


PATH_SAVE = '../figures/gp/'

def main():
    X_train = np.array([
        [-3.0],
        [-2.0],
        [-1.0],
        [2.0],
        [1.2],
        [1.1],
    ])
    Y_train = np.cos(X_train) + 10.0
    num_test = 200
    X_test = np.linspace(-3, 3, num_test)
    X_test = X_test.reshape((num_test, 1))
    Y_test_truth = np.cos(X_test) + 10.0
    mu, sigma, Sigma = gp.predict_optimized(X_train, Y_train, X_test)
    utils_plotting.plot_gp(X_train, Y_train, X_test, mu, sigma, Y_test_truth, PATH_SAVE, 'optimized')


if __name__ == '__main__':
    if not os.path.isdir(PATH_SAVE):
        os.makedirs(PATH_SAVE)
    main()

