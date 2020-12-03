# example_gp_mml_many_points
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 07, 2020

import numpy as np
import os

from bayeso.gp import gp
from bayeso.utils import utils_plotting


PATH_SAVE = '../figures/gp/'

def main():
    num_train = 200
    num_test = 1000
    X_train = np.random.randn(num_train, 1) * 5.0
    Y_train = np.cos(X_train) + 10.0
    X_test = np.linspace(-10, 10, num_test)
    X_test = X_test.reshape((num_test, 1))
    Y_test = np.cos(X_test) + 10.0

    mu, sigma, Sigma = gp.predict_with_optimized_hyps(X_train, Y_train, X_test, debug=True)
    utils_plotting.plot_gp_via_distribution(X_train, Y_train, X_test, mu, sigma, Y_test,'test_optimized_many_points')


if __name__ == '__main__':
    if not os.path.isdir(PATH_SAVE):
        os.makedirs(PATH_SAVE)
    main()
