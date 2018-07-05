# example_gp_mml_many_points
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 05, 2018

import numpy as np

from bayeso import gp
from bayeso.utils import utils_plotting


def main():
    num_train = 200
    num_test = 1000
    X_train = np.random.randn(num_train, 1) * 5.0
    Y_train = np.cos(X_train) + 10.0
    X_test = np.linspace(-10, 10, num_test)
    X_test = X_test.reshape((num_test, 1))
    Y_test_truth = np.cos(X_test) + 10.0
    mu, sigma = gp.predict_optimized(X_train, Y_train, X_test, debug=True)
    utils_plotting.plot_gp(X_train, Y_train, X_test, mu, sigma, Y_test_truth, '../results/gp/', 'test_optimized_many_points')


if __name__ == '__main__':
    main()

