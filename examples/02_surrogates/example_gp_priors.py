#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 16, 2023
#

import numpy as np
import os

from bayeso.gp import gp
from bayeso.utils import utils_plotting


def cosine(X):
    return np.cos(X)

def linear_down(X):
    list_up = []
    for elem_X in X:
        list_up.append([-0.5 * np.sum(elem_X)])
    return np.array(list_up)

def linear_up(X):
    list_up = []
    for elem_X in X:
        list_up.append([0.5 * np.sum(elem_X)])
    return np.array(list_up)

def main(fun_prior, path_save, str_prior):
    X_train = np.array([
        [-3.0],
        [-2.0],
        [-1.0],
    ])
    Y_train = np.cos(X_train) + 2.0
    num_test = 200
    X_test = np.linspace(-3, 6, num_test)
    X_test = X_test.reshape((num_test, 1))
    Y_test = np.cos(X_test) + 2.0

    mu, sigma, Sigma = gp.predict_with_optimized_hyps(X_train, Y_train, X_test, prior_mu=fun_prior)
    utils_plotting.plot_gp_via_distribution(X_train, Y_train, X_test, mu, sigma, Y_test, path_save, 'optimized_prior_{}'.format(str_prior))


if __name__ == '__main__':
    path_save = None

    if path_save is not None and not os.path.isdir(path_save):
        os.makedirs(path_save)
    main(cosine, path_save, 'cosine')
    main(linear_down, path_save, 'linear_down')
    main(linear_up, path_save, 'linear_up')
