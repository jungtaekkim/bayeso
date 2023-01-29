# example_gp_prior_cosine
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 12, 2020

import numpy as np
import os

from bayeso.gp import gp
from bayeso.utils import utils_plotting

PATH_SAVE = '../figures/gp/'


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

def main(fun_prior, str_prior):
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
    utils_plotting.plot_gp_via_distribution(X_train, Y_train, X_test, mu, sigma, Y_test, PATH_SAVE, 'optimized_prior_{}'.format(str_prior))


if __name__ == '__main__':
    if not os.path.isdir(PATH_SAVE):
        os.makedirs(PATH_SAVE)

    main(cosine, 'cosine')
    main(linear_down, 'linear_down')
    main(linear_up, 'linear_up')
