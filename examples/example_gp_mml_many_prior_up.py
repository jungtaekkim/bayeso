import numpy as np
import sys
sys.path.append('../')

from bayeso import gp
from bayeso import utils

def linear_up(X):
    list_up = []
    for elem_X in X:
        list_up.append([0.5 * np.sum(elem_X)])
    return np.array(list_up)

def main():
    X_train = np.array([
        [-3.0],
        [-2.0],
        [-1.5],
        [-1.0],
        [2.0],
        [1.2],
        [1.5],
    ])
    Y_train = np.cos(X_train) + 2.0
    num_test = 200
    X_test = np.linspace(-3, 6, num_test)
    X_test = X_test.reshape((num_test, 1))
    prior_mu = linear_up
    mu, sigma = gp.predict_optimized(X_train, Y_train, X_test, prior_mu=prior_mu)
    utils.plot_gp(X_train, Y_train, X_test, mu, sigma, '../results/gp/', 'test_optimized_many_prior_up')

if __name__ == '__main__':
    main()

