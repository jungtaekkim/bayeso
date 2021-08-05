# example_rf
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 03, 2021

import numpy as np
import os

from bayeso.trees import trees_random_forest
from bayeso.trees import trees_common
from bayeso.utils import utils_common
from bayeso.utils import utils_plotting

PATH_SAVE = '../figures/rf/'


def main():
    np.random.seed(42)
    X_train = np.array([
        [-3.0],
        [-1.0],
        [0.0],
        [1.0],
        [2.0],
        [4.0],
    ])
    Y_train = np.cos(X_train) + np.random.randn(X_train.shape[0], 1) * 0.2
    num_test = 200
    X_test = np.linspace(-5, 5, num_test)
    X_test = X_test.reshape((num_test, 1))
    Y_test = np.cos(X_test)

    num_trees = 100
    depth_max = 5
    size_min_leaf = 2
    num_features = 1

    trees = trees_random_forest.get_random_forest(
        X_train, Y_train, num_trees, depth_max, size_min_leaf, num_features
    )

    mu, sigma = trees_common.predict_by_trees(X_test, trees)

    utils_plotting.plot_gp_via_distribution(X_train, Y_train, X_test, mu, sigma, Y_test, path_save=PATH_SAVE, str_postfix='cos', time_pause=np.inf)

if __name__ == '__main__':
    if not os.path.isdir(PATH_SAVE):
        os.makedirs(PATH_SAVE)

    main()
