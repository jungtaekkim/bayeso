#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 16, 2023
#

import numpy as np
import os

from bayeso.trees import trees_generic_trees
from bayeso.trees import trees_common
from bayeso.utils import utils_common
from bayeso.utils import utils_plotting


def main(path_save):
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
    num_test = 10000
    X_test = np.linspace(-5, 5, num_test)
    X_test = X_test.reshape((num_test, 1))
    Y_test = np.cos(X_test)

    num_trees = 100
    depth_max = 5
    size_min_leaf = 2
    ratio_sampling = 0.8
    replace_samples = False
    num_features = 1
    split_random_location = True

    trees = trees_generic_trees.get_generic_trees(
        X_train, Y_train, num_trees, depth_max, size_min_leaf, ratio_sampling, replace_samples, num_features, split_random_location
    )

    mu, sigma = trees_common.predict_by_trees(X_test, trees)

    utils_plotting.plot_gp_via_distribution(X_train, Y_train, X_test, mu, sigma, Y_test, path_save=path_save, str_postfix='cos')


if __name__ == '__main__':
    path_save = None

    if path_save is not None and not os.path.isdir(path_save):
        os.makedirs(path_save)
    main(path_save)
