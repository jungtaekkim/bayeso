#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 03, 2021
#
"""
"""

import numpy as np

from bayeso.trees import trees_common


def get_random_forest(
    X, Y,
    num_trees,
    depth_max,
    size_min_leaf,
    num_features,
):
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert isinstance(num_trees, int)
    assert isinstance(depth_max, int)
    assert isinstance(size_min_leaf, int)
    assert isinstance(num_features, int)

    assert len(X.shape) == 2
    assert len(Y.shape) == 2
    assert X.shape[0] == Y.shape[0]
    assert Y.shape[1] == 1

    ratio_sample = 1.0
    replace_samples = True
    split_random_location = False

    list_trees = []

    for ind in range(0, num_trees):
        X_, Y_ = trees_common.subsample(X, Y, ratio_sample, replace_samples)

        root = trees_common._split(X_, Y_, num_features, split_random_location)
        trees_common.split(root, depth_max, size_min_leaf, num_features, split_random_location, 1)

        list_trees.append(root)

    assert len(list_trees) == num_trees
    return list_trees
