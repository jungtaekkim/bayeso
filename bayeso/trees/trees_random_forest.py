#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 6, 2021
#
"""It defines a random forest."""

import numpy as np

from bayeso.trees import trees_common
from bayeso.utils import utils_common


@utils_common.validate_types
def get_random_forest(
    X: np.ndarray, Y: np.ndarray,
    num_trees: int,
    depth_max: int,
    size_min_leaf: int,
    num_features: int,
) -> list:
    """
    It returns a random forest.

    :param X: inputs. Shape: (N, d).
    :type X: np.ndarray
    :param Y: outputs. Shape: (N, 1).
    :type Y: str.
    :param num_trees: the number of trees.
    :type num_trees: int.
    :param depth_max: maximum depth of tree.
    :type depth_max: int.
    :param size_min_leaf: minimum size of leaf.
    :type size_min_leaf: int.
    :param num_features: the number of split features.
    :type num_features: int.

    :returns: list of trees
    :rtype: list

    :raises: AssertionError

    """

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

    for _ in range(0, num_trees):
        X_, Y_ = trees_common.subsample(X, Y, ratio_sample, replace_samples)

        root = trees_common._split(X_, Y_, num_features, split_random_location)
        trees_common.split(root, depth_max, size_min_leaf, num_features, split_random_location, 1)

        list_trees.append(root)

    return list_trees
