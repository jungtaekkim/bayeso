#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 6, 2021
#
"""It defines generic trees."""

import numpy as np

from bayeso.trees import trees_common
from bayeso.utils import utils_common


@utils_common.validate_types
def get_generic_trees(
    X: np.ndarray, Y: np.ndarray,
    num_trees: int,
    depth_max: int,
    size_min_leaf: int,
    ratio_sampling: float,
    replace_samples: bool,
    num_features: int,
    split_random_location: bool,
) -> list:
    """
    It returns a list of generic trees.

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
    :param ratio_sampling: ratio of dataset subsampling.
    :type ratio_sampling: float
    :param replace_samples: flag for replacement.
    :type replace_samples: bool.
    :param num_features: the number of split features.
    :type num_features: int.
    :param split_random_location: flag for random split location.
    :type split_random_location: bool.

    :returns: list of trees
    :rtype: list

    :raises: AssertionError

    """

    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert isinstance(num_trees, int)
    assert isinstance(depth_max, int)
    assert isinstance(size_min_leaf, int)
    assert isinstance(ratio_sampling, float)
    assert isinstance(replace_samples, bool)
    assert isinstance(num_features, int)
    assert isinstance(split_random_location, bool)

    assert len(X.shape) == 2
    assert len(Y.shape) == 2
    assert X.shape[0] == Y.shape[0]
    assert Y.shape[1] == 1
    if replace_samples:
        assert ratio_sampling > 0.0
    else:
        assert 0.0 < ratio_sampling <= 1.0

    list_trees = []

    for _ in range(0, num_trees):
        X_, Y_ = trees_common.subsample(X, Y, ratio_sampling, replace_samples)

        root = trees_common._split(X_, Y_, num_features, split_random_location)
        trees_common.split(root, depth_max, size_min_leaf, num_features, split_random_location, 1)

        list_trees.append(root)

    return list_trees
