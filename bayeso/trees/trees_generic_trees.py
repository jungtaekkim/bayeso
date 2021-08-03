import numpy as np

from bayeso.trees import trees_common


def get_generic_trees(
    X, Y,
    num_trees,
    depth_max,
    size_min_leaf,
    ratio_sample,
    replace_samples,
    num_features,
    split_random_location,
):
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert isinstance(num_trees, int)
    assert isinstance(depth_max, int)
    assert isinstance(size_min_leaf, int)
    assert isinstance(ratio_sample, float)
    assert isinstance(replace_samples, bool)
    assert isinstance(num_features, int)
    assert isinstance(split_random_location, bool)

    assert len(X.shape) == 2
    assert len(Y.shape) == 2
    assert X.shape[0] == Y.shape[0]
    assert Y.shape[1] == 1
    if replace_samples:
        assert ratio_sample > 0.0
    else:
        assert ratio_sample > 0.0 and ratio_sample <= 1.0

    list_trees = []

    for ind in range(0, num_trees):
        X_, Y_ = trees_common.subsample(X, Y, ratio_sample, replace_samples)

        root = trees_common._split(X_, Y_, num_features, split_random_location)
        trees_common.split(root, depth_max, size_min_leaf, num_features, split_random_location, 1)

        list_trees.append(root)

    assert len(list_trees) == num_trees
    return list_trees
