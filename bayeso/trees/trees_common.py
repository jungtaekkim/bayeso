#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 6, 2021
#
"""
"""

import numpy as np

from bayeso import constants
from bayeso.utils import utils_common


@utils_common.validate_types
def get_inputs_from_leaf(leaf: list) -> np.ndarray:
    assert isinstance(leaf, list)

    X = [bx for bx, by in leaf]
    return np.array(X)

@utils_common.validate_types
def get_outputs_from_leaf(leaf: list) -> np.ndarray:
    assert isinstance(leaf, list)

    Y = [by for bx, by in leaf]
    return np.array(Y)

@utils_common.validate_types
def _mse(Y: np.ndarray) -> float:
    if Y.shape[0] > 0:
        mean = np.mean(Y, axis=0)
        mse_ = np.mean((Y - mean)**2)
    else:
        mse_ = 1e8
    return mse_

@utils_common.validate_types
def mse(left_right: tuple) -> float:
    assert isinstance(left_right, tuple)

    left, right = left_right

    Y_left = get_outputs_from_leaf(left)
    Y_right = get_outputs_from_leaf(right)

    mse_left = _mse(Y_left)
    mse_right = _mse(Y_right)

    return mse_left + mse_right

@utils_common.validate_types
def subsample(
    X: np.ndarray, Y: np.ndarray,
    ratio_sample: float, replace_samples: bool
) -> constants.TYPING_TUPLE_TWO_ARRAYS:
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert isinstance(ratio_sample, float)
    assert isinstance(replace_samples, bool)

    assert len(X.shape) == 2
    assert len(Y.shape) == 2
    assert X.shape[0] == Y.shape[0]
    assert Y.shape[1] == 1

    num_X = X.shape[0]
    num_samples = int(num_X * ratio_sample)

    indices = np.random.choice(num_X, num_samples, replace=replace_samples)

    X_ = X[indices]
    Y_ = Y[indices]

    assert X_.shape[0] == Y_.shape[0] == num_samples
    return X_, Y_

@utils_common.validate_types
def _split_left_right(
    X: np.ndarray, Y: np.ndarray,
    dim_to_split: int, val_to_split: float
) -> tuple:
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert isinstance(dim_to_split, int)
    assert isinstance(val_to_split, float)

    assert len(X.shape) == 2
    assert len(Y.shape) == 2
    assert X.shape[0] == Y.shape[0]
    assert Y.shape[1] == 1

    left = []
    right = []

    for bx, by in zip(X, Y):
        if bx[dim_to_split] < val_to_split:
            left.append((bx, by))
        else:
            right.append((bx, by))
    return left, right

@utils_common.validate_types
def _split(
    X: np.ndarray, Y: np.ndarray,
    num_features: int, split_random_location: bool
) -> dict:
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert isinstance(num_features, int)
    assert isinstance(split_random_location, bool)

    assert len(X.shape) == 2
    assert len(Y.shape) == 2
    assert X.shape[0] == Y.shape[0]
    assert X.shape[0] > 0
    assert Y.shape[1] == 1

    cur_ind = np.inf
    cur_val = np.inf
    cur_score = np.inf
    cur_left_right = None

    features = np.random.choice(X.shape[1], num_features, replace=False)

    for ind in features:
        dim_to_split = int(ind)
        num_evaluations = 1
        candidates_loc = np.sort(np.unique(X[:, dim_to_split]))

        if candidates_loc.shape[0] > 1:
            if split_random_location:
                min_bx = np.min(X[:, dim_to_split])
                max_bx = np.max(X[:, dim_to_split])
            else:
                num_evaluations = candidates_loc.shape[0] - 1

        for ind_loc in range(0, num_evaluations):
            if candidates_loc.shape[0] > 1:
                if split_random_location:
                    val_to_split = np.random.uniform(low=min_bx, high=max_bx)
                else:
                    val_to_split = np.mean(candidates_loc[ind_loc:ind_loc+2])
            else:
                val_to_split = X[0, dim_to_split]

            left_right = _split_left_right(X, Y, dim_to_split, val_to_split)
            left, right = left_right
            score = mse(left_right)

            if score < cur_score:
                cur_ind = dim_to_split
                cur_val = val_to_split
                cur_score = score
                cur_left_right = left_right

    return {
        'index': cur_ind,
        'value': cur_val,
        'left_right': cur_left_right
    }

@utils_common.validate_types
def split(
    node: dict,
    depth_max: int,
    size_min_leaf: int,
    num_features: int,
    split_random_location: bool,
    cur_depth: int
) -> constants.TYPE_NONE:
    assert isinstance(node, dict)
    assert isinstance(depth_max, int)
    assert isinstance(size_min_leaf, int)
    assert isinstance(num_features, int)
    assert isinstance(split_random_location, bool)
    assert isinstance(cur_depth, int)

    assert cur_depth > 0

    left, right = node['left_right']
    del(node['left_right'])

    if not left or not right:
        node['left'] = node['right'] = left + right
        return

    if cur_depth >= depth_max:
        node['left'], node['right'] = left, right
        return

    ##
    if len(left) <= size_min_leaf:
        node['left'] = left
    else:
        X_left = get_inputs_from_leaf(left)
        Y_left = get_outputs_from_leaf(left)

        node['left'] = _split(X_left, Y_left, num_features, split_random_location)
        split(node['left'], depth_max, size_min_leaf, num_features, split_random_location, cur_depth + 1)

    ##
    if len(right) <= size_min_leaf:
        node['right'] = right
    else:
        X_right = get_inputs_from_leaf(right)
        Y_right = get_outputs_from_leaf(right)

        node['right'] = _split(X_right, Y_right, num_features, split_random_location)
        split(node['right'], depth_max, size_min_leaf, num_features, split_random_location, cur_depth + 1)

@utils_common.validate_types
def _predict_by_tree(bx: np.ndarray, tree: dict) -> constants.TYPING_TUPLE_TWO_ARRAYS:
    assert isinstance(bx, np.ndarray)
    assert isinstance(tree, dict)

    assert len(bx.shape) == 1

    if bx[tree['index']] < tree['value']:
        if isinstance(tree['left'], dict):
            return _predict_by_tree(bx, tree['left'])
        else:
            cur_Y = get_outputs_from_leaf(tree['left'])
            return np.mean(cur_Y), np.std(cur_Y)
    else:
        if isinstance(tree['right'], dict):
            return _predict_by_tree(bx, tree['right'])
        else:
            cur_Y = get_outputs_from_leaf(tree['right'])
            return np.mean(cur_Y), np.std(cur_Y)

@utils_common.validate_types
def _predict_by_trees(bx: np.ndarray, list_trees: list) -> constants.TYPING_TUPLE_TWO_ARRAYS:
    assert isinstance(bx, np.ndarray)
    assert isinstance(list_trees, list)

    assert len(bx.shape) == 1

    list_mu_leaf = []
    list_sigma_leaf = []

    for tree in list_trees:
        mu_leaf, sigma_leaf = _predict_by_tree(bx, tree)

        list_mu_leaf.append(mu_leaf)
        list_sigma_leaf.append(sigma_leaf)

    mu = np.mean(list_mu_leaf)
    sigma = compute_sigma(np.array(list_mu_leaf), np.array(list_sigma_leaf))

    return mu, sigma

@utils_common.validate_types
def predict_by_trees(X: np.ndarray, list_trees: list) -> constants.TYPING_TUPLE_TWO_ARRAYS:
    assert isinstance(X, np.ndarray)
    assert isinstance(list_trees, list)

    assert len(X.shape) == 2

    preds_mu = []
    preds_sigma = []

    for bx in X:
        mu_bx, sigma_bx = _predict_by_trees(bx, list_trees)

        preds_mu.append(mu_bx)
        preds_sigma.append(sigma_bx)

    preds_mu = np.array(preds_mu)[..., np.newaxis]
    preds_sigma = np.array(preds_sigma)[..., np.newaxis]

    return preds_mu, preds_sigma

@utils_common.validate_types
def compute_sigma(
    preds_mu_leaf: np.ndarray,
    preds_sigma_leaf: np.ndarray,
    min_sigma: float=0.0
) -> np.ndarray:
    assert isinstance(preds_mu_leaf, np.ndarray)
    assert isinstance(preds_sigma_leaf, np.ndarray)
    assert isinstance(min_sigma, float)

    assert len(preds_mu_leaf.shape) == 1
    assert len(preds_sigma_leaf.shape) == 1
    assert preds_mu_leaf.shape[0] == preds_sigma_leaf.shape[0]

    preds_sigma_leaf_ = np.maximum(preds_sigma_leaf, np.zeros(preds_sigma_leaf.shape) + min_sigma)

    sigma = np.mean(preds_mu_leaf**2 + preds_sigma_leaf_**2)
    sigma -= np.mean(preds_mu_leaf)**2

    if sigma < 0.0:
        sigma = 0.0
    sigma = np.sqrt(sigma)

    return sigma
