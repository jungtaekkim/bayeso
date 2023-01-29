#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: October 13, 2021
#
"""It defines a common function for tree-based surrogates."""

import multiprocessing
import itertools
import numpy as np

from bayeso import constants
from bayeso.utils import utils_common


@utils_common.validate_types
def get_inputs_from_leaf(leaf: list) -> np.ndarray:
    """
    It returns an input from a leaf.

    :param leaf: pairs of input and output in a leaf.
    :type leaf: list

    :returns: an input. Shape: (n, d).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(leaf, list)

    X = [bx for bx, by in leaf]
    return np.array(X)

@utils_common.validate_types
def get_outputs_from_leaf(leaf: list) -> np.ndarray:
    """
    It returns an output from a leaf.

    :param leaf: pairs of input and output in a leaf.
    :type leaf: list

    :returns: an output. Shape: (n, 1).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(leaf, list)

    Y = [by for bx, by in leaf]
    return np.array(Y)

@utils_common.validate_types
def _mse(Y: np.ndarray) -> float:
    """
    It returns a mean squared loss over `Y`.

    :param Y: outputs in a leaf.
    :type Y: numpy.ndarray

    :returns: a loss value.
    :rtype: float

    :raises: AssertionError

    """

    assert isinstance(Y, np.ndarray)
    if len(Y.shape) == 2:
        assert Y.shape[1] == 1

    if Y.shape[0] > 0:
        mean = np.mean(Y, axis=0)
        mse_ = np.mean((Y - mean)**2)
    else:
        mse_ = 1e8

    return mse_

@utils_common.validate_types
def mse(left_right: tuple) -> float:
    """
    It returns a mean squared loss over `left_right`.

    :param left_right: a tuple of left and right leaves.
    :type left_right: tuple

    :returns: a loss value.
    :rtype: float

    :raises: AssertionError

    """

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
    ratio_sampling: float, replace_samples: bool
) -> constants.TYPING_TUPLE_TWO_ARRAYS:
    """
    It subsamples a bootstrap sample.

    :param X: inputs. Shape: (n, d).
    :type X: numpy.ndarray
    :param Y: outputs. Shape: (n, 1).
    :type Y: numpy.ndarray
    :param ratio_sampling: ratio of sampling.
    :type ratio_sampling: float
    :param replace_samples: a flag for sampling with replacement or without replacement.
    :type replace_samples: bool.

    :returns: a tuple of bootstrap sample. Shape: ((m, d), (m, 1)).
    :rtype: (numpy.ndarray, numpy.ndarray)

    :raises: AssertionError

    """

    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert isinstance(ratio_sampling, float)
    assert isinstance(replace_samples, bool)

    assert len(X.shape) == 2
    assert len(Y.shape) == 2
    assert X.shape[0] == Y.shape[0]
    assert Y.shape[1] == 1

    if replace_samples:
        assert ratio_sampling > 0.0
    else:
        assert 0.0 < ratio_sampling <= 1.0

    num_X = X.shape[0]
    num_samples = int(num_X * ratio_sampling)

    indices = np.random.choice(num_X, num_samples, replace=replace_samples)

    X_ = X[indices]
    Y_ = Y[indices]

    return X_, Y_

@utils_common.validate_types
def _split_left_right(
    X: np.ndarray, Y: np.ndarray,
    dim_to_split: int, val_to_split: float
) -> tuple:
    """
    It splits `X` and `Y` to left and right leaves.

    :param X: inputs. Shape: (n, d).
    :type X: numpy.ndarray
    :param Y: outputs. Shape: (n, 1).
    :type Y: numpy.ndarray
    :param dim_to_split: a dimension to split.
    :type dim_to_split: int.
    :param val_to_split: a value to split.
    :type val_to_split: float

    :returns: a tuple of left and right leaves.
    :rtype: tuple

    :raises: AssertionError

    """

    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert isinstance(dim_to_split, int)
    assert isinstance(val_to_split, (float, np.float16, np.float32, np.float64))

    assert len(X.shape) == 2
    assert len(Y.shape) == 2
    assert X.shape[0] == Y.shape[0]
    assert Y.shape[1] == 1

    indices_left = X[:, dim_to_split] < val_to_split
    left = list(zip(X[indices_left], Y[indices_left]))

    indices_right = X[:, dim_to_split] >= val_to_split
    right = list(zip(X[indices_right], Y[indices_right]))

    return left, right

@utils_common.validate_types
def _split_deterministic(X: np.ndarray, Y: np.ndarray, dim_to_split: int
) -> constants.TYPING_TUPLE_INT_FLOAT_TUPLE:
    candidates_loc = np.sort(np.unique(X[:, dim_to_split]))
    num_evaluations = 1
    if candidates_loc.shape[0] > 1:
        num_evaluations = candidates_loc.shape[0] - 1

    cur_ind = np.inf
    cur_val = np.inf
    cur_score = np.inf
    cur_left_right = None

    indices_loc = np.random.choice(
        num_evaluations, np.minimum(20, num_evaluations), replace=False)

    for ind_loc in indices_loc:
        if candidates_loc.shape[0] > 1:
            val_to_split = np.mean(candidates_loc[ind_loc:ind_loc+2])
        else:
            val_to_split = X[0, dim_to_split]

        left_right = _split_left_right(X, Y, dim_to_split, val_to_split)
        score = mse(left_right)

        if score < cur_score:
            cur_ind = dim_to_split
            cur_val = val_to_split
            cur_score = score
            cur_left_right = left_right

    return cur_ind, cur_val, cur_score, cur_left_right

@utils_common.validate_types
def _split_random(X: np.ndarray, Y: np.ndarray, dim_to_split: int
) -> constants.TYPING_TUPLE_INT_FLOAT_TUPLE:
    candidates_loc = np.sort(np.unique(X[:, dim_to_split]))

    if candidates_loc.shape[0] > 1:
        min_bx = np.min(X[:, dim_to_split])
        max_bx = np.max(X[:, dim_to_split])

        val_to_split = np.random.uniform(low=min_bx, high=max_bx)
    else:
        val_to_split = X[0, dim_to_split]

    left_right = _split_left_right(X, Y, dim_to_split, val_to_split)
    score = mse(left_right)

    return dim_to_split, val_to_split, score, left_right

@utils_common.validate_types
def _split(
    X: np.ndarray, Y: np.ndarray,
    num_features: int, split_random_location: bool
) -> dict:
    """
    It splits `X` and `Y` to left and right leaves as a dictionary
    including split dimension and split location.

    :param X: inputs. Shape: (n, d).
    :type X: numpy.ndarray
    :param Y: outputs. Shape: (n, 1).
    :type Y: numpy.ndarray
    :param num_features: the number of features to split.
    :type num_features: int.
    :param split_random_location: flag for setting a split location randomly or not.
    :type split_random_location: bool.

    :returns: a dictionary of left and right leaves, spilt dimension, and split location.
    :rtype: dict.

    :raises: AssertionError

    """

    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert isinstance(num_features, int)
    assert isinstance(split_random_location, bool)

    assert len(X.shape) == 2
    assert len(Y.shape) == 2
    assert X.shape[0] == Y.shape[0]
    assert X.shape[0] > 0
    assert Y.shape[1] == 1

    features = np.random.choice(X.shape[1], num_features, replace=False)

    cur_ind = np.inf
    cur_val = np.inf
    cur_score = np.inf
    cur_left_right = None

    for ind in features:
        dim_to_split = int(ind)

        if split_random_location:
            _ind, _val, _score, _left_right = _split_random(
                X, Y, dim_to_split)
        else:
            _ind, _val, _score, _left_right = _split_deterministic(
                X, Y, dim_to_split)

        if _score < cur_score:
            cur_ind = _ind
            cur_val = _val
            cur_score = _score
            cur_left_right = _left_right

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
    """
    It splits a root node to construct a tree.

    :param node: a root node.
    :type node: dict.
    :param depth_max: maximum depth of tree.
    :type depth_max: int.
    :param size_min_leaf: minimum size of leaf.
    :type size_min_leaf: int.
    :param num_features: the number of split features.
    :type num_features: int.
    :param split_random_location: flag for setting a split location randomly or not.
    :type split_random_location: bool.
    :param cur_depth: depth of the current node.
    :type cur_depth: int.

    :returns: None.
    :rtype: NoneType

    :raises: AssertionError

    """

    assert isinstance(node, dict)
    assert isinstance(depth_max, int)
    assert isinstance(size_min_leaf, int)
    assert isinstance(num_features, int)
    assert isinstance(split_random_location, bool)
    assert isinstance(cur_depth, int)

    assert cur_depth > 0

    left, right = node['left_right']
    del node['left_right']

    if not left or not right: # pragma: no cover
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
        split(node['left'], depth_max, size_min_leaf, num_features,
            split_random_location, cur_depth + 1)

    ##
    if len(right) <= size_min_leaf:
        node['right'] = right
    else:
        X_right = get_inputs_from_leaf(right)
        Y_right = get_outputs_from_leaf(right)

        node['right'] = _split(X_right, Y_right, num_features, split_random_location)
        split(node['right'], depth_max, size_min_leaf, num_features,
            split_random_location, cur_depth + 1)

@utils_common.validate_types
def _predict_by_tree(bx: np.ndarray, tree: dict) -> constants.TYPING_TUPLE_TWO_FLOATS:
    """
    It predicts a posterior distribution over `bx`, given `tree`.

    :param bx: an input. Shape: (d, ).
    :type bx: numpy.ndarray
    :param tree: a decision tree.
    :type tree: dict.

    :returns: posterior mean and standard devitation estimates.
    :rtype: (float, float)

    :raises: AssertionError

    """

    assert isinstance(bx, np.ndarray)
    assert isinstance(tree, dict)

    assert len(bx.shape) == 1

    if bx[tree['index']] < tree['value']:
        if isinstance(tree['left'], dict):
            return _predict_by_tree(bx, tree['left'])

        cur_Y = get_outputs_from_leaf(tree['left'])
        return np.mean(cur_Y), np.std(cur_Y)

    if isinstance(tree['right'], dict):
        return _predict_by_tree(bx, tree['right'])

    cur_Y = get_outputs_from_leaf(tree['right'])
    return np.mean(cur_Y), np.std(cur_Y)

@utils_common.validate_types
def _predict_by_trees(bx: np.ndarray, list_trees: list) -> constants.TYPING_TUPLE_TWO_FLOATS:
    """
    It predicts a posterior distribution over `bx`, given `list_trees`.

    :param bx: an input. Shape: (d, ).
    :type bx: numpy.ndarray
    :param list_trees: a list of decision trees.
    :type list_trees: list

    :returns: posterior mean and standard devitation estimates.
    :rtype: (float, float)

    :raises: AssertionError

    """

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
def unit_predict_by_trees(X: np.ndarray, list_trees: list) -> constants.TYPING_TUPLE_TWO_ARRAYS:
    """
    It predicts a posterior distribution over `X`, given `list_trees`.

    :param X: inputs. Shape: (n, d).
    :type X: numpy.ndarray
    :param list_trees: a list of decision trees.
    :type list_trees: list

    :returns: posterior mean and standard devitation estimates. Shape: ((n, 1), (n, 1)).
    :rtype: (numpy.ndarray, numpy.ndarray)

    :raises: AssertionError

    """

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
def predict_by_trees(X: np.ndarray, list_trees: list) -> constants.TYPING_TUPLE_TWO_ARRAYS:
    """
    It predicts a posterior distribution over `X`, given `list_trees`,
    using `multiprocessing`.

    :param X: inputs. Shape: (n, d).
    :type X: numpy.ndarray
    :param list_trees: a list of decision trees.
    :type list_trees: list

    :returns: posterior mean and standard devitation estimates. Shape: ((n, 1), (n, 1)).
    :rtype: (numpy.ndarray, numpy.ndarray)

    :raises: AssertionError

    """

    assert isinstance(X, np.ndarray)
    assert isinstance(list_trees, list)
    assert len(X.shape) == 2

    num_data_per_split = constants.NUM_DATA_PER_SPLIT_TREES
    num_cpu = multiprocessing.cpu_count()

    num_data = X.shape[0]

    if num_data <= num_data_per_split:
        preds_mu, preds_sigma = unit_predict_by_trees(X, list_trees)
    else:
        list_Xs = np.array_split(X, int(np.ceil(num_data / num_data_per_split)))

        with multiprocessing.Pool(num_cpu) as p:
            results = p.starmap(unit_predict_by_trees, zip(list_Xs, itertools.repeat(list_trees)))

        list_preds_mu, list_preds_sigma = zip(*results)

        preds_mu = np.concatenate(list_preds_mu, axis=0)
        preds_sigma = np.concatenate(list_preds_sigma, axis=0)

    return preds_mu, preds_sigma

@utils_common.validate_types
def compute_sigma(
    preds_mu_leaf: np.ndarray,
    preds_sigma_leaf: np.ndarray,
    min_sigma: float=0.0
) -> np.ndarray:
    """
    It computes predictive standard deviation estimates.

    :param preds_mu_leaf: predictive mean estimates of leaf. Shape: (n, ).
    :type preds_mu_leaf: numpy.ndarray
    :param preds_sigma_leaf: predictive standard deviation estimates of leaf. Shape: (n, ).
    :type preds_sigma_leaf: numpy.ndarray
    :param min_sigma: threshold for minimum standard deviation.
    :type min_sigma: float

    :returns: predictive standard deviation estimates. Shape: (n, ).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(preds_mu_leaf, np.ndarray)
    assert isinstance(preds_sigma_leaf, np.ndarray)
    assert isinstance(min_sigma, float)

    assert len(preds_mu_leaf.shape) == 1
    assert len(preds_sigma_leaf.shape) == 1
    assert preds_mu_leaf.shape[0] == preds_sigma_leaf.shape[0]

    preds_sigma_leaf_ = np.maximum(preds_sigma_leaf, np.zeros(preds_sigma_leaf.shape) + min_sigma)

    sigma = np.mean(preds_mu_leaf**2 + preds_sigma_leaf_**2)
    sigma -= np.mean(preds_mu_leaf)**2

    sigma = max(sigma, 0.0)
    sigma = np.sqrt(sigma)

    return sigma
