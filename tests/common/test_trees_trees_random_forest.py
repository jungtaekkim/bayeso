#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 20, 2021
#
"""test_trees_trees_random_forest"""

import typing
import pytest
import numpy as np

from bayeso import constants
from bayeso.trees import trees_random_forest as package_target


TEST_EPSILON = 1e-7

def test_get_random_forest_typing():
    annos = package_target.get_random_forest.__annotations__

    assert annos['X'] == np.ndarray
    assert annos['Y'] == np.ndarray
    assert annos['num_trees'] == int
    assert annos['depth_max'] == int
    assert annos['size_min_leaf'] == int
    assert annos['num_features'] == int
    assert annos['return'] == list

def test_get_random_forest():
    np.random.seed(42)

    X = np.reshape(np.arange(0, 160), (40, 4)).astype(np.float64)
    Y = np.random.randn(X.shape[0], 1)
    num_trees = 100
    depth_max = 4
    size_min_leaf = 2
    ratio_sampling = 0.95
    replace_samples = True
    num_features = 2
    split_random_location = True

    with pytest.raises(AssertionError) as error:
        trees = package_target.get_random_forest(X, Y, num_trees, depth_max, size_min_leaf, 1.23)
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_random_forest(X, Y, num_trees, depth_max, size_min_leaf, 'abc')
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_random_forest(X, Y, num_trees, depth_max, 'abc', num_features)
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_random_forest(X, Y, num_trees, 1.23, size_min_leaf, num_features)
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_random_forest(X, Y, num_trees, 'abc', size_min_leaf, num_features)
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_random_forest(X, Y, 12.0, depth_max, size_min_leaf, num_features)
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_random_forest(X, Y, 'abc', depth_max, size_min_leaf, num_features)
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_random_forest(X, 'abc', num_trees, depth_max, size_min_leaf, num_features)
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_random_forest(X, np.array([1, 2, 3, 4]), num_trees, depth_max, size_min_leaf, num_features)
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_random_forest('abc', Y, num_trees, depth_max, size_min_leaf, num_features)
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_random_forest(np.array([1, 2, 3, 4]), Y, num_trees, depth_max, size_min_leaf, num_features)

    trees = package_target.get_random_forest(X, Y, num_trees, depth_max, size_min_leaf, num_features)

    assert isinstance(trees, list)
    assert len(trees) == num_trees

    X = np.reshape(np.arange(0, 160), (40, 4)).astype(np.float32)
    trees = package_target.get_random_forest(X, Y, num_trees, depth_max, size_min_leaf, num_features)

    X = np.reshape(np.arange(0, 160), (40, 4)).astype(np.float16)
    trees = package_target.get_random_forest(X, Y, num_trees, depth_max, size_min_leaf, num_features)

    X = np.reshape(np.arange(0, 160), (40, 4)).astype(float)
    trees = package_target.get_random_forest(X, Y, num_trees, depth_max, size_min_leaf, num_features)
