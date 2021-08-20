#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 20, 2021
#
"""test_trees_generic_trees"""

import typing
import pytest
import numpy as np

from bayeso import constants
from bayeso.trees import trees_generic_trees as package_target


TEST_EPSILON = 1e-7

def test_get_generic_trees_typing():
    annos = package_target.get_generic_trees.__annotations__

    assert annos['X'] == np.ndarray
    assert annos['Y'] == np.ndarray
    assert annos['num_trees'] == int
    assert annos['depth_max'] == int
    assert annos['size_min_leaf'] == int
    assert annos['ratio_sampling'] == float
    assert annos['replace_samples'] == bool
    assert annos['num_features'] == int
    assert annos['split_random_location'] == bool
    assert annos['return'] == list

def test_get_generic_trees():
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
        trees = package_target.get_generic_trees(X, Y, num_trees, depth_max, size_min_leaf, ratio_sampling, replace_samples, num_features, 1)
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_generic_trees(X, Y, num_trees, depth_max, size_min_leaf, ratio_sampling, replace_samples, num_features, 'abc')
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_generic_trees(X, Y, num_trees, depth_max, size_min_leaf, ratio_sampling, replace_samples, 1.23, split_random_location)
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_generic_trees(X, Y, num_trees, depth_max, size_min_leaf, ratio_sampling, replace_samples, 'abc', split_random_location)
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_generic_trees(X, Y, num_trees, depth_max, size_min_leaf, ratio_sampling, 'abc', num_features, split_random_location)
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_generic_trees(X, Y, num_trees, depth_max, size_min_leaf, ratio_sampling, 123, num_features, split_random_location)
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_generic_trees(X, Y, num_trees, depth_max, size_min_leaf, 2, replace_samples, num_features, split_random_location)
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_generic_trees(X, Y, num_trees, depth_max, 0.123, ratio_sampling, replace_samples, num_features, split_random_location)
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_generic_trees(X, Y, num_trees, depth_max, 'abc', ratio_sampling, replace_samples, num_features, split_random_location)
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_generic_trees(X, Y, num_trees, 1.23, size_min_leaf, ratio_sampling, replace_samples, num_features, split_random_location)
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_generic_trees(X, Y, num_trees, 'abc', size_min_leaf, ratio_sampling, replace_samples, num_features, split_random_location)
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_generic_trees(X, Y, 12.0, depth_max, size_min_leaf, ratio_sampling, replace_samples, num_features, split_random_location)
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_generic_trees(X, Y, 'abc', depth_max, size_min_leaf, ratio_sampling, replace_samples, num_features, split_random_location)
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_generic_trees(X, 'abc', num_trees, depth_max, size_min_leaf, ratio_sampling, replace_samples, num_features, split_random_location)
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_generic_trees(X, np.array([1, 2, 3, 4]), num_trees, depth_max, size_min_leaf, ratio_sampling, replace_samples, num_features, split_random_location)
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_generic_trees('abc', Y, num_trees, depth_max, size_min_leaf, ratio_sampling, replace_samples, num_features, split_random_location)
    with pytest.raises(AssertionError) as error:
        trees = package_target.get_generic_trees(np.array([1, 2, 3, 4]), Y, num_trees, depth_max, size_min_leaf, ratio_sampling, replace_samples, num_features, split_random_location)

    trees = package_target.get_generic_trees(X, Y, num_trees, depth_max, size_min_leaf, ratio_sampling, replace_samples, num_features, split_random_location)

    assert isinstance(trees, list)
    assert len(trees) == num_trees

    X = np.reshape(np.arange(0, 160), (40, 4)).astype(np.float32)
    trees = package_target.get_generic_trees(X, Y, num_trees, depth_max, size_min_leaf, ratio_sampling, replace_samples, num_features, split_random_location)

    X = np.reshape(np.arange(0, 160), (40, 4)).astype(np.float16)
    trees = package_target.get_generic_trees(X, Y, num_trees, depth_max, size_min_leaf, ratio_sampling, replace_samples, num_features, split_random_location)

    X = np.reshape(np.arange(0, 160), (40, 4)).astype(float)
    trees = package_target.get_generic_trees(X, Y, num_trees, depth_max, size_min_leaf, ratio_sampling, replace_samples, num_features, split_random_location)
