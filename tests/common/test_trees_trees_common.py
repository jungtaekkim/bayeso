#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 17, 2021
#
"""test_trees_trees_common"""

import typing
import pytest
import numpy as np

from bayeso import constants
from bayeso.trees import trees_common as package_target


TEST_EPSILON = 1e-7

def test_get_inputs_from_leaf_typing():
    annos = package_target.get_inputs_from_leaf.__annotations__

    assert annos['leaf'] == list
    assert annos['return'] == np.ndarray

def test_get_inputs_from_leaf():
    leaf = [
        (np.array([1.0, 2.0, 3.0]), np.array([2.0])),
        (np.array([2.0, 1.0, 3.0]), np.array([1.0])),
        (np.array([3.0, 0.0, 3.0]), np.array([0.5])),
        (np.array([4.0, -1.0, 3.0]), np.array([1.5])),
    ]

    with pytest.raises(AssertionError) as error:
        package_target.get_inputs_from_leaf(123)
    with pytest.raises(AssertionError) as error:
        package_target.get_inputs_from_leaf('abc')

    inputs = package_target.get_inputs_from_leaf(leaf)
    assert len(inputs.shape) == 2
    assert inputs.shape[0] == len(leaf)
    assert inputs.shape[1] == leaf[0][0].shape[0]

def test_get_outputs_from_leaf_typing():
    annos = package_target.get_outputs_from_leaf.__annotations__

    assert annos['leaf'] == list
    assert annos['return'] == np.ndarray

def test_get_outputs_from_leaf():
    leaf = [
        (np.array([1.0, 2.0, 3.0]), np.array([2.0])),
        (np.array([2.0, 1.0, 3.0]), np.array([1.0])),
        (np.array([3.0, 0.0, 3.0]), np.array([0.5])),
        (np.array([4.0, -1.0, 3.0]), np.array([1.5])),
    ]

    with pytest.raises(AssertionError) as error:
        package_target.get_outputs_from_leaf(123)
    with pytest.raises(AssertionError) as error:
        package_target.get_outputs_from_leaf('abc')

    outputs = package_target.get_outputs_from_leaf(leaf)
    assert len(outputs.shape) == 2
    assert outputs.shape[0] == len(leaf)
    assert outputs.shape[1] == leaf[0][1].shape[0]

def test__mse_typing():
    annos = package_target._mse.__annotations__

    assert annos['Y'] == np.ndarray
    assert annos['return'] == float

def test__mse():
    Y = np.array([
        [1.0],
        [2.0],
        [6.0],
    ])

    with pytest.raises(AssertionError) as error:
        package_target._mse(123)
    with pytest.raises(AssertionError) as error:
        package_target._mse('abc')

    output = package_target._mse(np.zeros((0, 1)))
    assert output == 1e8

    output = package_target._mse(np.array([]))
    assert output == 1e8

    output = package_target._mse(Y)
    assert output == 14.0 / 3

def test_mse_typing():
    annos = package_target.mse.__annotations__

    assert annos['left_right'] == tuple
    assert annos['return'] == float

def test_mse():
    left = [
        (np.array([1.0, 2.0, 3.0]), np.array([0.5])),
        (np.array([2.0, 2.0, 1.0]), np.array([0.1])),
        (np.array([3.0, 0.0, 1.0]), np.array([0.2])),
        (np.array([4.0, 0.0, 3.0]), np.array([0.3])),
    ]
    right = [
        (np.array([10.0, 20.0, 30.0]), np.array([0.6])),
        (np.array([20.0, 20.0, 10.0]), np.array([0.9])),
        (np.array([30.0, 0.0, 10.0]), np.array([0.9])),
        (np.array([40.0, 0.0, 30.0]), np.array([0.8])),
    ]

    with pytest.raises(AssertionError) as error:
        package_target.mse(123)
    with pytest.raises(AssertionError) as error:
        package_target.mse('abc')
    with pytest.raises(AssertionError) as error:
        package_target.mse(np.zeros((4, 1)))

    output = package_target.mse(([], []))
    assert output == 2e8

    output = package_target.mse((left, []))
    assert output == 0.021875 + 1e8

    output = package_target.mse(([], right))
    assert output == 1e8 + 0.015

    output = package_target.mse((left, right))
    assert np.abs(output - (0.021875 + 0.015)) < TEST_EPSILON

def test_subsample_typing():
    annos = package_target.subsample.__annotations__

    assert annos['X'] == np.ndarray
    assert annos['Y'] == np.ndarray
    assert annos['ratio_sampling'] == float
    assert annos['replace_samples'] == bool
    assert annos['return'] == constants.TYPING_TUPLE_TWO_ARRAYS

def test_subsample():
    np.random.seed(42)

    X = np.reshape(np.arange(0, 40), (10, 4))
    Y = np.random.randn(10, 1)
    ratio_sampling = 0.5
    replace_samples = False

    with pytest.raises(AssertionError) as error:
        package_target.subsample(X, Y, ratio_sampling, 123)
    with pytest.raises(AssertionError) as error:
        package_target.subsample(X, Y, ratio_sampling, 'abc')
    with pytest.raises(AssertionError) as error:
        package_target.subsample(X, Y, 1, replace_samples)
    with pytest.raises(AssertionError) as error:
        package_target.subsample(X, Y, 'abc', replace_samples)
    with pytest.raises(AssertionError) as error:
        package_target.subsample(X, 'abc', ratio_sampling, replace_samples)
    with pytest.raises(AssertionError) as error:
        package_target.subsample(X, 123, ratio_sampling, replace_samples)
    with pytest.raises(AssertionError) as error:
        package_target.subsample('abc', Y, ratio_sampling, replace_samples)
    with pytest.raises(AssertionError) as error:
        package_target.subsample(123, Y, ratio_sampling, replace_samples)
    with pytest.raises(AssertionError) as error:
        package_target.subsample(X, Y, 4.0, False)
    with pytest.raises(AssertionError) as error:
        package_target.subsample(X, Y, -0.5, False)
    with pytest.raises(AssertionError) as error:
        package_target.subsample(X, Y, -0.5, True)

    X_, Y_ = package_target.subsample(X, Y, ratio_sampling, replace_samples)
    print(X_)
    print(Y_)

    X_truth = np.array([
        [24, 25, 26, 27],
        [8, 9, 10, 11],
        [32, 33, 34, 35],
        [28, 29, 30, 31],
        [36, 37, 38, 39],
    ])
    Y_truth = np.array([
        [1.57921282],
        [0.64768854],
        [-0.46947439],
        [0.76743473],
        [0.54256004],
    ])

    assert np.all(np.abs(X_truth - X_) < TEST_EPSILON)
    assert np.all(np.abs(Y_truth - Y_) < TEST_EPSILON)
    assert X_.shape[0] == Y_.shape[0] == X_truth.shape[0] == Y_truth.shape[0]
    assert X_.shape[0] == Y_.shape[0] == int(ratio_sampling * X.shape[0])

    X_, Y_ = package_target.subsample(X, Y, 1.2, True)
    print(X_)
    print(Y_)

    X_truth = np.array([
        [36, 37, 38, 39],
        [8, 9, 10, 11],
        [24, 25, 26, 27],
        [12, 13, 14, 15],
        [32, 33, 34, 35],
        [8, 9, 10, 11],
        [16, 17, 18, 19],
        [8, 9, 10, 11],
        [24, 25, 26, 27],
        [16, 17, 18, 19],
        [32, 33, 34, 35],
        [24, 25, 26, 27],
    ])
    Y_truth = np.array([
        [0.54256004],
        [0.64768854],
        [1.57921282],
        [1.52302986],
        [-0.46947439],
        [0.64768854],
        [-0.23415337],
        [0.64768854],
        [1.57921282],
        [-0.23415337],
        [-0.46947439],
        [1.57921282],
    ])

    assert np.all(np.abs(X_truth - X_) < TEST_EPSILON)
    assert np.all(np.abs(Y_truth - Y_) < TEST_EPSILON)
    assert X_.shape[0] == Y_.shape[0] == X_truth.shape[0] == Y_truth.shape[0]
    assert X_.shape[0] == Y_.shape[0] == int(1.2 * X.shape[0])

def test__split_left_right_typing():
    annos = package_target._split_left_right.__annotations__

    assert annos['X'] == np.ndarray
    assert annos['Y'] == np.ndarray
    assert annos['dim_to_split'] == int
    assert annos['val_to_split'] == float
    assert annos['return'] == tuple

def test__split_left_right():
    np.random.seed(42)

    X = np.reshape(np.arange(0, 40), (10, 4))
    Y = np.random.randn(10, 1)
    dim_to_split = 1
    val_to_split = 14.0

    with pytest.raises(AssertionError) as error:
        package_target._split_left_right(X, Y, dim_to_split, 'abc')
    with pytest.raises(AssertionError) as error:
        package_target._split_left_right(X, Y, dim_to_split, 123)
    with pytest.raises(AssertionError) as error:
        package_target._split_left_right(X, Y, 0.5, val_to_split)
    with pytest.raises(AssertionError) as error:
        package_target._split_left_right(X, Y, 'abc', val_to_split)
    with pytest.raises(AssertionError) as error:
        package_target._split_left_right(X, 'abc', dim_to_split, val_to_split)
    with pytest.raises(AssertionError) as error:
        package_target._split_left_right('abc', Y, dim_to_split, val_to_split)

    left, right = package_target._split_left_right(X, Y, dim_to_split, val_to_split)
    print(left)
    print(right)

    left_ = [
        (np.array([0, 1, 2, 3]), np.array([0.49671415])),
        (np.array([4, 5, 6, 7]), np.array([-0.1382643])),
        (np.array([8, 9, 10, 11]), np.array([0.64768854])),
        (np.array([12, 13, 14, 15]), np.array([1.52302986]))
    ]
    right_ = [
        (np.array([16, 17, 18, 19]), np.array([-0.23415337])),
        (np.array([20, 21, 22, 23]), np.array([-0.23413696])),
        (np.array([24, 25, 26, 27]), np.array([1.57921282])),
        (np.array([28, 29, 30, 31]), np.array([0.76743473])),
        (np.array([32, 33, 34, 35]), np.array([-0.46947439])),
        (np.array([36, 37, 38, 39]), np.array([0.54256004]))
    ]

    assert len(left_) == 4
    assert len(right_) == 6
    assert (len(left_) + len(right_)) == X.shape[0]

def test__split_typing():
    annos = package_target._split.__annotations__

    assert annos['X'] == np.ndarray
    assert annos['Y'] == np.ndarray
    assert annos['num_features'] == int
    assert annos['split_random_location'] == bool
    assert annos['return'] == dict

def test__split():
    np.random.seed(42)

    X = np.reshape(np.arange(0, 40), (10, 4))
    Y = np.random.randn(10, 1)
    num_features = 2
    split_random_location = False

    with pytest.raises(AssertionError) as error:
        package_target._split(X, Y, num_features, 'abc')
    with pytest.raises(AssertionError) as error:
        package_target._split(X, Y, 'abc', split_random_location)
    with pytest.raises(AssertionError) as error:
        package_target._split(X, Y, 2.0, split_random_location)
    with pytest.raises(AssertionError) as error:
        package_target._split(X, 'abc', num_features, split_random_location)
    with pytest.raises(AssertionError) as error:
        package_target._split('abc', Y, num_features, split_random_location)

    dict_split = package_target._split(X, Y, num_features, split_random_location)
    print(dict_split)
    print(dict_split['index'])
    print(dict_split['value'])
    print(dict_split['left_right'])

    assert isinstance(dict_split, dict)
    assert dict_split['index'] == 1
    assert dict_split['value'] == 35.0

    assert np.all(dict_split['left_right'][0][0][0] == np.array([0, 1, 2, 3]))
    assert np.abs(dict_split['left_right'][0][0][1] - np.array([0.49671415])) < TEST_EPSILON

    assert np.all(dict_split['left_right'][0][1][0] == np.array([4, 5, 6, 7]))
    assert np.abs(dict_split['left_right'][0][1][1] - np.array([-0.1382643])) < TEST_EPSILON

    assert np.all(dict_split['left_right'][0][2][0] == np.array([8, 9, 10, 11]))
    assert np.abs(dict_split['left_right'][0][2][1] - np.array([0.64768854])) < TEST_EPSILON

    assert np.all(dict_split['left_right'][0][3][0] == np.array([12, 13, 14, 15]))
    assert np.abs(dict_split['left_right'][0][3][1] - np.array([1.52302986])) < TEST_EPSILON

    assert np.all(dict_split['left_right'][0][4][0] == np.array([16, 17, 18, 19]))
    assert np.abs(dict_split['left_right'][0][4][1] - np.array([-0.23415337])) < TEST_EPSILON

    assert np.all(dict_split['left_right'][0][5][0] == np.array([20, 21, 22, 23]))
    assert np.abs(dict_split['left_right'][0][5][1] - np.array([-0.23413696])) < TEST_EPSILON

    assert np.all(dict_split['left_right'][0][6][0] == np.array([24, 25, 26, 27]))
    assert np.abs(dict_split['left_right'][0][6][1] - np.array([1.57921282])) < TEST_EPSILON

    assert np.all(dict_split['left_right'][0][7][0] == np.array([28, 29, 30, 31]))
    assert np.abs(dict_split['left_right'][0][7][1] - np.array([0.76743473])) < TEST_EPSILON

    assert np.all(dict_split['left_right'][0][8][0] == np.array([32, 33, 34, 35]))
    assert np.abs(dict_split['left_right'][0][8][1] - np.array([-0.46947439])) < TEST_EPSILON

    assert np.all(dict_split['left_right'][1][0][0] == np.array([36, 37, 38, 39]))
    assert np.abs(dict_split['left_right'][1][0][1] - np.array([0.54256004])) < TEST_EPSILON

    dict_split = package_target._split(X, Y, num_features, True)
    print(dict_split)
    print(dict_split['index'])
    print(dict_split['value'])
    print(dict_split['left_right'])

    assert isinstance(dict_split, dict)
    assert dict_split['index'] == 3
    assert dict_split['value'] == 37.159879341119996

    X = np.ones(X.shape)

    dict_split = package_target._split(X, Y, num_features, True)
    print(dict_split)
    print(dict_split['index'])
    print(dict_split['value'])
    print(dict_split['left_right'])

    assert isinstance(dict_split, dict)
    assert dict_split['index'] == 0
    assert dict_split['value'] == 1.0

def test_split_typing():
    annos = package_target.split.__annotations__

    assert annos['node'] == dict
    assert annos['depth_max'] == int
    assert annos['size_min_leaf'] == int
    assert annos['num_features'] == int
    assert annos['split_random_location'] == bool
    assert annos['cur_depth'] == int
    assert annos['return'] == constants.TYPE_NONE

def test_split():
    np.random.seed(42)

    X = np.reshape(np.arange(0, 40), (10, 4))
    Y = np.random.randn(10, 1)
    depth_max = 4
    size_min_leaf = 2
    num_features = 2
    split_random_location = False

    node = package_target._split(X, Y, num_features, split_random_location)

    with pytest.raises(AssertionError) as error:
        package_target.split(node, depth_max, size_min_leaf, num_features, split_random_location, 'abc')
    with pytest.raises(AssertionError) as error:
        package_target.split(node, depth_max, size_min_leaf, num_features, split_random_location, 1.0)
    with pytest.raises(AssertionError) as error:
        package_target.split(node, depth_max, size_min_leaf, num_features, 'abc', 1)
    with pytest.raises(AssertionError) as error:
        package_target.split(node, depth_max, size_min_leaf, 'abc', split_random_location, 1)
    with pytest.raises(AssertionError) as error:
        package_target.split(node, depth_max, 'abc', num_features, split_random_location, 1)
    with pytest.raises(AssertionError) as error:
        package_target.split(node, 'abc', size_min_leaf, num_features, split_random_location, 1)
    with pytest.raises(AssertionError) as error:
        package_target.split(X, depth_max, size_min_leaf, num_features, split_random_location, 1)
    with pytest.raises(AssertionError) as error:
        package_target.split('abc', depth_max, size_min_leaf, num_features, split_random_location, 1)

    package_target.split(node, depth_max, size_min_leaf, num_features, split_random_location, 1)
    assert isinstance(node, dict)

def test__predict_by_tree_typing():
    annos = package_target._predict_by_tree.__annotations__

    assert annos['bx'] == np.ndarray
    assert annos['tree'] == dict
    assert annos['return'] == constants.TYPING_TUPLE_TWO_FLOATS

def test__predict_by_tree():
    np.random.seed(42)

    X = np.reshape(np.arange(0, 40), (10, 4))
    Y = np.random.randn(10, 1)
    depth_max = 4
    size_min_leaf = 2
    num_features = 2
    split_random_location = True

    node = package_target._split(X, Y, num_features, split_random_location)
    package_target.split(node, depth_max, size_min_leaf, num_features, split_random_location, 1)

    with pytest.raises(AssertionError) as error:
        package_target._predict_by_tree(np.array([4.0, 2.0, 3.0, 1.0]), 'abc')
    with pytest.raises(AssertionError) as error:
        package_target._predict_by_tree(X, node)
    with pytest.raises(AssertionError) as error:
        package_target._predict_by_tree('abc', node)

    mean, std = package_target._predict_by_tree(np.array([4.0, 2.0, 3.0, 1.0]), node)
    print(mean)
    print(std)

    assert mean == 0.179224925920024
    assert std == 0.31748922709120864

def test__predict_by_trees_typing():
    annos = package_target._predict_by_trees.__annotations__

    assert annos['bx'] == np.ndarray
    assert annos['list_trees'] == list
    assert annos['return'] == constants.TYPING_TUPLE_TWO_FLOATS

def test__predict_by_trees():
    np.random.seed(42)

    X = np.reshape(np.arange(0, 40), (10, 4))
    Y = np.random.randn(10, 1)
    depth_max = 4
    size_min_leaf = 2
    num_features = 2
    split_random_location = True

    node_1 = package_target._split(X, Y, num_features, split_random_location)
    package_target.split(node_1, depth_max, size_min_leaf, num_features, split_random_location, 1)

    node_2 = package_target._split(X, Y, num_features, split_random_location)
    package_target.split(node_2, depth_max, size_min_leaf, num_features, split_random_location, 1)

    node_3 = package_target._split(X, Y, num_features, split_random_location)
    package_target.split(node_3, depth_max, size_min_leaf, num_features, split_random_location, 1)

    list_trees = [node_1, node_2, node_3]

    with pytest.raises(AssertionError) as error:
        package_target._predict_by_trees(np.array([4.0, 2.0, 3.0, 1.0]), node_1)
    with pytest.raises(AssertionError) as error:
        package_target._predict_by_trees(np.array([4.0, 2.0, 3.0, 1.0]), 'abc')
    with pytest.raises(AssertionError) as error:
        package_target._predict_by_trees(X, list_trees)
    with pytest.raises(AssertionError) as error:
        package_target._predict_by_trees('abc', list_trees)

    mean, std = package_target._predict_by_trees(np.array([4.0, 2.0, 3.0, 1.0]), list_trees)
    print(mean)
    print(std)

    assert mean == 0.12544669602080652
    assert std == 0.3333040901154691

def test_predict_by_trees_typing():
    annos = package_target.predict_by_trees.__annotations__

    assert annos['X'] == np.ndarray
    assert annos['list_trees'] == list
    assert annos['return'] == constants.TYPING_TUPLE_TWO_ARRAYS

def test_predict_by_trees():
    np.random.seed(42)

    X = np.reshape(np.arange(0, 40), (10, 4))
    Y = np.random.randn(10, 1)
    depth_max = 4
    size_min_leaf = 2
    num_features = 2
    split_random_location = True

    node_1 = package_target._split(X, Y, num_features, split_random_location)
    package_target.split(node_1, depth_max, size_min_leaf, num_features, split_random_location, 1)

    node_2 = package_target._split(X, Y, num_features, split_random_location)
    package_target.split(node_2, depth_max, size_min_leaf, num_features, split_random_location, 1)

    node_3 = package_target._split(X, Y, num_features, split_random_location)
    package_target.split(node_3, depth_max, size_min_leaf, num_features, split_random_location, 1)

    list_trees = [node_1, node_2, node_3]

    with pytest.raises(AssertionError) as error:
        package_target.predict_by_trees(np.array([4.0, 2.0, 3.0, 1.0]), node_1)
    with pytest.raises(AssertionError) as error:
        package_target.predict_by_trees(np.array([4.0, 2.0, 3.0, 1.0]), 'abc')
    with pytest.raises(AssertionError) as error:
        package_target.predict_by_trees(np.array([4.0, 2.0, 3.0, 1.0]), list_trees)
    with pytest.raises(AssertionError) as error:
        package_target.predict_by_trees('abc', list_trees)

    means, stds = package_target.predict_by_trees(X, list_trees)
    print(means)
    print(stds)

    means_truth = np.array([
        [0.33710618],
        [0.1254467],
        [0.68947573],
        [0.9866563],
        [0.16257799],
        [0.16258346],
        [0.85148454],
        [0.55084716],
        [0.30721653],
        [0.30721653],
    ])

    stds_truth = np.array([
        [0.29842457],
        [0.33330409],
        [0.44398864],
        [0.72536523],
        [0.74232577],
        [0.74232284],
        [0.83388663],
        [0.5615399],
        [0.64331582],
        [0.64331582],
    ])

    assert isinstance(means, np.ndarray)
    assert isinstance(stds, np.ndarray)
    assert len(means.shape) == 2
    assert len(stds.shape) == 2
    assert means.shape[0] == stds.shape[0] == X.shape[0]
    assert means.shape[1] == stds.shape[1] == 1

    assert np.all(np.abs(means - means_truth) < TEST_EPSILON)
    assert np.all(np.abs(stds - stds_truth) < TEST_EPSILON)

    X = np.random.randn(1000, 4)

    means, stds = package_target.predict_by_trees(X, list_trees)

    assert isinstance(means, np.ndarray)
    assert isinstance(stds, np.ndarray)
    assert len(means.shape) == 2
    assert len(stds.shape) == 2
    assert means.shape[0] == stds.shape[0] == X.shape[0]
    assert means.shape[1] == stds.shape[1] == 1

def test_compute_sigma_typing():
    annos = package_target.compute_sigma.__annotations__

    assert annos['preds_mu_leaf'] == np.ndarray
    assert annos['preds_sigma_leaf'] == np.ndarray
    assert annos['min_sigma'] == float
    assert annos['return'] == np.ndarray

def test_compute_sigma():
    means_leaf = np.array([
        1.0,
        2.0,
        3.0,
        9.0,
        8.0,
        4.0,
        5.0,
        6.0,
        7.0,
        10.0,
    ])

    stds_leaf = np.array([
        -1.0,
        0.0,
        1.0,
        2.0,
        1.0,
        1.0,
        4.0,
        3.0,
        4.0,
        -2.0,
    ])
    min_sigma = 0.0

    with pytest.raises(AssertionError) as error:
        package_target.compute_sigma(means_leaf, stds_leaf, min_sigma='abc')
    with pytest.raises(AssertionError) as error:
        package_target.compute_sigma(means_leaf, stds_leaf, min_sigma=4)
    with pytest.raises(AssertionError) as error:
        package_target.compute_sigma(means_leaf, 'abc', min_sigma=min_sigma)
    with pytest.raises(AssertionError) as error:
        package_target.compute_sigma(means_leaf, np.array([[1.0], [2.0], [1.0]]), min_sigma=min_sigma)
    with pytest.raises(AssertionError) as error:
        package_target.compute_sigma('abc', stds_leaf, min_sigma=min_sigma)
    with pytest.raises(AssertionError) as error:
        package_target.compute_sigma(np.array([[1.0], [2.0], [1.0]]), stds_leaf, min_sigma=min_sigma)

    sigma = package_target.compute_sigma(means_leaf, stds_leaf, min_sigma=min_sigma)
    print(sigma)

    sigma_truth = np.mean(means_leaf**2 + np.maximum(stds_leaf, min_sigma)**2)
    print(sigma_truth)
    sigma_truth -= np.mean(means_leaf)**2
    print(sigma_truth)
    sigma_truth = np.sqrt(sigma_truth)
    print(sigma_truth)

    assert sigma == sigma_truth == 3.612478373637688
