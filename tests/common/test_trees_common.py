#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 17, 2021
#
"""test_trees_common"""

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
