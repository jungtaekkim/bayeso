#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 6, 2021
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
