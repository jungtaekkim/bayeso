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
