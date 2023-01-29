#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: December 19, 2021
#
"""test_thompson_sampling"""

import pytest
import numpy as np

from bayeso import thompson_sampling as package_target
from bayeso import constants


TEST_EPSILON = 1e-5

def test_thompson_sampling_gp_iteration_typing():
    annos = package_target.thompson_sampling_gp_iteration.__annotations__

    assert annos['range_X'] == np.ndarray
    assert annos['X'] == np.ndarray
    assert annos['Y'] == np.ndarray
    assert annos['normalize_Y'] == bool
    assert annos['str_sampling_method'] == str
    assert annos['num_samples'] == int
    assert annos['debug'] == bool
    assert annos['return'] == np.ndarray

def test_thompson_sampling_gp_iteration():
    np.random.seed(42)
    arr_range = np.array([
        [0.0, 10.0],
        [-2.0, 2.0],
        [-5.0, 5.0],
    ])
    dim_X = arr_range.shape[0]
    num_X = 5
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)
    normalize_Y = True
    str_sampling_method = 'uniform'
    num_samples = 10
    debug = True

    with pytest.raises(AssertionError) as error:
        package_target.thompson_sampling_gp_iteration(arr_range, X, Y, normalize_Y=normalize_Y, str_sampling_method=str_sampling_method, num_samples=num_samples, debug=123)
    with pytest.raises(AssertionError) as error:
        package_target.thompson_sampling_gp_iteration(arr_range, X, Y, normalize_Y=normalize_Y, str_sampling_method=str_sampling_method, num_samples=num_samples, debug='abc')
    with pytest.raises(AssertionError) as error:
        package_target.thompson_sampling_gp_iteration(arr_range, X, Y, normalize_Y=normalize_Y, str_sampling_method=str_sampling_method, num_samples='abc', debug=debug)
    with pytest.raises(AssertionError) as error:
        package_target.thompson_sampling_gp_iteration(arr_range, X, Y, normalize_Y=normalize_Y, str_sampling_method=str_sampling_method, num_samples=1.23, debug=debug)
    with pytest.raises(AssertionError) as error:
        package_target.thompson_sampling_gp_iteration(arr_range, X, Y, normalize_Y=normalize_Y, str_sampling_method='123', num_samples=num_samples, debug=debug)
    with pytest.raises(AssertionError) as error:
        package_target.thompson_sampling_gp_iteration(arr_range, X, Y, normalize_Y=normalize_Y, str_sampling_method=123, num_samples=num_samples, debug=debug)
    with pytest.raises(AssertionError) as error:
        package_target.thompson_sampling_gp_iteration(arr_range, X, Y, normalize_Y=123, str_sampling_method=str_sampling_method, num_samples=num_samples, debug=debug)
    with pytest.raises(AssertionError) as error:
        package_target.thompson_sampling_gp_iteration(arr_range, X, Y, normalize_Y='abc', str_sampling_method=str_sampling_method, num_samples=num_samples, debug=debug)
    with pytest.raises(AssertionError) as error:
        package_target.thompson_sampling_gp_iteration(arr_range, X, 123, normalize_Y=normalize_Y, str_sampling_method=str_sampling_method, num_samples=num_samples, debug=debug)
    with pytest.raises(AssertionError) as error:
        package_target.thompson_sampling_gp_iteration(arr_range, 123, Y, normalize_Y=normalize_Y, str_sampling_method=str_sampling_method, num_samples=num_samples, debug=debug)
    with pytest.raises(AssertionError) as error:
        package_target.thompson_sampling_gp_iteration(arr_range, X, Y[:3], normalize_Y=normalize_Y, str_sampling_method=str_sampling_method, num_samples=num_samples, debug=debug)
    with pytest.raises(AssertionError) as error:
        package_target.thompson_sampling_gp_iteration(arr_range, X[:3], Y, normalize_Y=normalize_Y, str_sampling_method=str_sampling_method, num_samples=num_samples, debug=debug)
    with pytest.raises(AssertionError) as error:
        package_target.thompson_sampling_gp_iteration(arr_range[:2], X, Y, normalize_Y=normalize_Y, str_sampling_method=str_sampling_method, num_samples=num_samples, debug=debug)

    next_point = package_target.thompson_sampling_gp_iteration(arr_range, X, Y, normalize_Y=normalize_Y, str_sampling_method=str_sampling_method, num_samples=num_samples, debug=debug)

    assert isinstance(next_point, np.ndarray)
    assert len(next_point.shape) == 1
    assert next_point.shape[0] == dim_X

def test_thompson_sampling_gp_typing():
    annos = package_target.thompson_sampling_gp.__annotations__

    assert annos['range_X'] == np.ndarray
    assert annos['num_init'] == int
    assert annos['num_iter'] == int
    assert annos['fun_target'] == callable
    assert annos['normalize_Y'] == bool
    assert annos['str_sampling_method'] == str
    assert annos['num_samples'] == int
    assert annos['debug'] == bool
    assert annos['return'] == constants.TYPING_TUPLE_TWO_ARRAYS

def test_thompson_sampling_gp():
    np.random.seed(42)
    arr_range = np.array([
        [0.0, 10.0],
        [-2.0, 2.0],
        [-5.0, 5.0],
    ])
    dim_X = arr_range.shape[0]
    num_init = 2
    num_iter = 2
    fun_objective = lambda X: np.sum(X)
    normalize_Y = True
    str_sampling_method = 'halton'
    num_samples = 10
    debug = True

    with pytest.raises(AssertionError) as error:
        package_target.thompson_sampling_gp(arr_range, fun_objective, num_init, num_iter, normalize_Y=normalize_Y, str_sampling_method=str_sampling_method, num_samples=num_samples, debug=123)
    with pytest.raises(AssertionError) as error:
        package_target.thompson_sampling_gp(arr_range, fun_objective, num_init, num_iter, normalize_Y=normalize_Y, str_sampling_method=str_sampling_method, num_samples=num_samples, debug='abc')
    with pytest.raises(AssertionError) as error:
        package_target.thompson_sampling_gp(arr_range, fun_objective, num_init, num_iter, normalize_Y=normalize_Y, str_sampling_method=str_sampling_method, num_samples='abc', debug=debug)
    with pytest.raises(AssertionError) as error:
        package_target.thompson_sampling_gp(arr_range, fun_objective, num_init, num_iter, normalize_Y=normalize_Y, str_sampling_method=123, num_samples=num_samples, debug=debug)
    with pytest.raises(AssertionError) as error:
        package_target.thompson_sampling_gp(arr_range, fun_objective, num_init, num_iter, normalize_Y=123, str_sampling_method=str_sampling_method, num_samples=num_samples, debug=debug)
    with pytest.raises(AssertionError) as error:
        package_target.thompson_sampling_gp(arr_range, 'abc', num_init, num_iter, normalize_Y=normalize_Y, str_sampling_method=str_sampling_method, num_samples=num_samples, debug=debug)
    with pytest.raises(AssertionError) as error:
        package_target.thompson_sampling_gp(arr_range, 123, num_init, num_iter, normalize_Y=normalize_Y, str_sampling_method=str_sampling_method, num_samples=num_samples, debug=debug)
    with pytest.raises(AssertionError) as error:
        package_target.thompson_sampling_gp(arr_range, fun_objective, num_init, 1.23, normalize_Y=normalize_Y, str_sampling_method=str_sampling_method, num_samples=num_samples, debug=debug)
    with pytest.raises(AssertionError) as error:
        package_target.thompson_sampling_gp(arr_range, fun_objective, 'abc', num_iter, normalize_Y=normalize_Y, str_sampling_method=str_sampling_method, num_samples=num_samples, debug=debug)
    with pytest.raises(AssertionError) as error:
        package_target.thompson_sampling_gp(123, fun_objective, num_init, num_iter, normalize_Y=normalize_Y, str_sampling_method=str_sampling_method, num_samples=num_samples, debug=debug)
    with pytest.raises(AssertionError) as error:
        package_target.thompson_sampling_gp(np.array([1.0, 2.0, 3.0]), fun_objective, num_init, num_iter, normalize_Y=normalize_Y, str_sampling_method=str_sampling_method, num_samples=num_samples, debug=debug)

    X_, Y_ = package_target.thompson_sampling_gp(arr_range, fun_objective, num_init, num_iter, normalize_Y=normalize_Y, str_sampling_method=str_sampling_method, num_samples=num_samples, debug=debug)

    assert isinstance(X_, np.ndarray)
    assert isinstance(Y_, np.ndarray)
    assert len(X_.shape) == 2
    assert len(Y_.shape) == 2
    assert X_.shape[0] == Y_.shape[0] == num_init + num_iter
    assert X_.shape[1] == dim_X
    assert Y_.shape[1] == 1
