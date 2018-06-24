# test_bo
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: June 24, 2018

import numpy as np
import pytest

from bayeso import bo
from bayeso.utils import utils_bo


TEST_EPSILON = 1e-5


def test_get_grid():
    arr_range_1 = np.array([
        [0.0, 10.0],
        [-2.0, 2.0],
        [-5.0, 5.0],
    ])
    arr_range_2 = np.array([
        [0.0, 10.0],
        [2.0, 2.0],
        [5.0, 5.0],
    ])

    truth_arr_grid_1 = np.array([
        [0., -2., -5.],
        [0., -2., 0.],
        [0., -2., 5.],
        [5., -2., -5.],
        [5., -2., 0.],
        [5., -2., 5.],
        [10., -2., -5.],
        [10., -2., 0.],
        [10., -2., 5.],
        [0., 0., -5.],
        [0., 0., 0.],
        [0., 0., 5.],
        [5., 0., -5.],
        [5., 0., 0.],
        [5., 0., 5.],
        [10., 0., -5.],
        [10., 0., 0.],
        [10., 0., 5.],
        [0., 2., -5.],
        [0., 2., 0.],
        [0., 2., 5.],
        [5., 2., -5.],
        [5., 2., 0.],
        [5., 2., 5.],
        [10., 2., -5.],
        [10., 2., 0.],
        [10., 2., 5.],
    ])
    truth_arr_grid_2 = np.array([
        [0., 2., 5.],
        [0., 2., 5.],
        [0., 2., 5.],
        [5., 2., 5.],
        [5., 2., 5.],
        [5., 2., 5.],
        [10., 2., 5.],
        [10., 2., 5.],
        [10., 2., 5.],
        [0., 2., 5.],
        [0., 2., 5.],
        [0., 2., 5.],
        [5., 2., 5.],
        [5., 2., 5.],
        [5., 2., 5.],
        [10., 2., 5.],
        [10., 2., 5.],
        [10., 2., 5.],
        [0., 2., 5.],
        [0., 2., 5.],
        [0., 2., 5.],
        [5., 2., 5.],
        [5., 2., 5.],
        [5., 2., 5.],
        [10., 2., 5.],
        [10., 2., 5.],
        [10., 2., 5.],
    ])

    arr_grid_1 = utils_bo.get_grid(arr_range_1, 3)
    arr_grid_2 = utils_bo.get_grid(arr_range_2, 3)

    assert (arr_grid_1 == truth_arr_grid_1).all()
    assert (arr_grid_2 == truth_arr_grid_2).all()

def test_get_best_acquisition():
    fun_objective = lambda x: x**2 - 2.0 * x + 1.0
    arr_initials = np.expand_dims(np.arange(-5, 5), axis=1)

    with pytest.raises(AssertionError) as error:
        utils_bo.get_best_acquisition(1, fun_objective)
    with pytest.raises(AssertionError) as error:
        utils_bo.get_best_acquisition(arr_initials, None)
    with pytest.raises(AssertionError) as error:
        utils_bo.get_best_acquisition(np.arange(-5, 5), fun_objective)

    best_initial = utils_bo.get_best_acquisition(arr_initials, fun_objective)
    assert len(best_initial.shape)
    assert best_initial.shape[0] == 1
    assert best_initial.shape[1] == arr_initials.shape[1]
    assert best_initial == np.array([[1]])

def test_optimize_many_():
    np.random.seed(42)
    arr_range = np.array([
        [-5.0, 5.0],
    ])
    dim_X = arr_range.shape[0]
    num_X = 3
    num_iter = 10
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)
    fun_target = lambda x: x**2 - 2.0 * x + 1.0
    model_bo = bo.BO(arr_range)

    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many_(1, fun_target, X, Y, num_iter)
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many_(model_bo, 1, X, Y, num_iter)
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many_(model_bo, fun_target, 1, Y, num_iter)
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many_(model_bo, fun_target, X, 1, num_iter)
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many_(model_bo, fun_target, X, Y, 'abc')
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many_(model_bo, fun_target, np.random.randn(num_X), Y, num_iter)
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many_(model_bo, fun_target, X, np.random.randn(num_X), num_iter)
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many_(model_bo, fun_target, np.random.randn(2, dim_X), Y, num_iter)
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many_(model_bo, fun_target, X, np.random.randn(num_X, 2), num_iter)
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many_(model_bo, fun_target, X, Y, num_iter, str_initial_method_optimizer=1)
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many_(model_bo, fun_target, X, Y, num_iter, str_initial_method_optimizer='abc')
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many_(model_bo, fun_target, X, Y, num_iter, int_samples_ao='abc')

    X_final, Y_final = utils_bo.optimize_many_(model_bo, fun_target, X, Y, num_iter)
    assert X_final.shape[1] == X.shape[1] == dim_X
    assert X_final.shape[0] == Y_final.shape[0] == num_X + num_iter
    assert Y_final.shape[1] == 1

def test_optimize_many():
    np.random.seed(42)
    arr_range = np.array([
        [-5.0, 5.0],
    ])
    dim_X = arr_range.shape[0]
    num_X = 3
    num_iter = 10
    X = np.random.randn(num_X, dim_X)
    fun_target = lambda x: x**2 - 2.0 * x + 1.0
    model_bo = bo.BO(arr_range)

    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many(1, fun_target, X, num_iter)
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many(model_bo, 1, X, num_iter)
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many(model_bo, fun_target, 1, num_iter)
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many(model_bo, fun_target, X, 1.2)
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many(model_bo, fun_target, np.random.randn(num_X), num_iter)
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many(model_bo, fun_target, X, num_iter, str_initial_method_optimizer=1)
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many(model_bo, fun_target, X, num_iter, str_initial_method_optimizer='abc')
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many(model_bo, fun_target, X, num_iter, int_samples_ao='abc')

    X_final, Y_final = utils_bo.optimize_many(model_bo, fun_target, X, num_iter)
    assert X_final.shape[1] == X.shape[1] == dim_X
    assert X_final.shape[0] == Y_final.shape[0] == num_X + num_iter
    assert Y_final.shape[1] == 1

def test_optimize_many_with_random_init():
    np.random.seed(42)
    arr_range = np.array([
        [-5.0, 5.0],
    ])
    dim_X = arr_range.shape[0]
    num_X = 3
    num_iter = 10
    fun_target = lambda x: x**2 - 2.0 * x + 1.0
    model_bo = bo.BO(arr_range)

    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many_with_random_init(1, fun_target, num_X, num_iter)
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many_with_random_init(model_bo, 1, num_X, num_iter)
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many_with_random_init(model_bo, fun_target, 1.2, num_iter)
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many_with_random_init(model_bo, fun_target, num_X, 1.2)
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many_with_random_init(model_bo, fun_target, num_X, num_iter, str_initial_method_bo=1)
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many_with_random_init(model_bo, fun_target, num_X, num_iter, str_initial_method_bo='abc')
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many_with_random_init(model_bo, fun_target, num_X, num_iter, str_initial_method_bo='grid')
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many_with_random_init(model_bo, fun_target, num_X, num_iter, str_initial_method_optimizer=1)
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many_with_random_init(model_bo, fun_target, num_X, num_iter, str_initial_method_optimizer='abc')
    with pytest.raises(AssertionError) as error:
        utils_bo.optimize_many_with_random_init(model_bo, fun_target, num_X, num_iter, int_seed=1.2)

    X_final, Y_final = utils_bo.optimize_many_with_random_init(model_bo, fun_target, num_X, num_iter, str_initial_method_bo='uniform')
    assert X_final.shape[1] == dim_X
    assert X_final.shape[0] == Y_final.shape[0] == num_X + num_iter
    assert Y_final.shape[1] == 1
