# test_bo
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: June 23, 2018

import numpy as np
import pytest

from bayeso import bo


TEST_EPSILON = 1e-5

def test_load_bo():
    # legitimate cases
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
    # wrong cases
    arr_range_3 = np.array([
        [20.0, 10.0],
        [-2.0, 2.0],
        [-5.0, 5.0],
    ])
    arr_range_4 = np.array([
        [20.0, 10.0],
        [4.0, 2.0],
        [10.0, 5.0],
    ])

    with pytest.raises(AssertionError) as error:
        model_bo = bo.BO(1)
    with pytest.raises(AssertionError) as error:
        model_bo = bo.BO(np.arange(0, 10))
    with pytest.raises(AssertionError) as error:
        model_bo = bo.BO(arr_range_3)
    with pytest.raises(AssertionError) as error:
        model_bo = bo.BO(arr_range_4)
    with pytest.raises(AssertionError) as error:
        model_bo = bo.BO(arr_range_1, str_cov=1)
    with pytest.raises(AssertionError) as error:
        model_bo = bo.BO(arr_range_1, str_cov='abc')
    with pytest.raises(AssertionError) as error:
        model_bo = bo.BO(arr_range_1, str_acq=1)
    with pytest.raises(AssertionError) as error:
        model_bo = bo.BO(arr_range_1, str_acq='abc')
    with pytest.raises(AssertionError) as error:
        model_bo = bo.BO(arr_range_1, is_ard=1)
    with pytest.raises(AssertionError) as error:
        model_bo = bo.BO(arr_range_1, prior_mu=1)
    with pytest.raises(AssertionError) as error:
        model_bo = bo.BO(arr_range_1, debug=1)

    model_bo = bo.BO(arr_range_1)
    model_bo = bo.BO(arr_range_2)

def test_get_initial():
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
    fun_objective = lambda X: np.sum(X)
    model_bo = bo.BO(arr_range)

    with pytest.raises(AssertionError) as error:
        model_bo.get_initial(1)
    with pytest.raises(AssertionError) as error:
        model_bo.get_initial('grid', fun_objective=None)
    with pytest.raises(AssertionError) as error:
        model_bo.get_initial('uniform', fun_objective=1)
    with pytest.raises(AssertionError) as error:
        model_bo.get_initial('uniform', int_samples='abc')
    with pytest.raises(AssertionError) as error:
        model_bo.get_initial('uniform', int_seed='abc')
    with pytest.raises(NotImplementedError) as error:
        model_bo.get_initial('abc')

    arr_initials = model_bo.get_initial('grid', fun_objective=fun_objective)
    truth_arr_initials = np.array([
        [0.0, -2.0, -5.0],
    ])
    assert (np.abs(arr_initials - truth_arr_initials) < TEST_EPSILON).all()

    arr_initials = model_bo.get_initial('sobol', int_samples=3)
    arr_initials = model_bo.get_initial('sobol', int_samples=3, int_seed=42)
    truth_arr_initials = np.array([
        [5.0, 0.0, 0.0],
        [7.5, -1.0, 2.5],
        [2.5, 1.0, -2.5],
    ])
    assert (np.abs(arr_initials - truth_arr_initials) < TEST_EPSILON).all()

    arr_initials = model_bo.get_initial('uniform', int_samples=3, int_seed=42)
    truth_arr_initials = np.array([
        [3.74540119, 1.80285723, 2.31993942],
        [5.98658484, -1.37592544, -3.4400548],
        [0.58083612, 1.46470458, 1.01115012],
    ])
    assert (np.abs(arr_initials - truth_arr_initials) < TEST_EPSILON).all()

def test_optimize():
    np.random.seed(42)
    arr_range_1 = np.array([
        [0.0, 10.0],
        [-2.0, 2.0],
        [-5.0, 5.0],
    ])
    dim_X = arr_range_1.shape[0]
    num_X = 5
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)
    model_bo = bo.BO(arr_range_1)

    with pytest.raises(AssertionError) as error:
        model_bo.optimize(1, Y)
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(X, 1)
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(np.random.randn(num_X), Y)
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(np.random.randn(num_X, 1), Y)
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(X, np.random.randn(num_X))
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(X, np.random.randn(num_X, 2))
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(X, np.random.randn(3, 1))
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(X, Y, str_initial_method=1)
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(X, Y, str_initial_method='abc')
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(X, Y, int_samples='abc')
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(X, Y, is_normalized='abc')

    next_point, next_points, acquisitions, cov_X_X, inv_cov_X_X, hyps = model_bo.optimize(X, Y)
    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

    model_bo = bo.BO(arr_range_1, str_acq='pi')
    next_point, next_points, acquisitions, cov_X_X, inv_cov_X_X, hyps = model_bo.optimize(X, Y)
    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

    model_bo = bo.BO(arr_range_1, str_acq='ucb')
    next_point, next_points, acquisitions, cov_X_X, inv_cov_X_X, hyps = model_bo.optimize(X, Y)
    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]
