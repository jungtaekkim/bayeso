#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 8, 2021
#
"""test_wrappers_bo_class"""

import pytest
import numpy as np

from bayeso.wrappers import wrappers_bo_class as package_target


TEST_EPSILON = 1e-5

def test_load_bayesian_optimization():
    # legitimate cases
    range_1 = np.array([
        [0.0, 10.0],
        [-2.0, 2.0],
        [-5.0, 5.0],
    ])
    range_2 = np.array([
        [0.0, 10.0],
        [2.0, 2.0],
        [5.0, 5.0],
    ])
    # wrong cases
    range_3 = np.array([
        [20.0, 10.0],
        [-2.0, 2.0],
        [-5.0, 5.0],
    ])
    range_4 = np.array([
        [20.0, 10.0],
        [4.0, 2.0],
        [10.0, 5.0],
    ])

    fun_target = lambda X: np.linalg.norm(X, axis=1)[..., np.newaxis]
    num_iter = 20

    model_bo = package_target.BayesianOptimization(range_1, fun_target, num_iter)
    model_bo = package_target.BayesianOptimization(range_2, fun_target, num_iter, debug=True)

    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BayesianOptimization('abc', fun_target, num_iter)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BayesianOptimization(1, fun_target, num_iter)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BayesianOptimization(range_3, fun_target, num_iter)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BayesianOptimization(range_4, fun_target, num_iter)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BayesianOptimization(range_1, 123, num_iter)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BayesianOptimization(range_1, fun_target, 'abc')
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BayesianOptimization(range_1, fun_target, num_iter, str_cov=123)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BayesianOptimization(range_1, fun_target, num_iter, str_acq=123)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BayesianOptimization(range_1, fun_target, num_iter, normalize_Y=1)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BayesianOptimization(range_1, fun_target, num_iter, use_ard=1)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BayesianOptimization(range_1, fun_target, num_iter, prior_mu='abc')
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BayesianOptimization(range_1, fun_target, num_iter, str_initial_method_bo=123)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BayesianOptimization(range_1, fun_target, num_iter, str_sampling_method_ao=123)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BayesianOptimization(range_1, fun_target, num_iter, str_optimizer_method_gp=123)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BayesianOptimization(range_1, fun_target, num_iter, str_optimizer_method_bo=123)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BayesianOptimization(range_1, fun_target, num_iter, str_mlm_method=123)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BayesianOptimization(range_1, fun_target, num_iter, str_modelselection_method=123)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BayesianOptimization(range_1, fun_target, num_iter, num_samples_ao='abc')
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BayesianOptimization(range_1, fun_target, num_iter, debug=123)

def test_optimize_with_all_initial_information():
    np.random.seed(42)
    range_ = np.array([
        [-5.0, 5.0],
    ])
    dim_X = range_.shape[0]
    num_init = 3
    num_iter = 5
    X = np.random.randn(num_init, dim_X)
    Y = np.random.randn(num_init, 1)
    fun_target = lambda x: 2.0 * x + 1.0

    model_bo = package_target.BayesianOptimization(range_, fun_target, num_iter, debug=True)

    with pytest.raises(AssertionError) as error:
        model_bo.optimize_with_all_initial_information(X, 'abc')
    with pytest.raises(AssertionError) as error:
        model_bo.optimize_with_all_initial_information('abc', Y)

    X_, Y_, time_all_, time_surrogate_, time_acq_ = model_bo.optimize_with_all_initial_information(X, Y)

    assert len(X_.shape) == 2
    assert len(Y_.shape) == 2
    assert len(time_all_.shape) == 1
    assert len(time_surrogate_.shape) == 1
    assert len(time_acq_.shape) == 1
    assert X_.shape[1] == dim_X
    assert Y_.shape[1] == 1
    assert X_.shape[0] == Y_.shape[0] == num_init + num_iter
    assert time_all_.shape[0] == num_iter
    assert time_surrogate_.shape[0] == time_acq_.shape[0] == num_iter

def test_optimize_with_initial_inputs():
    np.random.seed(42)
    range_ = np.array([
        [-5.0, 5.0],
    ])
    dim_X = range_.shape[0]
    num_init = 3
    num_iter = 5
    X = np.random.randn(num_init, dim_X)
    fun_target = lambda x: 2.0 * x + 1.0

    model_bo = package_target.BayesianOptimization(range_, fun_target, num_iter, debug=True)

    with pytest.raises(AssertionError) as error:
        model_bo.optimize_with_initial_inputs('abc')
    with pytest.raises(AssertionError) as error:
        model_bo.optimize_with_initial_inputs(123)

    X_, Y_, time_all_, time_surrogate_, time_acq_ = model_bo.optimize_with_initial_inputs(X)

    assert len(X_.shape) == 2
    assert len(Y_.shape) == 2
    assert len(time_all_.shape) == 1
    assert len(time_surrogate_.shape) == 1
    assert len(time_acq_.shape) == 1
    assert X_.shape[1] == dim_X
    assert Y_.shape[1] == 1
    assert X_.shape[0] == Y_.shape[0] == num_init + num_iter
    assert time_all_.shape[0] == num_init + num_iter
    assert time_surrogate_.shape[0] == time_acq_.shape[0] == num_iter

def test_optimize():
    np.random.seed(42)
    range_ = np.array([
        [-5.0, 5.0],
    ])
    dim_X = range_.shape[0]
    num_init = 3
    num_iter = 5
    fun_target = lambda x: 2.0 * x + 1.0

    model_bo = package_target.BayesianOptimization(range_, fun_target, num_iter, debug=True)

    with pytest.raises(AssertionError) as error:
        model_bo.optimize('abc')
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(num_init, seed='abc')

    X_, Y_, time_all_, time_surrogate_, time_acq_ = model_bo.optimize(num_init)

    assert len(X_.shape) == 2
    assert len(Y_.shape) == 2
    assert len(time_all_.shape) == 1
    assert len(time_surrogate_.shape) == 1
    assert len(time_acq_.shape) == 1
    assert X_.shape[1] == dim_X
    assert Y_.shape[1] == 1
    assert X_.shape[0] == Y_.shape[0] == num_init + num_iter
    assert time_all_.shape[0] == num_init + num_iter
    assert time_surrogate_.shape[0] == time_acq_.shape[0] == num_iter
