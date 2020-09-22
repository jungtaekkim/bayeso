# test_wrappers_bo
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: September 21, 2020

import pytest
import numpy as np
import typing

from bayeso import bo
from bayeso.wrappers import wrappers_bo


def test_run_single_round_with_all_initial_information_typing():
    annos = wrappers_bo.run_single_round_with_all_initial_information.__annotations__

    assert annos['model_bo'] == bo.BO
    assert annos['fun_target'] == callable
    assert annos['X_train'] == np.ndarray
    assert annos['Y_train'] == np.ndarray
    assert annos['num_iter'] == int
    assert annos['str_sampling_method_ao'] == str
    assert annos['num_samples_ao'] == int
    assert annos['str_mlm_method'] == str
    assert annos['return'] == typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]

def test_run_single_round_with_all_initial_information():
    np.random.seed(42)
    arr_range = np.array([
        [-5.0, 5.0],
    ])
    dim_X = arr_range.shape[0]
    num_X = 3
    num_iter = 10
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)
    fun_target = lambda x: 2.0 * x + 1.0
    model_bo = bo.BO(arr_range)

    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round_with_all_initial_information(1, fun_target, X, Y, num_iter)
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round_with_all_initial_information(model_bo, 1, X, Y, num_iter)
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round_with_all_initial_information(model_bo, fun_target, 1, Y, num_iter)
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round_with_all_initial_information(model_bo, fun_target, X, 1, num_iter)
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round_with_all_initial_information(model_bo, fun_target, X, Y, 'abc')
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round_with_all_initial_information(model_bo, fun_target, np.random.randn(num_X), Y, num_iter)
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round_with_all_initial_information(model_bo, fun_target, X, np.random.randn(num_X), num_iter)
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round_with_all_initial_information(model_bo, fun_target, np.random.randn(2, dim_X), Y, num_iter)
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round_with_all_initial_information(model_bo, fun_target, X, np.random.randn(num_X, 2), num_iter)
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round_with_all_initial_information(model_bo, fun_target, X, Y, num_iter, str_sampling_method_ao=1)
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round_with_all_initial_information(model_bo, fun_target, X, Y, num_iter, str_sampling_method_ao='abc')
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round_with_all_initial_information(model_bo, fun_target, X, Y, num_iter, num_samples_ao='abc')

    X_final, Y_final, time_all_final, time_gp_final, time_acq_final = wrappers_bo.run_single_round_with_all_initial_information(model_bo, fun_target, X, Y, num_iter)
    assert len(X_final.shape) == 2
    assert len(Y_final.shape) == 2
    assert len(time_all_final.shape) == 1
    assert len(time_gp_final.shape) == 1
    assert len(time_acq_final.shape) == 1
    assert X_final.shape[1] == dim_X
    assert X_final.shape[0] == Y_final.shape[0] == num_X + num_iter
    assert time_all_final.shape[0] == num_iter
    assert Y_final.shape[1] == 1
    assert time_gp_final.shape[0] == time_acq_final.shape[0]

def test_run_single_round_with_initial_inputs_typing():
    annos = wrappers_bo.run_single_round_with_initial_inputs.__annotations__

    assert annos['model_bo'] == bo.BO
    assert annos['fun_target'] == callable
    assert annos['X_train'] == np.ndarray
    assert annos['num_iter'] == int
    assert annos['str_sampling_method_ao'] == str
    assert annos['num_samples_ao'] == int
    assert annos['str_mlm_method'] == str
    assert annos['return'] == typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]

def test_run_single_round_with_initial_inputs():
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
        wrappers_bo.run_single_round_with_initial_inputs(1, fun_target, X, num_iter)
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round_with_initial_inputs(model_bo, 1, X, num_iter)
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round_with_initial_inputs(model_bo, fun_target, 1, num_iter)
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round_with_initial_inputs(model_bo, fun_target, X, 1.2)
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round_with_initial_inputs(model_bo, fun_target, np.random.randn(num_X), num_iter)
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round_with_initial_inputs(model_bo, fun_target, X, num_iter, str_sampling_method_ao=1)
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round_with_initial_inputs(model_bo, fun_target, X, num_iter, str_sampling_method_ao='abc')
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round_with_initial_inputs(model_bo, fun_target, X, num_iter, num_samples_ao='abc')

    X_final, Y_final, time_all_final, time_gp_final, time_acq_final = wrappers_bo.run_single_round_with_initial_inputs(model_bo, fun_target, X, num_iter)
    assert len(X_final.shape) == 2
    assert len(Y_final.shape) == 2
    assert len(time_all_final.shape) == 1
    assert len(time_gp_final.shape) == 1
    assert len(time_acq_final.shape) == 1
    assert X_final.shape[1] == dim_X
    assert X_final.shape[0] == Y_final.shape[0] == time_all_final.shape[0] == num_X + num_iter
    assert Y_final.shape[1] == 1
    assert time_gp_final.shape[0] == time_acq_final.shape[0]

def test_run_single_round_typing():
    annos = wrappers_bo.run_single_round.__annotations__

    assert annos['model_bo'] == bo.BO
    assert annos['fun_target'] == callable
    assert annos['num_init'] == int
    assert annos['num_iter'] == int
    assert annos['str_sampling_method_ao'] == str
    assert annos['num_samples_ao'] == int
    assert annos['str_mlm_method'] == str
    assert annos['seed'] == typing.Union[int, type(None)]
    assert annos['return'] == typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]

def test_run_single_round():
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
        wrappers_bo.run_single_round(1, fun_target, num_X, num_iter)
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round(model_bo, 1, num_X, num_iter)
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round(model_bo, fun_target, 1.2, num_iter)
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round(model_bo, fun_target, num_X, 1.2)
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round(model_bo, fun_target, num_X, num_iter, str_initial_method_bo=1)
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round(model_bo, fun_target, num_X, num_iter, str_initial_method_bo='abc')
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round(model_bo, fun_target, num_X, num_iter, str_initial_method_bo='grid')
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round(model_bo, fun_target, num_X, num_iter, str_sampling_method_ao=1)
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round(model_bo, fun_target, num_X, num_iter, str_sampling_method_ao='abc')
    with pytest.raises(AssertionError) as error:
        wrappers_bo.run_single_round(model_bo, fun_target, num_X, num_iter, seed=1.2)

    X_final, Y_final, time_all_final, time_gp_final, time_acq_final = wrappers_bo.run_single_round(model_bo, fun_target, num_X, num_iter, str_initial_method_bo='uniform')
    assert len(X_final.shape) == 2
    assert len(Y_final.shape) == 2
    assert len(time_all_final.shape) == 1
    assert len(time_gp_final.shape) == 1
    assert len(time_acq_final.shape) == 1
    assert X_final.shape[1] == dim_X
    assert X_final.shape[0] == Y_final.shape[0] == time_all_final.shape[0] == num_X + num_iter
    assert Y_final.shape[1] == 1
    assert time_gp_final.shape[0] == time_acq_final.shape[0]
