#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: September 24, 2020
#
"""test_utils_bo"""

import typing
import pytest
import numpy as np

from bayeso import constants
from bayeso.utils import utils_bo as package_target


def test_normalize_min_max_typing():
    annos = package_target.normalize_min_max.__annotations__

    assert annos['Y'] == np.ndarray
    assert annos['return'] == np.ndarray

def test_normalize_min_max():
    Y = np.array([
        [1.0],
        [2.0],
        [10.0],
        [-5.0],
        [4.0],
        [2.0],
        [-4.0],
        [2.0],
    ])

    with pytest.raises(AssertionError) as error:
        package_target.normalize_min_max(123)
    with pytest.raises(AssertionError) as error:
        package_target.normalize_min_max('abc')
    with pytest.raises(AssertionError) as error:
        package_target.normalize_min_max(np.squeeze(Y))

    Y = package_target.normalize_min_max(Y)
    truth_Y = np.array([
        [6.0 / 15.0],
        [7.0 / 15.0],
        [1.0],
        [0.0],
        [9.0 / 15.0],
        [7.0 / 15.0],
        [1.0 / 15.0],
        [7.0 / 15.0],
    ]) * constants.MULTIPLIER_RESPONSE

    assert np.all(Y == truth_Y)

def test_get_best_acquisition_by_evaluation_typing():
    annos = package_target.get_best_acquisition_by_evaluation.__annotations__

    assert annos['initials'] == np.ndarray
    assert annos['fun_objective'] == typing.Callable
    assert annos['return'] == np.ndarray

def test_get_best_acquisition_by_evaluation():
    fun_objective = lambda x: x**2 - 2.0 * x + 1.0
    arr_initials = np.expand_dims(np.arange(-5, 5), axis=1)

    with pytest.raises(AssertionError) as error:
        package_target.get_best_acquisition_by_evaluation(1, fun_objective)
    with pytest.raises(AssertionError) as error:
        package_target.get_best_acquisition_by_evaluation(arr_initials, None)
    with pytest.raises(AssertionError) as error:
        package_target.get_best_acquisition_by_evaluation(np.arange(-5, 5), fun_objective)

    best_initial = package_target.get_best_acquisition_by_evaluation(arr_initials, fun_objective)
    assert len(best_initial.shape) == 2
    assert best_initial.shape[0] == 1
    assert best_initial.shape[1] == arr_initials.shape[1]
    assert best_initial == np.array([[1]])

    fun_objective = lambda x: np.linalg.norm(x, ord=2, axis=0)**2
    arr_initials = np.reshape(np.arange(-10, 10), (5, 4))

    with pytest.raises(AssertionError) as error:
        package_target.get_best_acquisition_by_evaluation(1, fun_objective)
    with pytest.raises(AssertionError) as error:
        package_target.get_best_acquisition_by_evaluation(arr_initials, None)
    with pytest.raises(AssertionError) as error:
        package_target.get_best_acquisition_by_evaluation(np.arange(-5, 5), fun_objective)

    best_initial = package_target.get_best_acquisition_by_evaluation(arr_initials, fun_objective)
    assert len(best_initial.shape) == 2
    assert best_initial.shape[0] == 1
    assert best_initial.shape[1] == arr_initials.shape[1]
    assert np.all(best_initial == np.array([[-2, -1, 0, 1]]))

def test_check_optimizer_method_bo_typing():
    annos = package_target.check_optimizer_method_bo.__annotations__

    assert annos['str_optimizer_method_bo'] == str
    assert annos['dim'] == int
    assert annos['debug'] == bool
    assert annos['return'] == str

def test_check_optimizer_method_bo():
    directminimize = None
    cma = None

    with pytest.raises(AssertionError) as error:
        package_target.check_optimizer_method_bo(2, 2, True)
    with pytest.raises(AssertionError) as error:
        package_target.check_optimizer_method_bo('DIRECT', 'abc', True)
    with pytest.raises(AssertionError) as error:
        package_target.check_optimizer_method_bo('DIRECT', 2, 'abc')
    with pytest.raises(AssertionError) as error:
        package_target.check_optimizer_method_bo('ABC', 2, True)

    package_target.check_optimizer_method_bo('L-BFGS-B', 2, False)
    package_target.check_optimizer_method_bo('DIRECT', 2, False)
    package_target.check_optimizer_method_bo('CMA-ES', 2, False)

def test_choose_fun_acquisition_typing():
    annos = package_target.choose_fun_acquisition.__annotations__

    assert annos['str_acq'] == str
    assert annos['noise'] == typing.Union[float, type(None)]
    assert annos['return'] == typing.Callable

def test_choose_fun_acquisition():
    dict_hyps = {'lengthscales': np.array([1.0, 1.0]), 'signal': 1.0, 'noise': 0.01}
    with pytest.raises(AssertionError) as error:
        package_target.choose_fun_acquisition(1, dict_hyps)
    with pytest.raises(AssertionError) as error:
        package_target.choose_fun_acquisition('abc', dict_hyps)
    with pytest.raises(AssertionError) as error:
        package_target.choose_fun_acquisition('pi', 1)

def test_check_hyps_convergence_typing():
    annos = package_target.check_hyps_convergence.__annotations__

    assert annos['list_hyps'] == typing.List[dict]
    assert annos['hyps'] == dict
    assert annos['str_cov'] == str
    assert annos['fix_noise'] == bool
    assert annos['ratio_threshold'] == float
    assert annos['return'] == bool

def test_check_hyps_convergence():
    dict_hyps_1 = {'lengthscales': np.array([1.0, 1.0]), 'signal': 1.0, 'noise': 0.01}
    dict_hyps_2 = {'lengthscales': np.array([2.0, 1.0]), 'signal': 1.0, 'noise': 0.01}

    with pytest.raises(AssertionError) as error:
        package_target.check_hyps_convergence(1, dict_hyps_1, 'se', True)
    with pytest.raises(AssertionError) as error:
        package_target.check_hyps_convergence([dict_hyps_1], 1, 'se', True)
    with pytest.raises(AssertionError) as error:
        package_target.check_hyps_convergence([dict_hyps_1], dict_hyps_1, 1, True)
    with pytest.raises(AssertionError) as error:
        package_target.check_hyps_convergence([dict_hyps_1], dict_hyps_1, 1, 'abc')

    assert package_target.check_hyps_convergence([dict_hyps_1], dict_hyps_1, 'se', False)
    assert not package_target.check_hyps_convergence([dict_hyps_2], dict_hyps_1, 'se', False)

def test_get_next_best_acquisition_typing():
    annos = package_target.get_next_best_acquisition.__annotations__

    assert annos['points'] == np.ndarray
    assert annos['acquisitions'] == np.ndarray
    assert annos['points_evaluated'] == np.ndarray
    assert annos['return'] == np.ndarray

def test_get_next_best_acquisition():
    arr_points = np.array([
        [0.0, 1.0],
        [1.0, -3.0],
        [-2.0, -4.0],
        [1.0, 3.0],
    ])
    arr_acquisitions = np.array([1.1, 0.2, 0.5, 0.6])
    cur_points = np.array([
        [-10.0, 1.0],
        [1.0, -3.0],
        [11.0, 2.0],
    ])

    with pytest.raises(AssertionError) as error:
        package_target.get_next_best_acquisition(1, arr_acquisitions, cur_points)
    with pytest.raises(AssertionError) as error:
        package_target.get_next_best_acquisition(np.arange(0, 4), arr_acquisitions, cur_points)
    with pytest.raises(AssertionError) as error:
        package_target.get_next_best_acquisition(arr_points, 1, cur_points)
    with pytest.raises(AssertionError) as error:
        package_target.get_next_best_acquisition(arr_points, np.ones((4, 2)), cur_points)
    with pytest.raises(AssertionError) as error:
        package_target.get_next_best_acquisition(arr_points, arr_acquisitions, 1)
    with pytest.raises(AssertionError) as error:
        package_target.get_next_best_acquisition(arr_points, arr_acquisitions, np.arange(0, 3))
    with pytest.raises(AssertionError) as error:
        package_target.get_next_best_acquisition(arr_points, np.arange(0, 10), cur_points)
    with pytest.raises(AssertionError) as error:
        package_target.get_next_best_acquisition(arr_points, arr_acquisitions, np.ones((3, 5)))

    next_point = package_target.get_next_best_acquisition(arr_points, arr_acquisitions, cur_points)
    assert (next_point == np.array([-2.0, -4.0])).all()

    cur_points = np.array([
        [-10.0, 1.0],
        [1.0, -3.0],
        [11.0, 2.0],
        [0.0, 1.0],
        [1.0, -3.0],
        [-2.0, -4.0],
        [1.0, 3.0],
    ])
    next_point = package_target.get_next_best_acquisition(arr_points, arr_acquisitions, cur_points)
    assert (next_point == np.array([1.0, 3.0])).all()

def test_get_best_acquisition_by_history_typing():
    annos = package_target.get_best_acquisition_by_history.__annotations__

    assert annos['X'] == np.ndarray
    assert annos['Y'] == np.ndarray
    assert annos['return'] == constants.TYPING_TUPLE_ARRAY_FLOAT

def test_get_best_acquisition_by_history():
    X = np.array([
        [1.0, 2.0, 1.0, 5.0],
        [2.0, 1.0, 2.0, 1.1],
        [4.0, 4.0, 4.0, 4.0],
        [2.0, 3.1, 2.2, 5.1],
        [4.2, 4.1, 9.1, 2.2],
    ])
    Y = np.array([
        [1.0],
        [2.0],
        [4.0],
        [0.0],
        [5.0],
    ])

    with pytest.raises(AssertionError) as error:
        package_target.get_best_acquisition_by_history(1, Y)
    with pytest.raises(AssertionError) as error:
        package_target.get_best_acquisition_by_history(X, 1)
    with pytest.raises(AssertionError) as error:
        package_target.get_best_acquisition_by_history(X[:4], Y)
    with pytest.raises(AssertionError) as error:
        package_target.get_best_acquisition_by_history(X, Y[:3])
    with pytest.raises(AssertionError) as error:
        package_target.get_best_acquisition_by_history(X, Y[:, 0])

    bx_best, y_best = package_target.get_best_acquisition_by_history(X, Y)

    assert np.all(bx_best == np.array([2.0, 3.1, 2.2, 5.1]))
    assert y_best == 0.0
