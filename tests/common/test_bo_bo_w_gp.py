#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: October 8, 2021
#
"""test_bo_bo_w_gp"""

import pytest
import numpy as np

from bayeso.bo import bo_w_gp as package_target
from bayeso import covariance
from bayeso.utils import utils_covariance


BO = package_target.BOwGP
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
        model_bo = BO(1)
    with pytest.raises(AssertionError) as error:
        model_bo = BO(np.arange(0, 10))
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_3)
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_4)
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_cov=1)
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_cov='abc')
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_acq=1)
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_acq='abc')
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, use_ard='abc')
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, use_ard=1)
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, prior_mu=1)
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_optimizer_method_gp=1)
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_optimizer_method_gp='abc')
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_optimizer_method_bo=1)
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_optimizer_method_bo='abc')
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_modelselection_method=1)
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_modelselection_method='abc')
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_exp=123)
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, debug=1)

    model_bo = BO(arr_range_1)
    model_bo = BO(arr_range_2)

def test_get_samples():
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
    model_bo = BO(arr_range, debug=True)

    with pytest.raises(AssertionError) as error:
        model_bo.get_samples(1)
    with pytest.raises(AssertionError) as error:
        model_bo.get_samples('grid', fun_objective=None)
    with pytest.raises(AssertionError) as error:
        model_bo.get_samples('uniform', fun_objective=1)
    with pytest.raises(AssertionError) as error:
        model_bo.get_samples('uniform', num_samples='abc')
    with pytest.raises(AssertionError) as error:
        model_bo.get_samples('uniform', seed='abc')
    with pytest.raises(AssertionError) as error:
        model_bo.get_samples('gaussian', seed='abc')
    with pytest.raises(AssertionError) as error:
        model_bo.get_samples('sobol', seed='abc')
    with pytest.raises(AssertionError) as error:
        model_bo.get_samples('halton', seed='abc')
    with pytest.raises(AssertionError) as error:
        model_bo.get_samples('abc')

    arr_initials = model_bo.get_samples('grid', num_samples=50, fun_objective=fun_objective)
    truth_arr_initials = np.array([
        [0.0, -2.0, -5.0],
    ])
    assert (np.abs(arr_initials - truth_arr_initials) < TEST_EPSILON).all()

    arr_initials_ = model_bo.get_samples('sobol', num_samples=3)
    arr_initials = model_bo.get_samples('sobol', num_samples=3, seed=42)

    print('sobol')
    for elem_1 in arr_initials:
        for elem_2 in elem_1:
            print(elem_2)

    truth_arr_initials = np.array([
        [
            8.78516613971442,
            -0.6113853892311454,
            -4.274874331895262,
        ],
        [
            2.7565963030792773,
            1.627942705526948,
            1.6141902864910662,
        ],
        [
            1.0978685109876096,
            -1.511254413984716,
            -1.5049133892171085,
        ],
    ])

    assert (np.abs(arr_initials - truth_arr_initials) < TEST_EPSILON).all()

    arr_initials_ = model_bo.get_samples('halton', num_samples=3)
    arr_initials = model_bo.get_samples('halton', num_samples=3, seed=42)

    print('halton')
    for elem_1 in arr_initials:
        for elem_2 in elem_1:
            print(elem_2)

    truth_arr_initials = np.array([
        [
            3.3124358790165855,
            -1.4099903230606423,
            -0.4462920191434243,
        ],
        [
            8.312435879016585,
            1.2566763436060246,
            -2.446292019143424,
        ],
        [
            0.8124358790165853,
            -0.07665698972730861,
            -4.446292019143424,
        ],
    ])

    assert (np.abs(arr_initials - truth_arr_initials) < TEST_EPSILON).all()

    arr_initials_ = model_bo.get_samples('uniform', num_samples=3)
    arr_initials = model_bo.get_samples('uniform', num_samples=3, seed=42)
    truth_arr_initials = np.array([
        [3.74540119, 1.80285723, 2.31993942],
        [5.98658484, -1.37592544, -3.4400548],
        [0.58083612, 1.46470458, 1.01115012],
    ])
    assert (np.abs(arr_initials - truth_arr_initials) < TEST_EPSILON).all()

    arr_initials_ = model_bo.get_samples('gaussian', num_samples=3)
    arr_initials = model_bo.get_samples('gaussian', num_samples=3, seed=42)
    truth_arr_initials = np.array([
        [6.241785382528082, -0.13826430117118466, 1.6192213452517312],
        [8.807574641020064, -0.23415337472333597, -0.5853423923729514],
        [8.948032038768478, 0.7674347291529088, -1.1736859648373803],
    ])
    assert (np.abs(arr_initials - truth_arr_initials) < TEST_EPSILON).all()

def test_get_initials():
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
    model_bo = BO(arr_range)

    with pytest.raises(AssertionError) as error:
        model_bo.get_initials(1, 10)
    with pytest.raises(AssertionError) as error:
        model_bo.get_initials('grid', 'abc')
    with pytest.raises(AssertionError) as error:
        model_bo.get_initials('grid', 10)
    with pytest.raises(AssertionError) as error:
        model_bo.get_initials('uniform', 10, seed='abc')
    with pytest.raises(AssertionError) as error:
        model_bo.get_initials('abc', 10)

    arr_initials = model_bo.get_initials('sobol', 3)
    arr_initials = model_bo.get_initials('sobol', 3, seed=42)

    for elem_1 in arr_initials:
        for elem_2 in elem_1:
            print(elem_2)

    truth_arr_initials = np.array([
        [
            8.78516613971442,
            -0.6113853892311454,
            -4.274874331895262,
        ],
        [
            2.7565963030792773,
            1.627942705526948,
            1.6141902864910662,
        ],
        [
            1.0978685109876096,
            -1.511254413984716,
            -1.5049133892171085,
        ],
    ])

    assert (np.abs(arr_initials - truth_arr_initials) < TEST_EPSILON).all()

    arr_initials = model_bo.get_initials('uniform', 3, seed=42)
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
    model_bo = BO(arr_range_1)

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
        model_bo.optimize(X, Y, str_sampling_method=1)
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(X, Y, str_sampling_method='abc')
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(X, Y, str_mlm_method=1)
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(X, Y, str_mlm_method='abc')
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(X, Y, num_samples='abc')

    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

def test_optimize_str_acq():
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

    model_bo = BO(arr_range_1, str_acq='pi')
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

    model_bo = BO(arr_range_1, str_acq='ucb')
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

    model_bo = BO(arr_range_1, str_acq='aei')
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

    model_bo = BO(arr_range_1, str_acq='pure_exploit')
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

    model_bo = BO(arr_range_1, str_acq='pure_explore')
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

def test_optimize_str_optimize_method_bo():
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

    model_bo = BO(arr_range_1, str_optimizer_method_bo='L-BFGS-B')
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

    # TODO: add DIRECT test, now it causes an error.

    model_bo = BO(arr_range_1, str_optimizer_method_bo='CMA-ES')
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

def test_optimize_str_mlm_method():
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

    model_bo = BO(arr_range_1)
    next_point, dict_info = model_bo.optimize(X, Y, str_mlm_method='converged')
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

    model_bo = BO(arr_range_1)
    next_point, dict_info = model_bo.optimize(X, Y, str_mlm_method='combined')
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

def test_optimize_str_modelselection_method():
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

    model_bo = BO(arr_range_1, str_modelselection_method='loocv')
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

def test_optimize_normalize_Y():
    np.random.seed(42)
    arr_range = np.array([
        [0.0, 10.0],
        [-2.0, 2.0],
        [-5.0, 5.0],
    ])
    dim_X = arr_range.shape[0]
    num_X = 1
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)

    model_bo = BO(arr_range, str_acq='ei', normalize_Y=True)
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

    X = np.array([
        [3.0, 0.0, 1.0],
        [2.0, -1.0, 4.0],
        [9.0, 1.5, 3.0],
    ])
    Y = np.array([
        [100.0],
        [100.0],
        [100.0],
    ])

    model_bo = BO(arr_range, str_acq='ei', normalize_Y=True)
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

def test_optimize_use_ard():
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

    model_bo = BO(arr_range, use_ard=False)
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]
    assert isinstance(hyps['lengthscales'], float)

    X = np.array([
        [3.0, 0.0, 1.0],
        [2.0, -1.0, 4.0],
        [9.0, 1.5, 3.0],
    ])
    Y = np.array([
        [100.0],
        [100.0],
        [100.0],
    ])

    model_bo = BO(arr_range, use_ard=True)
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]
    assert isinstance(hyps['lengthscales'], np.ndarray)
    assert len(hyps['lengthscales'].shape) == 1
    assert hyps['lengthscales'].shape[0] == 3

def test_compute_posteriors():
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

    model_bo = BO(arr_range_1, str_acq='ei')
    hyps = utils_covariance.get_hyps(model_bo.str_cov, dim=dim_X, use_ard=model_bo.use_ard)

    cov_X_X, inv_cov_X_X, _ = covariance.get_kernel_inverse(X, hyps, model_bo.str_cov)

    X_test = model_bo.get_samples('sobol', num_samples=10, seed=111)

    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(1, Y, X_test, cov_X_X, inv_cov_X_X, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, 1, X_test, cov_X_X, inv_cov_X_X, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, 1, cov_X_X, inv_cov_X_X, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, X_test, 1, inv_cov_X_X, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, X_test, cov_X_X, 1, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, X_test, cov_X_X, inv_cov_X_X, 1)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, X_test, cov_X_X, inv_cov_X_X, 1.0)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, X_test, cov_X_X, inv_cov_X_X, 'abc')

    pred_mean, pred_std = model_bo.compute_posteriors(X, Y, X_test, cov_X_X, inv_cov_X_X, hyps)

    assert len(pred_mean.shape) == 1
    assert len(pred_std.shape) == 1
    assert pred_mean.shape[0] == pred_mean.shape[0] == X_test.shape[0]

def test_compute_posteriors_set():
    np.random.seed(42)
    arr_range_1 = np.array([
        [0.0, 10.0],
        [-2.0, 2.0],
        [-5.0, 5.0],
    ])
    dim_X = arr_range_1.shape[0]
    num_X = 5
    num_instances = 4
    X = np.random.randn(num_X, num_instances, dim_X)
    Y = np.random.randn(num_X, 1)

    model_bo = BO(arr_range_1, str_acq='pi', str_cov='set_se', str_exp=None)
    hyps = utils_covariance.get_hyps(model_bo.str_cov, dim=dim_X, use_ard=model_bo.use_ard)

    cov_X_X, inv_cov_X_X, _ = covariance.get_kernel_inverse(X, hyps, model_bo.str_cov)
    
    X_test = np.array([
        [
            [1.0, 0.0, 0.0, 1.0],
            [2.0, -1.0, 2.0, 1.0],
            [3.0, -2.0, 4.0, 1.0],
        ],
        [
            [4.0, 2.0, -3.0, 1.0],
            [5.0, 0.0, -2.0, 1.0],
            [6.0, -2.0, -1.0, 1.0],
        ],
    ])

    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(1, Y, X_test, cov_X_X, inv_cov_X_X, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, 1, X_test, cov_X_X, inv_cov_X_X, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, 1, cov_X_X, inv_cov_X_X, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, X_test, 1, inv_cov_X_X, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, X_test, cov_X_X, 1, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, X_test, cov_X_X, inv_cov_X_X, 1)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, X_test, cov_X_X, inv_cov_X_X, 1.0)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, X_test, cov_X_X, inv_cov_X_X, 'abc')

    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, X_test, cov_X_X, inv_cov_X_X, hyps)

    pred_mean, pred_std = model_bo.compute_posteriors(X, Y, X_test[:, :, :dim_X], cov_X_X, inv_cov_X_X, hyps)

    assert len(pred_mean.shape) == 1
    assert len(pred_std.shape) == 1
    assert pred_mean.shape[0] == pred_mean.shape[0] == X_test.shape[0]

def test_compute_acquisitions():
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

    model_bo = BO(arr_range_1, str_acq='pi', str_exp='test')
    hyps = utils_covariance.get_hyps(model_bo.str_cov, dim=dim_X, use_ard=model_bo.use_ard)

    cov_X_X, inv_cov_X_X, _ = covariance.get_kernel_inverse(X, hyps, model_bo.str_cov)

    X_test = model_bo.get_samples('sobol', num_samples=10, seed=111)

    truth_X_test = np.array([
        [
            3.328958908095956,
            -1.8729291455820203,
            0.2839687094092369,
        ],
        [
            8.11741182114929,
            0.3799784183502197,
            -0.05574141861870885,
        ],
        [
            6.735238193068653,
            -0.9264274807646871,
            3.631770429201424,
        ],
        [
            2.13300823001191,
            1.3245289996266365,
            -3.547573888208717,
        ],
        [
            0.6936023756861687,
            -0.018464308232069016,
            -2.1043178741820157,
        ],
        [
            5.438151848502457,
            1.7285785367712379,
            2.0298107899725437,
        ],
        [
            9.085266247857362,
            -1.2144776917994022,
            -4.31197423255071,
        ],
        [
            4.468362366314977,
            0.5345162367448211,
            4.0739051485434175,
        ],
        [
            3.9395463559776545,
            -0.5726078534498811,
            -4.846788686700165,
        ],
        [
            9.92871844675392,
            1.1744442842900753,
            4.774723623413593,
        ],
    ])

    for elem_1 in X_test:
        for elem_2 in elem_1:
            print(elem_2)

    assert np.all(np.abs(X_test - truth_X_test) < TEST_EPSILON)

    with pytest.raises(AssertionError) as error:
        model_bo.compute_acquisitions(1, X, Y, cov_X_X, inv_cov_X_X, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_acquisitions(X_test, 1, Y, cov_X_X, inv_cov_X_X, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_acquisitions(X_test, X, 1, cov_X_X, inv_cov_X_X, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_acquisitions(X_test, X, Y, 1, inv_cov_X_X, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_acquisitions(X_test, X, Y, cov_X_X, 1, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_acquisitions(X_test, X, Y, cov_X_X, inv_cov_X_X, 1)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_acquisitions(X_test, X, Y, cov_X_X, inv_cov_X_X, 'abc')

    acqs = model_bo.compute_acquisitions(X_test, X, Y, cov_X_X, inv_cov_X_X, hyps)

    print('acqs')
    for elem_1 in acqs:
        print(elem_1)

    truth_acqs = np.array([
        0.9140836833364618,
        0.7893422923284443,
        0.7893819649518585,
        0.780516205172671,
        1.170379060386938,
        0.7889956503605072,
        0.7893345684226016,
        0.789773864915061,
        0.7908883762985802,
        0.7893339801719917,
    ])

    assert isinstance(acqs, np.ndarray)
    assert len(acqs.shape) == 1
    assert X_test.shape[0] == acqs.shape[0]
    assert np.all(np.abs(acqs - truth_acqs) < TEST_EPSILON)

def test_compute_acquisitions_set():
    np.random.seed(42)
    arr_range_1 = np.array([
        [0.0, 10.0],
        [-2.0, 2.0],
        [-5.0, 5.0],
    ])
    dim_X = arr_range_1.shape[0]
    num_X = 5
    num_instances = 4
    X = np.random.randn(num_X, num_instances, dim_X)
    Y = np.random.randn(num_X, 1)

    model_bo = BO(arr_range_1, str_acq='pi', str_cov='set_se', str_exp='test')
    hyps = utils_covariance.get_hyps(model_bo.str_cov, dim=dim_X, use_ard=model_bo.use_ard)

    cov_X_X, inv_cov_X_X, _ = covariance.get_kernel_inverse(X, hyps, model_bo.str_cov)
    
    X_test = np.array([
        [
            [1.0, 0.0, 0.0, 1.0],
            [2.0, -1.0, 2.0, 1.0],
            [3.0, -2.0, 4.0, 1.0],
        ],
        [
            [4.0, 2.0, -3.0, 1.0],
            [5.0, 0.0, -2.0, 1.0],
            [6.0, -2.0, -1.0, 1.0],
        ],
    ])

    with pytest.raises(AssertionError) as error:
        model_bo.compute_acquisitions(X_test, X, Y, cov_X_X, inv_cov_X_X, hyps)
