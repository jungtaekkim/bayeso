#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: September 24, 2020
#
"""test_bo"""

import pytest
import numpy as np

from bayeso import bo as package_target


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
        model_bo = package_target.BO(1)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BO(np.arange(0, 10))
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BO(arr_range_3)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BO(arr_range_4)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BO(arr_range_1, str_cov=1)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BO(arr_range_1, str_cov='abc')
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BO(arr_range_1, str_acq=1)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BO(arr_range_1, str_acq='abc')
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BO(arr_range_1, use_ard='abc')
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BO(arr_range_1, use_ard=1)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BO(arr_range_1, prior_mu=1)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BO(arr_range_1, str_optimizer_method_gp=1)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BO(arr_range_1, str_optimizer_method_gp='abc')
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BO(arr_range_1, str_optimizer_method_bo=1)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BO(arr_range_1, str_optimizer_method_bo='abc')
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BO(arr_range_1, str_modelselection_method=1)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BO(arr_range_1, str_modelselection_method='abc')
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BO(arr_range_1, str_surrogate='abc')
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BO(arr_range_1, str_surrogate=1)
    with pytest.raises(AssertionError) as error:
        model_bo = package_target.BO(arr_range_1, debug=1)

    model_bo = package_target.BO(arr_range_1)
    model_bo = package_target.BO(arr_range_2)

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
    model_bo = package_target.BO(arr_range, debug=True)

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

    arr_initials = model_bo.get_samples('grid', fun_objective=fun_objective)
    truth_arr_initials = np.array([
        [0.0, -2.0, -5.0],
    ])
    assert (np.abs(arr_initials - truth_arr_initials) < TEST_EPSILON).all()

    arr_initials_ = model_bo.get_samples('sobol', num_samples=3)
    arr_initials = model_bo.get_samples('sobol', num_samples=3, seed=42)
    truth_arr_initials = np.array([
        [5.051551531068981, 0.8090446023270488, -1.0847891168668866],
        [1.4649059670045972, -1.925125477835536, 1.4882571692578495],
        [3.202530408743769, 1.6943757990375161, -3.383688726462424],
    ])

    assert (np.abs(arr_initials - truth_arr_initials) < TEST_EPSILON).all()

    arr_initials_ = model_bo.get_samples('halton', num_samples=3)
    arr_initials = model_bo.get_samples('halton', num_samples=3, seed=42)
    truth_arr_initials = np.array([
        [4.325625705206888, 1.9045673771707823, 1.0981007622257621],
        [9.325625705206889, 0.5712340438374492, -2.9018992377742396],
        [1.825625705206888, -0.7620992894958845, 3.098100762225762],
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
    model_bo = package_target.BO(arr_range)

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
    truth_arr_initials = np.array([
        [5.051551531068981, 0.8090446023270488, -1.0847891168668866],
        [1.4649059670045972, -1.925125477835536, 1.4882571692578495],
        [3.202530408743769, 1.6943757990375161, -3.383688726462424],
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
    model_bo = package_target.BO(arr_range_1)

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
    time_gp = dict_info['time_gp']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_gp, float)
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

    model_bo = package_target.BO(arr_range_1, str_acq='pi')
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_gp = dict_info['time_gp']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_gp, float)
    assert isinstance(time_acq, float)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

    model_bo = package_target.BO(arr_range_1, str_acq='ucb')
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_gp = dict_info['time_gp']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_gp, float)
    assert isinstance(time_acq, float)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

    model_bo = package_target.BO(arr_range_1, str_acq='aei')
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_gp = dict_info['time_gp']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_gp, float)
    assert isinstance(time_acq, float)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

    model_bo = package_target.BO(arr_range_1, str_acq='pure_exploit')
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_gp = dict_info['time_gp']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_gp, float)
    assert isinstance(time_acq, float)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

    model_bo = package_target.BO(arr_range_1, str_acq='pure_explore')
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_gp = dict_info['time_gp']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_gp, float)
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

    model_bo = package_target.BO(arr_range_1, str_optimizer_method_bo='L-BFGS-B')
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_gp = dict_info['time_gp']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_gp, float)
    assert isinstance(time_acq, float)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

    # TODO: add DIRECT test, now it causes an error.

    model_bo = package_target.BO(arr_range_1, str_optimizer_method_bo='CMA-ES')
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_gp = dict_info['time_gp']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_gp, float)
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

    model_bo = package_target.BO(arr_range_1)
    next_point, dict_info = model_bo.optimize(X, Y, str_mlm_method='converged')
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_gp = dict_info['time_gp']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_gp, float)
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

    model_bo = package_target.BO(arr_range_1, str_modelselection_method='loocv')
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_gp = dict_info['time_gp']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_gp, float)
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

    model_bo = package_target.BO(arr_range, str_acq='ei', normalize_Y=True)
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_gp = dict_info['time_gp']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_gp, float)
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

    model_bo = package_target.BO(arr_range, str_acq='ei', normalize_Y=True)
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_gp = dict_info['time_gp']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_gp, float)
    assert isinstance(time_acq, float)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]
