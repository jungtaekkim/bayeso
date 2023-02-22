#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: October 8, 2021
#
"""test_bo_bo_w_trees"""

import pytest
import numpy as np

from bayeso.bo import bo_w_trees as package_target
from bayeso.trees import trees_random_forest
from bayeso.utils import utils_bo


BO = package_target.BOwTrees
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
        model_bo = BO(arr_range_1, str_surrogate=1)
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_surrogate='abc')
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_optimizer_method_bo=1)
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_optimizer_method_bo='abc')
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_exp=123)
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, debug=1)

    model_bo = BO(arr_range_1)
    model_bo = BO(arr_range_2, debug=True)

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
    model_bo = BO(arr_range, debug=True)

    with pytest.raises(AssertionError) as error:
        model_bo.get_samples(1)
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

    arr_initials = model_bo.get_samples('grid', num_samples=1)
    truth_arr_initials = np.array([
        [0.000, -2.000, -5.000],
    ])
    assert (np.abs(arr_initials - truth_arr_initials) < TEST_EPSILON).all()

    arr_initials = model_bo.get_samples('grid', num_samples=3)
    truth_arr_initials = np.array([
        [0.000, -2.000, -5.000],
        [0.000, -2.000, 0.000],
        [0.000, -2.000, 5.000],
        [5.000, -2.000, -5.000],
        [5.000, -2.000, 0.000],
        [5.000, -2.000, 5.000],
        [10.000, -2.000, -5.000],
        [10.000, -2.000, 0.000],
        [10.000, -2.000, 5.000],
        [0.000, 0.000, -5.000],
        [0.000, 0.000, 0.000],
        [0.000, 0.000, 5.000],
        [5.000, 0.000, -5.000],
        [5.000, 0.000, 0.000],
        [5.000, 0.000, 5.000],
        [10.000, 0.000, -5.000],
        [10.000, 0.000, 0.000],
        [10.000, 0.000, 5.000],
        [0.000, 2.000, -5.000],
        [0.000, 2.000, 0.000],
        [0.000, 2.000, 5.000],
        [5.000, 2.000, -5.000],
        [5.000, 2.000, 0.000],
        [5.000, 2.000, 5.000],
        [10.000, 2.000, -5.000],
        [10.000, 2.000, 0.000],
        [10.000, 2.000, 5.000],
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

    num_samples = 1000

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
        model_bo.optimize(X, Y, num_samples='abc')

    next_point, dict_info = model_bo.optimize(X, Y, num_samples=num_samples)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    trees = dict_info['trees']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(trees, list)
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

    num_samples = 10

    model_bo = BO(arr_range_1, str_acq='pi')
    next_point, dict_info = model_bo.optimize(X, Y, num_samples=num_samples)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    trees = dict_info['trees']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(trees, list)
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
    next_point, dict_info = model_bo.optimize(X, Y, num_samples=num_samples)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    trees = dict_info['trees']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(trees, list)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

    model_bo = BO(arr_range_1, str_acq='aei', debug=True)
    next_point, dict_info = model_bo.optimize(X, Y, num_samples=num_samples)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    trees = dict_info['trees']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(trees, list)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

    model_bo = BO(arr_range_1, str_acq='pure_exploit', debug=True)
    next_point, dict_info = model_bo.optimize(X, Y, num_samples=num_samples)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    trees = dict_info['trees']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(trees, list)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert len(next_point.shape) == 1
    assert len(next_points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

    model_bo = BO(arr_range_1, str_acq='pure_explore', debug=True)
    next_point, dict_info = model_bo.optimize(X, Y, num_samples=num_samples)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    trees = dict_info['trees']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(trees, list)
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

    num_samples = 10

    model_bo = BO(arr_range_1, str_optimizer_method_bo='random_search')
    next_point, dict_info = model_bo.optimize(X, Y, num_samples=num_samples)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    trees = dict_info['trees']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(trees, list)
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
    num_X = 5
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)

    num_samples = 10

    model_bo = BO(arr_range, normalize_Y=True, str_exp=None)
    next_point, dict_info = model_bo.optimize(X, Y, num_samples=num_samples)
    Y_original = dict_info['Y_original']
    Y_normalized = dict_info['Y_normalized']

    assert np.all(Y == Y_original)
    assert np.all(Y != Y_normalized)
    assert np.all(utils_bo.normalize_min_max(Y) == Y_normalized)

    model_bo = BO(arr_range, normalize_Y=False, str_exp=None)
    next_point, dict_info = model_bo.optimize(X, Y, num_samples=num_samples)
    Y_original = dict_info['Y_original']
    Y_normalized = dict_info['Y_normalized']

    assert np.all(Y == Y_normalized)
    assert np.all(Y == Y_original)

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

    model_bo = BO(arr_range_1, str_acq='ei', str_exp='test')

    num_trees = 10
    depth_max = 5
    size_min_leaf = 1
    num_features = int(np.sqrt(dim_X))

    trees = trees_random_forest.get_random_forest(
        X, Y, num_trees, depth_max, size_min_leaf, num_features
    )

    X_test = model_bo.get_samples('sobol', num_samples=10, seed=111)

    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(1, trees)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X_test, 1)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X_test, 'abc')

    pred_mean, pred_std = model_bo.compute_posteriors(X_test, trees)

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

    num_trees = 10
    depth_max = 5
    size_min_leaf = 1
    num_features = int(np.sqrt(dim_X))

    trees = trees_random_forest.get_random_forest(
        X, Y, num_trees, depth_max, size_min_leaf, num_features
    )

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
        model_bo.compute_acquisitions(1, X, Y, trees)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_acquisitions(X_test, 1, Y, trees)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_acquisitions(X_test, X, 1, trees)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_acquisitions(X_test, X, Y, 1)

    acqs = model_bo.compute_acquisitions(X_test, X, Y, trees)

    print('acqs')
    for elem_1 in acqs:
        print(elem_1)

    truth_acqs = np.array([
        0.8326459864487323,
        0.03890555794283132,
        0.38955006459944297,
        1.1058224991730214,
        1.1778382705844814,
        0.06437483526516512,
        2.0086546485945056,
        0.06437483526516512,
        1.5340211638096464,
        0.06437483526516512,
    ])

    assert isinstance(acqs, np.ndarray)
    assert len(acqs.shape) == 1
    assert X_test.shape[0] == acqs.shape[0]
    assert np.all(np.abs(acqs - truth_acqs) < TEST_EPSILON)

    acqs = model_bo.compute_acquisitions(X_test[0], X, Y, trees)

    assert isinstance(acqs, np.ndarray)
    assert len(acqs.shape) == 1
    assert acqs.shape[0] == 1
