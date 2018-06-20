# test_gp
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: May 30, 2018

import numpy as np
import pytest

from bayeso import gp
from bayeso.utils import utils_covariance
from bayeso import benchmarks


TEST_EPSILON = 1e-5

def test_get_prior_mu():
    fun_prior = lambda X: np.expand_dims(np.linalg.norm(X, axis=1), axis=1)
    fun_prior_1d = lambda X: np.linalg.norm(X, axis=1)
    X = np.reshape(np.arange(0, 90), (30, 3))

    with pytest.raises(AssertionError) as error:
        gp.get_prior_mu(1, X)
    with pytest.raises(AssertionError) as error:
        gp.get_prior_mu(fun_prior, 1)
    with pytest.raises(AssertionError) as error:
        gp.get_prior_mu(fun_prior, np.arange(0, 100))
    with pytest.raises(AssertionError) as error:
        gp.get_prior_mu(None, np.arange(0, 100))
    with pytest.raises(AssertionError) as error:
        gp.get_prior_mu(fun_prior_1d, X)

    assert (gp.get_prior_mu(None, X) == np.zeros((X.shape[0], 1))).all()
    assert (gp.get_prior_mu(fun_prior, X) == fun_prior(X)).all()

def test_get_kernels():
    dim_X = 3
    X = np.reshape(np.arange(0, 9), (3, dim_X))
    hyps = utils_covariance.get_hyps('se', dim_X)

    with pytest.raises(AssertionError) as error:
        gp.get_kernels(1, hyps, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_kernels(np.arange(0, 100), hyps, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_kernels(X, 1, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_kernels(X, hyps, 1)
    with pytest.raises(ValueError) as error:
        gp.get_kernels(X, hyps, 'abc')

    cov_X_X, inv_cov_X_X = gp.get_kernels(X, hyps, 'se')
    print(cov_X_X)
    print(inv_cov_X_X)
    truth_cov_X_X = [
        [1.01001000e+00, 1.37095909e-06, 3.53262857e-24],
        [1.37095909e-06, 1.01001000e+00, 1.37095909e-06],
        [3.53262857e-24, 1.37095909e-06, 1.01001000e+00]
    ]
    truth_inv_cov_X_X = [
        [9.90089207e-01, -1.34391916e-06, 1.82419797e-12],
        [-1.34391916e-06, 9.90089207e-01, -1.34391916e-06],
        [1.82419797e-12, -1.34391916e-06, 9.90089207e-01]
    ]
    assert (cov_X_X - truth_cov_X_X < TEST_EPSILON).all()
    assert (inv_cov_X_X - truth_inv_cov_X_X < TEST_EPSILON).all()

def test_log_ml():
    dim_X = 3
    str_cov = 'se'
    X = np.reshape(np.arange(0, 9), (3, dim_X))
    Y = np.expand_dims(np.arange(3, 10, 3), axis=1)
    dict_hyps = utils_covariance.get_hyps(str_cov, dim_X)
    arr_hyps = utils_covariance.convert_hyps(str_cov, dict_hyps)
    prior_mu_X = np.zeros((3, 1))

    with pytest.raises(AssertionError) as error:
        gp.log_ml(np.arange(0, 3), Y, arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp.log_ml(X, np.arange(0, 3), arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp.log_ml(X, Y, dict_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp.log_ml(X, Y, arr_hyps, 1, prior_mu_X)
    with pytest.raises(ValueError) as error:
        gp.log_ml(X, Y, arr_hyps, 'abc', prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp.log_ml(X, Y, arr_hyps, str_cov, np.arange(0, 3))
    with pytest.raises(AssertionError) as error:
        gp.log_ml(np.reshape(np.arange(0, 12), (4, dim_X)), Y, arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp.log_ml(X, np.expand_dims(np.arange(0, 4), axis=1), arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp.log_ml(X, Y, arr_hyps, str_cov, np.expand_dims(np.arange(0, 4), axis=1))

    log_ml = gp.log_ml(X, Y, arr_hyps, str_cov, prior_mu_X)
    print(log_ml)
    truth_log_ml = 65.14727922868668
    assert log_ml - truth_log_ml < TEST_EPSILON

def test_get_optimized_kernels():
    np.random.seed(42)
    dim_X = 3
    num_X = 10
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)
    prior_mu = None

    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernels(X, Y, prior_mu, 1)
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernels(X, Y, 1, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernels(X, 1, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernels(1, Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernels(np.ones(num_X), Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernels(X, np.ones(num_X), prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernels(np.ones((50, 3)), Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernels(X, np.ones((50, 1)), prior_mu, 'se')
    with pytest.raises(ValueError) as error:
        gp.get_optimized_kernels(X, Y, prior_mu, 'abc')
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernels(X, Y, prior_mu, 'se', str_optimizer_method=1)
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernels(X, Y, prior_mu, 'se', str_optimizer_method='abc')

    cov_X_X, inv_cov_X_X, hyps = gp.get_optimized_kernels(X, Y, prior_mu, 'se')
    truth_cov_X_X = np.array([
        [1.53127632e+00, 1.04874635e-05, 2.09034295e-06, 1.44258966e+00, 2.52307976e-01, 3.75875428e-06, 1.87551266e-10, 4.17034048e-03, 7.29577363e-06, 1.20701462e+00],
        [1.04874635e-05, 1.53127632e+00, 1.05080350e+00, 2.91901661e-05, 5.30478911e-09, 6.03269550e-22, 1.02645748e-29, 1.51633995e-15, 1.64770978e-21, 5.16232197e-07],
        [2.09034295e-06, 1.05080350e+00, 1.53127632e+00, 4.94203139e-06, 2.28629061e-10, 1.73070475e-23, 1.44719094e-31, 1.39076746e-16, 1.03916942e-22, 6.44230585e-08],
        [1.44258966e+00, 2.91901661e-05, 4.94203139e-06, 1.53127632e+00, 2.70331032e-01, 1.43587674e-06, 5.47194704e-11, 1.77746865e-03, 2.21483484e-06, 1.11120816e+00],
        [2.52307976e-01, 5.30478911e-09, 2.28629061e-10, 2.70331032e-01, 1.53127632e+00, 7.85067754e-04, 4.63149234e-07, 3.41661000e-02, 3.54362237e-04, 6.96759249e-01],
        [3.75875428e-06, 6.03269550e-22, 1.73070475e-23, 1.43587674e-06, 7.85067754e-04, 1.53127632e+00, 3.76343791e-01, 2.86895629e-01, 9.93342783e-01, 7.03757059e-05],
        [1.87551266e-10, 1.02645748e-29, 1.44719094e-31, 5.47194704e-11, 4.63149234e-07, 3.76343791e-01, 1.53127632e+00, 3.80456969e-03, 1.56471678e-01, 1.02372519e-08],
        [4.17034048e-03, 1.51633995e-15, 1.39076746e-16, 1.77746865e-03, 3.41661000e-02, 2.86895629e-01, 3.80456969e-03, 1.53127632e+00, 4.86638256e-01, 2.21760033e-02],
        [7.29577363e-06, 1.64770978e-21, 1.03916942e-22, 2.21483484e-06, 3.54362237e-04, 9.93342783e-01, 1.56471678e-01, 4.86638256e-01, 1.53127632e+00, 9.13778107e-05],
        [1.20701462e+00, 5.16232197e-07, 6.44230585e-08, 1.11120816e+00, 6.96759249e-01, 7.03757059e-05, 1.02372519e-08, 2.21760033e-02, 9.13778107e-05, 1.53127632e+00]
    ])
    truth_inv_cov_X_X = np.array([
        [7.95966309e+00, 1.02370001e-04, -6.17635493e-05, -5.96593637e+00, 7.90228155e-01, -4.59414541e-04, 1.24750155e-04, 1.11992399e-03, -1.45288571e-04, -2.30438887e+00],
        [1.02370001e-04, 1.23428393e+00, -8.46999233e-01, -1.22912626e-04, 1.43330161e-06, -8.48178154e-09, -7.66696323e-09, -3.07439279e-07, 1.02901858e-07, 7.47412928e-06],
        [-6.17635493e-05, -8.46999233e-01, 1.23428393e+00, 7.32847938e-05, -1.03580263e-06, 5.07072521e-09, 4.44781625e-09, 1.79330755e-07, -6.00806612e-08, -3.79401719e-06],
        [-5.96593637e+00, -1.22912626e-04, 7.32847938e-05, 5.94382982e+00, -3.07214243e-01, 4.23923300e-04, 2.04886136e-04, 9.49703020e-03, -3.25472728e-03, 5.28960213e-01],
        [7.90228155e-01, 1.43330161e-06, -1.03580263e-06, -3.07214243e-01, 9.57501030e-01, -9.28748630e-04, -1.85506626e-04, -1.22666319e-02, 4.34471531e-03, -8.35455986e-01],
        [-4.59414541e-04, -8.48178154e-09, 5.07072521e-09, 4.23923300e-04, -9.28748630e-04, 1.19511838e+00, -2.16229100e-01, 1.77995954e-02, -7.58837662e-01, 2.09680763e-04],
        [1.24750155e-04, -7.66696323e-09, 4.44781625e-09, 2.04886136e-04, -1.85506626e-04, -2.16229100e-01, 6.99730736e-01, 1.88310072e-02, 6.27827827e-02, -4.29129778e-04],
        [1.11992399e-03, -3.07439279e-07, 1.79330755e-07, 9.49703020e-03, -1.22666319e-02, 1.77995954e-02, 1.88310072e-02, 7.27908070e-01, -2.44795826e-01, -1.27207796e-02],
        [-1.45288571e-04, 1.02901858e-07, -6.00806612e-08, -3.25472728e-03, 4.34471531e-03, -7.58837662e-01, 6.27827827e-02, -2.44795826e-01, 1.21668916e+00, 4.00688109e-03],
        [-2.30438887e+00, 7.47412928e-06, -3.79401719e-06, 5.28960213e-01, -8.35455986e-01, 2.09680763e-04, -4.29129778e-04, -1.27207796e-02, 4.00688109e-03, 2.46594263e+00]
    ])
    truth_hyps_signal = 1.2374434602026578
    truth_hyps_noise = -6.699256349819536e-05
    truth_hyps_lengthscales = np.array([2.10477723e-01, 1.21297509e+00, 3.94824773e+02])
    assert (cov_X_X - truth_cov_X_X < TEST_EPSILON).all()
    assert (inv_cov_X_X - truth_inv_cov_X_X < TEST_EPSILON).all()
    assert hyps['signal'] - truth_hyps_signal < TEST_EPSILON
    assert hyps['noise'] - truth_hyps_noise < TEST_EPSILON
    assert (hyps['lengthscales'] - truth_hyps_lengthscales < TEST_EPSILON).all()

def test_predict_test_():
    np.random.seed(42)
    dim_X = 2
    num_X = 5
    num_X_test = 10
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)
    X_test = np.random.randn(num_X_test, dim_X)
    prior_mu = None

    cov_X_X, inv_cov_X_X, hyps = gp.predict_test_(X, Y, prior_mu, 'se')

