# test_gp
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 09, 2018

import numpy as np
import pytest

from bayeso import gp
from bayeso.utils import utils_covariance


TEST_EPSILON = 1e-7

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
    with pytest.raises(AssertionError) as error:
        gp.get_kernels(X, hyps, 'abc')
    with pytest.raises(AssertionError) as error:
        gp.get_kernels(X, hyps, 'se', debug=1)

    cov_X_X, inv_cov_X_X = gp.get_kernels(X, hyps, 'se')
    print(cov_X_X)
    print(inv_cov_X_X)
    truth_cov_X_X = [
        [1.00011000e+00, 1.37095909e-06, 3.53262857e-24],
        [1.37095909e-06, 1.00011000e+00, 1.37095909e-06],
        [3.53262857e-24, 1.37095909e-06, 1.00011000e+00]
    ]
    truth_inv_cov_X_X = [
        [9.99890012e-01, -1.37065753e-06, 1.87890871e-12],
        [-1.37065753e-06, 9.99890012e-01, -1.37065753e-06],
        [1.87890871e-12, -1.37065753e-06, 9.99890012e-01]
    ]
    assert (cov_X_X - truth_cov_X_X < TEST_EPSILON).all()
    assert (inv_cov_X_X - truth_inv_cov_X_X < TEST_EPSILON).all()
    assert cov_X_X.shape == inv_cov_X_X.shape

def test_get_kernel_cholesky():
    dim_X = 3
    X = np.reshape(np.arange(0, 9), (3, dim_X))
    hyps = utils_covariance.get_hyps('se', dim_X)

    with pytest.raises(AssertionError) as error:
        gp.get_kernel_cholesky(1, hyps, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_kernel_cholesky(np.arange(0, 10), hyps, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_kernel_cholesky(X, 1, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_kernel_cholesky(X, hyps, 1)
    with pytest.raises(AssertionError) as error:
        gp.get_kernel_cholesky(X, hyps, 'abc')
    with pytest.raises(AssertionError) as error:
        gp.get_kernel_cholesky(X, hyps, 'se', debug=1)

    cov_X_X, lower = gp.get_kernel_cholesky(X, hyps, 'se')
    print(cov_X_X)
    print(lower)
    truth_cov_X_X = [
        [1.00011000e+00, 1.37095909e-06, 3.53262857e-24],
        [1.37095909e-06, 1.00011000e+00, 1.37095909e-06],
        [3.53262857e-24, 1.37095909e-06, 1.00011000e+00],
    ]
    truth_lower = [
        [1.00005500e+00, 0.00000000e+00, 0.00000000e+00],
        [1.37088369e-06, 1.00005500e+00, 0.00000000e+00],
        [3.53243429e-24, 1.37088369e-06, 1.00005500e+00],
    ]
    assert (cov_X_X - truth_cov_X_X < TEST_EPSILON).all()
    assert (lower - truth_lower < TEST_EPSILON).all()
    assert cov_X_X.shape == lower.shape

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
    with pytest.raises(AssertionError) as error:
        gp.log_ml(X, Y, arr_hyps, str_cov, prior_mu_X, is_fixed_noise=1)
    with pytest.raises(AssertionError) as error:
        gp.log_ml(X, Y, arr_hyps, str_cov, prior_mu_X, debug=1)

    log_ml = gp.log_ml(X, Y, arr_hyps, str_cov, prior_mu_X)
    print(log_ml)
    truth_log_ml = 65.14727922868668
    assert log_ml - truth_log_ml < TEST_EPSILON

def test_get_optimized_kernel():
    np.random.seed(42)
    dim_X = 3
    num_X = 10
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)
    prior_mu = None

    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(X, Y, prior_mu, 1)
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(X, Y, 1, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(X, 1, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(1, Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(np.ones(num_X), Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(X, np.ones(num_X), prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(np.ones((50, 3)), Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(X, np.ones((50, 1)), prior_mu, 'se')
    with pytest.raises(ValueError) as error:
        gp.get_optimized_kernel(X, Y, prior_mu, 'abc')
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(X, Y, prior_mu, 'se', str_optimizer_method=1)
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(X, Y, prior_mu, 'se', is_fixed_noise=1)
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(X, Y, prior_mu, 'se', debug=1)

def test_predict_test_():
    np.random.seed(42)
    dim_X = 2
    num_X = 5
    num_X_test = 20
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)
    X_test = np.random.randn(num_X_test, dim_X)
    prior_mu = None
    cov_X_X, inv_cov_X_X, hyps = gp.get_optimized_kernel(X, Y, prior_mu, 'se')
    
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(X, Y, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov='se', prior_mu='abc')
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(X, Y, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov=1, prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(X, Y, X_test, cov_X_X, inv_cov_X_X, 1, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(X, Y, X_test, cov_X_X, 1, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(X, Y, X_test, 1, inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(X, Y, 1, cov_X_X, inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(X, 1, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(1, Y, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(np.random.randn(num_X, 1), Y, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(np.random.randn(10, dim_X), Y, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(X, np.random.randn(10, 1), X_test, cov_X_X, inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(X, Y, X_test, np.random.randn(3, 3), inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(X, Y, X_test, np.random.randn(10), inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(X, Y, X_test, cov_X_X, np.random.randn(10), hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(X, Y, X_test, np.random.randn(10), np.random.randn(10), hyps, str_cov='se', prior_mu=prior_mu)

def test_predict_test():
    np.random.seed(42)
    dim_X = 2
    num_X = 5
    num_X_test = 20
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)
    X_test = np.random.randn(num_X_test, dim_X)
    prior_mu = None
    cov_X_X, inv_cov_X_X, hyps = gp.get_optimized_kernel(X, Y, prior_mu, 'se')
    
    with pytest.raises(AssertionError) as error:
        gp.predict_test(X, Y, X_test, hyps, str_cov='se', prior_mu='abc')
    with pytest.raises(AssertionError) as error:
        gp.predict_test(X, Y, X_test, hyps, str_cov=1, prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test(X, Y, X_test, 1, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test(X, Y, 1, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test(X, 1, X_test, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test(1, Y, X_test, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test(np.random.randn(num_X, 1), Y, X_test, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test(np.random.randn(10, dim_X), Y, X_test, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test(X, np.random.randn(10, 1), X_test, hyps, str_cov='se', prior_mu=prior_mu)

def test_predict_optimized():
    np.random.seed(42)
    dim_X = 2
    num_X = 5
    num_X_test = 20
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)
    X_test = np.random.randn(num_X_test, dim_X)
    prior_mu = None
    
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(X, Y, X_test, str_cov='se', prior_mu='abc')
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(X, Y, X_test, str_cov=1, prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(X, Y, 1, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(X, 1, X_test, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(1, Y, X_test, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(np.random.randn(num_X, 1), Y, X_test, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(np.random.randn(10, dim_X), Y, X_test, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(X, np.random.randn(10, 1), X_test, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(X, Y, X_test, is_fixed_noise=1)
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(X, Y, X_test, debug=1)
