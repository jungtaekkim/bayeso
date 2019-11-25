# test_gp
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 09, 2018

import numpy as np
import pytest

from bayeso import gp
from bayeso import constants
from bayeso.utils import utils_covariance


TEST_EPSILON = 1e-7

def test_check_str_cov():
    with pytest.raises(AssertionError) as error:
        gp._check_str_cov(1, 'se', (2, 1))
    with pytest.raises(AssertionError) as error:
        gp._check_str_cov('test', 1, (2, 1))
    with pytest.raises(AssertionError) as error:
        gp._check_str_cov('test', 'se', 1)
    with pytest.raises(AssertionError) as error:
        gp._check_str_cov('test', 'se', (2, 100, 100))
    with pytest.raises(AssertionError) as error:
        gp._check_str_cov('test', 'se', (2, 100), shape_X2=(2, 100, 100))
    with pytest.raises(AssertionError) as error:
        gp._check_str_cov('test', 'set_se', (2, 100), shape_X2=(2, 100, 100))
    with pytest.raises(AssertionError) as error:
        gp._check_str_cov('test', 'set_se', (2, 100, 100), shape_X2=(2, 100))
    with pytest.raises(AssertionError) as error:
        gp._check_str_cov('test', 'se', (2, 1), shape_X2=1)

    with pytest.raises(ValueError) as error:
        gp._check_str_cov('test', 'abc', (2, 1))

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

def test_get_kernel_inverse():
    dim_X = 3
    X = np.reshape(np.arange(0, 9), (3, dim_X))
    hyps = utils_covariance.get_hyps('se', dim_X)

    with pytest.raises(AssertionError) as error:
        gp.get_kernel_inverse(1, hyps, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_kernel_inverse(np.arange(0, 100), hyps, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_kernel_inverse(X, 1, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_kernel_inverse(X, hyps, 1)
    with pytest.raises(ValueError) as error:
        gp.get_kernel_inverse(X, hyps, 'abc')
    with pytest.raises(AssertionError) as error:
        gp.get_kernel_inverse(X, hyps, 'se', debug=1)
    with pytest.raises(AssertionError) as error:
        gp.get_kernel_inverse(X, hyps, 'se', is_gradient='abc')
    with pytest.raises(AssertionError) as error:
        gp.get_kernel_inverse(X, hyps, 'se', is_fixed_noise='abc')

    cov_X_X, inv_cov_X_X, grad_cov_X_X = gp.get_kernel_inverse(X, hyps, 'se')
    print(cov_X_X)
    print(inv_cov_X_X)
    truth_cov_X_X = np.array([
        [1.00011000e+00, 1.37095909e-06, 3.53262857e-24],
        [1.37095909e-06, 1.00011000e+00, 1.37095909e-06],
        [3.53262857e-24, 1.37095909e-06, 1.00011000e+00]
    ])
    truth_inv_cov_X_X = np.array([
        [9.99890012e-01, -1.37065753e-06, 1.87890871e-12],
        [-1.37065753e-06, 9.99890012e-01, -1.37065753e-06],
        [1.87890871e-12, -1.37065753e-06, 9.99890012e-01]
    ])
    assert (np.abs(cov_X_X - truth_cov_X_X) < TEST_EPSILON).all()
    assert (np.abs(inv_cov_X_X - truth_inv_cov_X_X) < TEST_EPSILON).all()
    assert cov_X_X.shape == inv_cov_X_X.shape

    cov_X_X, inv_cov_X_X, grad_cov_X_X = gp.get_kernel_inverse(X, hyps, 'se', is_gradient=True)
    print(grad_cov_X_X)
    print(grad_cov_X_X.shape)

    truth_grad_cov_X_X = np.array([
        [
            [2.00002000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [2.74191817e-06, 3.70158953e-05, 3.70158953e-05, 3.70158953e-05],
            [7.06525714e-24, 3.81523886e-22, 3.81523886e-22, 3.81523886e-22]
        ], [
            [2.74191817e-06, 3.70158953e-05, 3.70158953e-05, 3.70158953e-05],
            [2.00002000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [2.74191817e-06, 3.70158953e-05, 3.70158953e-05, 3.70158953e-05]
        ], [
            [7.06525714e-24, 3.81523886e-22, 3.81523886e-22, 3.81523886e-22],
            [2.74191817e-06, 3.70158953e-05, 3.70158953e-05, 3.70158953e-05],
            [2.00002000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]
        ]
    ])
    assert (np.abs(cov_X_X - truth_cov_X_X) < TEST_EPSILON).all()
    assert (np.abs(inv_cov_X_X - truth_inv_cov_X_X) < TEST_EPSILON).all()
    assert (np.abs(grad_cov_X_X - truth_grad_cov_X_X) < TEST_EPSILON).all()
    assert cov_X_X.shape == inv_cov_X_X.shape == grad_cov_X_X.shape[:2]

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
    with pytest.raises(ValueError) as error:
        gp.get_kernel_cholesky(X, hyps, 'abc')
    with pytest.raises(AssertionError) as error:
        gp.get_kernel_cholesky(X, hyps, 'se', debug=1)

    cov_X_X, lower, _ = gp.get_kernel_cholesky(X, hyps, 'se')
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
    assert (np.abs(cov_X_X - truth_cov_X_X) < TEST_EPSILON).all()
    assert (np.abs(lower - truth_lower) < TEST_EPSILON).all()
    assert cov_X_X.shape == lower.shape

def test_neg_log_ml():
    dim_X = 3
    str_cov = 'se'
    X = np.reshape(np.arange(0, 9), (3, dim_X))
    Y = np.expand_dims(np.arange(3, 10, 3), axis=1)
    dict_hyps = utils_covariance.get_hyps(str_cov, dim_X)
    arr_hyps = utils_covariance.convert_hyps(str_cov, dict_hyps, is_fixed_noise=constants.IS_FIXED_GP_NOISE)
    prior_mu_X = np.zeros((3, 1))

    with pytest.raises(AssertionError) as error:
        gp.neg_log_ml(np.arange(0, 3), Y, arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp.neg_log_ml(X, np.arange(0, 3), arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp.neg_log_ml(X, Y, dict_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp.neg_log_ml(X, Y, arr_hyps, 1, prior_mu_X)
    with pytest.raises(ValueError) as error:
        gp.neg_log_ml(X, Y, arr_hyps, 'abc', prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp.neg_log_ml(X, Y, arr_hyps, str_cov, np.arange(0, 3))
    with pytest.raises(AssertionError) as error:
        gp.neg_log_ml(np.reshape(np.arange(0, 12), (4, dim_X)), Y, arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp.neg_log_ml(X, np.expand_dims(np.arange(0, 4), axis=1), arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp.neg_log_ml(X, Y, arr_hyps, str_cov, np.expand_dims(np.arange(0, 4), axis=1))
    with pytest.raises(AssertionError) as error:
        gp.neg_log_ml(X, Y, arr_hyps, str_cov, prior_mu_X, is_cholesky=1)
    with pytest.raises(AssertionError) as error:
        gp.neg_log_ml(X, Y, arr_hyps, str_cov, prior_mu_X, is_fixed_noise=1)
    with pytest.raises(AssertionError) as error:
        gp.neg_log_ml(X, Y, arr_hyps, str_cov, prior_mu_X, debug=1)

    neg_log_ml_ = gp.neg_log_ml(X, Y, arr_hyps, str_cov, prior_mu_X, is_gradient=False)
    print(neg_log_ml_)
    truth_log_ml_ = 65.74995266591506
    assert np.abs(neg_log_ml_ - truth_log_ml_) < TEST_EPSILON

    log_ml_ = gp.neg_log_ml(X, Y, arr_hyps, str_cov, prior_mu_X, is_cholesky=False)
    print(neg_log_ml_)
    truth_log_ml_ = 65.74995266566506
    assert np.abs(neg_log_ml_ - truth_log_ml_) < TEST_EPSILON

def test_neg_log_pseudo_l_loocv():
    dim_X = 3
    str_cov = 'se'
    X = np.reshape(np.arange(0, 9), (3, dim_X))
    Y = np.expand_dims(np.arange(3, 10, 3), axis=1)
    dict_hyps = utils_covariance.get_hyps(str_cov, dim_X)
    arr_hyps = utils_covariance.convert_hyps(str_cov, dict_hyps, is_fixed_noise=constants.IS_FIXED_GP_NOISE)
    prior_mu_X = np.zeros((3, 1))

    with pytest.raises(AssertionError) as error:
        gp.neg_log_pseudo_l_loocv(np.arange(0, 3), Y, arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp.neg_log_pseudo_l_loocv(X, np.arange(0, 3), arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp.neg_log_pseudo_l_loocv(X, Y, dict_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp.neg_log_pseudo_l_loocv(X, Y, arr_hyps, 1, prior_mu_X)
    with pytest.raises(ValueError) as error:
        gp.neg_log_pseudo_l_loocv(X, Y, arr_hyps, 'abc', prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp.neg_log_pseudo_l_loocv(X, Y, arr_hyps, str_cov, np.arange(0, 3))
    with pytest.raises(AssertionError) as error:
        gp.neg_log_pseudo_l_loocv(np.reshape(np.arange(0, 12), (4, dim_X)), Y, arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp.neg_log_pseudo_l_loocv(X, np.expand_dims(np.arange(0, 4), axis=1), arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp.neg_log_pseudo_l_loocv(X, Y, arr_hyps, str_cov, np.expand_dims(np.arange(0, 4), axis=1))
    with pytest.raises(AssertionError) as error:
        gp.neg_log_pseudo_l_loocv(X, Y, arr_hyps, str_cov, prior_mu_X, is_fixed_noise=1)
    with pytest.raises(AssertionError) as error:
        gp.neg_log_pseudo_l_loocv(X, Y, arr_hyps, str_cov, prior_mu_X, debug=1)

    neg_log_pseudo_l_ = gp.neg_log_pseudo_l_loocv(X, Y, arr_hyps, str_cov, prior_mu_X)
    print(neg_log_pseudo_l_)
    truth_log_pseudo_l_ = -65.75046897497609
    assert np.abs(neg_log_pseudo_l_ - truth_log_pseudo_l_) < TEST_EPSILON

def test_get_optimized_kernel():
    np.random.seed(42)
    dim_X = 3
    num_X = 10
    num_instances = 5
    X = np.random.randn(num_X, dim_X)
    X_set = np.random.randn(num_X, num_instances, dim_X)
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
        gp.get_optimized_kernel(X, Y, prior_mu, 'se', str_modelselection_method=1)
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(X, Y, prior_mu, 'se', is_fixed_noise=1)
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(X, Y, prior_mu, 'se', debug=1)

    # INFO: tests for set inputs
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(X_set, Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(X, Y, prior_mu, 'set_se')
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(X_set, Y, prior_mu, 'set_se', debug=1)

    gp.get_optimized_kernel(X, Y, prior_mu, 'se')
    gp.get_optimized_kernel(X, Y, prior_mu, 'se', str_optimizer_method='L-BFGS-B')
    gp.get_optimized_kernel(X, Y, prior_mu, 'se', str_modelselection_method='loocv')
    gp.get_optimized_kernel(X_set, Y, prior_mu, 'set_se')
    gp.get_optimized_kernel(X_set, Y, prior_mu, 'set_se', str_optimizer_method='L-BFGS-B')
    gp.get_optimized_kernel(X_set, Y, prior_mu, 'set_se', str_modelselection_method='loocv')

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
    
    mu_Xs, sigma_Xs = gp.predict_test(X, Y, X_test, hyps, str_cov='se', prior_mu=prior_mu)
    print(mu_Xs)
    print(sigma_Xs)

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
    
    mu_Xs, sigma_Xs = gp.predict_optimized(X, Y, X_test)
    print(mu_Xs)
    print(sigma_Xs)
