# test_gp_scipy
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 07, 2020

import numpy as np
import pytest

from bayeso import constants
from bayeso.gp import gp_scipy
from bayeso.utils import utils_covariance


TEST_EPSILON = 1e-7

def test_neg_log_ml():
    dim_X = 3
    str_cov = 'se'
    X = np.reshape(np.arange(0, 9), (3, dim_X))
    Y = np.expand_dims(np.arange(3, 10, 3), axis=1)
    is_fixed_noise = False

    dict_hyps = utils_covariance.get_hyps(str_cov, dim_X)
    arr_hyps = utils_covariance.convert_hyps(str_cov, dict_hyps, is_fixed_noise=is_fixed_noise)
    prior_mu_X = np.zeros((3, 1))

    with pytest.raises(AssertionError) as error:
        gp_scipy.neg_log_ml(np.arange(0, 3), Y, arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp_scipy.neg_log_ml(X, np.arange(0, 3), arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp_scipy.neg_log_ml(X, Y, dict_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp_scipy.neg_log_ml(X, Y, arr_hyps, 1, prior_mu_X)
    with pytest.raises(ValueError) as error:
        gp_scipy.neg_log_ml(X, Y, arr_hyps, 'abc', prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp_scipy.neg_log_ml(X, Y, arr_hyps, str_cov, np.arange(0, 3))
    with pytest.raises(AssertionError) as error:
        gp_scipy.neg_log_ml(np.reshape(np.arange(0, 12), (4, dim_X)), Y, arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp_scipy.neg_log_ml(X, np.expand_dims(np.arange(0, 4), axis=1), arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp_scipy.neg_log_ml(X, Y, arr_hyps, str_cov, np.expand_dims(np.arange(0, 4), axis=1))
    with pytest.raises(AssertionError) as error:
        gp_scipy.neg_log_ml(X, Y, arr_hyps, str_cov, prior_mu_X, is_cholesky=1)
    with pytest.raises(AssertionError) as error:
        gp_scipy.neg_log_ml(X, Y, arr_hyps, str_cov, prior_mu_X, is_fixed_noise=1)
    with pytest.raises(AssertionError) as error:
        gp_scipy.neg_log_ml(X, Y, arr_hyps, str_cov, prior_mu_X, debug=1)

    neg_log_ml_ = gp_scipy.neg_log_ml(X, Y, arr_hyps, str_cov, prior_mu_X, is_fixed_noise=is_fixed_noise, is_gradient=False, is_cholesky=True)
    print(neg_log_ml_)
    truth_log_ml_ = 21.916650988532854
    assert np.abs(neg_log_ml_ - truth_log_ml_) < TEST_EPSILON

    neg_log_ml_ = gp_scipy.neg_log_ml(X, Y, arr_hyps, str_cov, prior_mu_X, is_fixed_noise=is_fixed_noise, is_gradient=False, is_cholesky=False)
    print(neg_log_ml_)
    truth_log_ml_ = 21.91665090519953
    assert np.abs(neg_log_ml_ - truth_log_ml_) < TEST_EPSILON

    neg_log_ml_ = gp_scipy.neg_log_ml(X, Y, arr_hyps, str_cov, prior_mu_X, is_fixed_noise=is_fixed_noise, is_gradient=True, is_cholesky=False)
    print(neg_log_ml_)
    truth_log_ml_ = 21.91665090519953
    assert np.abs(neg_log_ml_ - truth_log_ml_) < TEST_EPSILON

    neg_log_ml_, neg_grad_log_ml_ = gp_scipy.neg_log_ml(X, Y, arr_hyps, str_cov, prior_mu_X, is_fixed_noise=is_fixed_noise, is_gradient=True, is_cholesky=True)
    print(neg_log_ml_)
    print(neg_grad_log_ml_)

    truth_log_ml_ = 21.916650988532854
    truth_grad_log_ml_ = np.array([
        -4.09907399e-01,
        -4.09912156e+01,
        -8.88182458e-04,
        -8.88182458e-04,
        -8.88182458e-04,
    ])
    assert np.abs(neg_log_ml_ - truth_log_ml_) < TEST_EPSILON
    assert np.all(np.abs(neg_grad_log_ml_ - truth_grad_log_ml_) < TEST_EPSILON)

def test_neg_log_pseudo_l_loocv():
    dim_X = 3
    str_cov = 'se'
    X = np.reshape(np.arange(0, 9), (3, dim_X))
    Y = np.expand_dims(np.arange(3, 10, 3), axis=1)
    dict_hyps = utils_covariance.get_hyps(str_cov, dim_X)
    arr_hyps = utils_covariance.convert_hyps(str_cov, dict_hyps, is_fixed_noise=constants.IS_FIXED_GP_NOISE)
    prior_mu_X = np.zeros((3, 1))

    with pytest.raises(AssertionError) as error:
        gp_scipy.neg_log_pseudo_l_loocv(np.arange(0, 3), Y, arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp_scipy.neg_log_pseudo_l_loocv(X, np.arange(0, 3), arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp_scipy.neg_log_pseudo_l_loocv(X, Y, dict_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp_scipy.neg_log_pseudo_l_loocv(X, Y, arr_hyps, 1, prior_mu_X)
    with pytest.raises(ValueError) as error:
        gp_scipy.neg_log_pseudo_l_loocv(X, Y, arr_hyps, 'abc', prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp_scipy.neg_log_pseudo_l_loocv(X, Y, arr_hyps, str_cov, np.arange(0, 3))
    with pytest.raises(AssertionError) as error:
        gp_scipy.neg_log_pseudo_l_loocv(np.reshape(np.arange(0, 12), (4, dim_X)), Y, arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp_scipy.neg_log_pseudo_l_loocv(X, np.expand_dims(np.arange(0, 4), axis=1), arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp_scipy.neg_log_pseudo_l_loocv(X, Y, arr_hyps, str_cov, np.expand_dims(np.arange(0, 4), axis=1))
    with pytest.raises(AssertionError) as error:
        gp_scipy.neg_log_pseudo_l_loocv(X, Y, arr_hyps, str_cov, prior_mu_X, is_fixed_noise=1)
    with pytest.raises(AssertionError) as error:
        gp_scipy.neg_log_pseudo_l_loocv(X, Y, arr_hyps, str_cov, prior_mu_X, debug=1)

    neg_log_pseudo_l_ = gp_scipy.neg_log_pseudo_l_loocv(X, Y, arr_hyps, str_cov, prior_mu_X)
    print(neg_log_pseudo_l_)
    truth_log_pseudo_l_ = 21.916822991658695
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
        gp_scipy.get_optimized_kernel(X, Y, prior_mu, 1)
    with pytest.raises(AssertionError) as error:
        gp_scipy.get_optimized_kernel(X, Y, 1, 'se')
    with pytest.raises(AssertionError) as error:
        gp_scipy.get_optimized_kernel(X, 1, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp_scipy.get_optimized_kernel(1, Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp_scipy.get_optimized_kernel(np.ones(num_X), Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp_scipy.get_optimized_kernel(X, np.ones(num_X), prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp_scipy.get_optimized_kernel(np.ones((50, 3)), Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp_scipy.get_optimized_kernel(X, np.ones((50, 1)), prior_mu, 'se')
    with pytest.raises(ValueError) as error:
        gp_scipy.get_optimized_kernel(X, Y, prior_mu, 'abc')
    with pytest.raises(AssertionError) as error:
        gp_scipy.get_optimized_kernel(X, Y, prior_mu, 'se', str_optimizer_method=1)
    with pytest.raises(AssertionError) as error:
        gp_scipy.get_optimized_kernel(X, Y, prior_mu, 'se', str_modelselection_method=1)
    with pytest.raises(AssertionError) as error:
        gp_scipy.get_optimized_kernel(X, Y, prior_mu, 'se', is_fixed_noise=1)
    with pytest.raises(AssertionError) as error:
        gp_scipy.get_optimized_kernel(X, Y, prior_mu, 'se', debug=1)

    # INFO: tests for set inputs
    with pytest.raises(AssertionError) as error:
        gp_scipy.get_optimized_kernel(X_set, Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp_scipy.get_optimized_kernel(X, Y, prior_mu, 'set_se')
    with pytest.raises(AssertionError) as error:
        gp_scipy.get_optimized_kernel(X_set, Y, prior_mu, 'set_se', debug=1)

    cov_X_X, inv_cov_X_X, hyps = gp_scipy.get_optimized_kernel(X, Y, prior_mu, 'se')
    print(hyps)
    cov_X_X, inv_cov_X_X, hyps = gp_scipy.get_optimized_kernel(X, Y, prior_mu, 'se', str_optimizer_method='BFGS')
    print(hyps)
    cov_X_X, inv_cov_X_X, hyps = gp_scipy.get_optimized_kernel(X, Y, prior_mu, 'se', str_optimizer_method='L-BFGS-B')
    print(hyps)
    cov_X_X, inv_cov_X_X, hyps = gp_scipy.get_optimized_kernel(X, Y, prior_mu, 'se', str_optimizer_method='Nelder-Mead')
    print(hyps)
    cov_X_X, inv_cov_X_X, hyps = gp_scipy.get_optimized_kernel(X, Y, prior_mu, 'se', str_modelselection_method='loocv')
    print(hyps)
    cov_X_X, inv_cov_X_X, hyps = gp_scipy.get_optimized_kernel(X_set, Y, prior_mu, 'set_se')
    print(hyps)
    cov_X_X, inv_cov_X_X, hyps = gp_scipy.get_optimized_kernel(X_set, Y, prior_mu, 'set_se', str_optimizer_method='L-BFGS-B')
    print(hyps)
    cov_X_X, inv_cov_X_X, hyps = gp_scipy.get_optimized_kernel(X_set, Y, prior_mu, 'set_se', str_modelselection_method='loocv')
    print(hyps)
