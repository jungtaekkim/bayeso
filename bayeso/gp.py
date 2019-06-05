# gp
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: June 03, 2019

import time
import numpy as np
import scipy 
import scipy.linalg
import scipy.optimize

from bayeso import covariance
from bayeso import constants
from bayeso.utils import utils_covariance


def _check_str_cov(str_fun, str_cov, shape_X1, shape_X2=None):
    assert isinstance(str_fun, str)
    assert isinstance(str_cov, str)
    assert isinstance(shape_X1, tuple)
    assert shape_X2 is None or isinstance(shape_X2, tuple)

    if str_cov in constants.ALLOWED_GP_COV_BASE:
        assert len(shape_X1) == 2
        if shape_X2 is not None:
            assert len(shape_X2) == 2
    elif str_cov in constants.ALLOWED_GP_COV_SET:
        assert len(shape_X1) == 3
        if shape_X2 is not None:
            assert len(shape_X2) == 3
    elif str_cov in constants.ALLOWED_GP_COV: # pragma: no cover
        raise ValueError('{}: missing conditions for str_cov.'.format(str_fun))
    else:
        raise ValueError('{}: invalid str_cov.'.format(str_fun))
    return

def get_prior_mu(prior_mu, X):
    assert isinstance(X, np.ndarray)
    assert callable(prior_mu) or prior_mu is None
    assert len(X.shape) == 2 or len(X.shape) == 3

    if prior_mu is None:
        prior_mu_X = np.zeros((X.shape[0], 1))
    else:
        prior_mu_X = prior_mu(X)
        assert len(prior_mu_X.shape) == 2
        assert X.shape[0] == prior_mu_X.shape[0]
    return prior_mu_X

def get_kernel_inverse(X_train, hyps, str_cov,
    debug=False
):
    assert isinstance(X_train, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(str_cov, str)
    assert isinstance(debug, bool)
    _check_str_cov('get_kernel_inverse', str_cov, X_train.shape)

    cov_X_X = covariance.cov_main(str_cov, X_train, X_train, hyps) + hyps['noise']**2 * np.eye(X_train.shape[0])
    cov_X_X = (cov_X_X + cov_X_X.T) / 2.0
    inv_cov_X_X = np.linalg.inv(cov_X_X)
    return cov_X_X, inv_cov_X_X

def get_kernel_cholesky(X_train, hyps, str_cov,
    debug=False
):
    assert isinstance(X_train, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(str_cov, str)
    assert isinstance(debug, bool)
    _check_str_cov('get_kernel_cholesky', str_cov, X_train.shape)
   
    cov_X_X = covariance.cov_main(str_cov, X_train, X_train, hyps) + hyps['noise']**2 * np.eye(X_train.shape[0])
    cov_X_X = (cov_X_X + cov_X_X.T) / 2.0
    lower = scipy.linalg.cholesky(cov_X_X, lower=True)
    return cov_X_X, lower

def log_ml(X_train, Y_train, hyps, str_cov, prior_mu_train,
    is_fixed_noise=constants.IS_FIXED_GP_NOISE,
    is_cholesky=True,
    debug=False
):
    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(hyps, np.ndarray)
    assert isinstance(str_cov, str)
    assert isinstance(prior_mu_train, np.ndarray)
    assert isinstance(is_fixed_noise, bool)
    assert isinstance(is_cholesky, bool)
    assert isinstance(debug, bool)
    assert len(Y_train.shape) == 2
    assert len(prior_mu_train.shape) == 2
    assert X_train.shape[0] == Y_train.shape[0] == prior_mu_train.shape[0]
    _check_str_cov('log_ml', str_cov, X_train.shape)

    hyps = utils_covariance.restore_hyps(str_cov, hyps, is_fixed_noise=is_fixed_noise)
    new_Y_train = Y_train - prior_mu_train
    if is_cholesky:
        cov_X_X, lower = get_kernel_cholesky(X_train, hyps, str_cov, debug=debug)

#        lower_new_Y_train = scipy.linalg.cho_solve((lower, True), new_Y_train)
        alpha = scipy.linalg.cho_solve((lower, True), new_Y_train)

        first_term = -0.5 * np.dot(new_Y_train.T, alpha)
        second_term = -1.0 * np.sum(np.log(np.diagonal(lower) + constants.JITTER_LOG))
    else:
        cov_X_X, inv_cov_X_X = get_kernel_inverse(X_train, hyps, str_cov, debug=debug)

        first_term = -0.5 * np.dot(np.dot(new_Y_train.T, inv_cov_X_X), new_Y_train)
        second_term = -0.5 * np.log(np.linalg.det(cov_X_X) + constants.JITTER_LOG)
        
    third_term = -float(X_train.shape[0]) / 2.0 * np.log(2.0 * np.pi)
    log_ml_ = np.squeeze(first_term + second_term + third_term)
    return log_ml_

def log_pseudo_l_loocv(X_train, Y_train, hyps, str_cov, prior_mu_train,
    is_fixed_noise=constants.IS_FIXED_GP_NOISE,
    debug=False
):
    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(hyps, np.ndarray)
    assert isinstance(str_cov, str)
    assert isinstance(prior_mu_train, np.ndarray)
    assert isinstance(is_fixed_noise, bool)
    assert isinstance(debug, bool)
    assert len(Y_train.shape) == 2
    assert len(prior_mu_train.shape) == 2
    assert X_train.shape[0] == Y_train.shape[0] == prior_mu_train.shape[0]
    _check_str_cov('log_pseudo_l_loocv', str_cov, X_train.shape)

    num_data = X_train.shape[0]
    hyps = utils_covariance.restore_hyps(str_cov, hyps, is_fixed_noise=is_fixed_noise)

    cov_X_X, inv_cov_X_X = get_kernel_inverse(X_train, hyps, str_cov, debug=debug)

    log_pseudo_l = 0.0
    for ind_data in range(0, num_data):
        cur_X_train = np.vstack((X_train[:ind_data], X_train[ind_data+1:]))
        cur_Y_train = np.vstack((Y_train[:ind_data], Y_train[ind_data+1:]))
        
        cur_X_test = np.expand_dims(X_train[ind_data], axis=0)
        cur_Y_test = Y_train[ind_data]

        cur_mu = np.squeeze(cur_Y_test) - np.dot(inv_cov_X_X, Y_train)[ind_data] / inv_cov_X_X[ind_data, ind_data]
        cur_sigma = np.sqrt(1.0 / (inv_cov_X_X[ind_data, ind_data] + constants.JITTER_COV))

        first_term = -0.5 * np.log(cur_sigma**2)
        second_term = -0.5 * (np.squeeze(cur_Y_test - cur_mu))**2 / (cur_sigma**2)
        third_term = -0.5 * np.log(2.0 * np.pi)
        cur_log_pseudo_l = first_term + second_term + third_term
        log_pseudo_l += cur_log_pseudo_l

    return log_pseudo_l

def get_optimized_kernel(X_train, Y_train, prior_mu, str_cov,
    str_optimizer_method=constants.STR_OPTIMIZER_METHOD_GP,
    str_modelselection_method=constants.STR_MODELSELECTION_METHOD,
    is_fixed_noise=constants.IS_FIXED_GP_NOISE,
    debug=False
):
    # TODO: check to input same is_fixed_noise to convert_hyps and restore_hyps
    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert callable(prior_mu) or prior_mu is None
    assert isinstance(str_cov, str)
    assert isinstance(str_optimizer_method, str)
    assert isinstance(str_modelselection_method, str)
    assert isinstance(is_fixed_noise, bool)
    assert isinstance(debug, bool)
    assert len(Y_train.shape) == 2
    assert X_train.shape[0] == Y_train.shape[0]
    _check_str_cov('get_optimized_kernel', str_cov, X_train.shape)
    assert str_optimizer_method in constants.ALLOWED_OPTIMIZER_METHOD_GP
    assert str_modelselection_method in constants.ALLOWED_MODELSELECTION_METHOD

    time_start = time.time()

    if debug:
        print('[DEBUG] get_optimized_kernel in gp.py: str_optimizer_method {}'.format(str_optimizer_method))
        print('[DEBUG] get_optimized_kernel in gp.py: str_modelselection_method {}'.format(str_modelselection_method))

    prior_mu_train = get_prior_mu(prior_mu, X_train)
    if str_cov in constants.ALLOWED_GP_COV_BASE:
        num_dim = X_train.shape[1]
    elif str_cov in constants.ALLOWED_GP_COV_SET:
        num_dim = X_train.shape[2]

    if str_modelselection_method == 'ml':
        neg_log_ml = lambda hyps: -1.0 * log_ml(X_train, Y_train, hyps, str_cov, prior_mu_train, is_fixed_noise=is_fixed_noise, debug=debug)
    elif str_modelselection_method == 'loocv':
        neg_log_ml = lambda hyps: -1.0 * log_pseudo_l_loocv(X_train, Y_train, hyps, str_cov, prior_mu_train, is_fixed_noise=is_fixed_noise, debug=debug)
    else: # pragma: no cover
        raise ValueError('get_optimized_kernel: missing conditions for str_modelselection_method.')

    hyps_converted = utils_covariance.convert_hyps(
        str_cov,
        utils_covariance.get_hyps(str_cov, num_dim),
        is_fixed_noise=is_fixed_noise,
    )

    if str_optimizer_method == 'BFGS':
        result_optimized = scipy.optimize.minimize(neg_log_ml, hyps_converted, method=str_optimizer_method)
        result_optimized = result_optimized.x
    elif str_optimizer_method == 'L-BFGS-B':
        bounds = utils_covariance.get_range_hyps(str_cov, num_dim, is_fixed_noise=is_fixed_noise)
        result_optimized = scipy.optimize.minimize(neg_log_ml, hyps_converted, method=str_optimizer_method, bounds=bounds)
        result_optimized = result_optimized.x
    # TODO: Fill this conditions
    elif str_optimizer_method == 'DIRECT': # pragma: no cover
        raise NotImplementedError('get_optimized_kernel: allowed str_optimizer_method, but it is not implemented.')
    elif str_optimizer_method == 'CMA-ES': # pragma: no cover
        raise NotImplementedError('get_optimized_kernel: allowed str_optimizer_method, but it is not implemented.')
    # INFO: It is allowed, but a condition is missed.
    else: # pragma: no cover
        raise ValueError('get_optimized_kernel: missing conditions for str_optimizer_method')

    hyps = utils_covariance.restore_hyps(str_cov, result_optimized, is_fixed_noise=is_fixed_noise)

    hyps, _ = utils_covariance.validate_hyps_dict(hyps, str_cov, num_dim)
    cov_X_X, inv_cov_X_X = get_kernel_inverse(X_train, hyps, str_cov, debug=debug)

    time_end = time.time()

    if debug:
        print('[DEBUG] get_optimized_kernel in gp.py: optimized hyps for gpr', hyps)
        print('[DEBUG] get_optimized_kernel in gp.py: time consumed', time_end - time_start, 'sec.')
    return cov_X_X, inv_cov_X_X, hyps

def predict_test_(X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps,
    str_cov=constants.STR_GP_COV,
    prior_mu=None,
    debug=False
):
    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(str_cov, str)
    assert isinstance(debug, bool)
    assert callable(prior_mu) or prior_mu is None
    assert len(Y_train.shape) == 2
    assert len(cov_X_X.shape) == 2
    assert len(inv_cov_X_X.shape) == 2
    assert (np.array(cov_X_X.shape) == np.array(inv_cov_X_X.shape)).all()
    _check_str_cov('predict_test_', str_cov, X_train.shape, shape_X2=X_test.shape)
    assert X_train.shape[0] == Y_train.shape[0]
    assert X_train.shape[1] == X_test.shape[1]

    prior_mu_train = get_prior_mu(prior_mu, X_train)
    prior_mu_test = get_prior_mu(prior_mu, X_test)
    cov_X_Xs = covariance.cov_main(str_cov, X_train, X_test, hyps)
    cov_Xs_Xs = covariance.cov_main(str_cov, X_test, X_test, hyps)
    cov_Xs_Xs = (cov_Xs_Xs + cov_Xs_Xs.T) / 2.0

    mu_Xs = np.dot(np.dot(cov_X_Xs.T, inv_cov_X_X), Y_train - prior_mu_train) + prior_mu_test
    Sigma_Xs = cov_Xs_Xs - np.dot(np.dot(cov_X_Xs.T, inv_cov_X_X), cov_X_Xs)
    return mu_Xs, np.expand_dims(np.sqrt(np.maximum(np.diag(Sigma_Xs), 0.0)), axis=1)

def predict_test(X_train, Y_train, X_test, hyps,
    str_cov=constants.STR_GP_COV,
    prior_mu=None,
    debug=False
):
    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(str_cov, str)
    assert isinstance(debug, bool)
    assert callable(prior_mu) or prior_mu is None
    assert len(Y_train.shape) == 2
    _check_str_cov('predict_test', str_cov, X_train.shape, shape_X2=X_test.shape)
    assert X_train.shape[0] == Y_train.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    
    cov_X_X, inv_cov_X_X = get_kernel_inverse(X_train, hyps, str_cov, debug=debug)
    mu_Xs, sigma_Xs = predict_test_(X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov=str_cov, prior_mu=prior_mu, debug=debug)
    return mu_Xs, sigma_Xs

def predict_optimized(X_train, Y_train, X_test,
    str_cov=constants.STR_GP_COV,
    prior_mu=None,
    is_fixed_noise=constants.IS_FIXED_GP_NOISE,
    debug=False
):
    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(str_cov, str)
    assert isinstance(is_fixed_noise, bool)
    assert isinstance(debug, bool)
    assert callable(prior_mu) or prior_mu is None
    assert len(Y_train.shape) == 2
    _check_str_cov('predict_optimized', str_cov, X_train.shape, shape_X2=X_test.shape)
    assert X_train.shape[0] == Y_train.shape[0]
    assert X_train.shape[1] == X_test.shape[1]

    time_start = time.time()

    cov_X_X, inv_cov_X_X, hyps = get_optimized_kernel(X_train, Y_train, prior_mu, str_cov, is_fixed_noise=is_fixed_noise, debug=debug)
    mu_Xs, sigma_Xs = predict_test_(X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov=str_cov, prior_mu=prior_mu, debug=debug)

    time_end = time.time()
    if debug:
        print('[DEBUG] predict_optimized in gp.py: time consumed', time_end - time_start, 'sec.')
    return mu_Xs, sigma_Xs
