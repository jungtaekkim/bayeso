# gp
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 04, 2018

import time
import numpy as np
import scipy 
import scipy.optimize

from bayeso import covariance
from bayeso import constants
from bayeso.utils import utils_covariance


def get_prior_mu(prior_mu, X):
    assert isinstance(X, np.ndarray)
    assert callable(prior_mu) or prior_mu is None
    assert len(X.shape) == 2

    if prior_mu is None:
        prior_mu_X = np.zeros((X.shape[0], 1))
    else:
        prior_mu_X = prior_mu(X)
        assert len(prior_mu_X.shape) == 2
        assert X.shape[0] == prior_mu_X.shape[0]
    return prior_mu_X

def get_kernels(X_train, hyps, str_cov, debug=False):
    assert isinstance(X_train, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(str_cov, str)
    assert isinstance(debug, bool)
    assert len(X_train.shape) == 2

    cov_X_X = covariance.cov_main(str_cov, X_train, X_train, hyps) + hyps['noise']**2 * np.eye(X_train.shape[0])
    cov_X_X = (cov_X_X + cov_X_X.T) / 2.0
    inv_cov_X_X = np.linalg.inv(cov_X_X)
    return cov_X_X, inv_cov_X_X

def get_kernel_cholesky(X_train, hyps, str_cov, debug=False):
    assert isinstance(X_train, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(str_cov, str)
    assert isinstance(debug, bool)
    assert len(X_train.shape) == 2
    
    cov_X_X = covariance.cov_main(str_cov, X_train, X_train, hyps) + hyps['noise']**2 * np.eye(X_train.shape[0])
    cov_X_X = (cov_X_X + cov_X_X.T) / 2.0
    lower = np.linalg.cholesky(cov_X_X)
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
    assert len(X_train.shape) == 2
    assert len(Y_train.shape) == 2
    assert len(prior_mu_train.shape) == 2
    assert X_train.shape[0] == Y_train.shape[0] == prior_mu_train.shape[0]

    hyps = utils_covariance.restore_hyps(str_cov, hyps, is_fixed_noise=is_fixed_noise)
    new_Y_train = Y_train - prior_mu_train
    if is_cholesky:
        cov_X_X, lower = get_kernel_cholesky(X_train, hyps, str_cov, debug)
        try:
            lower_new_Y_train, _, _, _ = np.linalg.lstsq(lower, new_Y_train, rcond=None)
        except:
            lower_new_Y_train, _, _, _ = np.linalg.lstsq(lower, new_Y_train, rcond=-1)
        try:
            alpha, _, _, _ = np.linalg.lstsq(lower.T, lower_new_Y_train, rcond=None)
        except:
            alpha, _, _, _ = np.linalg.lstsq(lower.T, lower_new_Y_train, rcond=-1)
        first_term = -0.5 * np.dot(new_Y_train.T, alpha)
        second_term = -1.0 * np.sum(np.log(np.diagonal(lower) + constants.JITTER_LOG))
    else:
        cov_X_X, inv_cov_X_X = get_kernels(X_train, hyps, str_cov, debug)

        first_term = -0.5 * np.dot(np.dot(new_Y_train.T, inv_cov_X_X), new_Y_train)
        second_term = -0.5 * np.log(np.linalg.det(cov_X_X) + constants.JITTER_LOG)
        
    third_term = -float(X_train.shape[1]) / 2.0 * np.log(2.0 * np.pi)
    log_ml = np.squeeze(first_term + second_term + third_term)
    return log_ml

def get_optimized_kernel(X_train, Y_train, prior_mu, str_cov,
    str_optimizer_method=constants.STR_OPTIMIZER_METHOD_GP,
    is_fixed_noise=constants.IS_FIXED_GP_NOISE,
    debug=False
):
    # TODO: check to input same is_fixed_noise to convert_hyps and restore_hyps
    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert callable(prior_mu) or prior_mu is None
    assert isinstance(str_cov, str)
    assert isinstance(str_optimizer_method, str)
    assert isinstance(is_fixed_noise, bool)
    assert isinstance(debug, bool)
    assert len(X_train.shape) == 2
    assert len(Y_train.shape) == 2
    assert X_train.shape[0] == Y_train.shape[0]

    time_start = time.time()

    prior_mu_train = get_prior_mu(prior_mu, X_train)
    num_dim = X_train.shape[1]
    neg_log_ml = lambda hyps: -1.0 * log_ml(X_train, Y_train, hyps, str_cov, prior_mu_train, is_fixed_noise=is_fixed_noise, debug=debug)
    result_optimized = scipy.optimize.minimize(
        neg_log_ml,
        utils_covariance.convert_hyps(
            str_cov,
            utils_covariance.get_hyps(str_cov, num_dim),
            is_fixed_noise=is_fixed_noise,
        ),
        method=str_optimizer_method,
    )
    result_optimized = result_optimized.x
    hyps = utils_covariance.restore_hyps(str_cov, result_optimized, is_fixed_noise=is_fixed_noise)
    hyps, _ = utils_covariance.validate_hyps_dict(hyps, str_cov, num_dim)
    cov_X_X, inv_cov_X_X = get_kernels(X_train, hyps, str_cov)

    time_end = time.time()

    if debug:
        print('[DEBUG] get_optimized_kernel in gp.py: optimized hyps for gpr', hyps)
        print('[DEBUG] get_optimized_kernel in gp.py: time consumed', time_end - time_start, 'sec.')
    return cov_X_X, inv_cov_X_X, hyps

def predict_test_(X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov=constants.STR_GP_COV, prior_mu=None):
    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(str_cov, str)
    assert callable(prior_mu) or prior_mu is None
    assert len(X_train.shape) == 2
    assert len(Y_train.shape) == 2
    assert len(X_test.shape) == 2
    assert len(cov_X_X.shape) == 2
    assert len(inv_cov_X_X.shape) == 2
    assert X_train.shape[0] == Y_train.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    assert (np.array(cov_X_X.shape) == np.array(inv_cov_X_X.shape)).all()

    prior_mu_train = get_prior_mu(prior_mu, X_train)
    prior_mu_test = get_prior_mu(prior_mu, X_test)
    cov_X_Xs = covariance.cov_main(str_cov, X_train, X_test, hyps)
    cov_Xs_Xs = covariance.cov_main(str_cov, X_test, X_test, hyps)
    cov_Xs_Xs = (cov_Xs_Xs + cov_Xs_Xs.T) / 2.0

    mu_Xs = np.dot(np.dot(cov_X_Xs.T, inv_cov_X_X), Y_train - prior_mu_train) + prior_mu_test
    Sigma_Xs = cov_Xs_Xs - np.dot(np.dot(cov_X_Xs.T, inv_cov_X_X), cov_X_Xs)
    return mu_Xs, np.expand_dims(np.sqrt(np.maximum(np.diag(Sigma_Xs), 0.0)), axis=1)

def predict_test(X_train, Y_train, X_test, hyps, str_cov=constants.STR_GP_COV, prior_mu=None):
    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(str_cov, str)
    assert callable(prior_mu) or prior_mu is None
    assert len(X_train.shape) == 2
    assert len(Y_train.shape) == 2
    assert len(X_test.shape) == 2
    assert X_train.shape[0] == Y_train.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    
    cov_X_X, inv_cov_X_X = get_kernels(X_train, hyps, str_cov)
    mu_Xs, sigma_Xs = predict_test_(X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov, prior_mu)
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
    assert len(X_train.shape) == 2
    assert len(Y_train.shape) == 2
    assert len(X_test.shape) == 2
    assert X_train.shape[0] == Y_train.shape[0]
    assert X_train.shape[1] == X_test.shape[1]

    time_start = time.time()

    cov_X_X, inv_cov_X_X, hyps = get_optimized_kernel(X_train, Y_train, prior_mu, str_cov, is_fixed_noise=is_fixed_noise, debug=debug)
    mu_Xs, sigma_Xs = predict_test_(X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov, prior_mu)

    time_end = time.time()
    if debug:
        print('[DEBUG] predict_optimized in gp.py: time consumed', time_end - time_start, 'sec.')
    return mu_Xs, sigma_Xs
