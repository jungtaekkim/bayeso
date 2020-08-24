# gp_scipy
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 07, 2020

import time
import numpy as np
import scipy.linalg
import scipy.optimize

from bayeso import constants
from bayeso.gp import gp_common
from bayeso.utils import utils_gp
from bayeso.utils import utils_covariance
from bayeso.utils import utils_logger

logger = utils_logger.get_logger('gp_scipy')


def neg_log_ml(X_train, Y_train, hyps, str_cov, prior_mu_train,
    is_fixed_noise=constants.IS_FIXED_GP_NOISE,
    is_cholesky=True,
    is_gradient=True,
    debug=False
):
    """
    This function computes a negative log marginal likelihood.

    :param X_train: inputs. Shape: (n, d) or (n, m, d).
    :type X_train: numpy.ndarray
    :param Y_train: outputs. Shape: (n, 1).
    :type Y_train: numpy.ndarray
    :param hyps: hyperparameters for Gaussian process. Shape: (h, ).
    :type hyps: numpy.ndarray
    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param prior_mu_train: the prior values computed by get_prior_mu(). Shape: (n, 1).
    :type prior_mu_train: numpy.ndarray
    :param is_fixed_noise: flag for fixing a noise.
    :type is_fixed_noise: bool., optional
    :param is_cholesky: flag for using a cholesky decomposition.
    :type is_cholesky: bool., optional
    :param is_gradient: flag for computing and returning gradients of negative log marginal likelihood.
    :type is_gradient: bool., optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    :returns: negative log marginal likelihood, or (negative log marginal likelihood, gradients of the likelihood).
    :rtype: float, or tuple of (float, float)

    :raises: AssertionError

    """

    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(hyps, np.ndarray)
    assert isinstance(str_cov, str)
    assert isinstance(prior_mu_train, np.ndarray)
    assert isinstance(is_fixed_noise, bool)
    assert isinstance(is_cholesky, bool)
    assert isinstance(is_gradient, bool)
    assert isinstance(debug, bool)
    assert len(Y_train.shape) == 2
    assert len(prior_mu_train.shape) == 2
    assert X_train.shape[0] == Y_train.shape[0] == prior_mu_train.shape[0]
    utils_gp.check_str_cov('neg_log_ml', str_cov, X_train.shape)

    hyps = utils_covariance.restore_hyps(str_cov, hyps, is_fixed_noise=is_fixed_noise)
    new_Y_train = Y_train - prior_mu_train
    if is_cholesky:
        cov_X_X, lower, grad_cov_X_X = gp_common.get_kernel_cholesky(X_train, hyps, str_cov, is_fixed_noise=is_fixed_noise, is_gradient=is_gradient, debug=debug)

        alpha = scipy.linalg.cho_solve((lower, True), new_Y_train)

        first_term = -0.5 * np.dot(new_Y_train.T, alpha)
        second_term = -1.0 * np.sum(np.log(np.diagonal(lower) + constants.JITTER_LOG))

        if is_gradient:
            assert grad_cov_X_X is not None

            first_term_grad = np.einsum("ik,jk->ijk", alpha, alpha)
            first_term_grad -= np.expand_dims(scipy.linalg.cho_solve((lower, True), np.eye(cov_X_X.shape[0])), axis=2)
            grad_log_ml_ = 0.5 * np.einsum("ijl,ijk->kl", first_term_grad, grad_cov_X_X)
            grad_log_ml_ = np.sum(grad_log_ml_, axis=1)
    else:
        # TODO: is_gradient is fixed.
        is_gradient = False
        cov_X_X, inv_cov_X_X, grad_cov_X_X = gp_common.get_kernel_inverse(X_train, hyps, str_cov, is_fixed_noise=is_fixed_noise, is_gradient=is_gradient, debug=debug)

        first_term = -0.5 * np.dot(np.dot(new_Y_train.T, inv_cov_X_X), new_Y_train)
        second_term = -0.5 * np.log(np.linalg.det(cov_X_X) + constants.JITTER_LOG)

    third_term = -float(X_train.shape[0]) / 2.0 * np.log(2.0 * np.pi)
    log_ml_ = np.squeeze(first_term + second_term + third_term)
    log_ml_ /= X_train.shape[0]

    if is_gradient:
        return -1.0 * log_ml_, -1.0 * grad_log_ml_ / X_train.shape[0]
    else:
        return -1.0 * log_ml_

def neg_log_pseudo_l_loocv(X_train, Y_train, hyps, str_cov, prior_mu_train,
    is_fixed_noise=constants.IS_FIXED_GP_NOISE,
    debug=False
):
    """
    It computes a negative log pseudo-likelihood using leave-one-out cross-validation.

    :param X_train: inputs. Shape: (n, d) or (n, m, d).
    :type X_train: numpy.ndarray
    :param Y_train: outputs. Shape: (n, 1).
    :type Y_train: numpy.ndarray
    :param hyps: hyperparameters for Gaussian process. Shape: (h, ).
    :type hyps: numpy.ndarray
    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param prior_mu_train: the prior values computed by get_prior_mu(). Shape: (n, 1).
    :type prior_mu_train: numpy.ndarray
    :param is_fixed_noise: flag for fixing a noise.
    :type is_fixed_noise: bool., optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    :returns: negative log pseudo-likelihood.
    :rtype: float

    :raises: AssertionError

    """

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
    utils_gp.check_str_cov('neg_log_pseudo_l_loocv', str_cov, X_train.shape)

    num_data = X_train.shape[0]
    hyps = utils_covariance.restore_hyps(str_cov, hyps, is_fixed_noise=is_fixed_noise)

    cov_X_X, inv_cov_X_X, grad_cov_X_X = gp_common.get_kernel_inverse(X_train, hyps, str_cov, is_fixed_noise=is_fixed_noise, debug=debug)

    log_pseudo_l_ = 0.0
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
        cur_log_pseudo_l_ = first_term + second_term + third_term
        log_pseudo_l_ += cur_log_pseudo_l_

    log_pseudo_l_ /= num_data
    log_pseudo_l_ *= -1.0

    return log_pseudo_l_

def get_optimized_kernel(X_train, Y_train, prior_mu, str_cov,
    str_optimizer_method=constants.STR_OPTIMIZER_METHOD_GP,
    str_modelselection_method=constants.STR_MODELSELECTION_METHOD,
    is_fixed_noise=constants.IS_FIXED_GP_NOISE,
    debug=False
):
    """
    This function computes the kernel matrix optimized by optimization method specified, its inverse matrix, and the optimized hyperparameters.

    :param X_train: inputs. Shape: (n, d) or (n, m, d).
    :type X_train: numpy.ndarray
    :param Y_train: outputs. Shape: (n, 1).
    :type Y_train: numpy.ndarray
    :param prior_mu: prior mean function or None.
    :type prior_mu: function or NoneType
    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param str_optimizer_method: the name of optimization method.
    :type str_optimizer_method: str., optional
    :param str_modelselection_method: the name of model selection method.
    :type str_modelselection_method: str., optional
    :param is_fixed_noise: flag for fixing a noise.
    :type is_fixed_noise: bool., optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    :returns: a tuple of kernel matrix over `X_train`, kernel matrix inverse, and dictionary of hyperparameters.
    :rtype: tuple of (numpy.ndarray, numpy.ndarray, dict.)

    :raises: AssertionError, ValueError

    """

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
    utils_gp.check_str_cov('get_optimized_kernel', str_cov, X_train.shape)
    assert str_optimizer_method in constants.ALLOWED_OPTIMIZER_METHOD_GP
    assert str_modelselection_method in constants.ALLOWED_MODELSELECTION_METHOD
    # TODO: fix this.
    if str_optimizer_method != 'Nelder-Mead':
        is_gradient = True
    else:
        is_gradient = False

    time_start = time.time()

    if debug: logger.debug('str_optimizer_method: {}'.format(str_optimizer_method))
    if debug: logger.debug('str_modelselection_method: {}'.format(str_modelselection_method))

    prior_mu_train = utils_gp.get_prior_mu(prior_mu, X_train)
    if str_cov in constants.ALLOWED_GP_COV_BASE:
        num_dim = X_train.shape[1]
    elif str_cov in constants.ALLOWED_GP_COV_SET:
        num_dim = X_train.shape[2]
        is_gradient = False

    if str_modelselection_method == 'ml':
        neg_log_ml_ = lambda hyps: neg_log_ml(X_train, Y_train, hyps, str_cov, prior_mu_train, is_fixed_noise=is_fixed_noise, is_gradient=is_gradient, debug=debug)
    elif str_modelselection_method == 'loocv':
        neg_log_ml_ = lambda hyps: neg_log_pseudo_l_loocv(X_train, Y_train, hyps, str_cov, prior_mu_train, is_fixed_noise=is_fixed_noise, debug=debug)
        is_gradient = False
    else: # pragma: no cover
        raise ValueError('get_optimized_kernel: missing conditions for str_modelselection_method.')

    hyps_converted = utils_covariance.convert_hyps(
        str_cov,
        utils_covariance.get_hyps(str_cov, num_dim),
        is_fixed_noise=is_fixed_noise,
    )

    if str_optimizer_method == 'BFGS':
        result_optimized = scipy.optimize.minimize(neg_log_ml_, hyps_converted, method=str_optimizer_method, jac=is_gradient, options={'disp': False})
        if debug: logger.debug('scipy message: {}'.format(result_optimized.message))

        result_optimized = result_optimized.x
    elif str_optimizer_method == 'L-BFGS-B':
        bounds = utils_covariance.get_range_hyps(str_cov, num_dim, is_fixed_noise=is_fixed_noise)
        result_optimized = scipy.optimize.minimize(neg_log_ml_, hyps_converted, method=str_optimizer_method, bounds=bounds, jac=is_gradient, options={'disp': False})
        if debug: logger.debug('scipy message: {}'.format(result_optimized.message))

        result_optimized = result_optimized.x
    elif str_optimizer_method == 'Nelder-Mead':
        result_optimized = scipy.optimize.minimize(neg_log_ml_, hyps_converted, method=str_optimizer_method, options={'disp': False})
        if debug: logger.debug('scipy message: {}'.format(result_optimized.message))

        result_optimized = result_optimized.x
    # TODO: Fill this conditions
    elif str_optimizer_method == 'DIRECT': # pragma: no cover
        raise NotImplementedError('get_optimized_kernel: allowed str_optimizer_method, but it is not implemented.')
    else: # pragma: no cover
        raise ValueError('get_optimized_kernel: missing conditions for str_optimizer_method')

    hyps = utils_covariance.restore_hyps(str_cov, result_optimized, is_fixed_noise=is_fixed_noise)

    hyps, _ = utils_covariance.validate_hyps_dict(hyps, str_cov, num_dim)
    cov_X_X, inv_cov_X_X, grad_cov_X_X = gp_common.get_kernel_inverse(X_train, hyps, str_cov, is_fixed_noise=is_fixed_noise, debug=debug)

    time_end = time.time()

    if debug: logger.debug('hyps optimized: {}'.format(utils_logger.get_str_hyps(hyps)))
    if debug: logger.debug('time consumed to construct gpr: {:.4f} sec.'.format(time_end - time_start))
    return cov_X_X, inv_cov_X_X, hyps
