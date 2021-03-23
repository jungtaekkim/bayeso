#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: March 22, 2021
#
"""It defines the functions related to kernels for
Gaussian process regression."""

import time
import numpy as np
import scipy.optimize

from bayeso import covariance
from bayeso import constants
from bayeso.gp import gp_likelihood
from bayeso.utils import utils_gp
from bayeso.utils import utils_covariance
from bayeso.utils import utils_common
from bayeso.utils import utils_logger

logger = utils_logger.get_logger('gp_kernel')


@utils_common.validate_types
def get_optimized_kernel(X_train: np.ndarray, Y_train: np.ndarray,
    prior_mu: constants.TYPING_UNION_CALLABLE_NONE, str_cov: str,
    str_optimizer_method: str=constants.STR_OPTIMIZER_METHOD_GP,
    str_modelselection_method: str=constants.STR_MODELSELECTION_METHOD,
    fix_noise: bool=constants.FIX_GP_NOISE,
    debug: bool=False
) -> constants.TYPING_TUPLE_TWO_ARRAYS_DICT:
    """
    This function computes the kernel matrix optimized by optimization
    method specified, its inverse matrix, and the optimized hyperparameters.

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
    :param fix_noise: flag for fixing a noise.
    :type fix_noise: bool., optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    :returns: a tuple of kernel matrix over `X_train`, kernel matrix
        inverse, and dictionary of hyperparameters.
    :rtype: tuple of (numpy.ndarray, numpy.ndarray, dict.)

    :raises: AssertionError, ValueError

    """

    # TODO: check to input same fix_noise to convert_hyps and restore_hyps
    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert callable(prior_mu) or prior_mu is None
    assert isinstance(str_cov, str)
    assert isinstance(str_optimizer_method, str)
    assert isinstance(str_modelselection_method, str)
    assert isinstance(fix_noise, bool)
    assert isinstance(debug, bool)
    assert len(Y_train.shape) == 2
    assert X_train.shape[0] == Y_train.shape[0]
    utils_covariance.check_str_cov('get_optimized_kernel', str_cov, X_train.shape)
    assert str_optimizer_method in constants.ALLOWED_OPTIMIZER_METHOD_GP
    assert str_modelselection_method in constants.ALLOWED_MODELSELECTION_METHOD
    # TODO: fix this.
    use_gradient = bool(str_optimizer_method != 'Nelder-Mead')

    time_start = time.time()

    if debug:
        logger.debug('str_optimizer_method: %s', str_optimizer_method)
        logger.debug('str_modelselection_method: %s', str_modelselection_method)

    prior_mu_train = utils_gp.get_prior_mu(prior_mu, X_train)
    if str_cov in constants.ALLOWED_GP_COV_BASE:
        num_dim = X_train.shape[1]
    elif str_cov in constants.ALLOWED_GP_COV_SET:
        num_dim = X_train.shape[2]
        use_gradient = False

    if str_modelselection_method == 'ml':
        neg_log_ml_ = lambda hyps: gp_likelihood.neg_log_ml(X_train, Y_train, hyps, str_cov,
            prior_mu_train, fix_noise=fix_noise, use_gradient=use_gradient,
            debug=debug)
    elif str_modelselection_method == 'loocv':
        neg_log_ml_ = lambda hyps: gp_likelihood.neg_log_pseudo_l_loocv(X_train, Y_train,
            hyps, str_cov, prior_mu_train, fix_noise=fix_noise, debug=debug)
        use_gradient = False
    else: # pragma: no cover
        raise ValueError('get_optimized_kernel: missing conditions for str_modelselection_method.')

    hyps_converted = utils_covariance.convert_hyps(
        str_cov,
        utils_covariance.get_hyps(str_cov, num_dim),
        fix_noise=fix_noise,
    )

    if str_optimizer_method in ['BFGS', 'SLSQP']:
        result_optimized = scipy.optimize.minimize(neg_log_ml_, hyps_converted,
            method=str_optimizer_method, jac=use_gradient, options={'disp': False})

        if debug:
            logger.debug('scipy message: %s', result_optimized.message)

        result_optimized = result_optimized.x
    elif str_optimizer_method in ['L-BFGS-B', 'SLSQP-Bounded']:
        if str_optimizer_method == 'SLSQP-Bounded':
            str_optimizer_method = 'SLSQP'

        bounds = utils_covariance.get_range_hyps(str_cov, num_dim, fix_noise=fix_noise)
        result_optimized = scipy.optimize.minimize(neg_log_ml_, hyps_converted,
            method=str_optimizer_method, bounds=bounds, jac=use_gradient,
            options={'disp': False})

        if debug:
            logger.debug('scipy message: %s', result_optimized.message)

        result_optimized = result_optimized.x
    elif str_optimizer_method in ['Nelder-Mead']:
        result_optimized = scipy.optimize.minimize(neg_log_ml_, hyps_converted,
            method=str_optimizer_method, options={'disp': False})

        if debug:
            logger.debug('scipy message: %s', result_optimized.message)

        result_optimized = result_optimized.x
    # TODO: Fill this conditions
    elif str_optimizer_method == 'DIRECT': # pragma: no cover
        raise NotImplementedError('get_optimized_kernel: allowed str_optimizer_method,\
            but it is not implemented.')
    else: # pragma: no cover
        raise ValueError('get_optimized_kernel: missing conditions for str_optimizer_method')

    hyps = utils_covariance.restore_hyps(str_cov, result_optimized, fix_noise=fix_noise)

    hyps, _ = utils_covariance.validate_hyps_dict(hyps, str_cov, num_dim)
    cov_X_X, inv_cov_X_X, _ = covariance.get_kernel_inverse(X_train,
        hyps, str_cov, fix_noise=fix_noise, debug=debug)

    time_end = time.time()

    if debug:
        logger.debug('hyps optimized: %s', utils_logger.get_str_hyps(hyps))
        logger.debug('time consumed to construct gpr: %.4f sec.', time_end - time_start)
    return cov_X_X, inv_cov_X_X, hyps