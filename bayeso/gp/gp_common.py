# gp_common
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 07, 2020

import numpy as np
import scipy.linalg

from bayeso import covariance
from bayeso import constants
from bayeso.utils import utils_gp
from bayeso.utils import utils_logger

logger = utils_logger.get_logger('gp_common')


def get_kernel_inverse(X_train, hyps, str_cov,
    is_fixed_noise=constants.IS_FIXED_GP_NOISE,
    is_gradient=False,
    debug=False
):
    """
    This function computes a kernel inverse without any matrix decomposition techniques.

    :param X_train: inputs. Shape: (n, d) or (n, m, d).
    :type X_train: numpy.ndarray
    :param hyps: dictionary of hyperparameters for Gaussian process.
    :type hyps: dict.
    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param is_fixed_noise: flag for fixing a noise.
    :type is_fixed_noise: bool., optional
    :param is_gradient: flag for computing and returning gradients of negative log marginal likelihood.
    :type is_gradient: bool., optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    :returns: a tuple of kernel matrix over `X_train`, kernel matrix inverse, and gradients of kernel matrix. If `is_gradient` is False, gradients of kernel matrix would be None.
    :rtype: tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)

    :raises: AssertionError

    """

    assert isinstance(X_train, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(str_cov, str)
    assert isinstance(is_gradient, bool)
    assert isinstance(is_fixed_noise, bool)
    assert isinstance(debug, bool)
    utils_gp.check_str_cov('get_kernel_inverse', str_cov, X_train.shape)

    cov_X_X = covariance.cov_main(str_cov, X_train, X_train, hyps, True) + hyps['noise']**2 * np.eye(X_train.shape[0])
    cov_X_X = (cov_X_X + cov_X_X.T) / 2.0
    inv_cov_X_X = np.linalg.inv(cov_X_X)

    if is_gradient:
        grad_cov_X_X = covariance.grad_cov_main(str_cov, X_train, X_train, hyps, is_fixed_noise, same_X_Xs=True)
    else:
        grad_cov_X_X = None

    return cov_X_X, inv_cov_X_X, grad_cov_X_X

def get_kernel_cholesky(X_train, hyps, str_cov,
    is_fixed_noise=constants.IS_FIXED_GP_NOISE,
    is_gradient=False,
    debug=False
):
    """
    This function computes a kernel inverse with Cholesky decomposition.

    :param X_train: inputs. Shape: (n, d) or (n, m, d).
    :type X_train: numpy.ndarray
    :param hyps: dictionary of hyperparameters for Gaussian process.
    :type hyps: dict.
    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param is_fixed_noise: flag for fixing a noise.
    :type is_fixed_noise: bool., optional
    :param is_gradient: flag for computing and returning gradients of negative log marginal likelihood.
    :type is_gradient: bool., optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    :returns: a tuple of kernel matrix over `X_train`, lower matrix computed by Cholesky decomposition, and gradients of kernel matrix. If `is_gradient` is False, gradients of kernel matrix would be None.
    :rtype: tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)

    :raises: AssertionError

    """

    assert isinstance(X_train, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(str_cov, str)
    assert isinstance(is_fixed_noise, bool)
    assert isinstance(is_gradient, bool)
    assert isinstance(debug, bool)
    utils_gp.check_str_cov('get_kernel_cholesky', str_cov, X_train.shape)

    cov_X_X = covariance.cov_main(str_cov, X_train, X_train, hyps, True) + hyps['noise']**2 * np.eye(X_train.shape[0])
    cov_X_X = (cov_X_X + cov_X_X.T) / 2.0
    try:
        lower = scipy.linalg.cholesky(cov_X_X, lower=True)
    except np.linalg.LinAlgError: # pragma: no cover
        cov_X_X += 1e-2 * np.eye(X_train.shape[0])
        lower = scipy.linalg.cholesky(cov_X_X, lower=True)

    if is_gradient:
        grad_cov_X_X = covariance.grad_cov_main(str_cov, X_train, X_train, hyps, is_fixed_noise, same_X_Xs=True)
    else:
        grad_cov_X_X = None
    return cov_X_X, lower, grad_cov_X_X
