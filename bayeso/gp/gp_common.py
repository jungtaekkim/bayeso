# gp_common
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 07, 2020

import numpy as np
import scipy.linalg

from bayeso import covariance
from bayeso import constants
from bayeso.utils import utils_logger

logger = utils_logger.get_logger('gp_common')


def _check_str_cov(str_fun, str_cov, shape_X1, shape_X2=None):
    """
    It is for validating the shape of X1 (and optionally the shape of X2).

    :param str_fun: the name of function.
    :type str_fun: str.
    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param shape_X1: the shape of X1.
    :type shape_X1: tuple
    :param shape_X2: None, or the shape of X2.
    :type shape_X2: NoneType or tuple, optional

    :returns: None, if it is valid. Raise an error, otherwise.
    :rtype: NoneType

    :raises: AssertionError, ValueError

    """

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
    """
    It computes the prior mean function values over inputs X.

    :param prior_mu: prior mean function or None.
    :type prior_mu: function or NoneType
    :param X: inputs for prior mean function. Shape: (n, d) or (n, m, d).
    :type X: numpy.ndarray

    :returns: zero array, or array of prior mean function values. Shape: (n, 1).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

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
    _check_str_cov('get_kernel_inverse', str_cov, X_train.shape)

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
    _check_str_cov('get_kernel_cholesky', str_cov, X_train.shape)

    cov_X_X = covariance.cov_main(str_cov, X_train, X_train, hyps, True) + hyps['noise']**2 * np.eye(X_train.shape[0])
    cov_X_X = (cov_X_X + cov_X_X.T) / 2.0
    try:
        lower = scipy.linalg.cholesky(cov_X_X, lower=True)
    except np.linalg.LinAlgError:
        cov_X_X += 1e-2 * np.eye(X_train.shape[0])
        lower = scipy.linalg.cholesky(cov_X_X, lower=True)

    if is_gradient:
        grad_cov_X_X = covariance.grad_cov_main(str_cov, X_train, X_train, hyps, is_fixed_noise, same_X_Xs=True)
    else:
        grad_cov_X_X = None
    return cov_X_X, lower, grad_cov_X_X
