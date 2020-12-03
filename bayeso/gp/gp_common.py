#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: September 24, 2020
#
"""It defines functions for Gaussian process regression."""

import numpy as np
import scipy.linalg

from bayeso import covariance
from bayeso import constants
from bayeso.utils import utils_gp
from bayeso.utils import utils_common


@utils_common.validate_types
def get_kernel_inverse(X_train: np.ndarray, hyps: dict, str_cov: str,
    fix_noise: bool=constants.FIX_GP_NOISE,
    use_gradient: bool=False,
    debug: bool=False
) -> constants.TYPING_TUPLE_THREE_ARRAYS:
    """
    This function computes a kernel inverse without any matrix decomposition techniques.

    :param X_train: inputs. Shape: (n, d) or (n, m, d).
    :type X_train: numpy.ndarray
    :param hyps: dictionary of hyperparameters for Gaussian process.
    :type hyps: dict.
    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param fix_noise: flag for fixing a noise.
    :type fix_noise: bool., optional
    :param use_gradient: flag for computing and returning gradients of
        negative log marginal likelihood.
    :type use_gradient: bool., optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    :returns: a tuple of kernel matrix over `X_train`, kernel matrix
        inverse, and gradients of kernel matrix. If `use_gradient` is False,
        gradients of kernel matrix would be None.
    :rtype: tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)

    :raises: AssertionError

    """

    assert isinstance(X_train, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(str_cov, str)
    assert isinstance(use_gradient, bool)
    assert isinstance(fix_noise, bool)
    assert isinstance(debug, bool)
    utils_gp.check_str_cov('get_kernel_inverse', str_cov, X_train.shape)

    cov_X_X = covariance.cov_main(str_cov, X_train, X_train, hyps, True) \
        + hyps['noise']**2 * np.eye(X_train.shape[0])
    cov_X_X = (cov_X_X + cov_X_X.T) / 2.0
    inv_cov_X_X = np.linalg.inv(cov_X_X)

    if use_gradient:
        grad_cov_X_X = covariance.grad_cov_main(str_cov, X_train, X_train,
            hyps, fix_noise, same_X_Xp=True)
    else:
        grad_cov_X_X = None

    return cov_X_X, inv_cov_X_X, grad_cov_X_X

@utils_common.validate_types
def get_kernel_cholesky(X_train: np.ndarray, hyps: dict, str_cov: str,
    fix_noise: bool=constants.FIX_GP_NOISE,
    use_gradient: bool=False,
    debug: bool=False
) -> constants.TYPING_TUPLE_THREE_ARRAYS:
    """
    This function computes a kernel inverse with Cholesky decomposition.

    :param X_train: inputs. Shape: (n, d) or (n, m, d).
    :type X_train: numpy.ndarray
    :param hyps: dictionary of hyperparameters for Gaussian process.
    :type hyps: dict.
    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param fix_noise: flag for fixing a noise.
    :type fix_noise: bool., optional
    :param use_gradient: flag for computing and returning gradients of
        negative log marginal likelihood.
    :type use_gradient: bool., optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    :returns: a tuple of kernel matrix over `X_train`, lower matrix computed
        by Cholesky decomposition, and gradients of kernel matrix. If
        `use_gradient` is False, gradients of kernel matrix would be None.
    :rtype: tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)

    :raises: AssertionError

    """

    assert isinstance(X_train, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(str_cov, str)
    assert isinstance(fix_noise, bool)
    assert isinstance(use_gradient, bool)
    assert isinstance(debug, bool)
    utils_gp.check_str_cov('get_kernel_cholesky', str_cov, X_train.shape)

    cov_X_X = covariance.cov_main(str_cov, X_train, X_train, hyps, True) \
        + hyps['noise']**2 * np.eye(X_train.shape[0])
    cov_X_X = (cov_X_X + cov_X_X.T) / 2.0
    try:
        lower = scipy.linalg.cholesky(cov_X_X, lower=True)
    except np.linalg.LinAlgError: # pragma: no cover
        cov_X_X += 1e-2 * np.eye(X_train.shape[0])
        lower = scipy.linalg.cholesky(cov_X_X, lower=True)

    if use_gradient:
        grad_cov_X_X = covariance.grad_cov_main(str_cov, X_train, X_train,
            hyps, fix_noise, same_X_Xp=True)
    else:
        grad_cov_X_X = None
    return cov_X_X, lower, grad_cov_X_X
