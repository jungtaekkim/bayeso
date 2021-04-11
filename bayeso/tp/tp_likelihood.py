#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: March 22, 2021
#
"""It defines the functions related to likelihood for
Student-:math:`t` process regression."""

import numpy as np
import scipy.special

from bayeso import covariance
from bayeso import constants
from bayeso.utils import utils_gp
from bayeso.utils import utils_covariance
from bayeso.utils import utils_common
from bayeso.utils import utils_logger

logger = utils_logger.get_logger('tp_likelihood')


@utils_common.validate_types
def neg_log_ml(X_train: np.ndarray, Y_train: np.ndarray, hyps: np.ndarray,
    str_cov: str, prior_mu_train: np.ndarray,
    use_ard: bool=constants.USE_ARD,
    fix_noise: bool=constants.FIX_GP_NOISE,
    use_gradient: bool=True,
    debug: bool=False
) -> constants.TYPING_UNION_FLOAT_FA:
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
    :param use_ard: flag for automatic relevance determination.
    :type use_ard: bool., optional
    :param fix_noise: flag for fixing a noise.
    :type fix_noise: bool., optional
    :param use_gradient: flag for computing and returning gradients of
        negative log marginal likelihood.
    :type use_gradient: bool., optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    :returns: negative log marginal likelihood, or (negative log marginal
        likelihood, gradients of the likelihood).
    :rtype: float, or tuple of (float, np.ndarray)

    :raises: AssertionError

    """

    utils_gp.validate_common_args(X_train, Y_train, str_cov, None, debug)
    assert isinstance(hyps, np.ndarray)
    assert isinstance(prior_mu_train, np.ndarray)
    assert isinstance(use_ard, bool)
    assert isinstance(fix_noise, bool)
    assert isinstance(use_gradient, bool)
    assert len(prior_mu_train.shape) == 2
    assert X_train.shape[0] == Y_train.shape[0] == prior_mu_train.shape[0]
    utils_covariance.check_str_cov('neg_log_ml', str_cov, X_train.shape)

    num_X = float(X_train.shape[0])
    hyps = utils_covariance.restore_hyps(str_cov, hyps,
        use_ard=use_ard,
        fix_noise=fix_noise, use_gp=False)
    new_Y_train = Y_train - prior_mu_train
    nu = hyps['dof']

    cov_X_X, inv_cov_X_X, grad_cov_X_X = covariance.get_kernel_inverse(X_train,
        hyps, str_cov, fix_noise=fix_noise, use_gradient=use_gradient,
        debug=debug)

    alpha = np.dot(inv_cov_X_X, new_Y_train)
    beta = np.squeeze(np.dot(np.dot(new_Y_train.T, inv_cov_X_X), new_Y_train))

    first_term = -0.5 * num_X * np.log((nu - 2.0) * np.pi)
    sign_second_term, second_term = np.linalg.slogdet(cov_X_X)

    # TODO: it should be checked.
    if sign_second_term <= 0: # pragma: no cover
        second_term = 0.0

    second_term = -0.5 * second_term

    third_term = np.log(scipy.special.gamma((nu + num_X) / 2.0) / scipy.special.gamma(nu / 2.0))
    fourth_term = -0.5 * (nu + num_X) * np.log(1.0 + beta / (nu - 2.0))

    log_ml_ = np.squeeze(first_term + second_term + third_term + fourth_term)
    log_ml_ /= num_X

    if use_gradient:
        assert grad_cov_X_X is not None
        grad_log_ml_ = np.zeros(grad_cov_X_X.shape[2] + 1)

        first_term_grad = ((nu + num_X) / (nu + beta - 2.0) * np.dot(alpha, alpha.T) - inv_cov_X_X)
        nu_grad = -num_X / (2.0 * (nu - 2.0))\
            + scipy.special.digamma((nu + num_X) / 2.0)\
            - scipy.special.digamma(nu / 2.0)\
            - 0.5 * np.log(1.0 + beta / (nu - 2.0))\
            + (nu + num_X) * beta / (2.0 * (nu - 2.0)**2 + 2.0 * beta * (nu - 2.0))

        if fix_noise:
            grad_log_ml_[0] = nu_grad
        else:
            grad_log_ml_[1] = nu_grad

        for ind in range(0, grad_cov_X_X.shape[2]):
            cur_grad = 0.5 * np.trace(np.dot(first_term_grad, grad_cov_X_X[:, :, ind]))
            if fix_noise:
                grad_log_ml_[ind + 1] = cur_grad
            else:
                if ind == 0:
                    cur_ind = 0
                else:
                    cur_ind = ind + 1

                grad_log_ml_[cur_ind] = cur_grad

    if use_gradient:
        return -1.0 * log_ml_, -1.0 * grad_log_ml_ / num_X

    return -1.0 * log_ml_
