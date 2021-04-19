#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: March 22, 2021
#
"""It defines the functions related to likelihood for
Gaussian process regression."""

import numpy as np
import scipy.linalg

from bayeso import covariance
from bayeso import constants
from bayeso.utils import utils_gp
from bayeso.utils import utils_covariance
from bayeso.utils import utils_common
from bayeso.utils import utils_logger

logger = utils_logger.get_logger('gp_likelihood')


@utils_common.validate_types
def neg_log_ml(X_train: np.ndarray, Y_train: np.ndarray, hyps: np.ndarray,
    str_cov: str, prior_mu_train: np.ndarray,
    use_ard: bool=constants.USE_ARD,
    fix_noise: bool=constants.FIX_GP_NOISE,
    use_cholesky: bool=True,
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
    :param use_cholesky: flag for using a cholesky decomposition.
    :type use_cholesky: bool., optional
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

    # TODO: add use_ard.
    utils_gp.validate_common_args(X_train, Y_train, str_cov, None, debug)
    assert isinstance(hyps, np.ndarray)
    assert isinstance(prior_mu_train, np.ndarray)
    assert isinstance(use_ard, bool)
    assert isinstance(fix_noise, bool)
    assert isinstance(use_cholesky, bool)
    assert isinstance(use_gradient, bool)
    assert len(prior_mu_train.shape) == 2
    assert X_train.shape[0] == Y_train.shape[0] == prior_mu_train.shape[0]
    utils_covariance.check_str_cov('neg_log_ml', str_cov, X_train.shape)

    hyps = utils_covariance.restore_hyps(str_cov, hyps, use_ard=use_ard, fix_noise=fix_noise)
    new_Y_train = Y_train - prior_mu_train
    if use_cholesky:
        cov_X_X, lower, grad_cov_X_X = covariance.get_kernel_cholesky(X_train,
            hyps, str_cov, fix_noise=fix_noise, use_gradient=use_gradient,
            debug=debug)

        alpha = scipy.linalg.cho_solve((lower, True), new_Y_train)

        first_term = -0.5 * np.dot(new_Y_train.T, alpha)
        second_term = -1.0 * np.sum(np.log(np.diagonal(lower) + constants.JITTER_LOG))

        if use_gradient:
            assert grad_cov_X_X is not None

            first_term_grad = np.einsum("ik,jk->ijk", alpha, alpha)
            first_term_grad -= np.expand_dims(scipy.linalg.cho_solve((lower, True),
                np.eye(cov_X_X.shape[0])), axis=2)
            grad_log_ml_ = 0.5 * np.einsum("ijl,ijk->kl", first_term_grad, grad_cov_X_X)
            grad_log_ml_ = np.sum(grad_log_ml_, axis=1)
    else:
        # TODO: use_gradient is fixed.
        use_gradient = False
        cov_X_X, inv_cov_X_X, grad_cov_X_X = covariance.get_kernel_inverse(X_train,
            hyps, str_cov, fix_noise=fix_noise, use_gradient=use_gradient,
            debug=debug)

        first_term = -0.5 * np.dot(np.dot(new_Y_train.T, inv_cov_X_X), new_Y_train)
        sign_second_term, second_term = np.linalg.slogdet(cov_X_X)

        # TODO: It should be checked.
        if sign_second_term <= 0: # pragma: no cover
            second_term = 0.0

        second_term = -0.5 * second_term

    third_term = -float(X_train.shape[0]) / 2.0 * np.log(2.0 * np.pi)
    log_ml_ = np.squeeze(first_term + second_term + third_term)
    log_ml_ /= X_train.shape[0]

    if use_gradient:
        return -1.0 * log_ml_, -1.0 * grad_log_ml_ / X_train.shape[0]

    return -1.0 * log_ml_

@utils_common.validate_types
def neg_log_pseudo_l_loocv(X_train: np.ndarray, Y_train: np.ndarray, hyps: np.ndarray,
    str_cov: str, prior_mu_train: np.ndarray,
    fix_noise: bool=constants.FIX_GP_NOISE,
    debug: bool=False
) -> float:
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
    :param fix_noise: flag for fixing a noise.
    :type fix_noise: bool., optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    :returns: negative log pseudo-likelihood.
    :rtype: float

    :raises: AssertionError

    """

    # TODO: add use_ard.
    utils_gp.validate_common_args(X_train, Y_train, str_cov, None, debug)
    assert isinstance(hyps, np.ndarray)
    assert isinstance(prior_mu_train, np.ndarray)
    assert isinstance(fix_noise, bool)
    assert len(prior_mu_train.shape) == 2
    assert X_train.shape[0] == Y_train.shape[0] == prior_mu_train.shape[0]
    utils_covariance.check_str_cov('neg_log_pseudo_l_loocv', str_cov, X_train.shape)

    num_data = X_train.shape[0]
    hyps = utils_covariance.restore_hyps(str_cov, hyps, fix_noise=fix_noise)

    _, inv_cov_X_X, _ = covariance.get_kernel_inverse(X_train, hyps,
        str_cov, fix_noise=fix_noise, debug=debug)

    log_pseudo_l_ = 0.0
    for ind_data in range(0, num_data):
        # TODO: check this.
#        cur_X_train = np.vstack((X_train[:ind_data], X_train[ind_data+1:]))
#        cur_Y_train = np.vstack((Y_train[:ind_data], Y_train[ind_data+1:]))

#        cur_X_test = np.expand_dims(X_train[ind_data], axis=0)
        cur_Y_test = Y_train[ind_data]

        cur_mu = np.squeeze(cur_Y_test) \
            - np.dot(inv_cov_X_X, Y_train)[ind_data] / inv_cov_X_X[ind_data, ind_data]
        cur_sigma = np.sqrt(1.0 / (inv_cov_X_X[ind_data, ind_data] + constants.JITTER_COV))

        first_term = -0.5 * np.log(cur_sigma**2)
        second_term = -0.5 * (np.squeeze(cur_Y_test - cur_mu))**2 / (cur_sigma**2)
        third_term = -0.5 * np.log(2.0 * np.pi)
        cur_log_pseudo_l_ = first_term + second_term + third_term
        log_pseudo_l_ += cur_log_pseudo_l_

    log_pseudo_l_ /= num_data
    log_pseudo_l_ *= -1.0

    return log_pseudo_l_
