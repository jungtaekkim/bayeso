#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: March 22, 2021
#
"""It defines Gaussian process regression."""

import time
import numpy as np
import scipy.stats

from bayeso import covariance
from bayeso import constants
from bayeso.gp import gp_kernel
from bayeso.utils import utils_gp
from bayeso.utils import utils_covariance
from bayeso.utils import utils_common
from bayeso.utils import utils_logger

logger = utils_logger.get_logger('gp')


@utils_common.validate_types
def sample_functions(mu: np.ndarray, Sigma: np.ndarray,
    num_samples: int=1
) -> np.ndarray:
    """
    It samples `num_samples` functions from multivariate Gaussian distribution (mu, Sigma).

    :param mu: mean vector. Shape: (n, ).
    :type mu: numpy.ndarray
    :param Sigma: covariance matrix. Shape: (n, n).
    :type Sigma: numpy.ndarray
    :param num_samples: the number of sampled functions
    :type num_samples: int., optional

    :returns: sampled functions. Shape: (num_samples, n).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(mu, np.ndarray)
    assert isinstance(Sigma, np.ndarray)
    assert isinstance(num_samples, int)
    assert len(mu.shape) == 1
    assert len(Sigma.shape) == 2
    assert mu.shape[0] == Sigma.shape[0] == Sigma.shape[1]

    rv = scipy.stats.multivariate_normal(mean=mu, cov=Sigma)
    list_rvs = [rv.rvs() for _ in range(0, num_samples)]
    return np.array(list_rvs)

@utils_common.validate_types
def predict_with_cov(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray,
    cov_X_X: np.ndarray, inv_cov_X_X: np.ndarray, hyps: dict,
    str_cov: str=constants.STR_COV,
    prior_mu: constants.TYPING_UNION_CALLABLE_NONE=None,
    debug: bool=False
) -> constants.TYPING_TUPLE_THREE_ARRAYS:
    """
    This function returns posterior mean and posterior standard deviation
    functions over `X_test`, computed by Gaussian process regression with
    `X_train`, `Y_train`, `cov_X_X`, `inv_cov_X_X`, and `hyps`.

    :param X_train: inputs. Shape: (n, d) or (n, m, d).
    :type X_train: numpy.ndarray
    :param Y_train: outputs. Shape: (n, 1).
    :type Y_train: numpy.ndarray
    :param X_test: inputs. Shape: (l, d) or (l, m, d).
    :type X_test: numpy.ndarray
    :param cov_X_X: kernel matrix over `X_train`. Shape: (n, n).
    :type cov_X_X: numpy.ndarray
    :param inv_cov_X_X: kernel matrix inverse over `X_train`. Shape: (n, n).
    :type inv_cov_X_X: numpy.ndarray
    :param hyps: dictionary of hyperparameters for Gaussian process.
    :type hyps: dict.
    :param str_cov: the name of covariance function.
    :type str_cov: str., optional
    :param prior_mu: None, or prior mean function.
    :type prior_mu: NoneType, or callable, optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    :returns: a tuple of posterior mean function over `X_test`, posterior
        standard deviation function over `X_test`, and posterior covariance
        matrix over `X_test`. Shape: ((l, 1), (l, 1), (l, l)).
    :rtype: tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)

    :raises: AssertionError

    """

    utils_gp.validate_common_args(X_train, Y_train, str_cov, prior_mu, debug, X_test)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert len(cov_X_X.shape) == 2
    assert len(inv_cov_X_X.shape) == 2
    assert (np.array(cov_X_X.shape) == np.array(inv_cov_X_X.shape)).all()
    utils_covariance.check_str_cov('predict_with_cov', str_cov,
        X_train.shape, shape_X2=X_test.shape)

    prior_mu_train = utils_gp.get_prior_mu(prior_mu, X_train)
    prior_mu_test = utils_gp.get_prior_mu(prior_mu, X_test)
    cov_X_Xs = covariance.cov_main(str_cov, X_train, X_test, hyps, False)
    cov_Xs_Xs = covariance.cov_main(str_cov, X_test, X_test, hyps, True)
    cov_Xs_Xs = (cov_Xs_Xs + cov_Xs_Xs.T) / 2.0

    mu_Xs = np.dot(np.dot(cov_X_Xs.T, inv_cov_X_X), Y_train - prior_mu_train) + prior_mu_test
    Sigma_Xs = cov_Xs_Xs - np.dot(np.dot(cov_X_Xs.T, inv_cov_X_X), cov_X_Xs)
    return mu_Xs, np.expand_dims(np.sqrt(np.maximum(np.diag(Sigma_Xs), 0.0)), axis=1), Sigma_Xs

@utils_common.validate_types
def predict_with_hyps(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, hyps: dict,
    str_cov: str=constants.STR_COV,
    prior_mu: constants.TYPING_UNION_CALLABLE_NONE=None,
    debug: bool=False
) -> constants.TYPING_TUPLE_THREE_ARRAYS:
    """
    This function returns posterior mean and posterior standard deviation
    functions over `X_test`, computed by Gaussian process regression with
    `X_train`, `Y_train`, and `hyps`.

    :param X_train: inputs. Shape: (n, d) or (n, m, d).
    :type X_train: numpy.ndarray
    :param Y_train: outputs. Shape: (n, 1).
    :type Y_train: numpy.ndarray
    :param X_test: inputs. Shape: (l, d) or (l, m, d).
    :type X_test: numpy.ndarray
    :param hyps: dictionary of hyperparameters for Gaussian process.
    :type hyps: dict.
    :param str_cov: the name of covariance function.
    :type str_cov: str., optional
    :param prior_mu: None, or prior mean function.
    :type prior_mu: NoneType, or callable, optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    :returns: a tuple of posterior mean function over `X_test`, posterior
        standard deviation function over `X_test`, and posterior covariance
        matrix over `X_test`. Shape: ((l, 1), (l, 1), (l, l)).
    :rtype: tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)

    :raises: AssertionError

    """

    utils_gp.validate_common_args(X_train, Y_train, str_cov, prior_mu, debug, X_test)
    assert isinstance(hyps, dict)
    utils_covariance.check_str_cov('predict_with_hyps', str_cov,
        X_train.shape, shape_X2=X_test.shape)

    cov_X_X, inv_cov_X_X, _ = covariance.get_kernel_inverse(X_train,
        hyps, str_cov, debug=debug)
    mu_Xs, sigma_Xs, Sigma_Xs = predict_with_cov(X_train, Y_train, X_test,
        cov_X_X, inv_cov_X_X, hyps, str_cov=str_cov,
        prior_mu=prior_mu, debug=debug)

    return mu_Xs, sigma_Xs, Sigma_Xs

@utils_common.validate_types
def predict_with_optimized_hyps(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray,
    str_cov: str=constants.STR_COV,
    str_optimizer_method: str=constants.STR_OPTIMIZER_METHOD_GP,
    prior_mu: constants.TYPING_UNION_CALLABLE_NONE=None,
    fix_noise: float=constants.FIX_GP_NOISE,
    debug: bool=False
) -> constants.TYPING_TUPLE_THREE_ARRAYS:
    """
    This function returns posterior mean and posterior standard deviation
    functions over `X_test`, computed by the Gaussian process regression
    optimized with `X_train` and `Y_train`.

    :param X_train: inputs. Shape: (n, d) or (n, m, d).
    :type X_train: numpy.ndarray
    :param Y_train: outputs. Shape: (n, 1).
    :type Y_train: numpy.ndarray
    :param X_test: inputs. Shape: (l, d) or (l, m, d).
    :type X_test: numpy.ndarray
    :param str_cov: the name of covariance function.
    :type str_cov: str., optional
    :param str_optimizer_method: the name of optimization method.
    :type str_optimizer_method: str., optional
    :param prior_mu: None, or prior mean function.
    :type prior_mu: NoneType, or callable, optional
    :param fix_noise: flag for fixing a noise.
    :type fix_noise: bool., optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    :returns: a tuple of posterior mean function over `X_test`, posterior
        standard deviation function over `X_test`, and posterior covariance
        matrix over `X_test`. Shape: ((l, 1), (l, 1), (l, l)).
    :rtype: tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)

    :raises: AssertionError

    """

    utils_gp.validate_common_args(X_train, Y_train, str_cov, prior_mu, debug, X_test)
    assert isinstance(str_optimizer_method, str)
    assert isinstance(fix_noise, bool)
    utils_covariance.check_str_cov('predict_with_optimized_kernel', str_cov,
        X_train.shape, shape_X2=X_test.shape)
    assert str_optimizer_method in constants.ALLOWED_OPTIMIZER_METHOD_GP

    time_start = time.time()

    cov_X_X, inv_cov_X_X, hyps = gp_kernel.get_optimized_kernel(X_train, Y_train,
        prior_mu, str_cov, str_optimizer_method=str_optimizer_method,
        fix_noise=fix_noise, debug=debug)
    mu_Xs, sigma_Xs, Sigma_Xs = predict_with_cov(X_train, Y_train, X_test,
        cov_X_X, inv_cov_X_X, hyps, str_cov=str_cov, prior_mu=prior_mu,
        debug=debug)

    time_end = time.time()
    if debug:
        logger.debug('time consumed to construct gpr: %.4f sec.', time_end - time_start)
    return mu_Xs, sigma_Xs, Sigma_Xs
