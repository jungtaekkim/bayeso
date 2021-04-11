#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: March 22, 2021
#
"""It defines Student-:math:`t` process regression."""

import time
import numpy as np
import scipy.stats

from bayeso import covariance
from bayeso import constants
from bayeso.tp import tp_kernel
from bayeso.utils import utils_gp
from bayeso.utils import utils_covariance
from bayeso.utils import utils_common
from bayeso.utils import utils_logger

logger = utils_logger.get_logger('tp')


@utils_common.validate_types
def sample_functions(nu: float, mu: np.ndarray, Sigma: np.ndarray,
    num_samples: int=1
) -> np.ndarray:
    """
    It samples `num_samples` functions from multivariate Student-$t$ distribution (nu, mu, Sigma).

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

    assert isinstance(nu, float)
    assert isinstance(mu, np.ndarray)
    assert isinstance(Sigma, np.ndarray)
    assert isinstance(num_samples, int)
    assert len(mu.shape) == 1
    assert len(Sigma.shape) == 2
    assert mu.shape[0] == Sigma.shape[0] == Sigma.shape[1]

    if nu == np.inf:
        x = np.array([1.0] * num_samples)
    else:
        x = np.random.chisquare(nu, num_samples) / nu

    rv = scipy.stats.multivariate_normal(mean=np.zeros(mu.shape[0]), cov=Sigma)
    list_samples = [rv.rvs() for _ in range(0, num_samples)]

    samples = np.array(list_samples)
    samples = mu[np.newaxis, ...] + samples / np.sqrt(x)[..., np.newaxis]
    assert samples.shape == (num_samples, mu.shape[0])

    return samples

@utils_common.validate_types
def predict_with_cov(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray,
    cov_X_X: np.ndarray, inv_cov_X_X: np.ndarray, hyps: dict,
    str_cov: str=constants.STR_COV,
    prior_mu: constants.TYPING_UNION_CALLABLE_NONE=None,
    debug: bool=False
) -> constants.TYPING_TUPLE_FLOAT_THREE_ARRAYS:
    """
    This function returns degree of freedom, posterior mean,
    posterior standard variance, and
    posterior covariance functions over `X_test`,
    computed by Student-$t$ process regression with
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
    :param hyps: dictionary of hyperparameters for Student-$t$ process.
    :type hyps: dict.
    :param str_cov: the name of covariance function.
    :type str_cov: str., optional
    :param prior_mu: None, or prior mean function.
    :type prior_mu: NoneType, or callable, optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    :returns: a tuple of degree of freedom,
        posterior mean function over `X_test`,
        posterior standrad variance function over `X_test`, and
        posterior covariance matrix over `X_test`.
        Shape: ((), (l, 1), (l, 1), (l, l)).
    :rtype: tuple of (float, numpy.ndarray, numpy.ndarray, numpy.ndarray)

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

    num_X = X_train.shape[0]
    new_Y_train = Y_train - prior_mu_train
    nu = hyps['dof']
    beta = np.squeeze(np.dot(np.dot(new_Y_train.T, inv_cov_X_X), new_Y_train))

    nu_Xs = nu + float(num_X)
    mu_Xs = np.dot(np.dot(cov_X_Xs.T, inv_cov_X_X), new_Y_train) + prior_mu_test
    Sigma_Xs = cov_Xs_Xs - np.dot(np.dot(cov_X_Xs.T, inv_cov_X_X), cov_X_Xs)
    Sigma_Xs = (nu + beta - 2.0) / (nu + num_X - 2.0) * Sigma_Xs

    sigma_Xs = np.expand_dims(np.sqrt(np.maximum(np.diag(Sigma_Xs), 0.0)), axis=1)

    return nu_Xs, mu_Xs, sigma_Xs, Sigma_Xs

@utils_common.validate_types
def predict_with_hyps(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, hyps: dict,
    str_cov: str=constants.STR_COV,
    prior_mu: constants.TYPING_UNION_CALLABLE_NONE=None,
    debug: bool=False
) -> constants.TYPING_TUPLE_FLOAT_THREE_ARRAYS:
    """
    This function returns degree of freedom, posterior mean,
    posterior standard variance, and
    posterior covariance functions over `X_test`,
    computed by Student-$t$ process regression with
    `X_train`, `Y_train`, and `hyps`.

    :param X_train: inputs. Shape: (n, d) or (n, m, d).
    :type X_train: numpy.ndarray
    :param Y_train: outputs. Shape: (n, 1).
    :type Y_train: numpy.ndarray
    :param X_test: inputs. Shape: (l, d) or (l, m, d).
    :type X_test: numpy.ndarray
    :param hyps: dictionary of hyperparameters for Student-$t$ process.
    :type hyps: dict.
    :param str_cov: the name of covariance function.
    :type str_cov: str., optional
    :param prior_mu: None, or prior mean function.
    :type prior_mu: NoneType, or callable, optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    :returns: a tuple of degree of freedom,
        posterior mean function over `X_test`,
        posterior standrad variance function over `X_test`, and
        posterior covariance matrix over `X_test`.
        Shape: ((), (l, 1), (l, 1), (l, l)).
    :rtype: tuple of (float, numpy.ndarray, numpy.ndarray, numpy.ndarray)

    :raises: AssertionError

    """

    utils_gp.validate_common_args(X_train, Y_train, str_cov, prior_mu, debug, X_test)
    assert isinstance(hyps, dict)
    utils_covariance.check_str_cov('predict_with_hyps', str_cov, X_train.shape,
        shape_X2=X_test.shape)

    cov_X_X, inv_cov_X_X, _ = covariance.get_kernel_inverse(X_train,
        hyps, str_cov, debug=debug)
    nu_Xs, mu_Xs, sigma_Xs, Sigma_Xs = predict_with_cov(X_train, Y_train, X_test,
        cov_X_X, inv_cov_X_X, hyps, str_cov=str_cov,
        prior_mu=prior_mu, debug=debug)

    return nu_Xs, mu_Xs, sigma_Xs, Sigma_Xs

@utils_common.validate_types
def predict_with_optimized_hyps(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray,
    str_cov: str=constants.STR_COV,
    str_optimizer_method: str=constants.STR_OPTIMIZER_METHOD_TP,
    prior_mu: constants.TYPING_UNION_CALLABLE_NONE=None,
    fix_noise: float=constants.FIX_GP_NOISE,
    debug: bool=False
) -> constants.TYPING_TUPLE_FLOAT_THREE_ARRAYS:
    """
    This function returns degree of freedom, posterior mean,
    posterior standard variance, and
    posterior covariance functions over `X_test`,
    computed by the Student-$t$ process regression
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

    :returns: a tuple of degree of freedom,
        posterior mean function over `X_test`,
        posterior standrad variance function over `X_test`, and
        posterior covariance matrix over `X_test`.
        Shape: ((), (l, 1), (l, 1), (l, l)).
    :rtype: tuple of (float, numpy.ndarray, numpy.ndarray, numpy.ndarray)

    :raises: AssertionError

    """

    utils_gp.validate_common_args(X_train, Y_train, str_cov, prior_mu, debug, X_test)
    assert isinstance(str_optimizer_method, str)
    assert isinstance(fix_noise, bool)
    utils_covariance.check_str_cov('predict_with_optimized_kernel', str_cov,
        X_train.shape, shape_X2=X_test.shape)
    assert str_optimizer_method in constants.ALLOWED_OPTIMIZER_METHOD_GP

    time_start = time.time()

    cov_X_X, inv_cov_X_X, hyps = tp_kernel.get_optimized_kernel(X_train, Y_train,
        prior_mu, str_cov, str_optimizer_method=str_optimizer_method,
        fix_noise=fix_noise, debug=debug)
    nu_Xs, mu_Xs, sigma_Xs, Sigma_Xs = predict_with_cov(X_train, Y_train, X_test,
        cov_X_X, inv_cov_X_X, hyps, str_cov=str_cov, prior_mu=prior_mu,
        debug=debug)

    time_end = time.time()
    if debug:
        logger.debug('time consumed to construct tpr: %.4f sec.', time_end - time_start)
    return nu_Xs, mu_Xs, sigma_Xs, Sigma_Xs
