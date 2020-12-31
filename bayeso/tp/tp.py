#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: December 30, 2020
#
"""It defines Student-$t$ process regression."""

import time
import numpy as np
import scipy.stats
import scipy.linalg
import scipy.optimize
import scipy.special

from bayeso import covariance
from bayeso import constants
from bayeso.utils import utils_gp
from bayeso.utils import utils_covariance
from bayeso.utils import utils_common
from bayeso.utils import utils_logger

logger = utils_logger.get_logger('tp')


@utils_common.validate_types
def neg_log_ml(X_train: np.ndarray, Y_train: np.ndarray, hyps: np.ndarray,
    str_cov: str, prior_mu_train: np.ndarray,
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

    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(hyps, np.ndarray)
    assert isinstance(str_cov, str)
    assert isinstance(prior_mu_train, np.ndarray)
    assert isinstance(fix_noise, bool)
    assert isinstance(use_gradient, bool)
    assert isinstance(debug, bool)
    assert len(Y_train.shape) == 2
    assert len(prior_mu_train.shape) == 2
    assert X_train.shape[0] == Y_train.shape[0] == prior_mu_train.shape[0]
    utils_covariance.check_str_cov('neg_log_ml', str_cov, X_train.shape)

    num_X = float(X_train.shape[0])
    hyps = utils_covariance.restore_hyps(str_cov, hyps,
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
    # TODO: let me think.
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
        nu_grad = -num_X / (2.0 * (nu - 2.0)) + scipy.special.digamma((nu + num_X) / 2.0) - scipy.special.digamma(nu / 2.0) - 0.5 * np.log(1.0 + beta / (nu - 2.0)) + (nu + num_X) * beta / (2.0 * (nu - 2.0)**2 + 2.0 * beta * (nu - 2.0))

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
def get_optimized_kernel(X_train: np.ndarray, Y_train: np.ndarray,
    prior_mu: constants.TYPING_UNION_CALLABLE_NONE, str_cov: str,
    str_optimizer_method: str=constants.STR_OPTIMIZER_METHOD_TP,
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
    :param fix_noise: flag for fixing a noise.
    :type fix_noise: bool., optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    :returns: a tuple of kernel matrix over `X_train`, kernel matrix
        inverse, and dictionary of hyperparameters.
    :rtype: tuple of (numpy.ndarray, numpy.ndarray, dict.)

    :raises: AssertionError, ValueError

    """

    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert callable(prior_mu) or prior_mu is None
    assert isinstance(str_cov, str)
    assert isinstance(str_optimizer_method, str)
    assert isinstance(fix_noise, bool)
    assert isinstance(debug, bool)
    assert len(Y_train.shape) == 2
    assert X_train.shape[0] == Y_train.shape[0]
    utils_covariance.check_str_cov('get_optimized_kernel', str_cov, X_train.shape)
    assert str_optimizer_method in constants.ALLOWED_OPTIMIZER_METHOD_TP

    # TODO: Fix it later.
    use_gradient = True

    time_start = time.time()

    if debug:
        logger.debug('str_optimizer_method: %s', str_optimizer_method)

    prior_mu_train = utils_gp.get_prior_mu(prior_mu, X_train)
    if str_cov in constants.ALLOWED_GP_COV_BASE:
        num_dim = X_train.shape[1]
    elif str_cov in constants.ALLOWED_GP_COV_SET:
        num_dim = X_train.shape[2]
        use_gradient = False

    neg_log_ml_ = lambda hyps: neg_log_ml(X_train, Y_train, hyps, str_cov,
        prior_mu_train, fix_noise=fix_noise, use_gradient=use_gradient,
        debug=debug)

    hyps_converted = utils_covariance.convert_hyps(
        str_cov,
        utils_covariance.get_hyps(str_cov, num_dim, use_gp=False),
        fix_noise=fix_noise,
        use_gp=False
    )

    if str_optimizer_method in ['L-BFGS-B', 'SLSQP']:
        bounds = utils_covariance.get_range_hyps(str_cov, num_dim,
            fix_noise=fix_noise, use_gp=False)
        result_optimized = scipy.optimize.minimize(neg_log_ml_, hyps_converted,
            method=str_optimizer_method, bounds=bounds, jac=use_gradient,
            options={'disp': False})

        if debug:
            logger.debug('scipy message: %s', result_optimized.message)

        result_optimized = result_optimized.x
    else: # pragma: no cover
        raise ValueError('get_optimized_kernel: missing conditions for str_optimizer_method')

    hyps = utils_covariance.restore_hyps(str_cov, result_optimized, fix_noise=fix_noise, use_gp=False)

    hyps, _ = utils_covariance.validate_hyps_dict(hyps, str_cov, num_dim, use_gp=False)
    cov_X_X, inv_cov_X_X, _ = covariance.get_kernel_inverse(X_train,
        hyps, str_cov, fix_noise=fix_noise, debug=debug)

    time_end = time.time()

    if debug:
        logger.debug('hyps optimized: %s', utils_logger.get_str_hyps(hyps))
        logger.debug('time consumed to construct gpr: %.4f sec.', time_end - time_start)
    return cov_X_X, inv_cov_X_X, hyps

@utils_common.validate_types
def predict_with_cov(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray,
    cov_X_X: np.ndarray, inv_cov_X_X: np.ndarray, hyps: dict,
    str_cov: str=constants.STR_GP_COV,
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
    :type prior_mu: NoneType, or function, optional
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

    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(str_cov, str)
    assert isinstance(debug, bool)
    assert callable(prior_mu) or prior_mu is None
    assert len(Y_train.shape) == 2
    assert len(cov_X_X.shape) == 2
    assert len(inv_cov_X_X.shape) == 2
    assert (np.array(cov_X_X.shape) == np.array(inv_cov_X_X.shape)).all()
    utils_covariance.check_str_cov('predict_with_cov', str_cov,
        X_train.shape, shape_X2=X_test.shape)
    assert X_train.shape[0] == Y_train.shape[0]
    assert X_train.shape[1] == X_test.shape[1]

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
    str_cov: str=constants.STR_GP_COV,
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
    :type prior_mu: NoneType, or function, optional
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

    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(str_cov, str)
    assert isinstance(debug, bool)
    assert callable(prior_mu) or prior_mu is None
    assert len(Y_train.shape) == 2
    utils_covariance.check_str_cov('predict_with_hyps', str_cov, X_train.shape, shape_X2=X_test.shape)
    assert X_train.shape[0] == Y_train.shape[0]
    assert X_train.shape[1] == X_test.shape[1]

    cov_X_X, inv_cov_X_X, _ = covariance.get_kernel_inverse(X_train,
        hyps, str_cov, debug=debug)
    nu_Xs, mu_Xs, sigma_Xs, Sigma_Xs = predict_with_cov(X_train, Y_train, X_test,
        cov_X_X, inv_cov_X_X, hyps, str_cov=str_cov,
        prior_mu=prior_mu, debug=debug)

    return nu_Xs, mu_Xs, sigma_Xs, Sigma_Xs

@utils_common.validate_types
def predict_with_optimized_hyps(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray,
    str_cov: str=constants.STR_GP_COV,
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
    :type prior_mu: NoneType, or function, optional
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

    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(str_cov, str)
    assert isinstance(str_optimizer_method, str)
    assert isinstance(fix_noise, bool)
    assert isinstance(debug, bool)
    assert callable(prior_mu) or prior_mu is None
    assert len(Y_train.shape) == 2
    utils_covariance.check_str_cov('predict_with_optimized_kernel', str_cov,
        X_train.shape, shape_X2=X_test.shape)
    assert X_train.shape[0] == Y_train.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    assert str_optimizer_method in constants.ALLOWED_OPTIMIZER_METHOD_GP

    time_start = time.time()

    cov_X_X, inv_cov_X_X, hyps = get_optimized_kernel(X_train, Y_train,
        prior_mu, str_cov, str_optimizer_method=str_optimizer_method,
        fix_noise=fix_noise, debug=debug)
    nu_Xs, mu_Xs, sigma_Xs, Sigma_Xs = predict_with_cov(X_train, Y_train, X_test,
        cov_X_X, inv_cov_X_X, hyps, str_cov=str_cov, prior_mu=prior_mu,
        debug=debug)

    time_end = time.time()
    if debug:
        logger.debug('time consumed to construct gpr: %.4f sec.', time_end - time_start)
    return nu_Xs, mu_Xs, sigma_Xs, Sigma_Xs
