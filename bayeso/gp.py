# gp
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: January 31, 2020

import time
import numpy as np
import scipy 
import scipy.linalg
import scipy.optimize

from bayeso import covariance
from bayeso import constants
from bayeso.utils import utils_covariance


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
    lower = scipy.linalg.cholesky(cov_X_X, lower=True)

    if is_gradient:
        grad_cov_X_X = covariance.grad_cov_main(str_cov, X_train, X_train, hyps, is_fixed_noise, same_X_Xs=True)
    else:
        grad_cov_X_X = None
    return cov_X_X, lower, grad_cov_X_X

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
    _check_str_cov('neg_log_ml', str_cov, X_train.shape)

    hyps = utils_covariance.restore_hyps(str_cov, hyps, is_fixed_noise=is_fixed_noise)
    new_Y_train = Y_train - prior_mu_train
    if is_cholesky:
        cov_X_X, lower, grad_cov_X_X = get_kernel_cholesky(X_train, hyps, str_cov, is_fixed_noise=is_fixed_noise, is_gradient=is_gradient, debug=debug)

#        lower_new_Y_train = scipy.linalg.cho_solve((lower, True), new_Y_train)
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
        cov_X_X, inv_cov_X_X, grad_cov_X_X = get_kernel_inverse(X_train, hyps, str_cov, is_fixed_noise=is_fixed_noise, is_gradient=is_gradient, debug=debug)

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
    _check_str_cov('neg_log_pseudo_l_loocv', str_cov, X_train.shape)

    num_data = X_train.shape[0]
    hyps = utils_covariance.restore_hyps(str_cov, hyps, is_fixed_noise=is_fixed_noise)

    cov_X_X, inv_cov_X_X, grad_cov_X_X = get_kernel_inverse(X_train, hyps, str_cov, is_fixed_noise=is_fixed_noise, debug=debug)

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
    _check_str_cov('get_optimized_kernel', str_cov, X_train.shape)
    assert str_optimizer_method in constants.ALLOWED_OPTIMIZER_METHOD_GP
    assert str_modelselection_method in constants.ALLOWED_MODELSELECTION_METHOD
    # TODO: fix this.
    is_gradient = True

    time_start = time.time()

    if debug:
        print('[DEBUG] get_optimized_kernel in gp.py: str_optimizer_method {}'.format(str_optimizer_method))
        print('[DEBUG] get_optimized_kernel in gp.py: str_modelselection_method {}'.format(str_modelselection_method))

    prior_mu_train = get_prior_mu(prior_mu, X_train)
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
        result_optimized = result_optimized.x
    elif str_optimizer_method == 'L-BFGS-B':
        bounds = utils_covariance.get_range_hyps(str_cov, num_dim, is_fixed_noise=is_fixed_noise)
        result_optimized = scipy.optimize.minimize(neg_log_ml_, hyps_converted, method=str_optimizer_method, bounds=bounds, jac=is_gradient, options={'disp': False})
        result_optimized = result_optimized.x
    # TODO: Fill this conditions
    elif str_optimizer_method == 'DIRECT': # pragma: no cover
        raise NotImplementedError('get_optimized_kernel: allowed str_optimizer_method, but it is not implemented.')
    elif str_optimizer_method == 'CMA-ES': # pragma: no cover
        raise NotImplementedError('get_optimized_kernel: allowed str_optimizer_method, but it is not implemented.')
    else: # pragma: no cover
        raise ValueError('get_optimized_kernel: missing conditions for str_optimizer_method')

    hyps = utils_covariance.restore_hyps(str_cov, result_optimized, is_fixed_noise=is_fixed_noise)

    hyps, _ = utils_covariance.validate_hyps_dict(hyps, str_cov, num_dim)
    cov_X_X, inv_cov_X_X, grad_cov_X_X = get_kernel_inverse(X_train, hyps, str_cov, is_fixed_noise=is_fixed_noise, debug=debug)

    time_end = time.time()

    if debug:
        print('[DEBUG] get_optimized_kernel in gp.py: optimized hyps for gpr', hyps)
        print('[DEBUG] get_optimized_kernel in gp.py: time consumed', time_end - time_start, 'sec.')
    return cov_X_X, inv_cov_X_X, hyps

def predict_test_(X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps,
    str_cov=constants.STR_GP_COV,
    prior_mu=None,
    debug=False
):
    """
    This function returns posterior mean and posterior standard deviation functions over `X_test`, computed by Gaussian process regression with `X_train`, `Y_train`, `cov_X_X`, `inv_cov_X_X`, and `hyps`.

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
    :type prior_mu: NoneType, or function, optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    :returns: a tuple of posterior mean function over `X_test` and posterior standard deviation function over `X_test`. Shape: ((l, 1), (l, 1)).
    :rtype: tuple of (numpy.ndarray, numpy.ndarray)

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
    _check_str_cov('predict_test_', str_cov, X_train.shape, shape_X2=X_test.shape)
    assert X_train.shape[0] == Y_train.shape[0]
    assert X_train.shape[1] == X_test.shape[1]

    prior_mu_train = get_prior_mu(prior_mu, X_train)
    prior_mu_test = get_prior_mu(prior_mu, X_test)
    cov_X_Xs = covariance.cov_main(str_cov, X_train, X_test, hyps, False)
    cov_Xs_Xs = covariance.cov_main(str_cov, X_test, X_test, hyps, True)
    cov_Xs_Xs = (cov_Xs_Xs + cov_Xs_Xs.T) / 2.0

    mu_Xs = np.dot(np.dot(cov_X_Xs.T, inv_cov_X_X), Y_train - prior_mu_train) + prior_mu_test
    Sigma_Xs = cov_Xs_Xs - np.dot(np.dot(cov_X_Xs.T, inv_cov_X_X), cov_X_Xs)
    return mu_Xs, np.expand_dims(np.sqrt(np.maximum(np.diag(Sigma_Xs), 0.0)), axis=1)

def predict_test(X_train, Y_train, X_test, hyps,
    str_cov=constants.STR_GP_COV,
    prior_mu=None,
    debug=False
):
    """
    This function returns posterior mean and posterior standard deviation functions over `X_test`, computed by Gaussian process regression with `X_train`, `Y_train`, and `hyps`.

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
    :type prior_mu: NoneType, or function, optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    :returns: a tuple of posterior mean function over `X_test` and posterior standard deviation function over `X_test`. Shape: ((l, 1), (l, 1)).
    :rtype: tuple of (numpy.ndarray, numpy.ndarray)

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
    _check_str_cov('predict_test', str_cov, X_train.shape, shape_X2=X_test.shape)
    assert X_train.shape[0] == Y_train.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    
    cov_X_X, inv_cov_X_X, grad_cov_X_X = get_kernel_inverse(X_train, hyps, str_cov, debug=debug)
    mu_Xs, sigma_Xs = predict_test_(X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov=str_cov, prior_mu=prior_mu, debug=debug)
    return mu_Xs, sigma_Xs

def predict_optimized(X_train, Y_train, X_test,
    str_cov=constants.STR_GP_COV,
    prior_mu=None,
    is_fixed_noise=constants.IS_FIXED_GP_NOISE,
    debug=False
):
    """
    This function returns posterior mean and posterior standard deviation functions over `X_test`, computed by the Gaussian process regression optimized with `X_train` and `Y_train`.

    :param X_train: inputs. Shape: (n, d) or (n, m, d).
    :type X_train: numpy.ndarray
    :param Y_train: outputs. Shape: (n, 1).
    :type Y_train: numpy.ndarray
    :param X_test: inputs. Shape: (l, d) or (l, m, d).
    :type X_test: numpy.ndarray
    :param str_cov: the name of covariance function.
    :type str_cov: str., optional
    :param prior_mu: None, or prior mean function.
    :type prior_mu: NoneType, or function, optional
    :param is_fixed_noise: flag for fixing a noise.
    :type is_fixed_noise: bool., optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    :returns: a tuple of posterior mean function over `X_test` and posterior standard deviation function over `X_test`. Shape: ((l, 1), (l, 1)).
    :rtype: tuple of (numpy.ndarray, numpy.ndarray)

    :raises: AssertionError

    """

    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(str_cov, str)
    assert isinstance(is_fixed_noise, bool)
    assert isinstance(debug, bool)
    assert callable(prior_mu) or prior_mu is None
    assert len(Y_train.shape) == 2
    _check_str_cov('predict_optimized', str_cov, X_train.shape, shape_X2=X_test.shape)
    assert X_train.shape[0] == Y_train.shape[0]
    assert X_train.shape[1] == X_test.shape[1]

    time_start = time.time()

    cov_X_X, inv_cov_X_X, hyps = get_optimized_kernel(X_train, Y_train, prior_mu, str_cov, is_fixed_noise=is_fixed_noise, debug=debug)
    mu_Xs, sigma_Xs = predict_test_(X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov=str_cov, prior_mu=prior_mu, debug=debug)

    time_end = time.time()
    if debug:
        print('[DEBUG] predict_optimized in gp.py: time consumed', time_end - time_start, 'sec.')
    return mu_Xs, sigma_Xs
