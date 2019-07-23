# covariance
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: April 11, 2019

import numpy as np
import scipy.spatial.distance as scisd

from bayeso import constants
from bayeso.utils import utils_covariance


def choose_fun_cov(str_cov):
    if str_cov == 'se':
        fun_cov = cov_se
    elif str_cov == 'matern32':
        fun_cov = cov_matern32
    elif str_cov == 'matern52':
        fun_cov = cov_matern52
    else:
        raise NotImplementedError('cov_main: allowed str_cov condition, but it is not implemented.')
    return fun_cov

def cov_se(bx, bxp, lengthscales, signal):
    assert isinstance(bx, np.ndarray)
    assert isinstance(bxp, np.ndarray)
    assert isinstance(lengthscales, np.ndarray) or isinstance(lengthscales, float)
    assert isinstance(signal, float)
    if isinstance(lengthscales, np.ndarray):
        assert bx.shape[0] == bxp.shape[0] == lengthscales.shape[0]
    else:
        assert bx.shape[0] == bxp.shape[0]
    return signal**2 * np.exp(-0.5 * np.linalg.norm((bx - bxp) / lengthscales, ord=2)**2)

def cov_matern32(bx, bxp, lengthscales, signal):
    assert isinstance(bx, np.ndarray)
    assert isinstance(bxp, np.ndarray)
    assert isinstance(lengthscales, np.ndarray) or isinstance(lengthscales, float)
    if isinstance(lengthscales, np.ndarray):
        assert bx.shape[0] == bxp.shape[0] == lengthscales.shape[0]
    else:
        assert bx.shape[0] == bxp.shape[0]
    assert isinstance(signal, float)

    dist = np.linalg.norm((bx - bxp) / lengthscales, ord=2)
    return signal**2 * (1.0 + np.sqrt(3.0) * dist) * np.exp(-1.0 * np.sqrt(3.0) * dist)

def cov_matern52(bx, bxp, lengthscales, signal):
    assert isinstance(bx, np.ndarray)
    assert isinstance(bxp, np.ndarray)
    assert isinstance(lengthscales, np.ndarray) or isinstance(lengthscales, float)
    if isinstance(lengthscales, np.ndarray):
        assert bx.shape[0] == bxp.shape[0] == lengthscales.shape[0]
    else:
        assert bx.shape[0] == bxp.shape[0]
    assert isinstance(signal, float)

    dist = np.linalg.norm((bx - bxp) / lengthscales, ord=2)
    return signal**2 * (1.0 + np.sqrt(5.0) * dist + 5.0 / 3.0 * dist**2) * np.exp(-1.0 * np.sqrt(5.0) * dist)

def cov_set(str_cov, X, Xs, lengthscales, signal):
    assert isinstance(str_cov, str)
    assert isinstance(X, np.ndarray)
    assert isinstance(Xs, np.ndarray)
    assert isinstance(lengthscales, np.ndarray) or isinstance(lengthscales, float)
    assert isinstance(signal, float)
    assert len(X.shape) == 2
    assert len(Xs.shape) == 2
    if isinstance(lengthscales, np.ndarray):
        assert X.shape[1] == Xs.shape[1] == lengthscales.shape[0]
    else:
        assert X.shape[1] == Xs.shape[1]
    assert str_cov in constants.ALLOWED_GP_COV_BASE
    num_X = X.shape[0]
    num_Xs = Xs.shape[0]
    num_d_X = X.shape[1]
    num_d_Xs = Xs.shape[1]

    fun_cov = choose_fun_cov(str_cov)
    cov_ = 0.0
    for ind_X in range(0, num_X):
        for ind_Xs in range(0, num_Xs):
            cov_ += fun_cov(X[ind_X], Xs[ind_Xs], lengthscales, signal)
    cov_ /= num_X * num_Xs

    return cov_

def cov_main(str_cov, X, Xs, hyps,
    jitter=constants.JITTER_COV
):
    assert isinstance(str_cov, str)
    assert isinstance(X, np.ndarray)
    assert isinstance(Xs, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(jitter, float)
    assert str_cov in constants.ALLOWED_GP_COV

    num_X = X.shape[0]
    num_Xs = Xs.shape[0]

    cov_ = np.zeros((num_X, num_Xs))
    if num_X == num_Xs:
        cov_ += np.eye(num_X) * jitter

    if str_cov == 'se' or str_cov == 'matern32' or str_cov == 'matern52':
        assert len(X.shape) == 2
        assert len(Xs.shape) == 2
        num_d_X = X.shape[1]
        num_d_Xs = Xs.shape[1]
        assert num_d_X == num_d_Xs

        hyps, is_valid = utils_covariance.validate_hyps_dict(hyps, str_cov, num_d_X)
        # TODO: ValueError is appropriate? We can just raise AssertionError in validate_hyps_dict. I am not sure.
        if not is_valid:
            raise ValueError('cov_main: invalid hyperparameters.')

        fun_cov = choose_fun_cov(str_cov)

        for ind_X in range(0, num_X):
            for ind_Xs in range(0, num_Xs):
                cov_[ind_X, ind_Xs] += fun_cov(X[ind_X], Xs[ind_Xs], hyps['lengthscales'], hyps['signal'])
    elif str_cov in constants.ALLOWED_GP_COV_SET:
        list_str_cov = str_cov.split('_')
        str_cov = list_str_cov[1]
        assert len(X.shape) == 3
        assert len(Xs.shape) == 3
        num_d_X = X.shape[2]
        num_d_Xs = Xs.shape[2]
        assert num_d_X == num_d_Xs

        hyps, is_valid = utils_covariance.validate_hyps_dict(hyps, str_cov, num_d_X)
        # TODO: Please check this is_valid.
        if not is_valid: # pragma: no cover
            raise ValueError('cov_main: invalid hyperparameters.')

        for ind_X in range(0, num_X):
            for ind_Xs in range(0, num_Xs):
                cov_[ind_X, ind_Xs] += cov_set(str_cov, X[ind_X], Xs[ind_Xs], hyps['lengthscales'], hyps['signal'])
    else:
        raise NotImplementedError('cov_main: allowed str_cov, but it is not implemented.')
    return cov_

def grad_cov_main(str_cov, X, Xs, hyps, is_fixed_noise,
    jitter=constants.JITTER_COV,
):
    # TODO: X and Xs should be same?
    assert isinstance(str_cov, str)
    assert isinstance(X, np.ndarray)
    assert isinstance(Xs, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(is_fixed_noise, bool)
    assert isinstance(jitter, float)
    assert str_cov in constants.ALLOWED_GP_COV

    num_X = X.shape[0]
    num_Xs = Xs.shape[0]
    num_dim = X.shape[1]
    if isinstance(hyps['lengthscales'], np.ndarray):
        is_scalar_lengthscales = False
        num_hyps = num_dim + 1
    else:
        is_scalar_lengthscales = True
        num_hyps = 2
    if not is_fixed_noise:
        num_hyps += 1

    cov_ = cov_main(str_cov, X, Xs, hyps, jitter=jitter)
    grad_cov_ = np.zeros((num_X, num_Xs, num_hyps))

    # TODO: I guess some gradients are wrong.
    if str_cov == 'se':
        for ind_X in range(0, num_X):
            for ind_Xs in range(0, num_Xs):
                X_Xs_l = (X[ind_X] - Xs[ind_Xs]) / hyps['lengthscales']
                dist = np.linalg.norm(X_Xs_l, ord=2)

                ind_next = 0
                if not is_fixed_noise:
                    # TODO: is it 1.0?
                    grad_cov_[ind_X, ind_Xs, 0] = 1.0
                    ind_next += 1
                grad_cov_[ind_X, ind_Xs, ind_next] = cov_[ind_X, ind_Xs] / hyps['signal']**2
                grad_cov_[ind_X, ind_Xs, ind_next+1:] = cov_[ind_X, ind_Xs] * dist**2
    elif str_cov == 'matern32':
        for ind_X in range(0, num_X):
            for ind_Xs in range(0, num_Xs):
                X_Xs_l = (X[ind_X] - Xs[ind_Xs]) / hyps['lengthscales']
                dist = np.linalg.norm(X_Xs_l, ord=2)

                ind_next = 0
                if not is_fixed_noise:
                    grad_cov_[ind_X, ind_Xs, 0] = 1.0
                    ind_next += 1
                grad_cov_[ind_X, ind_Xs, ind_next] = cov_[ind_X, ind_Xs] / hyps['signal']**2
                grad_cov_[ind_X, ind_Xs, ind_next+1:] = 3.0 * hyps['signal']**2 * np.exp(-np.sqrt(3) * dist) * dist**2
    elif str_cov == 'matern52':
        for ind_X in range(0, num_X):
            for ind_Xs in range(0, num_Xs):
                X_Xs_l = (X[ind_X] - Xs[ind_Xs]) / hyps['lengthscales']
                dist = np.linalg.norm((X[ind_X] - Xs[ind_Xs]) / hyps['lengthscales'], ord=2)

                ind_next = 0
                if not is_fixed_noise:
                    grad_cov_[ind_X, ind_Xs, 0] = 1.0
                    ind_next += 1

                grad_cov_[ind_X, ind_Xs, ind_next] = cov_[ind_X, ind_Xs] / hyps['signal']**2
                grad_cov_[ind_X, ind_Xs, ind_next+1:] = 5.0 / 3.0 * hyps['signal']**2 * (1 + np.sqrt(5) * dist) * np.exp(-np.sqrt(5) * dist) * dist**2
    else:
        raise NotImplementedError('grad_cov_main: a missing str_cov')

    return grad_cov_
