# covariance
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: November 25, 2019

import numpy as np
import scipy.spatial.distance as scisd

from bayeso import constants
from bayeso.utils import utils_covariance


def choose_fun_cov(str_cov, is_grad=False):
    assert isinstance(str_cov, str)
    assert isinstance(is_grad, bool)

    if str_cov == 'se':
        if is_grad:
            fun_cov = grad_cov_se
        else:
            fun_cov = cov_se
    elif str_cov == 'matern32':
        if is_grad:
            fun_cov = grad_cov_matern32
        else:
            fun_cov = cov_matern32
    elif str_cov == 'matern52':
        if is_grad:
            fun_cov = grad_cov_matern52
        else:
            fun_cov = cov_matern52
    else:
        raise NotImplementedError('cov_main: allowed str_cov and is_grad conditions, but it is not implemented.')
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

def grad_cov_se(cov_, X, Xs, hyps, num_hyps, is_fixed_noise):
    assert isinstance(cov_, np.ndarray)
    assert isinstance(X, np.ndarray)
    assert isinstance(Xs, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(num_hyps, int)
    assert isinstance(is_fixed_noise, bool)

    num_X = X.shape[0]
    num_Xs = Xs.shape[0]

    grad_cov_ = np.zeros((num_X, num_Xs, num_hyps))

    for ind_X in range(0, num_X):
        for ind_Xs in range(0, num_Xs):
            X_Xs_l = (X[ind_X] - Xs[ind_Xs]) / hyps['lengthscales']
            dist = np.linalg.norm(X_Xs_l, ord=2)

            ind_next = 0
            if not is_fixed_noise:
                if ind_X == ind_Xs:
                    grad_cov_[ind_X, ind_Xs, 0] = 2.0 * hyps['noise']
                ind_next += 1
            grad_cov_[ind_X, ind_Xs, ind_next] = 2.0 * cov_[ind_X, ind_Xs] / hyps['signal']
            grad_cov_[ind_X, ind_Xs, ind_next+1:] = cov_[ind_X, ind_Xs] * dist**2 * hyps['lengthscales']**(-1)

    return grad_cov_

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

def grad_cov_matern32(cov_, X, Xs, hyps, num_hyps, is_fixed_noise):
    assert isinstance(cov_, np.ndarray)
    assert isinstance(X, np.ndarray)
    assert isinstance(Xs, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(num_hyps, int)
    assert isinstance(is_fixed_noise, bool)

    num_X = X.shape[0]
    num_Xs = Xs.shape[0]

    grad_cov_ = np.zeros((num_X, num_Xs, num_hyps))

    for ind_X in range(0, num_X):
        for ind_Xs in range(0, num_Xs):
            X_Xs_l = (X[ind_X] - Xs[ind_Xs]) / hyps['lengthscales']
            dist = np.linalg.norm(X_Xs_l, ord=2)

            ind_next = 0
            if not is_fixed_noise:
                if ind_X == ind_Xs:
                    grad_cov_[ind_X, ind_Xs, 0] = 2.0 * hyps['noise']
                ind_next += 1
            grad_cov_[ind_X, ind_Xs, ind_next] = 2.0 * cov_[ind_X, ind_Xs] / hyps['signal']
            grad_cov_[ind_X, ind_Xs, ind_next+1:] = 3.0 * hyps['signal']**2 * np.exp(-np.sqrt(3) * dist) * dist**2 * hyps['lengthscales']**(-1)

    return grad_cov_

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

def grad_cov_matern52(cov_, X, Xs, hyps, num_hyps, is_fixed_noise):
    assert isinstance(cov_, np.ndarray)
    assert isinstance(X, np.ndarray)
    assert isinstance(Xs, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(num_hyps, int)
    assert isinstance(is_fixed_noise, bool)

    num_X = X.shape[0]
    num_Xs = Xs.shape[0]

    grad_cov_ = np.zeros((num_X, num_Xs, num_hyps))

    for ind_X in range(0, num_X):
        for ind_Xs in range(0, num_Xs):
            X_Xs_l = (X[ind_X] - Xs[ind_Xs]) / hyps['lengthscales']
            dist = np.linalg.norm((X[ind_X] - Xs[ind_Xs]) / hyps['lengthscales'], ord=2)

            ind_next = 0
            if not is_fixed_noise:
                if ind_X == ind_Xs:
                    grad_cov_[ind_X, ind_Xs, 0] = 2.0 * hyps['noise']
                ind_next += 1
            grad_cov_[ind_X, ind_Xs, ind_next] = 2.0 * cov_[ind_X, ind_Xs] / hyps['signal']
            grad_cov_[ind_X, ind_Xs, ind_next+1:] = 5.0 / 3.0 * hyps['signal']**2 * (1.0 + np.sqrt(5) * dist) * np.exp(-np.sqrt(5) * dist) * dist**3 * hyps['lengthscales']**(-1)

    return grad_cov_

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

def cov_main(str_cov, X, Xs, hyps, same_X_Xs,
    jitter=constants.JITTER_COV
):
    assert isinstance(str_cov, str)
    assert isinstance(X, np.ndarray)
    assert isinstance(Xs, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(same_X_Xs, bool)
    assert isinstance(jitter, float)
    assert str_cov in constants.ALLOWED_GP_COV

    num_X = X.shape[0]
    num_Xs = Xs.shape[0]

    cov_ = np.zeros((num_X, num_Xs))
    if same_X_Xs:
        assert num_X == num_Xs
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

        if not same_X_Xs:
            for ind_X in range(0, num_X):
                for ind_Xs in range(0, num_Xs):
                    cov_[ind_X, ind_Xs] += fun_cov(X[ind_X], Xs[ind_Xs], hyps['lengthscales'], hyps['signal'])
        else:
            for ind_X in range(0, num_X):
                for ind_Xs in range(ind_X, num_Xs):
                    cov_[ind_X, ind_Xs] += fun_cov(X[ind_X], Xs[ind_Xs], hyps['lengthscales'], hyps['signal'])
                    if ind_X < ind_Xs:
                        cov_[ind_Xs, ind_X] = cov_[ind_X, ind_Xs]
    elif str_cov in constants.ALLOWED_GP_COV_SET:
        list_str_cov = str_cov.split('_')
        str_cov = list_str_cov[1]
        assert len(X.shape) == 3
        assert len(Xs.shape) == 3
        num_d_X = X.shape[2]
        num_d_Xs = Xs.shape[2]
        assert num_d_X == num_d_Xs

        hyps, is_valid = utils_covariance.validate_hyps_dict(hyps, str_cov, num_d_X)
        if not is_valid:
            raise ValueError('cov_main: invalid hyperparameters.')

        if not same_X_Xs:
            for ind_X in range(0, num_X):
                for ind_Xs in range(0, num_Xs):
                    cov_[ind_X, ind_Xs] += cov_set(str_cov, X[ind_X], Xs[ind_Xs], hyps['lengthscales'], hyps['signal'])
        else:
            for ind_X in range(0, num_X):
                for ind_Xs in range(ind_X, num_Xs):
                    cov_[ind_X, ind_Xs] += cov_set(str_cov, X[ind_X], Xs[ind_Xs], hyps['lengthscales'], hyps['signal'])
                    if ind_X < ind_Xs:
                        cov_[ind_Xs, ind_X] = cov_[ind_X, ind_Xs]
    else:
        raise NotImplementedError('cov_main: allowed str_cov, but it is not implemented.')
    return cov_

def grad_cov_main(str_cov, X, Xs, hyps, is_fixed_noise,
    same_X_Xs=True,
    jitter=constants.JITTER_COV,
):
    assert isinstance(str_cov, str)
    assert isinstance(X, np.ndarray)
    assert isinstance(Xs, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(is_fixed_noise, bool)
    assert isinstance(same_X_Xs, bool)
    assert isinstance(jitter, float)
    assert str_cov in constants.ALLOWED_GP_COV
    # TODO: X and Xs should be same?
    assert same_X_Xs

    num_dim = X.shape[1]

    if isinstance(hyps['lengthscales'], np.ndarray):
        is_scalar_lengthscales = False
        num_hyps = num_dim + 1
    else:
        is_scalar_lengthscales = True
        num_hyps = 2
    if not is_fixed_noise:
        num_hyps += 1

    cov_ = cov_main(str_cov, X, Xs, hyps, same_X_Xs, jitter=jitter)
    fun_grad_cov = choose_fun_cov(str_cov, is_grad=True)

    # TODO: I guess some gradients are wrong.
    grad_cov_ = fun_grad_cov(cov_, X, Xs, hyps, num_hyps, is_fixed_noise)

    return grad_cov_
