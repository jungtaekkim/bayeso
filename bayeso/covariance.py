# covariance
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: April 11, 2019

import numpy as np

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
