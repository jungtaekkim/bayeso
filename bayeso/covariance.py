# covariance
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 09, 2018

import numpy as np

from bayeso import constants


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

def cov_main(str_cov, X, Xs, hyps, jitter=constants.JITTER_COV):
    assert isinstance(str_cov, str)
    assert isinstance(X, np.ndarray)
    assert isinstance(Xs, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(jitter, float)
    assert len(X.shape) == 2
    assert len(Xs.shape) == 2
    assert str_cov in constants.ALLOWED_GP_COV

    num_X = X.shape[0]
    num_d_X = X.shape[1]
    num_Xs = Xs.shape[0]
    num_d_Xs = Xs.shape[1]
    assert num_d_X == num_d_Xs

    cov_ = np.zeros((num_X, num_Xs))
    if num_X == num_Xs:
        cov_ += np.eye(num_X) * jitter
    if str_cov == 'se':
        if hyps.get('lengthscales') is None or hyps.get('signal') is None:
            raise ValueError('cov_main: insufficient hyperparameters.')
        for ind_X in range(0, num_X):
            for ind_Xs in range(0, num_Xs):
                cov_[ind_X, ind_Xs] += cov_se(X[ind_X], Xs[ind_Xs], hyps['lengthscales'], hyps['signal'])
    elif str_cov == 'matern52' or str_cov == 'matern32':
        raise NotImplementedError('cov_main: matern52 or matern32.')
    else:
        raise NotImplementedError('cov_main: allowed str_cov, but it is not implemented.')
    return cov_
