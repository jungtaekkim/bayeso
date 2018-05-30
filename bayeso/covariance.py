# covariance
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: May 30, 2018

import numpy as np


def cov_se(x, xp, lengthscales, signal):
    return signal**2 * np.exp(-0.5 * np.linalg.norm((x - xp)/lengthscales)**2)

def cov_main(str_cov, X, Xs, hyps, jitter=1e-5):
    num_X = X.shape[0]
    num_d_X = X.shape[1]
    num_Xs = Xs.shape[0]
    num_d_Xs = Xs.shape[1]
    if num_d_X != num_d_Xs:
        print('ERROR: matrix dimensions: ', num_d_X, num_d_Xs)
        raise ValueError('matrix dimensions are different.')

    cov_ = np.zeros((num_X, num_Xs))
    if num_X == num_Xs:
        cov_ += np.eye(num_X) * jitter
    if str_cov == 'se':
        if hyps.get('lengthscales') is None or hyps.get('signal') is None:
            raise ValueError('hyperparameters are insufficient.')
        for ind_X in range(0, num_X):
            for ind_Xs in range(0, num_Xs):
                cov_[ind_X, ind_Xs] += cov_se(X[ind_X], Xs[ind_Xs], hyps['lengthscales'], hyps['signal'])
    else:
        raise ValueError('kernel is inappropriate.')
    return cov_


