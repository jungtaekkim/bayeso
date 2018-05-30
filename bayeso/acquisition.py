# acquisition
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: May 30, 2018

import numpy as np
import scipy.stats

from bayeso import constants


def pi(pred_mean, pred_std, Y_train, jitter=constants.JITTER_ACQ):
    with np.errstate(divide='ignore'):
        val_z = (np.min(Y_train) - pred_mean) / (pred_std + jitter)
    return scipy.stats.norm.cdf(val_z)

def ei(pred_mean, pred_std, Y_train, jitter=constants.JITTER_ACQ):
    with np.errstate(divide='ignore'):
        val_z = (np.min(Y_train) - pred_mean) / (pred_std + jitter)
    return (np.min(Y_train) - pred_mean) * scipy.stats.norm.cdf(val_z) + pred_std * scipy.stats.norm.pdf(val_z)

def ucb(pred_mean, pred_std, kappa=2.0, Y_train=None, is_increased=True):
    kappa_ = kappa * np.log(Y_train.shape[0])
    return -pred_mean + kappa_ * pred_std

if __name__ == '__main__':
    pass
