import numpy as np
import scipy.stats

JITTER = 1e-5

def pi(pred_mean, pred_std, Y_train):
    with np.errstate(divide='ignore'):
        val_z = (np.min(Y_train) - pred_mean) / (pred_std + JITTER)
    return scipy.stats.norm.cdf(val_z)

def ei(pred_mean, pred_std, Y_train):
    with np.errstate(divide='ignore'):
        val_z = (np.min(Y_train) - pred_mean) / (pred_std + JITTER)
    return (np.min(Y_train) - pred_mean) * scipy.stats.norm.cdf(val_z) + pred_std * scipy.stats.norm.pdf(val_z)

def ucb(pred_mean, pred_std, Y_train=None):
    kappa = 10.0
    return -pred_mean + kappa * pred_std


if __name__ == '__main__':
    pass

