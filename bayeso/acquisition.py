#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: September 24, 2020
#
"""It defines acquisition functions, each of which
is employed to determine where next to evaluate."""

import numpy as np
import scipy.stats

from bayeso import constants
from bayeso.utils import utils_common


@utils_common.validate_types
def pi(pred_mean: np.ndarray, pred_std: np.ndarray, Y_train: np.ndarray,
    jitter: float=constants.JITTER_ACQ
) -> np.ndarray:
    """
    It is a probability of improvement criterion.

    :param pred_mean: posterior predictive mean function over `X_test`.
        Shape: (l, ).
    :type pred_mean: numpy.ndarray
    :param pred_std: posterior predictive standard deviation function over
        `X_test`. Shape: (l, ).
    :type pred_std: numpy.ndarray
    :param Y_train: outputs of `X_train`. Shape: (n, 1).
    :type Y_train: numpy.ndarray
    :param jitter: jitter for `pred_std`.
    :type jitter: float, optional

    :returns: acquisition function values. Shape: (l, ).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(pred_mean, np.ndarray)
    assert isinstance(pred_std, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(jitter, float)
    assert len(pred_mean.shape) == 1
    assert len(pred_std.shape) == 1
    assert len(Y_train.shape) == 2
    assert pred_mean.shape[0] == pred_std.shape[0]

    with np.errstate(divide='ignore'):
        val_z = (np.min(Y_train) - pred_mean) / (pred_std + jitter)
    return scipy.stats.norm.cdf(val_z)

@utils_common.validate_types
def ei(pred_mean: np.ndarray, pred_std: np.ndarray, Y_train: np.ndarray,
    jitter: float=constants.JITTER_ACQ
) -> np.ndarray:
    """
    It is an expected improvement criterion.

    :param pred_mean: posterior predictive mean function over `X_test`. Shape: (l, ).
    :type pred_mean: numpy.ndarray
    :param pred_std: posterior predictive standard deviation function over `X_test`. Shape: (l, ).
    :type pred_std: numpy.ndarray
    :param Y_train: outputs of `X_train`. Shape: (n, 1).
    :type Y_train: numpy.ndarray
    :param jitter: jitter for `pred_std`.
    :type jitter: float, optional

    :returns: acquisition function values. Shape: (l, ).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(pred_mean, np.ndarray)
    assert isinstance(pred_std, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(jitter, float)
    assert len(pred_mean.shape) == 1
    assert len(pred_std.shape) == 1
    assert len(Y_train.shape) == 2
    assert pred_mean.shape[0] == pred_std.shape[0]

    with np.errstate(divide='ignore'):
        val_z = (np.min(Y_train) - pred_mean) / (pred_std + jitter)
    return (np.min(Y_train) - pred_mean) * scipy.stats.norm.cdf(val_z) \
        + pred_std * scipy.stats.norm.pdf(val_z)

@utils_common.validate_types
def ucb(pred_mean: np.ndarray, pred_std: np.ndarray,
    Y_train: constants.TYPING_UNION_ARRAY_NONE=None,
    kappa: float=2.0,
    increase_kappa: bool=True
) -> np.ndarray:
    """
    It is a Gaussian process upper confidence bound criterion.

    :param pred_mean: posterior predictive mean function over `X_test`.
        Shape: (l, ).
    :type pred_mean: numpy.ndarray
    :param pred_std: posterior predictive standard deviation function over
        `X_test`. Shape: (l, ).
    :type pred_std: numpy.ndarray
    :param Y_train: outputs of `X_train`. Shape: (n, 1).
    :type Y_train: numpy.ndarray, optional
    :param kappa: trade-off hyperparameter between exploration and
        exploitation.
    :type kappa: float, optional
    :param increase_kappa: flag for increasing a kappa value as `Y_train`
        grows. If `Y_train` is None, it is ignored, which means `kappa` is
        fixed.
    :type increase_kappa: bool., optional

    :returns: acquisition function values. Shape: (l, ).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(pred_mean, np.ndarray)
    assert isinstance(pred_std, np.ndarray)
    assert isinstance(Y_train, (np.ndarray, type(None)))
    assert isinstance(kappa, float)
    assert isinstance(increase_kappa, bool)
    assert len(pred_mean.shape) == 1
    assert len(pred_std.shape) == 1
    if Y_train is not None:
        assert len(Y_train.shape) == 2
    assert pred_mean.shape[0] == pred_std.shape[0]

    if increase_kappa and Y_train is not None:
        kappa_ = kappa * np.log(Y_train.shape[0])
    else:
        kappa_ = kappa
    return -pred_mean + kappa_ * pred_std

@utils_common.validate_types
def aei(pred_mean: np.ndarray, pred_std: np.ndarray, Y_train: np.ndarray, noise: float,
    jitter: float=constants.JITTER_ACQ
) -> np.ndarray:
    """
    It is an augmented expected improvement criterion.

    :param pred_mean: posterior predictive mean function over `X_test`. Shape: (l, ).
    :type pred_mean: numpy.ndarray
    :param pred_std: posterior predictive standard deviation function over `X_test`. Shape: (l, ).
    :type pred_std: numpy.ndarray
    :param Y_train: outputs of `X_train`. Shape: (n, 1).
    :type Y_train: numpy.ndarray
    :param noise: noise for augmenting exploration.
    :type noise: float
    :param jitter: jitter for `pred_std`.
    :type jitter: float, optional

    :returns: acquisition function values. Shape: (l, ).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(pred_mean, np.ndarray)
    assert isinstance(pred_std, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(noise, float)
    assert isinstance(jitter, float)
    assert len(pred_mean.shape) == 1
    assert len(pred_std.shape) == 1
    assert len(Y_train.shape) == 2
    assert pred_mean.shape[0] == pred_std.shape[0]

    with np.errstate(divide='ignore'):
        val_z = (np.min(Y_train) - pred_mean) / (pred_std + jitter)
    val_ei = (np.min(Y_train) - pred_mean) * scipy.stats.norm.cdf(val_z) \
        + pred_std * scipy.stats.norm.pdf(val_z)
    val_aei = val_ei * (1.0 - noise / np.sqrt(pred_std**2 + noise**2))
    return val_aei

@utils_common.validate_types
def pure_exploit(pred_mean: np.ndarray) -> np.ndarray:
    """
    It is a pure exploitation criterion.

    :param pred_mean: posterior predictive mean function over `X_test`. Shape: (l, ).
    :type pred_mean: numpy.ndarray

    :returns: acquisition function values. Shape: (l, ).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(pred_mean, np.ndarray)
    assert len(pred_mean.shape) == 1

    return -pred_mean

@utils_common.validate_types
def pure_explore(pred_std: np.ndarray) -> np.ndarray:
    """
    It is a pure exploration criterion.

    :param pred_std: posterior predictive standard deviation function over `X_test`. Shape: (l, ).
    :type pred_std: numpy.ndarray

    :returns: acquisition function values. Shape: (l, ).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(pred_std, np.ndarray)
    assert len(pred_std.shape) == 1

    return pred_std
