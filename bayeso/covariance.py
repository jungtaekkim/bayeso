#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: December 29, 2020
#
"""It defines covariance functions and their associated functions."""

import numpy as np
import scipy.spatial.distance as scisd
import scipy.linalg

from bayeso import constants
from bayeso.utils import utils_covariance
from bayeso.utils import utils_common


@utils_common.validate_types
def choose_fun_cov(str_cov: str) -> constants.TYPING_CALLABLE:
    """
    It chooses a covariance function.

    :param str_cov: the name of covariance function.
    :type str_cov: str.

    :returns: covariance function.
    :rtype: callable

    :raises: AssertionError

    """

    assert isinstance(str_cov, str)

    if str_cov in ('eq', 'se'):
        fun_cov = cov_se
    elif str_cov == 'matern32':
        fun_cov = cov_matern32
    elif str_cov == 'matern52':
        fun_cov = cov_matern52
    else:
        raise NotImplementedError('choose_fun_cov: allowed str_cov condition,\
            but it is not implemented.')
    return fun_cov

@utils_common.validate_types
def choose_fun_grad_cov(str_cov: str) -> constants.TYPING_CALLABLE:
    """
    It chooses a function for computing gradients of covariance function.

    :param str_cov: the name of covariance function.
    :type str_cov: str.

    :returns: function for computing gradients of covariance function.
    :rtype: callable

    :raises: AssertionError

    """

    assert isinstance(str_cov, str)

    if str_cov in ('eq', 'se'):
        fun_grad_cov = grad_cov_se
    elif str_cov == 'matern32':
        fun_grad_cov = grad_cov_matern32
    elif str_cov == 'matern52':
        fun_grad_cov = grad_cov_matern52
    else:
        raise NotImplementedError('choose_fun_grad_cov: allowed str_cov condition,\
            but it is not implemented.')
    return fun_grad_cov

@utils_common.validate_types
def get_kernel_inverse(X_train: np.ndarray, hyps: dict, str_cov: str,
    fix_noise: bool=constants.FIX_GP_NOISE,
    use_gradient: bool=False,
    debug: bool=False
) -> constants.TYPING_TUPLE_THREE_ARRAYS:
    """
    This function computes a kernel inverse without any matrix decomposition techniques.

    :param X_train: inputs. Shape: (n, d) or (n, m, d).
    :type X_train: numpy.ndarray
    :param hyps: dictionary of hyperparameters for Gaussian process.
    :type hyps: dict.
    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param fix_noise: flag for fixing a noise.
    :type fix_noise: bool., optional
    :param use_gradient: flag for computing and returning gradients of
        negative log marginal likelihood.
    :type use_gradient: bool., optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    :returns: a tuple of kernel matrix over `X_train`, kernel matrix
        inverse, and gradients of kernel matrix. If `use_gradient` is False,
        gradients of kernel matrix would be None.
    :rtype: tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)

    :raises: AssertionError

    """

    assert isinstance(X_train, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(str_cov, str)
    assert isinstance(use_gradient, bool)
    assert isinstance(fix_noise, bool)
    assert isinstance(debug, bool)
    utils_covariance.check_str_cov('get_kernel_inverse', str_cov, X_train.shape)

    cov_X_X = cov_main(str_cov, X_train, X_train, hyps, True) \
        + hyps['noise']**2 * np.eye(X_train.shape[0])
    cov_X_X = (cov_X_X + cov_X_X.T) / 2.0
    inv_cov_X_X = np.linalg.inv(cov_X_X)

    if use_gradient:
        grad_cov_X_X = grad_cov_main(str_cov, X_train, X_train,
            hyps, fix_noise, same_X_Xp=True)
    else:
        grad_cov_X_X = None

    return cov_X_X, inv_cov_X_X, grad_cov_X_X

@utils_common.validate_types
def get_kernel_cholesky(X_train: np.ndarray, hyps: dict, str_cov: str,
    fix_noise: bool=constants.FIX_GP_NOISE,
    use_gradient: bool=False,
    debug: bool=False
) -> constants.TYPING_TUPLE_THREE_ARRAYS:
    """
    This function computes a kernel inverse with Cholesky decomposition.

    :param X_train: inputs. Shape: (n, d) or (n, m, d).
    :type X_train: numpy.ndarray
    :param hyps: dictionary of hyperparameters for Gaussian process.
    :type hyps: dict.
    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param fix_noise: flag for fixing a noise.
    :type fix_noise: bool., optional
    :param use_gradient: flag for computing and returning gradients of
        negative log marginal likelihood.
    :type use_gradient: bool., optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    :returns: a tuple of kernel matrix over `X_train`, lower matrix computed
        by Cholesky decomposition, and gradients of kernel matrix. If
        `use_gradient` is False, gradients of kernel matrix would be None.
    :rtype: tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)

    :raises: AssertionError

    """

    assert isinstance(X_train, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(str_cov, str)
    assert isinstance(fix_noise, bool)
    assert isinstance(use_gradient, bool)
    assert isinstance(debug, bool)
    utils_covariance.check_str_cov('get_kernel_cholesky', str_cov, X_train.shape)

    cov_X_X = cov_main(str_cov, X_train, X_train, hyps, True) \
        + hyps['noise']**2 * np.eye(X_train.shape[0])
    cov_X_X = (cov_X_X + cov_X_X.T) / 2.0
    try:
        lower = scipy.linalg.cholesky(cov_X_X, lower=True)
    except np.linalg.LinAlgError: # pragma: no cover
        cov_X_X += 1e-2 * np.eye(X_train.shape[0])
        lower = scipy.linalg.cholesky(cov_X_X, lower=True)

    if use_gradient:
        grad_cov_X_X = grad_cov_main(str_cov, X_train, X_train,
            hyps, fix_noise, same_X_Xp=True)
    else:
        grad_cov_X_X = None
    return cov_X_X, lower, grad_cov_X_X

@utils_common.validate_types
def cov_se(X: np.ndarray, Xp: np.ndarray, lengthscales: constants.TYPING_UNION_ARRAY_FLOAT,
    signal: float
) -> np.ndarray:
    """
    It computes squared exponential kernel over `X` and `Xp`, where
    `lengthscales` and `signal` are given.

    :param X: inputs. Shape: (n, d).
    :type X: numpy.ndarray
    :param Xp: another inputs. Shape: (m, d).
    :type Xp: numpy.ndarray
    :param lengthscales: length scales. Shape: (d, ) or ().
    :type lengthscales: numpy.ndarray, or float
    :param signal: coefficient for signal.
    :type signal: float

    :returns: kernel values over `X` and `Xp`. Shape: (n, m).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(X, np.ndarray)
    assert isinstance(Xp, np.ndarray)
    assert isinstance(lengthscales, (np.ndarray, float))
    assert isinstance(signal, float)
    assert len(X.shape) == 2
    assert len(Xp.shape) == 2
    if isinstance(lengthscales, np.ndarray):
        assert X.shape[1] == Xp.shape[1] == lengthscales.shape[0]
    else:
        assert X.shape[1] == Xp.shape[1]
    dist = scisd.cdist(X / lengthscales, Xp / lengthscales, metric='euclidean')
    cov_X_Xp = signal**2 * np.exp(-0.5 * dist**2)
    return cov_X_Xp

@utils_common.validate_types
def grad_cov_se(cov_X_Xp: np.ndarray, X: np.ndarray, Xp: np.ndarray, hyps: dict,
    num_hyps: int, fix_noise: bool
) -> np.ndarray:
    """
    It computes gradients of squared exponential kernel over `X` and `Xp`,
    where `hyps` is given.

    :param cov_X_Xp: covariance matrix. Shape: (n, m).
    :type cov_X_Xp: numpy.ndarray
    :param X: one inputs. Shape: (n, d).
    :type X: numpy.ndarray
    :param Xp: another inputs. Shape: (m, d).
    :type Xp: numpy.ndarray
    :param hyps: dictionary of hyperparameters for covariance function.
    :type hyps: dict.
    :param num_hyps: the number of hyperparameters == l.
    :type num_hyps: int.
    :param fix_noise: flag for fixing a noise.
    :type fix_noise: bool.

    :returns: gradient matrix over hyperparameters. Shape: (n, m, l).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(cov_X_Xp, np.ndarray)
    assert isinstance(X, np.ndarray)
    assert isinstance(Xp, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(num_hyps, int)
    assert isinstance(fix_noise, bool)

    num_X = X.shape[0]
    num_Xp = Xp.shape[0]

    grad_cov_X_Xp = np.zeros((num_X, num_Xp, num_hyps))
    dist = scisd.cdist(X / hyps['lengthscales'], Xp / hyps['lengthscales'], metric='euclidean')

    if fix_noise:
        ind_next = 0
    else:
        grad_cov_X_Xp[:, :, 0] += 2.0 * hyps['noise'] * np.eye(num_X, M=num_Xp)
        ind_next = 1

    grad_cov_X_Xp[:, :, ind_next] += 2.0 * cov_X_Xp / hyps['signal']

    term_pre = cov_X_Xp * dist**2

    if isinstance(hyps['lengthscales'], np.ndarray) and len(hyps['lengthscales'].shape) == 1:
        for ind_ in range(0, hyps['lengthscales'].shape[0]):
            grad_cov_X_Xp[:, :, ind_next+ind_+1] += term_pre * hyps['lengthscales'][ind_]**(-1)
    else:
        grad_cov_X_Xp[:, :, ind_next+1] += term_pre * hyps['lengthscales']**(-1)

    return grad_cov_X_Xp

@utils_common.validate_types
def cov_matern32(X: np.ndarray, Xp: np.ndarray, lengthscales: constants.TYPING_UNION_ARRAY_FLOAT,
    signal: float
) -> np.ndarray:
    """
    It computes Matern 3/2 kernel over `X` and `Xp`, where `lengthscales` and `signal` are given.

    :param X: inputs. Shape: (n, d).
    :type X: numpy.ndarray
    :param Xp: another inputs. Shape: (m, d).
    :type Xp: numpy.ndarray
    :param lengthscales: length scales. Shape: (d, ) or ().
    :type lengthscales: numpy.ndarray, or float
    :param signal: coefficient for signal.
    :type signal: float

    :returns: kernel values over `X` and `Xp`. Shape: (n, m).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(X, np.ndarray)
    assert isinstance(Xp, np.ndarray)
    assert isinstance(lengthscales, (np.ndarray, float))
    assert len(X.shape) == 2
    assert len(Xp.shape) == 2
    if isinstance(lengthscales, np.ndarray):
        assert X.shape[1] == Xp.shape[1] == lengthscales.shape[0]
    else:
        assert X.shape[1] == Xp.shape[1]
    assert isinstance(signal, float)

    dist = scisd.cdist(X / lengthscales, Xp / lengthscales, metric='euclidean')
    cov_ = signal**2 * (1.0 + np.sqrt(3.0) * dist) * np.exp(-1.0 * np.sqrt(3.0) * dist)
    return cov_

@utils_common.validate_types
def grad_cov_matern32(cov_X_Xp: np.ndarray, X: np.ndarray, Xp: np.ndarray, hyps: dict,
    num_hyps: int, fix_noise: bool
) -> np.ndarray:
    """
    It computes gradients of Matern 3/2 kernel over `X` and `Xp`, where `hyps` is given.

    :param cov_X_Xp: covariance matrix. Shape: (n, m).
    :type cov_X_Xp: numpy.ndarray
    :param X: one inputs. Shape: (n, d).
    :type X: numpy.ndarray
    :param Xp: another inputs. Shape: (m, d).
    :type Xp: numpy.ndarray
    :param hyps: dictionary of hyperparameters for covariance function.
    :type hyps: dict.
    :param num_hyps: the number of hyperparameters == l.
    :type num_hyps: int.
    :param fix_noise: flag for fixing a noise.
    :type fix_noise: bool.

    :returns: gradient matrix over hyperparameters. Shape: (n, m, l).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(cov_X_Xp, np.ndarray)
    assert isinstance(X, np.ndarray)
    assert isinstance(Xp, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(num_hyps, int)
    assert isinstance(fix_noise, bool)

    num_X = X.shape[0]
    num_Xp = Xp.shape[0]

    grad_cov_X_Xp = np.zeros((num_X, num_Xp, num_hyps))
    dist = scisd.cdist(X / hyps['lengthscales'], Xp / hyps['lengthscales'], metric='euclidean')

    if fix_noise:
        ind_next = 0
    else:
        grad_cov_X_Xp[:, :, 0] += 2.0 * hyps['noise'] * np.eye(num_X, M=num_Xp)
        ind_next = 1

    grad_cov_X_Xp[:, :, ind_next] += 2.0 * cov_X_Xp / hyps['signal']

    term_pre = 3.0 * hyps['signal']**2 * np.exp(-np.sqrt(3) * dist) * dist**2

    if isinstance(hyps['lengthscales'], np.ndarray) and len(hyps['lengthscales'].shape) == 1:
        for ind_ in range(0, hyps['lengthscales'].shape[0]):
            grad_cov_X_Xp[:, :, ind_next+ind_+1] += term_pre * hyps['lengthscales'][ind_]**(-1)
    else:
        grad_cov_X_Xp[:, :, ind_next+1] += term_pre * hyps['lengthscales']**(-1)

    return grad_cov_X_Xp

@utils_common.validate_types
def cov_matern52(X: np.ndarray, Xp:np.ndarray, lengthscales: constants.TYPING_UNION_ARRAY_FLOAT,
    signal: float
) -> np.ndarray:
    """
    It computes Matern 5/2 kernel over `X` and `Xp`, where `lengthscales`
    and `signal` are given.

    :param X: inputs. Shape: (n, d).
    :type X: numpy.ndarray
    :param Xp: another inputs. Shape: (m, d).
    :type Xp: numpy.ndarray
    :param lengthscales: length scales. Shape: (d, ) or ().
    :type lengthscales: numpy.ndarray, or float
    :param signal: coefficient for signal.
    :type signal: float

    :returns: kernel values over `X` and `Xp`. Shape: (n, m).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(X, np.ndarray)
    assert isinstance(Xp, np.ndarray)
    assert isinstance(lengthscales, (np.ndarray, float))
    assert len(X.shape) == 2
    assert len(Xp.shape) == 2
    if isinstance(lengthscales, np.ndarray):
        assert X.shape[1] == Xp.shape[1] == lengthscales.shape[0]
    else:
        assert X.shape[1] == Xp.shape[1]
    assert isinstance(signal, float)

    dist = scisd.cdist(X / lengthscales, Xp / lengthscales, metric='euclidean')
    cov_X_Xp = signal**2 * (1.0 + np.sqrt(5.0) * dist + 5.0 / 3.0 * dist**2) \
        * np.exp(-1.0 * np.sqrt(5.0) * dist)
    return cov_X_Xp

@utils_common.validate_types
def grad_cov_matern52(cov_X_Xp: np.ndarray, X: np.ndarray, Xp: np.ndarray, hyps: dict,
    num_hyps: int, fix_noise: bool
) -> np.ndarray:
    """
    It computes gradients of Matern 5/2 kernel over `X` and `Xp`, where `hyps` is given.

    :param cov_X_Xp: covariance matrix. Shape: (n, m).
    :type cov_X_Xp: numpy.ndarray
    :param X: one inputs. Shape: (n, d).
    :type X: numpy.ndarray
    :param Xp: another inputs. Shape: (m, d).
    :type Xp: numpy.ndarray
    :param hyps: dictionary of hyperparameters for covariance function.
    :type hyps: dict.
    :param num_hyps: the number of hyperparameters == l.
    :type num_hyps: int.
    :param fix_noise: flag for fixing a noise.
    :type fix_noise: bool.

    :returns: gradient matrix over hyperparameters. Shape: (n, m, l).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(cov_X_Xp, np.ndarray)
    assert isinstance(X, np.ndarray)
    assert isinstance(Xp, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(num_hyps, int)
    assert isinstance(fix_noise, bool)

    num_X = X.shape[0]
    num_Xp = Xp.shape[0]

    grad_cov_X_Xp = np.zeros((num_X, num_Xp, num_hyps))
    dist = scisd.cdist(X / hyps['lengthscales'], Xp / hyps['lengthscales'], metric='euclidean')

    if fix_noise:
        ind_next = 0
    else:
        grad_cov_X_Xp[:, :, 0] += 2.0 * hyps['noise'] * np.eye(num_X, M=num_Xp)
        ind_next = 1

    grad_cov_X_Xp[:, :, ind_next] += 2.0 * cov_X_Xp / hyps['signal']

    term_pre = 5.0 / 3.0 * hyps['signal']**2 * (1.0 + np.sqrt(5) * dist) \
        * np.exp(-np.sqrt(5) * dist) * dist**3

    if isinstance(hyps['lengthscales'], np.ndarray) and len(hyps['lengthscales'].shape) == 1:
        for ind_ in range(0, hyps['lengthscales'].shape[0]):
            grad_cov_X_Xp[:, :, ind_next+ind_+1] += term_pre * hyps['lengthscales'][ind_]**(-1)
    else:
        grad_cov_X_Xp[:, :, ind_next+1] += term_pre * hyps['lengthscales']**(-1)

    return grad_cov_X_Xp

@utils_common.validate_types
def cov_set(str_cov: str, X: np.ndarray, Xp: np.ndarray,
    lengthscales: constants.TYPING_UNION_ARRAY_FLOAT, signal: float
) -> np.ndarray:
    """
    It computes set kernel matrix over `X` and `Xp`, where `lengthscales` and `signal` are given.

    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param X: one inputs. Shape: (n, m, d).
    :type X: numpy.ndarray
    :param Xp: another inputs. Shape: (l, m, d).
    :type Xp: numpy.ndarray
    :param lengthscales: length scales. Shape: (d, ) or ().
    :type lengthscales: numpy.ndarray, or float
    :param signal: coefficient for signal.
    :type signal: float

    :returns: set kernel matrix over `X` and `Xp`. Shape: (n, l).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(str_cov, str)
    assert isinstance(X, np.ndarray)
    assert isinstance(Xp, np.ndarray)
    assert isinstance(lengthscales, (np.ndarray, float))
    assert isinstance(signal, float)
    assert len(X.shape) == 2
    assert len(Xp.shape) == 2
    if isinstance(lengthscales, np.ndarray):
        assert X.shape[1] == Xp.shape[1] == lengthscales.shape[0]
    else:
        assert X.shape[1] == Xp.shape[1]
    assert str_cov in constants.ALLOWED_COV_BASE
    num_X = X.shape[0]
    num_Xp = Xp.shape[0]

    fun_cov = choose_fun_cov(str_cov)
    cov_X_Xp = fun_cov(X, Xp, lengthscales, signal)
    cov_X_Xp = np.sum(cov_X_Xp)

    cov_X_Xp /= num_X * num_Xp

    return cov_X_Xp

@utils_common.validate_types
def cov_main(str_cov: str, X: np.ndarray, Xp: np.ndarray, hyps: dict, same_X_Xp: bool,
    jitter: float=constants.JITTER_COV
) -> np.ndarray:
    """
    It computes kernel matrix over `X` and `Xp`, where `hyps` is given.

    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param X: one inputs. Shape: (n, d).
    :type X: numpy.ndarray
    :param Xp: another inputs. Shape: (m, d).
    :type Xp: numpy.ndarray
    :param hyps: dictionary of hyperparameters for covariance function.
    :type hyps: dict.
    :param same_X_Xp: flag for checking `X` and `Xp` are same.
    :type same_X_Xp: bool.
    :param jitter: jitter for diagonal entries.
    :type jitter: float, optional

    :returns: kernel matrix over `X` and `Xp`. Shape: (n, m).
    :rtype: numpy.ndarray

    :raises: AssertionError, ValueError

    """

    assert isinstance(str_cov, str)
    assert isinstance(X, np.ndarray)
    assert isinstance(Xp, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(same_X_Xp, bool)
    assert isinstance(jitter, float)
    assert str_cov in constants.ALLOWED_COV

    num_X = X.shape[0]
    num_Xp = Xp.shape[0]

    cov_X_Xp = np.zeros((num_X, num_Xp))
    if same_X_Xp:
        assert num_X == num_Xp
        cov_X_Xp += np.eye(num_X) * jitter

    if str_cov in constants.ALLOWED_COV_BASE:
        assert len(X.shape) == 2
        assert len(Xp.shape) == 2
        dim_X = X.shape[1]
        dim_Xp = Xp.shape[1]
        assert dim_X == dim_Xp

        hyps = utils_covariance.validate_hyps_dict(hyps, str_cov, dim_X)

        fun_cov = choose_fun_cov(str_cov)
        cov_X_Xp += fun_cov(X, Xp, hyps['lengthscales'], hyps['signal'])

        assert cov_X_Xp.shape == (num_X, num_Xp)
    elif str_cov in constants.ALLOWED_COV_SET:
        list_str_cov = str_cov.split('_')
        str_cov = list_str_cov[1]

        assert len(X.shape) == 3
        assert len(Xp.shape) == 3

        dim_X = X.shape[2]
        dim_Xp = Xp.shape[2]

        assert dim_X == dim_Xp

        hyps = utils_covariance.validate_hyps_dict(hyps, str_cov, dim_X)

        if not same_X_Xp:
            for ind_X in range(0, num_X):
                for ind_Xp in range(0, num_Xp):
                    cov_X_Xp[ind_X, ind_Xp] += cov_set(str_cov, X[ind_X], Xp[ind_Xp],
                        hyps['lengthscales'], hyps['signal'])
        else:
            for ind_X in range(0, num_X):
                for ind_Xp in range(ind_X, num_Xp):
                    cov_X_Xp[ind_X, ind_Xp] += cov_set(str_cov, X[ind_X], Xp[ind_Xp],
                        hyps['lengthscales'], hyps['signal'])
                    if ind_X < ind_Xp:
                        cov_X_Xp[ind_Xp, ind_X] = cov_X_Xp[ind_X, ind_Xp]
    else:
        raise NotImplementedError('cov_main: allowed str_cov, but it is not implemented.')

    return cov_X_Xp

@utils_common.validate_types
def grad_cov_main(str_cov: str, X: np.ndarray, Xp: np.ndarray, hyps: dict, fix_noise: bool,
    same_X_Xp: bool=True,
    jitter: float=constants.JITTER_COV,
) -> np.ndarray:
    """
    It computes gradients of kernel matrix over hyperparameters, where `hyps` is given.

    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param X: one inputs. Shape: (n, d).
    :type X: numpy.ndarray
    :param Xp: another inputs. Shape: (m, d).
    :type Xp: numpy.ndarray
    :param hyps: dictionary of hyperparameters for covariance function.
    :type hyps: dict.
    :param fix_noise: flag for fixing a noise.
    :type fix_noise: bool.
    :param same_X_Xp: flag for checking `X` and `Xp` are same.
    :type same_X_Xp: bool., optional
    :param jitter: jitter for diagonal entries.
    :type jitter: float, optional

    :returns: gradient matrix over hyperparameters. Shape: (n, m, l) where
        l is the number of hyperparameters.
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(str_cov, str)
    assert isinstance(X, np.ndarray)
    assert isinstance(Xp, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(fix_noise, bool)
    assert isinstance(same_X_Xp, bool)
    assert isinstance(jitter, float)
    assert str_cov in constants.ALLOWED_COV
    # TODO: X and Xp should be same?
    assert same_X_Xp

    dim_X = X.shape[1]

    if isinstance(hyps['lengthscales'], np.ndarray):
        num_hyps = dim_X + 1
    else:
        num_hyps = 2

    if not fix_noise:
        num_hyps += 1

    cov_X_Xp = cov_main(str_cov, X, Xp, hyps, same_X_Xp, jitter=jitter)

    fun_grad_cov = choose_fun_grad_cov(str_cov)
    grad_cov_X_Xp = fun_grad_cov(cov_X_Xp, X, Xp, hyps, num_hyps, fix_noise)

    return grad_cov_X_Xp
