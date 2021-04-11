#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: November 5, 2020
#
"""It is utilities for Bayesian optimization."""

import numpy as np
try:
    from scipydirect import minimize as directminimize
except: # pragma: no cover
    directminimize = None
try:
    import cma
except: # pragma: no cover
    cma = None

from bayeso import acquisition
from bayeso import constants
from bayeso.utils import utils_covariance
from bayeso.utils import utils_common
from bayeso.utils import utils_logger

logger = utils_logger.get_logger('utils_bo')


@utils_common.validate_types
def get_best_acquisition_by_evaluation(initials: np.ndarray,
    fun_objective: constants.TYPING_CALLABLE
) -> np.ndarray:
    """
    It returns the best acquisition with respect to values of `fun_objective`.
    Here, the best acquisition is a minimizer of `fun_objective`.

    :param initials: inputs. Shape: (n, d).
    :type initials: numpy.ndarray
    :param fun_objective: an objective function.
    :type fun_objective: callable

    :returns: the best example of `initials`. Shape: (1, d).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(initials, np.ndarray)
    assert callable(fun_objective)
    assert len(initials.shape) == 2

    acq_val_best = np.inf
    initial_best = None
    for initial in initials:
        acq_val = fun_objective(initial)
        if acq_val < acq_val_best:
            initial_best = initial
            acq_val_best = acq_val
    initial_best = initial_best[np.newaxis, ...]
    return initial_best

@utils_common.validate_types
def get_best_acquisition_by_history(X: np.ndarray, Y: np.ndarray
) -> constants.TYPING_TUPLE_ARRAY_FLOAT:
    """
    It returns the best acquisition that has shown minimum result, and its minimum result.

    :param X: historical queries. Shape: (n, d).
    :type X: numpy.ndarray
    :param Y: the observations of `X`. Shape: (n, 1).
    :type Y: numpy.ndarray

    :returns: a tuple of the best query and its result.
    :rtype: (numpy.ndarray, float)

    :raises: AssertionError

    """

    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert len(X.shape) == 2
    assert len(Y.shape) == 2
    assert X.shape[0] == Y.shape[0]
    assert Y.shape[1] == 1

    ind_best = np.argmin(Y)
    bx_best = X[ind_best]
    y_best = Y[ind_best, 0]

    return bx_best, y_best

@utils_common.validate_types
def get_next_best_acquisition(points: np.ndarray, acquisitions: np.ndarray,
    points_evaluated: np.ndarray
) -> np.ndarray:
    """
    It returns the next best acquired example.

    :param points: inputs for acquisition function. Shape: (n, d).
    :type points: numpy.ndarray
    :param acquisitions: acquisition function values over `points`. Shape: (n, ).
    :type acquisitions: numpy.ndarray
    :param points_evaluated: examples evaluated so far. Shape: (m, d).
    :type points_evaluated: numpy.ndarray

    :returns: next best acquired point. Shape: (d, ).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(points_evaluated, np.ndarray)
    assert len(points.shape) == 2
    assert len(acquisitions.shape) == 1
    assert len(points_evaluated.shape) == 2
    assert points.shape[0] == acquisitions.shape[0]
    assert points.shape[1] == points_evaluated.shape[1]

    for cur_point in points_evaluated:
        ind_same, = np.where(np.linalg.norm(points - cur_point, axis=1) < 1e-2)
        points = np.delete(points, ind_same, axis=0)
        acquisitions = np.delete(acquisitions, ind_same)
    cur_best = np.inf
    next_point = None

    if points.shape[0] > 0:
        for arr_point, cur_acq in zip(points, acquisitions):
            if cur_acq < cur_best:
                cur_best = cur_acq
                next_point = arr_point
    else:
        next_point = points_evaluated[-1]
    return next_point

@utils_common.validate_types
def check_optimizer_method_bo(str_optimizer_method_bo: str, dim: int, debug: bool) -> str:
    """
    It checks the availability of optimization methods.
    It helps to run Bayesian optimization, even though additional
    optimization methods are not installed or there exist the conditions
    some of optimization methods cannot be run.

    :param str_optimizer_method_bo: the name of optimization method for
        Bayesian optimization.
    :type str_optimizer_method_bo: str.
    :param dim: dimensionality of the problem we solve.
    :type dim: int.
    :param debug: flag for printing log messages.
    :type debug: bool.

    :returns: available `str_optimizer_method_bo`.
    :rtype: str.

    :raises: AssertionError

    """

    assert isinstance(str_optimizer_method_bo, str)
    assert isinstance(dim, int)
    assert isinstance(debug, bool)
    assert str_optimizer_method_bo in constants.ALLOWED_OPTIMIZER_METHOD_BO

    if str_optimizer_method_bo == 'DIRECT' and directminimize is None: # pragma: no cover
        logger.warning('DIRECT is selected, but it is not installed.')
        str_optimizer_method_bo = 'L-BFGS-B'
    elif str_optimizer_method_bo == 'CMA-ES' and cma is None: # pragma: no cover
        logger.warning('CMA-ES is selected, but it is not installed.')
        str_optimizer_method_bo = 'L-BFGS-B'
    # TODO: It should be checked.
    elif str_optimizer_method_bo == 'CMA-ES' and dim == 1: # pragma: no cover
        logger.warning('CMA-ES is selected, but a dimension of bounds is 1.')
        str_optimizer_method_bo = 'L-BFGS-B'
    return str_optimizer_method_bo

@utils_common.validate_types
def choose_fun_acquisition(str_acq: str, hyps: dict) -> constants.TYPING_CALLABLE:
    """
    It chooses and returns an acquisition function.

    :param str_acq: the name of acquisition function.
    :type str_acq: str.
    :param hyps: dictionary of hyperparameters for acquisition function.
    :type hyps: dict.

    :returns: acquisition function.
    :rtype: callable

    :raises: AssertionError

    """

    assert isinstance(str_acq, str)
    assert isinstance(hyps, dict)
    assert str_acq in constants.ALLOWED_BO_ACQ

    if str_acq == 'pi':
        fun_acquisition = acquisition.pi
    elif str_acq == 'ei':
        fun_acquisition = acquisition.ei
    elif str_acq == 'ucb':
        fun_acquisition = acquisition.ucb
    elif str_acq == 'aei':
        fun_acquisition = lambda pred_mean, pred_std, Y_train: acquisition.aei(
            pred_mean, pred_std, Y_train, hyps['noise'])
    elif str_acq == 'pure_exploit':
        fun_acquisition = lambda pred_mean, pred_std, Y_train: acquisition.pure_exploit(
            pred_mean)
    elif str_acq == 'pure_explore':
        fun_acquisition = lambda pred_mean, pred_std, Y_train: acquisition.pure_explore(pred_std)
    else:
        raise NotImplementedError('_choose_fun_acquisition: allowed str_acq,\
            but it is not implemented.')
    return fun_acquisition

@utils_common.validate_types
def check_hyps_convergence(list_hyps: constants.TYPING_LIST[dict], hyps: dict,
    str_cov: str, fix_noise: bool,
    ratio_threshold: float=0.05
) -> bool:
    """
    It checks convergence of hyperparameters for Gaussian process regression.

    :param list_hyps: list of historical hyperparameters for Gaussian process regression.
    :type list_hyps: list
    :param hyps: dictionary of hyperparameters for acquisition function.
    :type hyps: dict.
    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param fix_noise: flag for fixing a noise.
    :type fix_noise: bool.
    :param ratio_threshold: ratio of threshold for checking convergence.
    :type ratio_threshold: float, optional

    :returns: flag for checking convergence. If converged, it is True.
    :rtype: bool.

    :raises: AssertionError

    """

    assert isinstance(list_hyps, list)
    assert isinstance(hyps, dict)
    assert isinstance(str_cov, str)
    assert isinstance(fix_noise, bool)
    assert isinstance(ratio_threshold, float)

    converged = False
    if len(list_hyps) > 0:
        hyps_converted = utils_covariance.convert_hyps(str_cov, hyps, fix_noise=fix_noise)
        target_hyps_converted = utils_covariance.convert_hyps(str_cov, list_hyps[-1],
            fix_noise=fix_noise)

        threshold = np.linalg.norm(target_hyps_converted) * ratio_threshold
        if np.linalg.norm(hyps_converted - target_hyps_converted, ord=2) < threshold:
            converged = True
    return converged
