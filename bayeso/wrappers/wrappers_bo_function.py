#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: September 24, 2020
#
"""It defines wrappers for Bayesian optimization."""

import time
import numpy as np

from bayeso import bo
from bayeso import constants
from bayeso.utils import utils_bo
from bayeso.utils import utils_common
from bayeso.utils import utils_logger

logger = utils_logger.get_logger('wrappers_bo')


@utils_common.validate_types
def run_single_round_with_all_initial_information(model_bo: bo.BO,
    fun_target: constants.TYPING_CALLABLE,
    X_train: np.ndarray, Y_train: np.ndarray,
    num_iter: int,
    str_sampling_method_ao: str=constants.STR_SAMPLING_METHOD_AO,
    num_samples_ao: int=constants.NUM_SAMPLES_AO,
    str_mlm_method: str=constants.STR_MLM_METHOD
) -> constants.TYPING_TUPLE_FIVE_ARRAYS:
    """
    It optimizes `fun_target` for `num_iter` iterations with given `model_bo`.
    It returns the optimization results and execution times.

    :param model_bo: Bayesian optimization model.
    :type model_bo: bayeso.bo.BO
    :param fun_target: a target function.
    :type fun_target: callable
    :param X_train: initial inputs. Shape: (n, d) or (n, m, d).
    :type X_train: numpy.ndarray
    :param Y_train: initial outputs. Shape: (n, 1).
    :type Y_train: numpy.ndarray
    :param num_iter: the number of iterations for Bayesian optimization.
    :type num_iter: int.
    :param str_sampling_method_ao: the name of initialization method for
        acquisition function optimization.
    :type str_sampling_method_ao: str., optional
    :param num_samples_ao: the number of samples for acquisition function
        optimization. If L-BFGS-B is used as an acquisition function
        optimization method, it is employed.
    :type num_samples_ao: int., optional
    :param str_mlm_method: the name of marginal likelihood maximization
        method for Gaussian process regression.
    :type str_mlm_method: str., optional

    :returns: tuple of acquired examples, their function values, overall
        execution times per iteration, execution time consumed in Gaussian
        process regression, and execution time consumed in acquisition
        function optimization. Shape: ((n + `num_iter`, d), (n + `num_iter`, 1),
        (`num_iter`, ), (`num_iter`, ), (`num_iter`, )), or ((n + `num_iter`, m, d),
        (n + `num_iter`, m, 1), (`num_iter`, ), (`num_iter`, ), (`num_iter`, )).
    :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)

    :raises: AssertionError

    """

    assert isinstance(model_bo, bo.BO)
    assert callable(fun_target)
    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(num_iter, int)
    assert isinstance(str_sampling_method_ao, str)
    assert isinstance(num_samples_ao, int)
    assert isinstance(str_mlm_method, str)
    assert len(X_train.shape) == 2
    assert len(Y_train.shape) == 2
    assert X_train.shape[0] == Y_train.shape[0]
    assert Y_train.shape[1] == 1
    assert str_mlm_method in constants.ALLOWED_MLM_METHOD

    time_start = time.time()

    X_final = X_train
    Y_final = Y_train
    time_all_final = []
    time_gp_final = []
    time_acq_final = []
    for ind_iter in range(0, num_iter):
        logger.info('Iteration %d', ind_iter + 1)
        time_iter_start = time.time()

        next_point, dict_info = model_bo.optimize(X_final, Y_final,
            str_sampling_method=str_sampling_method_ao,
            num_samples=num_samples_ao, str_mlm_method=str_mlm_method)
        next_points = dict_info['next_points']
        acquisitions = dict_info['acquisitions']
        time_gp = dict_info['time_gp']
        time_acq = dict_info['time_acq']

        if model_bo.debug:
            logger.debug('next_point: %s', utils_logger.get_str_array(next_point))

        if np.where(np.linalg.norm(next_point - X_final, axis=1)\
            < constants.TOLERANCE_DUPLICATED_ACQ)[0].shape[0] > 0: # pragma: no cover
            next_point = utils_bo.get_next_best_acquisition(next_points, acquisitions, X_final)
            if model_bo.debug:
                logger.debug('next_point is repeated, so next best is selected.\
                    next_point: %s', utils_logger.get_str_array(next_point))
        X_final = np.vstack((X_final, next_point))

        time_to_evaluate_start = time.time()
        Y_final = np.vstack((Y_final, fun_target(next_point)))
        time_to_evaluate_end = time.time()
        if model_bo.debug:
            logger.debug('time consumed to evaluate: %.4f sec.',
                time_to_evaluate_end - time_to_evaluate_start)

        time_iter_end = time.time()
        time_all_final.append(time_iter_end - time_iter_start)
        time_gp_final.append(time_gp)
        time_acq_final.append(time_acq)

    time_end = time.time()

    if model_bo.debug:
        logger.debug('overall time consumed in single BO round: %.4f sec.', time_end - time_start)

    time_all_final = np.array(time_all_final)
    time_gp_final = np.array(time_gp_final)
    time_acq_final = np.array(time_acq_final)
    return X_final, Y_final, time_all_final, time_gp_final, time_acq_final

@utils_common.validate_types
def run_single_round_with_initial_inputs(model_bo: bo.BO,
    fun_target: constants.TYPING_CALLABLE,
    X_train: np.ndarray, num_iter: int,
    str_sampling_method_ao: str=constants.STR_SAMPLING_METHOD_AO,
    num_samples_ao: int=constants.NUM_SAMPLES_AO,
    str_mlm_method: str=constants.STR_MLM_METHOD,
) -> constants.TYPING_TUPLE_FIVE_ARRAYS:
    """
    It optimizes `fun_target` for `num_iter` iterations with given
    `model_bo` and initial inputs `X_train`.
    It returns the optimization results and execution times.

    :param model_bo: Bayesian optimization model.
    :type model_bo: bayeso.bo.BO
    :param fun_target: a target function.
    :type fun_target: callable
    :param X_train: initial inputs. Shape: (n, d) or (n, m, d).
    :type X_train: numpy.ndarray
    :param num_iter: the number of iterations for Bayesian optimization.
    :type num_iter: int.
    :param str_sampling_method_ao: the name of initialization method for
        acquisition function optimization.
    :type str_sampling_method_ao: str., optional
    :param num_samples_ao: the number of samples for acquisition function
        optimization. If L-BFGS-B is used as an acquisition function
        optimization method, it is employed.
    :type num_samples_ao: int., optional
    :param str_mlm_method: the name of marginal likelihood maximization
        method for Gaussian process regression.
    :type str_mlm_method: str., optional

    :returns: tuple of acquired examples, their function values, overall
        execution times per iteration, execution time consumed in Gaussian
        process regression, and execution time consumed in acquisition
        function optimization. Shape: ((n + `num_iter`, d), (n + `num_iter`, 1),
        (n + `num_iter`, ), (`num_iter`, ), (`num_iter`, )), or ((n + `num_iter`, m, d),
        (n + `num_iter`, m, 1), (n + `num_iter`, ), (`num_iter`, ), (`num_iter`, )).
    :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)

    :raises: AssertionError

    """

    assert isinstance(model_bo, bo.BO)
    assert callable(fun_target)
    assert isinstance(X_train, np.ndarray)
    assert isinstance(num_iter, int)
    assert isinstance(str_sampling_method_ao, str)
    assert isinstance(num_samples_ao, int)
    assert isinstance(str_mlm_method, str)
    assert len(X_train.shape) == 2
    assert str_mlm_method in constants.ALLOWED_MLM_METHOD

    Y_train = []
    time_initials = []
    for elem in X_train:
        time_initial_start = time.time()
        Y_train.append(fun_target(elem))
        time_initial_end = time.time()
        time_initials.append(time_initial_end - time_initial_start)
    time_initials = np.array(time_initials)

    Y_train = np.array(Y_train)
    Y_train = np.reshape(Y_train, (Y_train.shape[0], 1))
    X_final, Y_final, time_all_final, time_gp_final, time_acq_final \
        = run_single_round_with_all_initial_information(
            model_bo,
            fun_target,
            X_train,
            Y_train,
            num_iter,
            str_sampling_method_ao=str_sampling_method_ao,
            num_samples_ao=num_samples_ao,
            str_mlm_method=str_mlm_method
        )
    return X_final, Y_final, \
        np.concatenate((time_initials, time_all_final)), \
        time_gp_final, time_acq_final

@utils_common.validate_types
def run_single_round(model_bo: bo.BO, fun_target: constants.TYPING_CALLABLE,
    num_init: int, num_iter: int,
    str_initial_method_bo: str=constants.STR_INITIALIZING_METHOD_BO,
    str_sampling_method_ao: str=constants.STR_SAMPLING_METHOD_AO,
    num_samples_ao: int=constants.NUM_SAMPLES_AO,
    str_mlm_method: str=constants.STR_MLM_METHOD,
    seed: constants.TYPING_UNION_INT_NONE=None
) -> constants.TYPING_TUPLE_FIVE_ARRAYS:
    """
    It optimizes `fun_target` for `num_iter` iterations with given
    `model_bo` and `num_init` initial examples.
    Initial examples are sampled by `get_initials` method in `model_bo`.
    It returns the optimization results and execution times.

    :param model_bo: Bayesian optimization model.
    :type model_bo: bayeso.bo.BO
    :param fun_target: a target function.
    :type fun_target: callable
    :param num_init: the number of initial examples for Bayesian optimization.
    :type num_init: int.
    :param num_iter: the number of iterations for Bayesian optimization.
    :type num_iter: int.
    :param str_initial_method_bo: the name of initialization method for
        sampling initial examples in Bayesian optimization.
    :type str_initial_method_bo: str., optional
    :param str_sampling_method_ao: the name of initialization method for
        acquisition function optimization.
    :type str_sampling_method_ao: str., optional
    :param num_samples_ao: the number of samples for acquisition function
        optimization. If L-BFGS-B is used as an acquisition function
        optimization method, it is employed.
    :type num_samples_ao: int., optional
    :param str_mlm_method: the name of marginal likelihood maximization
        method for Gaussian process regression.
    :type str_mlm_method: str., optional
    :param seed: None, or random seed.
    :type seed: NoneType or int., optional

    :returns: tuple of acquired examples, their function values, overall
        execution times per iteration, execution time consumed in Gaussian
        process regression, and execution time consumed in acquisition
        function optimization. Shape: ((`num_init` + `num_iter`, d),
        (`num_init` + `num_iter`, 1), (`num_init` + `num_iter`, ), (`num_iter`, ),
        (`num_iter`, )), or ((`num_init` + `num_iter`, m, d), (`num_init` + `num_iter`, m, 1),
        (`num_init` + `num_iter`, ), (`num_iter`, ), (`num_iter`, )),
        where d is a dimensionality of the problem we are solving and m is
        a cardinality of sets.
    :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)

    :raises: AssertionError

    """

    assert isinstance(model_bo, bo.BO)
    assert callable(fun_target)
    assert isinstance(num_init, int)
    assert isinstance(num_iter, int)
    assert isinstance(str_initial_method_bo, str)
    assert isinstance(str_sampling_method_ao, str)
    assert isinstance(num_samples_ao, int)
    assert isinstance(str_mlm_method, str)
    assert isinstance(seed, (int, type(None)))
    assert str_initial_method_bo in constants.ALLOWED_INITIALIZING_METHOD_BO
    assert str_mlm_method in constants.ALLOWED_MLM_METHOD

    logger.info('range_X:\n%s', utils_logger.get_str_array(model_bo.range_X))
    logger.info('str_cov: %s', model_bo.str_cov)
    logger.info('str_acq: %s', model_bo.str_acq)
    logger.info('str_optimizer_method_gp: %s', model_bo.str_optimizer_method_gp)
    logger.info('str_optimizer_method_bo: %s', model_bo.str_optimizer_method_bo)
    logger.info('str_modelselection_method: %s', model_bo.str_modelselection_method)
    logger.info('num_init: %d', num_init)
    logger.info('num_iter: %d', num_iter)
    logger.info('str_initial_method_bo: %s', str_initial_method_bo)
    logger.info('str_sampling_method_ao: %s', str_sampling_method_ao)
    logger.info('num_samples_ao: %d', num_samples_ao)
    logger.info('str_mlm_method: %s', str_mlm_method)
    logger.info('seed: %s', seed)

    time_start = time.time()

    X_init = model_bo.get_initials(str_initial_method_bo, num_init, seed=seed)
    if model_bo.debug:
        logger.debug('X_init:\n%s', utils_logger.get_str_array(X_init))

    X_final, Y_final, time_all_final, time_gp_final, time_acq_final \
        = run_single_round_with_initial_inputs(
            model_bo, fun_target, X_init, num_iter,
            str_sampling_method_ao=str_sampling_method_ao,
            num_samples_ao=num_samples_ao,
            str_mlm_method=str_mlm_method
        )

    time_end = time.time()

    if model_bo.debug:
        logger.debug('overall time consumed including initializations: %.4f sec.',
            time_end - time_start)

    return X_final, Y_final, time_all_final, time_gp_final, time_acq_final
