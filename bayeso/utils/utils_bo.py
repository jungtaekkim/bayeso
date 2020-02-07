# utils_bo
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 07, 2020

import numpy as np
import time

from bayeso import bo
from bayeso import constants


def get_next_best_acquisition(arr_points, arr_acquisitions, cur_points):
    """
    It returns the next best acquired example.

    :param arr_points: inputs for acquisition function. Shape: (n, d).
    :type arr_points: numpy.ndarray
    :param arr_acquisitions: acquisition function values over `arr_points`. Shape: (n, ).
    :type arr_acquisitions: numpy.ndarray
    :param cur_points: examples evaluated so far. Shape: (m, d).
    :type cur_points: numpy.ndarray

    :returns: next best acquired point. Shape: (d, ).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(arr_points, np.ndarray)
    assert isinstance(arr_acquisitions, np.ndarray)
    assert isinstance(cur_points, np.ndarray)
    assert len(arr_points.shape) == 2
    assert len(arr_acquisitions.shape) == 1
    assert len(cur_points.shape) == 2
    assert arr_points.shape[0] == arr_acquisitions.shape[0]
    assert arr_points.shape[1] == cur_points.shape[1]
   
    for cur_point in cur_points:
        ind_same, = np.where(np.linalg.norm(arr_points - cur_point, axis=1) < 1e-2)
        arr_points = np.delete(arr_points, ind_same, axis=0)
        arr_acquisitions = np.delete(arr_acquisitions, ind_same)
    cur_best = np.inf
    next_point = None

    if arr_points.shape[0] > 0:
        for arr_point, cur_acq in zip(arr_points, arr_acquisitions):
            if cur_acq < cur_best:
                cur_best = cur_acq
                next_point = arr_point
    else:
        next_point = cur_points[-1]
    return next_point

def optimize_many_(model_bo, fun_target, X_train, Y_train, int_iter,
    str_initial_method_ao=constants.STR_AO_INITIALIZATION,
    int_samples_ao=constants.NUM_ACQ_SAMPLES,
    str_mlm_method=constants.STR_MLM_METHOD
):
    """
    It optimizes `fun_target` for `int_iter` iterations with given `model_bo`.
    It returns the optimization results and execution times.

    :param model_bo: Bayesian optimization model.
    :type model_bo: bayeso.bo.BO
    :param fun_target: a target function.
    :type fun_target: function
    :param X_train: initial inputs. Shape: (n, d) or (n, m, d).
    :type X_train: numpy.ndarray
    :param Y_train: initial outputs. Shape: (n, 1).
    :type Y_train: numpy.ndarray
    :param int_iter: the number of iterations for Bayesian optimization.
    :type int_iter: int.
    :param str_initial_method_ao: the name of initialization method for acquisition function optimization.
    :type str_initial_method_ao: str., optional
    :param int_samples_ao: the number of samples for acquisition function optimization. If L-BFGS-B is used as an acquisition function optimization method, it is employed.
    :type int_samples_ao: int., optional
    :param str_mlm_method: the name of marginal likelihood maximization method for Gaussian process regression.
    :type str_mlm_method: str., optional

    :returns: tuple of acquired examples, their function values, overall execution times per iteration, execution time consumed in Gaussian process regression, and execution time consumed in acquisition function optimization. Shape: ((n + `int_iter`, d), (n + `int_iter`, 1), (`int_iter`, ), (`int_iter`, ), (`int_iter`, )), or ((n + `int_iter`, m, d), (n + `int_iter`, m, 1), (`int_iter`, ), (`int_iter`, ), (`int_iter`, )).
    :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)

    :raises: AssertionError

    """

    assert isinstance(model_bo, bo.BO)
    assert callable(fun_target)
    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(int_iter, int)
    assert isinstance(str_initial_method_ao, str)
    assert isinstance(int_samples_ao, int)
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
    for ind_iter in range(0, int_iter):
        print('Iteration {}'.format(ind_iter + 1))
        time_iter_start = time.time()

        next_point, next_points, acquisitions, _, _, _, times = model_bo.optimize(X_final, Y_final, str_initial_method_ao=str_initial_method_ao, int_samples=int_samples_ao, str_mlm_method=str_mlm_method)

        if model_bo.debug:
            print('[DEBUG] optimize_many_ in utils_bo.py: next_point', next_point)

        # TODO: check this code, which uses norm.
#        if np.where(np.sum(next_point == X_final, axis=1) == X_final.shape[1])[0].shape[0] > 0:
        if np.where(np.linalg.norm(next_point - X_final, axis=1) < 1e-3)[0].shape[0] > 0: # pragma: no cover
            next_point = get_next_best_acquisition(next_points, acquisitions, X_final)
            if model_bo.debug:
                print('[DEBUG] optimize_many_ in utils_bo.py: next_point is repeated, so next best is selected. next_point', next_point)
        X_final = np.vstack((X_final, next_point))

        time_to_evaluate_start = time.time()
        Y_final = np.vstack((Y_final, fun_target(next_point)))
        time_to_evaluate_end = time.time()
        if model_bo.debug:
            print('[DEBUG] optimize_many_ in utils_bo.py: time consumed to evaluate', time_to_evaluate_end - time_to_evaluate_start, 'sec.')

        time_iter_end = time.time()
        time_all_final.append(time_iter_end - time_iter_start)
        time_gp_final.append(times['gp'])
        time_acq_final.append(times['acq'])

    time_end = time.time()

    if model_bo.debug:
        print('[DEBUG] optimize_many_ in utils_bo.py: time consumed', time_end - time_start, 'sec.')

    time_all_final = np.array(time_all_final)
    time_gp_final = np.array(time_gp_final)
    time_acq_final = np.array(time_acq_final)
    return X_final, Y_final, time_all_final, time_gp_final, time_acq_final

def optimize_many(model_bo, fun_target, X_train, int_iter,
    str_initial_method_ao=constants.STR_AO_INITIALIZATION,
    int_samples_ao=constants.NUM_ACQ_SAMPLES,
    str_mlm_method=constants.STR_MLM_METHOD,
):
    """
    It optimizes `fun_target` for `int_iter` iterations with given `model_bo` and initial inputs `X_train`.
    It returns the optimization results and execution times.

    :param model_bo: Bayesian optimization model.
    :type model_bo: bayeso.bo.BO
    :param fun_target: a target function.
    :type fun_target: function
    :param X_train: initial inputs. Shape: (n, d) or (n, m, d).
    :type X_train: numpy.ndarray
    :param int_iter: the number of iterations for Bayesian optimization.
    :type int_iter: int.
    :param str_initial_method_ao: the name of initialization method for acquisition function optimization.
    :type str_initial_method_ao: str., optional
    :param int_samples_ao: the number of samples for acquisition function optimization. If L-BFGS-B is used as an acquisition function optimization method, it is employed.
    :type int_samples_ao: int., optional
    :param str_mlm_method: the name of marginal likelihood maximization method for Gaussian process regression.
    :type str_mlm_method: str., optional

    :returns: tuple of acquired examples, their function values, overall execution times per iteration, execution time consumed in Gaussian process regression, and execution time consumed in acquisition function optimization. Shape: ((n + `int_iter`, d), (n + `int_iter`, 1), (n + `int_iter`, ), (`int_iter`, ), (`int_iter`, )), or ((n + `int_iter`, m, d), (n + `int_iter`, m, 1), (n + `int_iter`, ), (`int_iter`, ), (`int_iter`, )).
    :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)

    :raises: AssertionError

    """

    assert isinstance(model_bo, bo.BO)
    assert callable(fun_target)
    assert isinstance(X_train, np.ndarray)
    assert isinstance(int_iter, int)
    assert isinstance(str_initial_method_ao, str)
    assert isinstance(int_samples_ao, int)
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
    X_final, Y_final, time_all_final, time_gp_final, time_acq_final = optimize_many_(
        model_bo,
        fun_target,
        X_train,
        Y_train,
        int_iter,
        str_initial_method_ao=str_initial_method_ao,
        int_samples_ao=int_samples_ao,
        str_mlm_method=str_mlm_method
    )
    return X_final, Y_final, np.concatenate((time_initials, time_all_final)), time_gp_final, time_acq_final

def optimize_many_with_random_init(model_bo, fun_target, int_init, int_iter,
    str_initial_method_bo=constants.STR_BO_INITIALIZATION,
    str_initial_method_ao=constants.STR_AO_INITIALIZATION,
    int_samples_ao=constants.NUM_ACQ_SAMPLES,
    str_mlm_method=constants.STR_MLM_METHOD,
    int_seed=None
):
    """
    It optimizes `fun_target` for `int_iter` iterations with given `model_bo` and `int_init` initial examples.
    Initial examples are sampled by `get_initial` method in `model_bo`.
    It returns the optimization results and execution times.

    :param model_bo: Bayesian optimization model.
    :type model_bo: bayeso.bo.BO
    :param fun_target: a target function.
    :type fun_target: function
    :param int_init: the number of initial examples for Bayesian optimization.
    :type int_init: int.
    :param int_iter: the number of iterations for Bayesian optimization.
    :type int_iter: int.
    :param str_initial_method_bo: the name of initialization method for sampling initial examples in Bayesian optimization.
    :type str_initial_method_bo: str., optional
    :param str_initial_method_ao: the name of initialization method for acquisition function optimization.
    :type str_initial_method_ao: str., optional
    :param int_samples_ao: the number of samples for acquisition function optimization. If L-BFGS-B is used as an acquisition function optimization method, it is employed.
    :type int_samples_ao: int., optional
    :param str_mlm_method: the name of marginal likelihood maximization method for Gaussian process regression.
    :type str_mlm_method: str., optional
    :param int_seed: None, or random seed.
    :type int_seed: NoneType or int., optional

    :returns: tuple of acquired examples, their function values, overall execution times per iteration, execution time consumed in Gaussian process regression, and execution time consumed in acquisition function optimization. Shape: ((`int_init` + `int_iter`, d), (`int_init` + `int_iter`, 1), (`int_init` + `int_iter`, ), (`int_iter`, ), (`int_iter`, )), or ((`int_init` + `int_iter`, m, d), (`int_init` + `int_iter`, m, 1), (`int_init` + `int_iter`, ), (`int_iter`, ), (`int_iter`, )), where d is a dimensionality of the problem we are solving and m is a cardinality of sets.
    :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)

    :raises: AssertionError

    """

    assert isinstance(model_bo, bo.BO)
    assert callable(fun_target)
    assert isinstance(int_init, int)
    assert isinstance(int_iter, int)
    assert isinstance(str_initial_method_bo, str)
    assert isinstance(str_initial_method_ao, str)
    assert isinstance(int_samples_ao, int)
    assert isinstance(str_mlm_method, str)
    assert isinstance(int_seed, int) or int_seed is None
    assert str_initial_method_bo in constants.ALLOWED_INITIALIZATIONS_BO
    assert str_mlm_method in constants.ALLOWED_MLM_METHOD

    print('[INFO] arr_range {}'.format(model_bo.arr_range))
    print('[INFO] str_cov {}'.format(model_bo.str_cov))
    print('[INFO] str_acq {}'.format(model_bo.str_acq))
    print('[INFO] str_optimizer_method_gp {}'.format(model_bo.str_optimizer_method_gp))
    print('[INFO] str_optimizer_method_bo {}'.format(model_bo.str_optimizer_method_bo))
    print('[INFO] str_modelselection_method {}'.format(model_bo.str_modelselection_method))
    print('[INFO] int_init {}'.format(int_init))
    print('[INFO] int_iter {}'.format(int_iter))
    print('[INFO] str_initial_method_bo {}'.format(str_initial_method_bo))
    print('[INFO] str_initial_method_ao {}'.format(str_initial_method_ao))
    print('[INFO] int_samples_ao {}'.format(int_samples_ao))
    print('[INFO] str_mlm_method {}'.format(str_mlm_method))
    print('[INFO] int_seed {}'.format(int_seed))

    time_start = time.time()

    X_init = model_bo.get_initial(str_initial_method_bo, fun_objective=fun_target, int_samples=int_init, int_seed=int_seed)
    if model_bo.debug:
        print('[DEBUG] optimize_many_with_random_init in utils_bo.py: X_init')
        print(X_init)
    X_final, Y_final, time_all_final, time_gp_final, time_acq_final = optimize_many(
        model_bo, fun_target, X_init, int_iter,
        str_initial_method_ao=str_initial_method_ao,
        int_samples_ao=int_samples_ao,
        str_mlm_method=str_mlm_method
    )

    time_end = time.time()

    if model_bo.debug:
        print('[DEBUG] optimize_many_with_random_init in utils_bo.py: time consumed', time_end - time_start, 'sec.')

    return X_final, Y_final, time_all_final, time_gp_final, time_acq_final
