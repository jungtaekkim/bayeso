# utils_bo
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: June 04, 2019

import numpy as np
import time

from bayeso import bo
from bayeso import constants


def get_next_best_acquisition(arr_points, arr_acquisitions, cur_points):
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
        next_point = cur_points[cur_points.shape[0]-1]
    return next_point

def optimize_many_(model_bo, fun_target, X_train, Y_train, int_iter,
    str_initial_method_ao=constants.STR_AO_INITIALIZATION,
    int_samples_ao=constants.NUM_ACQ_SAMPLES,
    str_mlm_method=constants.STR_MLM_METHOD,
    str_modelselection_method=constants.STR_MODELSELECTION_METHOD
):
    assert isinstance(model_bo, bo.BO)
    assert callable(fun_target)
    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(int_iter, int)
    assert isinstance(str_initial_method_ao, str)
    assert isinstance(int_samples_ao, int)
    assert isinstance(str_mlm_method, str)
    assert isinstance(str_modelselection_method, str)
    assert len(X_train.shape) == 2
    assert len(Y_train.shape) == 2
    assert X_train.shape[0] == Y_train.shape[0]
    assert Y_train.shape[1] == 1
    assert str_mlm_method in constants.ALLOWED_MLM_METHOD
    assert str_modelselection_method in constants.ALLOWED_MODELSELECTION_METHOD

    time_start = time.time()

    X_final = X_train
    Y_final = Y_train
    time_final = []
    for ind_iter in range(0, int_iter):
        time_iter_start = time.time()

        if model_bo.debug:
            print('[DEBUG] optimize_many_ in utils_bo.py: current iteration', ind_iter + 1)
        next_point, next_points, acquisitions, _, _, _ = model_bo.optimize(X_final, Y_final, str_initial_method=str_initial_method_ao, int_samples=int_samples_ao, str_mlm_method=str_mlm_method, str_modelselection_method=str_modelselection_method)
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
        time_final.append(time_iter_end - time_iter_start)

    time_end = time.time()

    if model_bo.debug:
        print('[DEBUG] optimize_many_ in utils_bo.py: time consumed', time_end - time_start, 'sec.')
    return X_final, Y_final, np.array(time_final)

def optimize_many(model_bo, fun_target, X_train, int_iter,
    str_initial_method_ao=constants.STR_AO_INITIALIZATION,
    int_samples_ao=constants.NUM_ACQ_SAMPLES,
    str_mlm_method=constants.STR_MLM_METHOD,
    str_modelselection_method=constants.STR_MODELSELECTION_METHOD
):
    assert isinstance(model_bo, bo.BO)
    assert callable(fun_target)
    assert isinstance(X_train, np.ndarray)
    assert isinstance(int_iter, int)
    assert isinstance(str_initial_method_ao, str)
    assert isinstance(int_samples_ao, int)
    assert isinstance(str_mlm_method, str)
    assert isinstance(str_modelselection_method, str)
    assert len(X_train.shape) == 2
    assert str_mlm_method in constants.ALLOWED_MLM_METHOD
    assert str_modelselection_method in constants.ALLOWED_MODELSELECTION_METHOD

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
    X_final, Y_final, time_final = optimize_many_(
        model_bo,
        fun_target,
        X_train,
        Y_train,
        int_iter,
        str_initial_method_ao=str_initial_method_ao,
        int_samples_ao=int_samples_ao,
        str_mlm_method=str_mlm_method,
        str_modelselection_method=str_modelselection_method,
    )
    return X_final, Y_final, np.concatenate((time_initials, time_final))

def optimize_many_with_random_init(model_bo, fun_target, int_init, int_iter,
    str_initial_method_bo=constants.STR_BO_INITIALIZATION,
    str_initial_method_ao=constants.STR_AO_INITIALIZATION,
    int_samples_ao=constants.NUM_ACQ_SAMPLES,
    str_mlm_method=constants.STR_MLM_METHOD,
    str_modelselection_method=constants.STR_MODELSELECTION_METHOD,
    int_seed=None
):
    assert isinstance(model_bo, bo.BO)
    assert callable(fun_target)
    assert isinstance(int_init, int)
    assert isinstance(int_iter, int)
    assert isinstance(str_initial_method_bo, str)
    assert isinstance(str_initial_method_ao, str)
    assert isinstance(int_samples_ao, int)
    assert isinstance(str_mlm_method, str)
    assert isinstance(str_modelselection_method, str)
    assert isinstance(int_seed, int) or int_seed is None
    assert str_initial_method_bo in constants.ALLOWED_INITIALIZATIONS_BO
    assert str_mlm_method in constants.ALLOWED_MLM_METHOD
    assert str_modelselection_method in constants.ALLOWED_MODELSELECTION_METHOD

    print('[INFO] arr_range {}'.format(model_bo.arr_range))
    print('[INFO] str_cov {}'.format(model_bo.str_cov))
    print('[INFO] str_acq {}'.format(model_bo.str_acq))
    print('[INFO] str_optimizer_method_gp {}'.format(model_bo.str_optimizer_method_gp))
    print('[INFO] str_optimizer_method_bo {}'.format(model_bo.str_optimizer_method_bo))
    print('[INFO] int_init {}'.format(int_init))
    print('[INFO] int_iter {}'.format(int_iter))
    print('[INFO] str_initial_method_bo {}'.format(str_initial_method_bo))
    print('[INFO] str_initial_method_ao {}'.format(str_initial_method_ao))
    print('[INFO] int_samples_ao {}'.format(int_samples_ao))
    print('[INFO] str_mlm_method {}'.format(str_mlm_method))
    print('[INFO] str_modelselection_method {}'.format(str_modelselection_method))
    print('[INFO] int_seed {}'.format(int_seed))

    time_start = time.time()

    X_init = model_bo.get_initial(str_initial_method_bo, fun_objective=fun_target, int_samples=int_init, int_seed=int_seed)
    if model_bo.debug:
        print('[DEBUG] optimize_many_with_random_init in utils_bo.py: X_init')
        print(X_init)
    X_final, Y_final, time_final = optimize_many(model_bo, fun_target, X_init, int_iter,
        str_initial_method_ao=str_initial_method_ao,
        int_samples_ao=int_samples_ao,
        str_mlm_method=str_mlm_method,
        str_modelselection_method=str_modelselection_method
    )

    time_end = time.time()

    if model_bo.debug:
        print('[DEBUG] optimize_many_with_random_init in utils_bo.py: time consumed', time_end - time_start, 'sec.')

    return X_final, Y_final, time_final
