# utils_bo
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: June 24, 2018

import numpy as np
import time

from bayeso import bo
from bayeso import constants


def get_grid(arr_ranges, int_grid):
    assert isinstance(arr_ranges, np.ndarray)
    assert isinstance(int_grid, int)
    assert len(arr_ranges.shape)
    assert arr_ranges.shape[1] == 2

    list_grid = []
    for range_ in arr_ranges:
        list_grid.append(np.linspace(range_[0], range_[1], int_grid))
    list_grid_mesh = list(np.meshgrid(*list_grid))
    list_grid = []
    for elem in list_grid_mesh:
        list_grid.append(elem.flatten(order='C'))
    arr_grid = np.vstack(tuple(list_grid))
    arr_grid = arr_grid.T
    return arr_grid

def get_best_acquisition(arr_initials, fun_objective):
    assert isinstance(arr_initials, np.ndarray)
    assert callable(fun_objective)
    assert len(arr_initials.shape) == 2

    cur_best = np.inf
    cur_initial = None
    for arr_initial in arr_initials:
        cur_acq = fun_objective(arr_initial)
        if cur_acq < cur_best:
            cur_initial = arr_initial
            cur_best = cur_acq
    return np.expand_dims(cur_initial, axis=0)

def optimize_many_(model_bo, fun_target, X_train, Y_train, int_iter,
    str_initial_method_optimizer=constants.STR_OPTIMIZER_INITIALIZATION,
    int_samples_ao=constants.NUM_ACQ_SAMPLES,
):
    assert isinstance(model_bo, bo.BO)
    assert callable(fun_target)
    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(int_iter, int)
    assert isinstance(str_initial_method_optimizer, str)
    assert isinstance(int_samples_ao, int)
    assert len(X_train.shape) == 2
    assert len(Y_train.shape) == 2
    assert X_train.shape[0] == Y_train.shape[0]
    assert Y_train.shape[1] == 1

    X_final = X_train
    Y_final = Y_train
    for _ in range(0, int_iter):
        next_point, _, _, _ = model_bo.optimize(X_final, Y_final, str_initial_method=str_initial_method_optimizer, int_samples=int_samples_ao)
        X_final = np.vstack((X_final, next_point))
        Y_final = np.vstack((Y_final, fun_target(next_point)))
    return X_final, Y_final

def optimize_many(model_bo, fun_target, X_train, int_iter,
    str_initial_method_optimizer=constants.STR_OPTIMIZER_INITIALIZATION,
    int_samples_ao=constants.NUM_ACQ_SAMPLES,
):
    assert isinstance(model_bo, bo.BO)
    assert callable(fun_target)
    assert isinstance(X_train, np.ndarray)
    assert isinstance(int_iter, int)
    assert isinstance(str_initial_method_optimizer, str)
    assert isinstance(int_samples_ao, int)
    assert len(X_train.shape) == 2

    Y_train = []
    for elem in X_train:
        Y_train.append(fun_target(elem))
    Y_train = np.array(Y_train)
    Y_train = np.reshape(Y_train, (Y_train.shape[0], 1))
    X_final, Y_final = optimize_many_(
        model_bo,
        fun_target,
        X_train,
        Y_train,
        int_iter,
        str_initial_method_optimizer=str_initial_method_optimizer,
        int_samples_ao=int_samples_ao,
    )
    return X_final, Y_final

def optimize_many_with_random_init(model_bo, fun_target, int_init, int_iter,
    str_initial_method_bo=constants.STR_BO_INITIALIZATION,
    str_initial_method_optimizer=constants.STR_OPTIMIZER_INITIALIZATION,
    int_samples_ao=constants.NUM_ACQ_SAMPLES,
    int_seed=None,
):
    assert isinstance(model_bo, bo.BO)
    assert callable(fun_target)
    assert isinstance(int_init, int)
    assert isinstance(int_iter, int)
    assert isinstance(str_initial_method_bo, str)
    assert isinstance(str_initial_method_optimizer, str)
    assert isinstance(int_samples_ao, int)
    assert isinstance(int_seed, int) or int_seed is None
    assert str_initial_method_bo in constants.ALLOWED_INITIALIZATIONS_BO

    time_start = time.time()

    X_init = model_bo.get_initial(str_initial_method_bo, fun_objective=fun_target, int_samples=int_init, int_seed=int_seed)
    if model_bo.debug:
        print('[DEBUG] optimize_many_with_random_init: X_init')
        print(X_init)
    X_final, Y_final = optimize_many(model_bo, fun_target, X_init, int_iter,
        str_initial_method_optimizer=str_initial_method_optimizer,
        int_samples_ao=int_samples_ao,
    )

    time_end = time.time()

    if model_bo.debug:
        print('[DEBUG] optimize_many_with_random_init', time_end - time_start, 'sec.')

    return X_final, Y_final
