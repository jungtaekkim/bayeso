# utils_bo
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: June 22, 2018

import numpy as np


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
    cur_best = np.inf
    cur_initial = None
    for arr_initial in arr_initials:
        cur_acq = fun_objective(arr_initial)
        if cur_acq < cur_best:
            cur_initial = arr_initial
            cur_best = cur_acq
    return np.expand_dims(cur_initial, axis=0)

def optimize_many_(model_bo, fun_target, X_train, Y_train, num_iter):
    X_final = X_train
    Y_final = Y_train
    for _ in range(0, num_iter):
        result_bo, _, _, _ = model_bo.optimize(X_final, Y_final)
        X_final = np.vstack((X_final, result_bo))
        Y_final = np.vstack((Y_final, fun_target(result_bo)))
    return X_final, Y_final

def optimize_many(model_bo, fun_target, X_train, num_iter):
    Y_train = []
    for elem in X_train:
        Y_train.append(fun_target(elem))
    Y_train = np.array(Y_train)
    Y_train = np.reshape(Y_train, (Y_train.shape[0], 1))
    X_final, Y_final = optimize_many_(model_bo, fun_target, X_train, Y_train, num_iter)
    return X_final, Y_final

def optimize_many_with_random_init(model_bo, fun_target, num_init, num_iter, int_seed=None):
    list_init = []
    for ind_init in range(0, num_init):
        if int_seed is None or int_seed == 0:
            print('REMIND: seed is None or 0.')
            list_init.append(model_bo._get_initial(is_random=True, is_grid=False))
        else:
            list_init.append(model_bo._get_initial(is_random=True, is_grid=False, int_seed=int_seed**2 * (ind_init+1)))
    X_init = np.array(list_init)
    X_final, Y_final = optimize_many(model_bo, fun_target, X_init, num_iter)
    return X_final, Y_final

