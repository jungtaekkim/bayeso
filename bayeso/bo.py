import numpy as np
from scipy.optimize import minimize

import gp
import acquisition
import utils

NUM_GRID = 50

class BO():
    def __init__(self, arr_range, str_cov='se', is_ard=True, str_acq='ei', prior_mu=None):
        self.arr_range = arr_range
        self.str_cov = str_cov
        self.str_acq = str_acq
        self.is_ard = is_ard
        self.prior_mu = prior_mu

    def _get_initial_random(self, int_seed=None):
        if int_seed is not None:
            np.random.seed(int_seed)
        list_initial = []
        for elem in self.arr_range:
            list_initial.append(np.random.uniform(elem[0], elem[1]))
        arr_initial = np.array(list_initial)
        return arr_initial

    def _get_initial(self, is_random=False, is_grid=False, fun_obj=None, int_seed=None):
        if is_random:
            arr_initial = self._get_initial_random(int_seed)
        elif is_grid:
            if fun_obj is None:
                print('WARNING: fun_obj is not given.')
                arr_initial = self._get_initial_random(int_seed)
            else:
                list_grid = []
                for elem in self.arr_range:
                    list_grid.append(np.linspace(elem[0], elem[1], NUM_GRID))
                arr_grid = np.array(list_grid)
                arr_initial = None
                initial_best = np.inf
                count_same = 0

                for ind_initial in range(0, NUM_GRID**self.arr_range.shape[0]):
                    cur_initial = []
                    for ind_cur in range(0, self.arr_range.shape[0]):
                        cur_initial.append(arr_grid[ind_cur, int(ind_initial / (NUM_GRID**ind_cur) % NUM_GRID)])
                    cur_initial = np.array(cur_initial)
                    cur_acq = fun_obj(cur_initial)
                    if cur_acq < initial_best:
                        initial_best = cur_acq
                        arr_initial = cur_initial
                    elif cur_acq == initial_best:
                        count_same += 1
                if count_same == NUM_GRID**self.arr_range.shape[0] - 1:
                    arr_initial = self._get_initial_random()
        else:
            raise ValueError('initialization is inappropriate.')
        return arr_initial

    def _optimize_objective(self, fun_acq, X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps):
        X_test = np.atleast_2d(X_test)
        pred_mean, pred_std = gp.predict_test_(X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps, self.str_cov, self.prior_mu)
        result_acq = fun_acq(pred_mean, pred_std, Y_train)
        return result_acq

    def optimize(self, X_train, Y_train):
        cov_X_X, inv_cov_X_X, hyps = gp.get_optimized_kernels(X_train, Y_train, self.prior_mu, self.str_cov)

        # NEED: to add acquisition function
        if self.str_acq == 'pi':
            fun_acq = acquisition.pi
        elif self.str_acq == 'ei':
            fun_acq = acquisition.ei
        elif self.str_acq == 'ucb':
            fun_acq = acquisition.ucb
        else:
            raise ValueError('acquisition function is not properly set.')
      
        fun_obj = lambda X_test: -1000.0 * self._optimize_objective(fun_acq, X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps)

        list_bounds = []
        for elem in self.arr_range:
            list_bounds.append(tuple(elem))

        result_optimized = minimize(fun_obj, x0=self._get_initial(False, True, fun_obj), bounds=list_bounds, method='L-BFGS-B', options={'disp': True})
        print('INFORM: optimized result for acq. ', result_optimized.x)
        return result_optimized.x, cov_X_X, inv_cov_X_X, hyps

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
    X_final, Y_final = optimize_many_(model_bo, fun_target, X_train, Y_train, num_ter)
    return X_final, Y_final

def optimize_many_with_random_init(model_bo, fun_target, num_init, num_iter, int_seed=None):
    list_init = []
    for ind_init in range(0, num_init):
        if int_seed is None or int_seed == 0:
            print('REMIND: seed is None or 0.')
            list_init.append(model_bo._get_initial(is_random=True))
        else:
            list_init.append(model_bo._get_initial(is_random=True, int_seed=int_seed**2 * (ind_init+1)))
    X_init = np.array(list_init)
    X_final, Y_final = optimize_many(model_bo, fun_target, X_init, num_iter)
    return X_final, Y_final
        

if __name__ == '__main__':
    pass


