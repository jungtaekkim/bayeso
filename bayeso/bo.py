import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
try:
    import GPflow
except:
    import gpflow
    GPflow = gpflow

import gp
import acquisition
import utils

NUM_GRID = 100

class BO():
    def __init__(self, arr_range, str_kernel='se', is_ard=True, str_acq='ei', fun_mean=None):
        self.arr_range = arr_range
        self.str_kernel = str_kernel
        self.str_acq = str_acq
        self.is_ard = is_ard
        if fun_mean is None:
            self.fun_mean = GPflow.mean_functions.Zero()
        else:
            list_mean = []
            for elem in fun_mean:
                if isinstance(elem[2], float):
                    list_mean.append(GPflow.mean_functions.Constant(elem[2]))
            switched_mean = GPflow.mean_functions.SwitchedMeanFunction(list_mean)
            self.fun_mean = switched_mean
        self.orig_fun_mean = fun_mean

    def _get_initial_random(self, int_seed=None):
        if int_seed is not None:
            np.random.seed(int_seed)
        list_initial = []
        for elem in self.arr_range:
            list_initial.append(np.random.uniform(elem[0], elem[1]))
        arr_initial = np.array(list_initial)
        return arr_initial

    def _get_initial_first(self):
        arr_initial = self.arr_range[:, 0]
        return arr_initial

    def _get_initial(self, is_random=False, is_grid=False, fun_obj=None, int_seed=None):
        if is_random:
            arr_initial = self._get_initial_random(int_seed)
        elif is_grid:
            if fun_obj is None:
                print('WARNING: no fun_obj')
                arr_initial = self._get_initial_random(int_seed)
            else:
                list_grid = []
                for elem in self.arr_range:
                    list_grid.append(np.linspace(elem[0], elem[1], NUM_GRID))
                arr_grid = np.array(list_grid)
                arr_initial = None
                initial_best = tf.constant(np.inf, dtype=tf.float64)
                count_same = 0

                raise NotImplementedError('in _get_initial()')
                for ind_initial in range(0, NUM_GRID**self.arr_range.shape[0]):
                    temp_ind = ind_initial
                    cur_initial = []
                    for ind_cur in range(0, self.arr_range.shape[0]):
                        cur_initial.append(arr_grid[ind_cur, int(temp_ind%NUM_GRID)])
                        temp_ind /= NUM_GRID
                    cur_initial = np.array(cur_initial)
                    cur_acq = fun_obj(cur_initial)
                    def f1():
                        tf.assign(initial_best, cur_acq)
                        tf.assign(arr_initial, cur_initial)
                    def f2():
                        tf.assign(count_same, count_same+1)

                    if tf.less(cur_acq, initial_best):
                        initial_best = cur_acq
                        arr_initial = cur_initial
                    elif tf.equal(cur_acq, initial_best):
                        count_same += 1
                if count_same == NUM_GRID - 1:
                    arr_initial = self._get_initial_random()
        else:
            arr_initial = self._get_initial_first()
        return arr_initial

    def _add_kernel_indicator(self, X_train):
        list_indicator = []
        for elem_1 in X_train:
            flag_in = False
            ind_in = 0
            for ind_elem_2, elem_2 in enumerate(self.orig_fun_mean):
#                print 'Compared: ', elem_1, elem_2
                if (elem_2[0] <= elem_1).all() and (elem_1 <= elem_2[1]).all():
                    flag_in = True
                    ind_in = ind_elem_2
            list_indicator.append(ind_in)
#        print 'Indicator', X_train.shape, len(list_indicator)
        return np.hstack((X_train, 1.0 * np.array(list_indicator).reshape(-1, 1)))

    def _optimize_objective(self, fun_acq, fun_gp, model_gp, X_test, Y_train=None):
        if self.orig_fun_mean is not None:
            X_test = self._add_kernel_indicator(X_test)
        pred_mean, pred_std = fun_gp(model_gp, X_test)
        print(pred_mean)
        print(pred_std)
        if Y_train is not None:
            result_acq = fun_acq(pred_mean, pred_std, Y_train)
        else:
            result_acq = fun_acq(pred_mean, pred_std)
        print(result_acq)
        print(result_acq.shape)
        if result_acq.shape == (1, 1):
            return result_acq[0, 0]
        else:
            return result_acq

    def optimize(self, X_train, Y_train):
        # NEED: to add kernel
        if self.str_kernel == 'se':
            fun_ker = GPflow.kernels.RBF(X_train.shape[1], ARD=self.is_ard)
        else:
            print('WARNING: need to add kernel')
            fun_ker = GPflow.kernels.RBF(X_train.shape[1], ARD=self.is_ard)

        if self.orig_fun_mean is not None:
            X_train_ = self._add_kernel_indicator(X_train)
        else:
            X_train_ = X_train

        model_gp = gp.build_model_gp(X_train_, Y_train, fun_ker, self.fun_mean)

        # NEED: to add acquisition function
        if self.str_acq == 'pi':
            fun_acq = acquisition.pi
        elif self.str_acq == 'ei':
            fun_acq = acquisition.ei
        elif self.str_acq == 'ucb':
            fun_acq = acquisition.ucb
        else:
            print('WARNING: need to add new acquisition function. set to EI.')
            fun_acq = acquisition.ei
      
        if self.str_acq == 'pi' or self.str_acq == 'ei':
            fun_obj = lambda X_test: -100.0 * self._optimize_objective(fun_acq, gp.predict_test, model_gp, np.atleast_2d(X_test), Y_train=Y_train)
        elif self.str_acq == 'ucb':
            fun_obj = lambda X_test: -100.0 * self._optimize_objective(fun_acq, gp.predict_test, model_gp, np.atleast_2d(X_test), Y_train=None)
        else:
            print('ERROR: need to add a condition for acquisition function.')
            raise NotImplementedError()

        list_bounds = []
        for elem in self.arr_range:
            list_bounds.append(tuple(elem))

        result_optimized = tf.train.GradientDescentOptimizer(x0=self._get_initial(is_random=True), bounds=list_bounds).minimize(fun_obj)
#        result_optimized = minimize(fun_obj, x0=self._get_initial(False, True, fun_obj), bounds=list_bounds, options={'disp': True})
        return result_optimized

def optimize_many(model_bo, fun_target, X_train, Y_train, num_iter):
    X_final = X_train
    Y_final = Y_train
    for _ in range(0, num_iter):
        result_bo = model_bo.optimize(X_final, Y_final)
        X_final = np.vstack((X_final, result_bo.x))
        Y_final = np.vstack((Y_final, fun_target(result_bo.x)))
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
    Y_init = fun_target(X_init)
    X_final, Y_final = optimize_many(model_bo, fun_target, X_init, Y_init, num_iter)
    return X_final, Y_final
        

if __name__ == '__main__':
    import matplotlib.pyplot as plt
#    X_train = np.array([[-2.0], [0.0], [0.2], [0.1], [0.05], [0.15], [1.0], [1.5], [2.05], [1.9], [2.0], [2.1], [3.0], [-1.0]])
    X_train = np.array([[-2.5], [2.5]])
    Y_train = np.sin(X_train)
    X_test = np.linspace(-3, 3, 100)
    X_test = X_test.reshape((100, 1))
    sess = tf.Session()
    model_bo = BO(np.array([[-3.0, 3.0]]), str_kernel='se', is_ard=True, str_acq='ei', fun_mean=np.array([[np.array([-3.0]), np.array([1.0]), 0.1], [np.array([1.0]), np.array([3.0]), -0.1]]))
    print(model_bo.fun_mean)
    model_gp = gp.build_model_gp(model_bo._add_kernel_indicator(X_train), Y_train, GPflow.kernels.RBF(X_train.shape[1], ARD=True), fun_mean=model_bo.fun_mean)
#    model_gp = gp.build_model_gp(X_train, Y_train, GPflow.kernels.RBF(X_train.shape[1], ARD=True))
    pred_mean, pred_std = gp.predict_test(model_gp, model_bo._add_kernel_indicator(X_test))
#    pred_mean, pred_std = gp.predict_test(model_gp, X_test)

    list_all_X_ = []
    list_all_Y_ = []
    for ind_all in range(0, 3):
        X_, Y_ = optimize_many_with_random_init(model_bo, np.sin, 3, 5, 1 * (ind_all+1))
        list_all_X_.append(X_)
        list_all_Y_.append(Y_)
    utils.plot_minimum_mean_std([list_all_Y_, list_all_Y_], ['abc', 'abcd'], './', 'abc', 3)
    utils.plot_minimum_mean([list_all_Y_, list_all_Y_], ['abc', 'abcd'], './', 'abc', 3)
        




