# bo
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 09, 2018

import numpy as np
import time
from scipy.optimize import minimize
import sobol_seq

from bayeso import gp
from bayeso import acquisition
from bayeso.utils import utils_common
from bayeso.utils import utils_bo
from bayeso import constants


# TODO: I am not sure, but flatten() should be replaced.
class BO():
    def __init__(self, arr_range,
        str_cov=constants.STR_GP_COV,
        str_acq=constants.STR_BO_ACQ,
        is_ard=True,
        prior_mu=None,
        debug=False,
    ):
        # TODO: use is_ard.
        # TODO: add debug cases.
        assert isinstance(arr_range, np.ndarray)
        assert isinstance(str_cov, str)
        assert isinstance(str_acq, str)
        assert isinstance(is_ard, bool)
        assert isinstance(debug, bool)
        assert callable(prior_mu) or prior_mu is None
        assert len(arr_range.shape) == 2
        assert arr_range.shape[1] == 2
        assert (arr_range[:, 0] <= arr_range[:, 1]).all()
        assert str_cov in constants.ALLOWED_GP_COV
        assert str_acq in constants.ALLOWED_BO_ACQ

        self.arr_range = arr_range
        self.num_dim = arr_range.shape[0]
        self.str_cov = str_cov
        self.str_acq = str_acq
        self.is_ard = is_ard
        self.debug = debug
        self.prior_mu = prior_mu

    def _get_initial_grid(self, int_grid=constants.NUM_BO_GRID):
        assert isinstance(int_grid, int)

        arr_initials = utils_bo.get_grid(self.arr_range, int_grid)
        return arr_initials

    def _get_initial_uniform(self, int_samples, int_seed=None):
        assert isinstance(int_seed, int) or int_seed is None
        
        if int_seed is not None:
            np.random.seed(int_seed)
        list_initials = []
        for _ in range(0, int_samples):
            list_initial = []
            for elem in self.arr_range:
                list_initial.append(np.random.uniform(elem[0], elem[1]))
            list_initials.append(np.array(list_initial))
        arr_initials = np.array(list_initials)
        return arr_initials

    # TODO: I am not sure, but noise can be added.
    def _get_initial_sobol(self, int_samples, int_seed=None):
        assert isinstance(int_seed, int) or int_seed is None

        if int_seed is None:
            int_seed = np.random.randint(0, 10000)
        if self.debug:
            print('[DEBUG] _get_initial_sobol in bo.py: int_seed', int_seed)
        arr_samples = sobol_seq.i4_sobol_generate(self.num_dim, int_samples, int_seed)
        arr_samples = arr_samples * (self.arr_range[:, 1].flatten() - self.arr_range[:, 0].flatten()) + self.arr_range[:, 0].flatten()
        return arr_samples

    def _get_initial_latin(self, int_samples):
        raise NotImplementedError('_get_initial_latin in bo.py')

    # TODO: int_grid should be able to be input.
    def get_initial(self, str_initial_method,
        fun_objective=None,
        int_samples=constants.NUM_ACQ_SAMPLES,
        int_seed=None,
    ):
        assert isinstance(str_initial_method, str)
        assert callable(fun_objective) or fun_objective is None
        assert isinstance(int_samples, int)
        assert isinstance(int_seed, int) or int_seed is None

        if str_initial_method == 'grid':
            assert fun_objective is not None
            if self.debug:
                print('[DEBUG] get_initial in bo.py: int_samples is ignored, because grid is chosen.')
            arr_initials = self._get_initial_grid()
            arr_initials = utils_bo.get_best_acquisition(arr_initials, fun_objective)
        elif str_initial_method == 'uniform':
            arr_initials = self._get_initial_uniform(int_samples, int_seed=int_seed)
        elif str_initial_method == 'sobol':
            arr_initials = self._get_initial_sobol(int_samples, int_seed=int_seed)
        elif str_initial_method == 'latin':
            raise NotImplementedError('get_initial: latin')
        else:
            raise NotImplementedError('get_initial: allowed str_initial_method, but it is not implemented.')
        if self.debug:
            print('[DEBUG] get_initial in bo.py: arr_initials')
            print(arr_initials)
        return arr_initials

    def _optimize_objective(self, fun_acquisition, X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps):
        X_test = np.atleast_2d(X_test)
        pred_mean, pred_std = gp.predict_test_(X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps, self.str_cov, self.prior_mu)
        # no matter which acquisition functions are given, we input pred_mean, pred_std, and Y_train.
        acquisitions = fun_acquisition(pred_mean=pred_mean.flatten(), pred_std=pred_std.flatten(), Y_train=Y_train)
        return acquisitions

    def _optimize(self, fun_negative_acquisition, str_initial_method, int_samples):
        list_bounds = []
        for elem in self.arr_range:
            list_bounds.append(tuple(elem))
        arr_initials = self.get_initial(str_initial_method, fun_objective=fun_negative_acquisition, int_samples=int_samples)
        list_next_point = []
        for arr_initial in arr_initials:
            next_point = minimize(
                fun_negative_acquisition,
                x0=arr_initial,
                bounds=list_bounds,
                method=constants.STR_OPTIMIZER_METHOD_BO,
                options={'disp': False}
            )
            list_next_point.append(next_point.x)
            if self.debug:
                print('[DEBUG] _optimize in bo.py: optimized point for acq', next_point.x)
        next_points = np.array(list_next_point)
        next_point = utils_bo.get_best_acquisition(next_points, fun_negative_acquisition)
        return next_point.flatten(), next_points

    def optimize(self, X_train, Y_train,
        str_initial_method=constants.STR_AO_INITIALIZATION,
        int_samples=constants.NUM_ACQ_SAMPLES,
        is_normalized=True,
    ):
        # TODO: is_normalized cases
        assert isinstance(X_train, np.ndarray)
        assert isinstance(Y_train, np.ndarray)
        assert isinstance(str_initial_method, str)
        assert isinstance(int_samples, int)
        assert isinstance(is_normalized, bool)
        assert len(X_train.shape) == 2
        assert len(Y_train.shape) == 2
        assert Y_train.shape[1] == 1
        assert X_train.shape[0] == Y_train.shape[0]
        assert X_train.shape[1] == self.num_dim
        assert int_samples > 0
        assert str_initial_method in constants.ALLOWED_INITIALIZATIONS_AO

        time_start = time.time()

        if is_normalized:
            Y_train = (Y_train - np.min(Y_train)) / (np.max(Y_train) - np.min(Y_train)) * constants.MULTIPLIER_RESPONSE

        cov_X_X, inv_cov_X_X, hyps = gp.get_optimized_kernel(X_train, Y_train, self.prior_mu, self.str_cov, debug=self.debug)

        if self.str_acq == 'pi':
            fun_acquisition = acquisition.pi
        elif self.str_acq == 'ei':
            fun_acquisition = acquisition.ei
        elif self.str_acq == 'ucb':
            fun_acquisition = acquisition.ucb
        elif self.str_acq == 'aei':
            fun_acquisition = lambda pred_mean, pred_std, Y_train: acquisition.aei(pred_mean, pred_std, Y_train, hyps['noise'])
        elif self.str_acq == 'pure_exploit':
            fun_acquisition = acquisition.pure_exploit
        elif self.str_acq == 'pure_explore':
            fun_acquisition = acquisition.pure_explore
        else:
            raise NotImplementedError('optimize: allowed str_acq, but it is not implemented.')
      
        fun_negative_acquisition = lambda X_test: -1.0 * constants.MULTIPLIER_ACQ * self._optimize_objective(fun_acquisition, X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps)
        next_point, next_points = self._optimize(fun_negative_acquisition, str_initial_method=str_initial_method, int_samples=int_samples)
        acquisitions = fun_negative_acquisition(next_points)

        time_end = time.time()

        if self.debug:
            print('[DEBUG] optimize in bo.py: time consumed', time_end - time_start, 'sec.')
        return next_point, next_points, acquisitions, cov_X_X, inv_cov_X_X, hyps
