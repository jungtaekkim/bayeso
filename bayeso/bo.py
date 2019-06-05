# bo
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: June 03, 2019

import numpy as np
import time
from scipy.optimize import minimize
try:
    from scipydirect import minimize as directminimize
except: # pragma: no cover
    directminimize = None
try:
    import cma
except: # pragma: no cover
    cma = None
import sobol_seq

from bayeso import gp
from bayeso import acquisition
from bayeso.utils import utils_common
from bayeso.utils import utils_covariance
from bayeso import constants


def get_grid(arr_ranges, int_grid):
    assert isinstance(arr_ranges, np.ndarray)
    assert isinstance(int_grid, int)
    assert len(arr_ranges.shape) == 2
    assert arr_ranges.shape[1] == 2
    assert (arr_ranges[:, 0] <= arr_ranges[:, 1]).all()

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

def _check_optimizer_method_bo(str_optimizer_method_bo, num_dim, debug):
    assert isinstance(str_optimizer_method_bo, str)
    assert isinstance(num_dim, int)
    assert isinstance(debug, bool)
    assert str_optimizer_method_bo in constants.ALLOWED_OPTIMIZER_METHOD_BO

    if str_optimizer_method_bo == 'DIRECT' and directminimize is None: # pragma: no cover
        if debug:
            print('[DEBUG] _check_optimizer_method_bo in bo.py: DIRECT is selected, but it is not installed.')
        str_optimizer_method_bo = 'L-BFGS-B'
    elif str_optimizer_method_bo == 'CMA-ES' and cma is None: # pragma: no cover
        if debug:
            print('[DEBUG] _check_optimizer_method_bo in bo.py: CMA-ES is selected, but it is not installed.')
        str_optimizer_method_bo = 'L-BFGS-B'
    # TODO: It should be checked.
    elif str_optimizer_method_bo == 'CMA-ES' and num_dim == 1: # pragma: no cover
        if debug:
            print('[DEBUG] _check_optimizer_method_bo in bo.py: CMA-ES is selected, but a dimension of bounds is 1.')
        str_optimizer_method_bo = 'L-BFGS-B'
    return str_optimizer_method_bo

def _choose_fun_acquisition(str_acq, hyps):
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
        fun_acquisition = lambda pred_mean, pred_std, Y_train: acquisition.aei(pred_mean, pred_std, Y_train, hyps['noise'])
    elif str_acq == 'pure_exploit':
        fun_acquisition = acquisition.pure_exploit
    elif str_acq == 'pure_explore':
        fun_acquisition = acquisition.pure_explore
    else:
        raise NotImplementedError('_choose_fun_acquisition: allowed str_acq, but it is not implemented.')
    return fun_acquisition

def _check_hyps_convergence(list_hyps, hyps, str_cov, is_fixed_noise, ratio_threshold=0.05):
    assert isinstance(list_hyps, list)
    assert isinstance(hyps, dict)
    assert isinstance(str_cov, str)
    assert isinstance(is_fixed_noise, bool)
    assert isinstance(ratio_threshold, float)

    is_converged = False
    if len(list_hyps) > 0:
        hyps_converted = utils_covariance.convert_hyps(str_cov, hyps, is_fixed_noise=is_fixed_noise)
        target_hyps_converted = utils_covariance.convert_hyps(str_cov, list_hyps[-1], is_fixed_noise=is_fixed_noise)

        cur_norm = np.linalg.norm(hyps_converted - target_hyps_converted, ord=2)
        threshold = np.linalg.norm(target_hyps_converted) * ratio_threshold
        if np.linalg.norm(hyps_converted - target_hyps_converted, ord=2) < threshold:
            is_converged = True
    return is_converged

# TODO: I am not sure, but flatten() should be replaced.
class BO():
    def __init__(self, arr_range,
        str_cov=constants.STR_GP_COV,
        str_acq=constants.STR_BO_ACQ,
        is_ard=True,
        prior_mu=None,
        str_optimizer_method_gp=constants.STR_OPTIMIZER_METHOD_GP,
        str_optimizer_method_bo=constants.STR_OPTIMIZER_METHOD_BO,
        debug=False
    ):
        # TODO: use is_ard.
        # TODO: add debug cases.
        assert isinstance(arr_range, np.ndarray)
        assert isinstance(str_cov, str)
        assert isinstance(str_acq, str)
        assert isinstance(is_ard, bool)
        assert isinstance(str_optimizer_method_bo, str)
        assert isinstance(str_optimizer_method_gp, str)
        assert isinstance(debug, bool)
        assert callable(prior_mu) or prior_mu is None
        assert len(arr_range.shape) == 2
        assert arr_range.shape[1] == 2
        assert (arr_range[:, 0] <= arr_range[:, 1]).all()
        assert str_cov in constants.ALLOWED_GP_COV
        assert str_acq in constants.ALLOWED_BO_ACQ
        assert str_optimizer_method_gp in constants.ALLOWED_OPTIMIZER_METHOD_GP
        assert str_optimizer_method_bo in constants.ALLOWED_OPTIMIZER_METHOD_BO

        self.arr_range = arr_range
        self.num_dim = arr_range.shape[0]
        self.str_cov = str_cov
        self.str_acq = str_acq
        self.is_ard = is_ard
        self.str_optimizer_method_bo = _check_optimizer_method_bo(str_optimizer_method_bo, arr_range.shape[0], debug)
        self.str_optimizer_method_gp = str_optimizer_method_gp
        self.debug = debug
        self.prior_mu = prior_mu

        self.is_optimize_hyps = True
        self.historical_hyps = []

    def _get_initial_grid(self, int_grid=constants.NUM_BO_GRID):
        assert isinstance(int_grid, int)

        arr_initials = get_grid(self.arr_range, int_grid)
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
            arr_initials = get_best_acquisition(arr_initials, fun_objective)
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
        pred_mean, pred_std = gp.predict_test_(X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov=self.str_cov, prior_mu=self.prior_mu, debug=self.debug)
        # no matter which acquisition functions are given, we input pred_mean, pred_std, and Y_train.
        acquisitions = fun_acquisition(pred_mean=pred_mean.flatten(), pred_std=pred_std.flatten(), Y_train=Y_train)
        return acquisitions

    def _get_bounds(self):
        list_bounds = []
        for elem in self.arr_range:
            list_bounds.append(tuple(elem))
        return list_bounds

    def _optimize(self, fun_negative_acquisition, str_initial_method, int_samples):
        list_next_point = []
        if self.str_optimizer_method_bo == 'L-BFGS-B':
            list_bounds = self._get_bounds()
            arr_initials = self.get_initial(str_initial_method, fun_objective=fun_negative_acquisition, int_samples=int_samples)
            for arr_initial in arr_initials:
                next_point = minimize(
                    fun_negative_acquisition,
                    x0=arr_initial,
                    bounds=list_bounds,
                    method=self.str_optimizer_method_bo,
                    options={'disp': False}
                )
                next_point_x = next_point.x
                list_next_point.append(next_point_x)
                if self.debug:
                    print('[DEBUG] _optimize in bo.py: optimized point for acq', next_point_x)
        elif self.str_optimizer_method_bo == 'DIRECT': # pragma: no cover
            list_bounds = self._get_bounds()
            next_point = directminimize(
                fun_negative_acquisition,
                bounds=list_bounds,
            )
            next_point_x = next_point.x
            list_next_point.append(next_point_x)
        elif self.str_optimizer_method_bo == 'CMA-ES':
            list_bounds = self._get_bounds()
            list_bounds = np.array(list_bounds)
            def fun_wrapper(f):
                def g(bx):
                    return f(bx)[0]
                return g
            arr_initials = self.get_initial(str_initial_method, fun_objective=fun_negative_acquisition, int_samples=1)
            next_point_x = cma.fmin(fun_wrapper(fun_negative_acquisition), arr_initials[0], 0.5, options={'bounds': [list_bounds[:, 0], list_bounds[:, 1]], 'verbose': -1})[0]
            list_next_point.append(next_point_x)

        next_points = np.array(list_next_point)
        next_point = get_best_acquisition(next_points, fun_negative_acquisition)
        return next_point.flatten(), next_points

    # TODO: str_mlm_method should be moved to __init__.
    def optimize(self, X_train, Y_train,
        str_initial_method=constants.STR_AO_INITIALIZATION,
        int_samples=constants.NUM_ACQ_SAMPLES,
        is_normalized=constants.IS_NORMALIZED_RESPONSE,
        str_mlm_method=constants.STR_MLM_METHOD,
        str_modelselection_method=constants.STR_MODELSELECTION_METHOD
    ):
        # TODO: is_normalized cases
        assert isinstance(X_train, np.ndarray)
        assert isinstance(Y_train, np.ndarray)
        assert isinstance(str_initial_method, str)
        assert isinstance(int_samples, int)
        assert isinstance(is_normalized, bool)
        assert isinstance(str_mlm_method, str)
        assert isinstance(str_modelselection_method, str)
        assert len(X_train.shape) == 2
        assert len(Y_train.shape) == 2
        assert Y_train.shape[1] == 1
        assert X_train.shape[0] == Y_train.shape[0]
        assert X_train.shape[1] == self.num_dim
        assert int_samples > 0
        assert str_initial_method in constants.ALLOWED_INITIALIZATIONS_AO
        assert str_mlm_method in constants.ALLOWED_MLM_METHOD
        assert str_modelselection_method in constants.ALLOWED_MODELSELECTION_METHOD

        time_start = time.time()

        if is_normalized:
            Y_train = (Y_train - np.min(Y_train)) / (np.max(Y_train) - np.min(Y_train)) * constants.MULTIPLIER_RESPONSE

        if str_mlm_method == 'regular':
            cov_X_X, inv_cov_X_X, hyps = gp.get_optimized_kernel(X_train, Y_train, self.prior_mu, self.str_cov, str_optimizer_method=self.str_optimizer_method_gp, str_modelselection_method=str_modelselection_method, debug=self.debug)
        elif str_mlm_method == 'converged':
            if self.is_optimize_hyps:
                cov_X_X, inv_cov_X_X, hyps = gp.get_optimized_kernel(X_train, Y_train, self.prior_mu, self.str_cov, str_optimizer_method=self.str_optimizer_method_gp, str_modelselection_method=str_modelselection_method, debug=self.debug)
                self.is_optimize_hyps = not _check_hyps_convergence(self.historical_hyps, hyps, self.str_cov, constants.IS_FIXED_GP_NOISE)
            # TODO: Can we test this else statement?
            else: # pragma: no cover
                print('[DEBUG] optimize in bo.py: hyps are converged.')
                hyps = self.historical_hyps[-1]
                cov_X_X, inv_cov_X_X = gp.get_kernel_inverse(X_train, hyps, self.str_cov, debug=self.debug)
        elif str_mlm_method == 'probabilistic': # pragma: no cover
            raise NotImplementedError('optimize: it will be added.')
        else: # pragma: no cover
            raise ValueError('optimize: missing condition for str_mlm_method.')

        self.historical_hyps.append(hyps)

        fun_acquisition = _choose_fun_acquisition(self.str_acq, hyps)
      
        fun_negative_acquisition = lambda X_test: -1.0 * constants.MULTIPLIER_ACQ * self._optimize_objective(fun_acquisition, X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps)
        next_point, next_points = self._optimize(fun_negative_acquisition, str_initial_method=str_initial_method, int_samples=int_samples)
        acquisitions = fun_negative_acquisition(next_points)

        time_end = time.time()

        if self.debug:
            print('[DEBUG] optimize in bo.py: time consumed', time_end - time_start, 'sec.')
        return next_point, next_points, acquisitions, cov_X_X, inv_cov_X_X, hyps
