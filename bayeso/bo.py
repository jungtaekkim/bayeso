# bo
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 04, 2020

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


def get_grids(arr_ranges, int_grids):
    """
    It returns grids of given `arr_ranges`, where each of dimension has `int_grids` partitions.

    :param arr_ranges: ranges. Shape: (d, 2).
    :type arr_ranges: numpy.ndarray
    :param int_grids: the number of partitions per dimension.
    :type int_grids: int.

    :returns: grids of given `arr_ranges`. Shape: (`int_grids`:math:`^{\\text{d}}`, d).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(arr_ranges, np.ndarray)
    assert isinstance(int_grids, int)
    assert len(arr_ranges.shape) == 2
    assert arr_ranges.shape[1] == 2
    assert (arr_ranges[:, 0] <= arr_ranges[:, 1]).all()

    list_grids = []
    for range_ in arr_ranges:
        list_grids.append(np.linspace(range_[0], range_[1], int_grids))
    list_grids_mesh = list(np.meshgrid(*list_grids))
    list_grids = []
    for elem in list_grids_mesh:
        list_grids.append(elem.flatten(order='C'))
    arr_grids = np.vstack(tuple(list_grids))
    arr_grids = arr_grids.T
    return arr_grids

def get_best_acquisition(arr_initials, fun_objective):
    """
    It returns the best example with respect to values of `fun_objective`.
    Here, the best acquisition is a minimizer of `fun_objective`.

    :param arr_initials: inputs. Shape: (n, d).
    :type arr_initials: numpy.ndarray
    :param fun_objective: an objective function.
    :type fun_objective: function

    :returns: the best example of `arr_initials`. Shape: (1, d).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

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
    """
    It checks the availability of optimization methods.
    It helps to run Bayesian optimization, even though additional optimization methods are not installed or there exist the conditions some of optimization methods cannot be run.

    :param str_optimizer_method_bo: the name of optimization method for Bayesian optimization.
    :type str_optimizer_method_bo: str.
    :param num_dim: dimensionality of the problem we solve.
    :type num_dim: int.
    :param debug: flag for printing log messages.
    :type debug: bool.

    :returns: available `str_optimizer_method_bo`.
    :rtype: str.

    :raises: AssertionError

    """

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
    """
    It chooses and returns an acquisition function.

    :param str_acq: the name of acquisition function.
    :type str_acq: str.
    :param hyps: dictionary of hyperparameters for acquisition function.
    :type hyps: dict.

    :returns: acquisition function.
    :rtype: function

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
        fun_acquisition = lambda pred_mean, pred_std, Y_train: acquisition.aei(pred_mean, pred_std, Y_train, hyps['noise'])
    elif str_acq == 'pure_exploit':
        fun_acquisition = acquisition.pure_exploit
    elif str_acq == 'pure_explore':
        fun_acquisition = acquisition.pure_explore
    else:
        raise NotImplementedError('_choose_fun_acquisition: allowed str_acq, but it is not implemented.')
    return fun_acquisition

def _check_hyps_convergence(list_hyps, hyps, str_cov, is_fixed_noise, ratio_threshold=0.05):
    """
    It checks convergence of hyperparameters for Gaussian process regression.

    :param list_hyps: list of historical hyperparameters for Gaussian process regression.
    :type list_hyps: list
    :param hyps: dictionary of hyperparameters for acquisition function.
    :type hyps: dict.
    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param is_fixed_noise: flag for fixing a noise.
    :type is_fixed_noise: bool.
    :param ratio_threshold: ratio of threshold for checking convergence.
    :type ratio_threshold: float, optional

    :returns: flag for checking convergence. If converged, it is True.
    :rtype: bool.

    :raises: AssertionError

    """

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
class BO(object):
    """
    It is a Bayesian optimization class.

    :param arr_range: a search space. Shape: (d, 2).
    :type arr_range: numpy.ndarray
    :param str_cov: the name of covariance function.
    :type str_cov: str., optional
    :param str_acq: the name of acquisition function.
    :type str_acq: str., optional
    :param is_normalized: flag for normalizing outputs.
    :type is_normalized: bool., optional
    :param is_ard: flag for automatic relevance determination.
    :type is_ard: bool., optional
    :param prior_mu: None, or prior mean function.
    :type prior_mu: NoneType, or function, optional
    :param str_optimizer_method_gp: the name of optimization method for Gaussian process regression.
    :type str_optimizer_method_gp: str., optional
    :param str_optimizer_method_bo: the name of optimization method for Bayesian optimization.
    :type str_optimizer_method_bo: str., optional
    :param str_modelselection_method: the name of model selection method for Gaussian process regression.
    :type str_modelselection_method: str., optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    """

    def __init__(self, arr_range,
        str_cov=constants.STR_GP_COV,
        str_acq=constants.STR_BO_ACQ,
        is_normalized=constants.IS_NORMALIZED_RESPONSE,
        is_ard=True,
        prior_mu=None,
        str_optimizer_method_gp=constants.STR_OPTIMIZER_METHOD_GP,
        str_optimizer_method_bo=constants.STR_OPTIMIZER_METHOD_BO,
        str_modelselection_method=constants.STR_MODELSELECTION_METHOD,
        debug=False
    ):
        """
        Constructor method

        """

        # TODO: use is_ard.
        assert isinstance(arr_range, np.ndarray)
        assert isinstance(str_cov, str)
        assert isinstance(str_acq, str)
        assert isinstance(is_normalized, bool)
        assert isinstance(is_ard, bool)
        assert isinstance(str_optimizer_method_bo, str)
        assert isinstance(str_optimizer_method_gp, str)
        assert isinstance(str_modelselection_method, str)
        assert isinstance(debug, bool)
        assert callable(prior_mu) or prior_mu is None
        assert len(arr_range.shape) == 2
        assert arr_range.shape[1] == 2
        assert (arr_range[:, 0] <= arr_range[:, 1]).all()
        assert str_cov in constants.ALLOWED_GP_COV
        assert str_acq in constants.ALLOWED_BO_ACQ
        assert str_optimizer_method_gp in constants.ALLOWED_OPTIMIZER_METHOD_GP
        assert str_optimizer_method_bo in constants.ALLOWED_OPTIMIZER_METHOD_BO
        assert str_modelselection_method in constants.ALLOWED_MODELSELECTION_METHOD

        self.arr_range = arr_range
        self.num_dim = arr_range.shape[0]
        self.str_cov = str_cov
        self.str_acq = str_acq
        self.is_ard = is_ard
        self.is_normalized = is_normalized
        self.str_optimizer_method_bo = _check_optimizer_method_bo(str_optimizer_method_bo, arr_range.shape[0], debug)
        self.str_optimizer_method_gp = str_optimizer_method_gp
        self.str_modelselection_method = str_modelselection_method
        self.debug = debug
        self.prior_mu = prior_mu

        self.is_optimize_hyps = True
        self.historical_hyps = []

    def _get_initial_grid(self, int_grid=constants.NUM_BO_GRID):
        """
        It returns grids of `self.arr_range`.

        :param int_grid: the number of grids.
        :type int_grid: int., optional

        :returns: grids of `self.arr_range`. Shape: (`int_grid`:math:`^{\\text{d}}`, d).
        :rtype: numpy.ndarray

        :raises: AssertionError

        """

        assert isinstance(int_grid, int)

        arr_initials = get_grids(self.arr_range, int_grid)
        return arr_initials

    def _get_initial_uniform(self, int_samples, int_seed=None):
        """
        It returns `int_samples` examples uniformly sampled.

        :param int_samples: the number of samples.
        :type int_samples: int.
        :param int_seed: None, or random seed.
        :type int_seed: NoneType or int., optional

        :returns: random examples. Shape: (`int_samples`, d).
        :rtype: numpy.ndarray

        :raises: AssertionError

        """

        assert isinstance(int_samples, int)
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

    # TODO: noise should be added.
    def _get_initial_sobol(self, int_samples, int_seed=None):
        """
        It returns `int_samples` examples sampled from Sobol sequence.

        :param int_samples: the number of samples.
        :type int_samples: int.
        :param int_seed: None, or random seed.
        :type int_seed: NoneType or int., optional

        :returns: examples sampled from Sobol sequence. Shape: (`int_samples`, d).
        :rtype: numpy.ndarray

        :raises: AssertionError

        """

        assert isinstance(int_samples, int)
        assert isinstance(int_seed, int) or int_seed is None

        if int_seed is None:
            int_seed = np.random.randint(0, 10000)
        if self.debug:
            print('[DEBUG] _get_initial_sobol in bo.py: int_seed', int_seed)
        arr_samples = sobol_seq.i4_sobol_generate(self.num_dim, int_samples, int_seed)
        arr_samples = arr_samples * (self.arr_range[:, 1].flatten() - self.arr_range[:, 0].flatten()) + self.arr_range[:, 0].flatten()
        return arr_samples

    def _get_initial_latin(self, int_samples):
        """
        It returns `int_samples` examples sampled from Latin hypercube.

        :param int_samples: the number of samples.
        :type int_samples: int.

        :returns: examples sampled from Latin hypercube. Shape: (`int_samples`, d).
        :rtype: numpy.ndarray

        :raises: AssertionError

        """

        raise NotImplementedError('_get_initial_latin in bo.py')

    # TODO: int_grid should be able to be input.
    # TODO: change this method name. More general name is needed like get_samples.
    def get_initial(self, str_initial_method,
        fun_objective=None,
        int_samples=constants.NUM_ACQ_SAMPLES,
        int_seed=None,
    ):
        """
        It returns a single example or `int_samples` examples, sampled by a certian method `str_initial_method`.

        :param str_initial_method: the name of sampling method.
        :type str_initial_method: str.
        :param fun_objective: None, or objective function.
        :type fun_objective: NoneType or function, optional
        :param int_samples: the number of samples.
        :type int_samples: int., optional
        :param int_seed: None, or random seed.
        :type int_seed: NoneType or int., optional

        :returns: sampled examples. Shape: (1, d) or (`int_samples`, d).
        :rtype: numpy.ndarray

        :raises: AssertionError

        """

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
        """
        It returns acquisition function values over `X_test`.

        :param fun_acquisition: acquisition function.
        :type fun_acquisition: function
        :param X_train: inputs. Shape: (n, d) or (n, m, d).
        :type X_train: numpy.ndarray
        :param Y_train: outputs. Shape: (n, 1).
        :type Y_train: numpy.ndarray
        :param X_test: inputs. Shape: (l, d) or (l, m, d).
        :type X_test: numpy.ndarray
        :param cov_X_X: kernel matrix over `X_train`. Shape: (n, n).
        :type cov_X_X: numpy.ndarray
        :param inv_cov_X_X: kernel matrix inverse over `X_train`. Shape: (n, n).
        :type inv_cov_X_X: numpy.ndarray
        :param hyps: dictionary of hyperparameters for Gaussian process.
        :type hyps: dict.

        :returns: acquisition function values over `X_test`. Shape: (l, ).
        :rtype: numpy.ndarray

        """

        X_test = np.atleast_2d(X_test)
        pred_mean, pred_std = gp.predict_test_(X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov=self.str_cov, prior_mu=self.prior_mu, debug=self.debug)
        # no matter which acquisition functions are given, we input pred_mean, pred_std, and Y_train.
        acquisitions = fun_acquisition(pred_mean=pred_mean.flatten(), pred_std=pred_std.flatten(), Y_train=Y_train)
        return acquisitions

    def _get_bounds(self):
        """
        It returns list of range tuples, obtained from `self.arr_range`.

        :returns: list of range tuples.
        :rtype: list

        """

        list_bounds = []
        for elem in self.arr_range:
            list_bounds.append(tuple(elem))
        return list_bounds

    def _optimize(self, fun_negative_acquisition, str_initial_method, int_samples):
        """
        It optimizes `fun_negative_function` with `self.str_optimizer_method_bo`.
        `int_samples` examples are determined by `str_initial_method`, to start acquisition function optimization.

        :param fun_objective: negative acquisition function.
        :type fun_objective: function
        :param str_initial_method: the name of sampling method.
        :type str_initial_method: str.
        :param int_samples: the number of samples.
        :type int_samples: int.

        :returns: tuple of next point to evaluate and all candidates determined by acquisition function optimization. Shape: ((d, ), (`int_samples`, d)).
        :rtype: (numpy.ndarray, numpy.ndarray)

        """

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
                maxf=88888,
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
            cur_sigma0 = np.mean(list_bounds[:, 1] - list_bounds[:, 0]) / 4.0
            next_point_x = cma.fmin(fun_wrapper(fun_negative_acquisition), arr_initials[0], cur_sigma0, options={'bounds': [list_bounds[:, 0], list_bounds[:, 1]], 'verbose': -1, 'maxfevals': 1e5})[0]
            list_next_point.append(next_point_x)

        next_points = np.array(list_next_point)
        next_point = get_best_acquisition(next_points, fun_negative_acquisition)
        return next_point.flatten(), next_points

    def optimize(self, X_train, Y_train,
        str_initial_method_ao=constants.STR_AO_INITIALIZATION,
        int_samples=constants.NUM_ACQ_SAMPLES,
        str_mlm_method=constants.STR_MLM_METHOD,
    ):
        """
        It computes acquired example, candidates of acquired examples, acquisition function values for the candidates, covariance matrix, inverse matrix of the covariance matrix, hyperparameters optimized, and execution times.

        :param X_train: inputs. Shape: (n, d) or (n, m, d).
        :type X_train: numpy.ndarray
        :param Y_train: outputs. Shape: (n, 1).
        :type Y_train: numpy.ndarray
        :param str_initial_method_ao: the name of initialization method for acquisition function optimization.
        :type str_initial_method_ao: str., optional
        :param int_samples: the number of samples.
        :type int_samples: int., optional
        :param str_mlm_method: the name of marginal likelihood maximization method for Gaussian process regression.
        :type str_mlm_method: str., optional

        :returns: acquired example, candidates of acquired examples, acquisition function values over the candidates, covariance matrix by `hyps`, inverse matrix of the covariance matrix, hyperparameters optimized, and execution times. Shape: ((d, ), (`int_samples`, d), (`int_samples`, ), (n, n), (n, n), dict., dict.).
        :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, dict., dict.)

        :raises: AssertionError

        """

        assert isinstance(X_train, np.ndarray)
        assert isinstance(Y_train, np.ndarray)
        assert isinstance(str_initial_method_ao, str)
        assert isinstance(int_samples, int)
        assert isinstance(str_mlm_method, str)
        assert len(X_train.shape) == 2
        assert len(Y_train.shape) == 2
        assert Y_train.shape[1] == 1
        assert X_train.shape[0] == Y_train.shape[0]
        assert X_train.shape[1] == self.num_dim
        assert int_samples > 0
        assert str_initial_method_ao in constants.ALLOWED_INITIALIZATIONS_AO
        assert str_mlm_method in constants.ALLOWED_MLM_METHOD

        time_start = time.time()

        if self.is_normalized:
            Y_train = (Y_train - np.min(Y_train)) / (np.max(Y_train) - np.min(Y_train)) * constants.MULTIPLIER_RESPONSE

        time_start_gp = time.time()
        if str_mlm_method == 'regular':
            cov_X_X, inv_cov_X_X, hyps = gp.get_optimized_kernel(X_train, Y_train, self.prior_mu, self.str_cov, str_optimizer_method=self.str_optimizer_method_gp, str_modelselection_method=self.str_modelselection_method, debug=self.debug)
        elif str_mlm_method == 'converged':
            is_fixed_noise = constants.IS_FIXED_GP_NOISE

            if self.is_optimize_hyps:
                cov_X_X, inv_cov_X_X, hyps = gp.get_optimized_kernel(X_train, Y_train, self.prior_mu, self.str_cov, str_optimizer_method=self.str_optimizer_method_gp, str_modelselection_method=self.str_modelselection_method, debug=self.debug)
                self.is_optimize_hyps = not _check_hyps_convergence(self.historical_hyps, hyps, self.str_cov, is_fixed_noise)
            else: # pragma: no cover
                print('[DEBUG] optimize in bo.py: hyps are converged.')
                hyps = self.historical_hyps[-1]
                cov_X_X, inv_cov_X_X, _ = gp.get_kernel_inverse(X_train, hyps, self.str_cov, is_fixed_noise=is_fixed_noise, debug=self.debug)
        elif str_mlm_method == 'probabilistic': # pragma: no cover
            raise NotImplementedError('optimize: it will be added.')
        else: # pragma: no cover
            raise ValueError('optimize: missing condition for str_mlm_method.')
        time_end_gp = time.time()

        self.historical_hyps.append(hyps)

        fun_acquisition = _choose_fun_acquisition(self.str_acq, hyps)

        time_start_acq = time.time()
        fun_negative_acquisition = lambda X_test: -1.0 * constants.MULTIPLIER_ACQ * self._optimize_objective(fun_acquisition, X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps)
        next_point, next_points = self._optimize(fun_negative_acquisition, str_initial_method=str_initial_method_ao, int_samples=int_samples)
        time_end_acq = time.time()

        acquisitions = fun_negative_acquisition(next_points)

        time_end = time.time()

        times = {
            'overall': time_end - time_start,
            'gp': time_end_gp - time_start_gp,
            'acq': time_end_acq - time_start_acq,
        }

        if self.debug:
            print('[DEBUG] optimize in bo.py: time consumed', time_end - time_start, 'sec.')

        return next_point, next_points, acquisitions, cov_X_X, inv_cov_X_X, hyps, times
