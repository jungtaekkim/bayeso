# bo
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: September 16, 2020

import numpy as np
import time
import typing
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

from bayeso.gp import gp
from bayeso.gp import gp_common
from bayeso.utils import utils_bo
from bayeso.utils import utils_covariance
from bayeso.utils import utils_common
from bayeso.utils import utils_logger
from bayeso import constants

logger = utils_logger.get_logger('bo')


class BO(object):
    """
    It is a Bayesian optimization class.

    :param arr_range: a search space. Shape: (d, 2).
    :type arr_range: numpy.ndarray
    :param str_cov: the name of covariance function.
    :type str_cov: str., optional
    :param str_acq: the name of acquisition function.
    :type str_acq: str., optional
    :param normalize_Y: flag for normalizing outputs.
    :type normalize_Y: bool., optional
    :param use_ard: flag for automatic relevance determination.
    :type use_ard: bool., optional
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
        normalize_Y=constants.IS_NORMALIZED_RESPONSE,
        use_ard=True,
        prior_mu=None,
        str_optimizer_method_gp=constants.STR_OPTIMIZER_METHOD_GP,
        str_optimizer_method_bo=constants.STR_OPTIMIZER_METHOD_BO,
        str_modelselection_method=constants.STR_MODELSELECTION_METHOD,
        debug=False
    ):
        """
        Constructor method

        """

        # TODO: use use_ard.
        assert isinstance(arr_range, np.ndarray)
        assert isinstance(str_cov, str)
        assert isinstance(str_acq, str)
        assert isinstance(normalize_Y, bool)
        assert isinstance(use_ard, bool)
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
        self.use_ard = use_ard
        self.normalize_Y = normalize_Y
        self.str_optimizer_method_bo = utils_bo.check_optimizer_method_bo(str_optimizer_method_bo, arr_range.shape[0], debug)
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

        arr_initials = utils_common.get_grids(self.arr_range, int_grid)
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
        if self.debug: logger.debug('seed: {}'.format(int_seed))

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
            if self.debug: logger.debug('int_samples is ignored, because grid is chosen.')
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

        if self.debug: logger.debug('arr_initials:\n{}'.format(utils_logger.get_str_array(arr_initials)))

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
        pred_mean, pred_std, _ = gp.predict_test_(X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov=self.str_cov, prior_mu=self.prior_mu, debug=self.debug)
        # no matter which acquisition functions are given, we input pred_mean, pred_std, and Y_train.
        acquisitions = fun_acquisition(pred_mean=np.ravel(pred_mean), pred_std=np.ravel(pred_std), Y_train=Y_train)
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
                if self.debug: logger.debug('acquired sample: {}'.format(utils_logger.get_str_array(next_point_x)))
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
        next_point = utils_bo.get_best_acquisition(next_points, fun_negative_acquisition)[0]
        return next_point, next_points

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

        :returns: acquired example and dictionary of information. Shape: ((d, ), dict.).
        :rtype: (numpy.ndarray, dict.)

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

        if self.normalize_Y and not np.max(Y_train) == np.min(Y_train):
            Y_train = (Y_train - np.min(Y_train)) / (np.max(Y_train) - np.min(Y_train)) * constants.MULTIPLIER_RESPONSE

        time_start_gp = time.time()
        if str_mlm_method == 'regular':
            cov_X_X, inv_cov_X_X, hyps = gp.get_optimized_kernel(X_train, Y_train, self.prior_mu, self.str_cov, str_optimizer_method=self.str_optimizer_method_gp, str_modelselection_method=self.str_modelselection_method, debug=self.debug)
        elif str_mlm_method == 'converged':
            is_fixed_noise = constants.IS_FIXED_GP_NOISE

            if self.is_optimize_hyps:
                cov_X_X, inv_cov_X_X, hyps = gp.get_optimized_kernel(X_train, Y_train, self.prior_mu, self.str_cov, str_optimizer_method=self.str_optimizer_method_gp, str_modelselection_method=self.str_modelselection_method, debug=self.debug)
                self.is_optimize_hyps = not utils_bo.check_hyps_convergence(self.historical_hyps, hyps, self.str_cov, is_fixed_noise)
            else: # pragma: no cover
                if self.debug: logger.debug('hyps converged.')
                hyps = self.historical_hyps[-1]
                cov_X_X, inv_cov_X_X, _ = gp_common.get_kernel_inverse(X_train, hyps, self.str_cov, is_fixed_noise=is_fixed_noise, debug=self.debug)
        else: # pragma: no cover
            raise ValueError('optimize: missing condition for str_mlm_method.')
        time_end_gp = time.time()

        self.historical_hyps.append(hyps)

        fun_acquisition = utils_bo.choose_fun_acquisition(self.str_acq, hyps)

        time_start_acq = time.time()
        fun_negative_acquisition = lambda X_test: -1.0 * constants.MULTIPLIER_ACQ * self._optimize_objective(fun_acquisition, X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps)
        next_point, next_points = self._optimize(fun_negative_acquisition, str_initial_method=str_initial_method_ao, int_samples=int_samples)
        time_end_acq = time.time()

        acquisitions = fun_negative_acquisition(next_points)

        time_end = time.time()

        dict_info = {
            'next_points': next_points,
            'acquisitions': acquisitions,
            'cov_X_X': cov_X_X,
            'inv_cov_X_X': inv_cov_X_X,
            'hyps': hyps,
            'time_overall': time_end - time_start,
            'time_gp': time_end_gp - time_start_gp,
            'time_acq': time_end_acq - time_start_acq,
        }

        if self.debug: logger.debug('overall time consumed to acquire: {:.4f} sec.'.format(time_end - time_start))

        return next_point, dict_info
