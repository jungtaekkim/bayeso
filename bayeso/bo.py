#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: December 29, 2020
#
"""It defines a class of Bayesian optimization."""

import time
import numpy as np
from scipy.optimize import minimize
try:
    from scipydirect import minimize as directminimize
except: # pragma: no cover
    directminimize = None
try:
    import cma
except: # pragma: no cover
    cma = None
import qmcpy

from bayeso import covariance
from bayeso import constants
from bayeso.gp import gp
from bayeso.gp import gp_kernel
from bayeso.utils import utils_bo
from bayeso.utils import utils_common
from bayeso.utils import utils_logger

logger = utils_logger.get_logger('bo')


class BO:
    """
    It is a Bayesian optimization class.

    :param range_X: a search space. Shape: (d, 2).
    :type range_X: numpy.ndarray
    :param str_cov: the name of covariance function.
    :type str_cov: str., optional
    :param str_acq: the name of acquisition function.
    :type str_acq: str., optional
    :param normalize_Y: flag for normalizing outputs.
    :type normalize_Y: bool., optional
    :param use_ard: flag for automatic relevance determination.
    :type use_ard: bool., optional
    :param prior_mu: None, or prior mean function.
    :type prior_mu: NoneType, or callable, optional
    :param str_surrogate: the name of surrogate model.
    :type str_surrogate: str., optional
    :param str_optimizer_method_gp: the name of optimization method for
        Gaussian process regression.
    :type str_optimizer_method_gp: str., optional
    :param str_optimizer_method_bo: the name of optimization method for
        Bayesian optimization.
    :type str_optimizer_method_bo: str., optional
    :param str_modelselection_method: the name of model selection method
        for Gaussian process regression.
    :type str_modelselection_method: str., optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    """

    def __init__(self, range_X: np.ndarray,
        str_cov: str=constants.STR_COV,
        str_acq: str=constants.STR_BO_ACQ,
        normalize_Y: bool=constants.NORMALIZE_RESPONSE,
        use_ard: bool=constants.USE_ARD,
        prior_mu: constants.TYPING_UNION_CALLABLE_NONE=None,
        str_surrogate: str=constants.STR_SURROGATE,
        str_optimizer_method_gp: str=constants.STR_OPTIMIZER_METHOD_GP,
        str_optimizer_method_bo: str=constants.STR_OPTIMIZER_METHOD_AO,
        str_modelselection_method: str=constants.STR_MODELSELECTION_METHOD,
        debug: bool=False
    ):
        """
        Constructor method

        """

        assert isinstance(range_X, np.ndarray)
        assert isinstance(str_cov, str)
        assert isinstance(str_acq, str)
        assert isinstance(normalize_Y, bool)
        assert isinstance(use_ard, bool)
        assert isinstance(str_optimizer_method_bo, str)
        assert isinstance(str_optimizer_method_gp, str)
        assert isinstance(str_modelselection_method, str)
        assert isinstance(debug, bool)
        assert callable(prior_mu) or prior_mu is None
        assert len(range_X.shape) == 2
        assert range_X.shape[1] == 2
        assert (range_X[:, 0] <= range_X[:, 1]).all()
        assert str_cov in constants.ALLOWED_COV
        assert str_acq in constants.ALLOWED_BO_ACQ
        assert str_optimizer_method_gp in constants.ALLOWED_OPTIMIZER_METHOD_GP
        assert str_optimizer_method_bo in constants.ALLOWED_OPTIMIZER_METHOD_BO
        assert str_modelselection_method in constants.ALLOWED_MODELSELECTION_METHOD
        assert str_surrogate in constants.ALLOWED_SURROGATE

        self.range_X = range_X
        self.num_dim = range_X.shape[0]
        self.str_cov = str_cov
        self.str_acq = str_acq
        self.use_ard = use_ard
        self.normalize_Y = normalize_Y
        self.str_optimizer_method_bo = utils_bo.check_optimizer_method_bo(
            str_optimizer_method_bo, range_X.shape[0], debug)
        self.str_optimizer_method_gp = str_optimizer_method_gp
        self.str_modelselection_method = str_modelselection_method
        self.debug = debug
        self.prior_mu = prior_mu

        self.is_optimize_hyps = True
        self.historical_hyps = []

    def _get_samples_grid(self, num_grids: int=constants.NUM_GRIDS_AO) -> np.ndarray:
        """
        It returns grids of `self.range_X`.

        :param num_grids: the number of grids.
        :type num_grids: int., optional

        :returns: grids of `self.range_X`. Shape: (`num_grids`:math:`^{\\text{d}}`, d).
        :rtype: numpy.ndarray

        :raises: AssertionError

        """

        assert isinstance(num_grids, int)

        initials = utils_common.get_grids(self.range_X, num_grids)
        return initials

    def _get_samples_uniform(self, num_samples: int,
        seed: constants.TYPING_UNION_INT_NONE=None
    ) -> np.ndarray:
        """
        It returns `num_samples` examples uniformly sampled.

        :param num_samples: the number of samples.
        :type num_samples: int.
        :param seed: None, or random seed.
        :type seed: NoneType or int., optional

        :returns: random examples. Shape: (`num_samples`, d).
        :rtype: numpy.ndarray

        :raises: AssertionError

        """

        assert isinstance(num_samples, int)
        assert isinstance(seed, (int, type(None)))

        if seed is not None:
            state_random = np.random.RandomState(seed)
        else:
            state_random = np.random.RandomState()

        list_initials = []
        for _ in range(0, num_samples):
            list_initial = []
            for elem in self.range_X:
                list_initial.append(state_random.uniform(elem[0], elem[1]))
            list_initials.append(np.array(list_initial))
        initials = np.array(list_initials)
        return initials

    def _get_samples_gaussian(self, num_samples: int,
        seed: constants.TYPING_UNION_INT_NONE=None
    ) -> np.ndarray:
        """
        It returns `num_samples` examples sampled from Gaussian distribution.

        :param num_samples: the number of samples.
        :type num_samples: int.
        :param seed: None, or random seed.
        :type seed: NoneType or int., optional

        :returns: random examples. Shape: (`num_samples`, d).
        :rtype: numpy.ndarray

        :raises: AssertionError

        """

        assert isinstance(num_samples, int)
        assert isinstance(seed, (int, type(None)))

        if seed is not None:
            state_random = np.random.RandomState(seed)
        else:
            state_random = np.random.RandomState()

        list_initials = []

        for _ in range(0, num_samples):
            list_initial = []
            for elem in self.range_X:
                new_mean = (elem[1] + elem[0]) / 2.0
                new_std = (elem[1] - elem[0]) / 4.0

                cur_sample = state_random.randn() * new_std + new_mean
                cur_sample = np.clip(cur_sample, elem[0], elem[1])

                list_initial.append(cur_sample)
            list_initials.append(np.array(list_initial))
        initials = np.array(list_initials)
        return initials

    def _get_samples_sobol(self, num_samples: int,
        seed: constants.TYPING_UNION_INT_NONE=None
    ) -> np.ndarray:
        """
        It returns `num_samples` examples sampled from Sobol' sequence.

        :param num_samples: the number of samples.
        :type num_samples: int.
        :param seed: None, or random seed.
        :type seed: NoneType or int., optional

        :returns: examples sampled from Sobol' sequence. Shape: (`num_samples`, d).
        :rtype: numpy.ndarray

        :raises: AssertionError

        """

        assert isinstance(num_samples, int)
        assert isinstance(seed, (int, type(None)))

        sampler = qmcpy.Sobol(self.num_dim, seed=seed, graycode=True)
        samples = sampler.gen_samples(num_samples)

        samples = samples * (self.range_X[:, 1].flatten() - self.range_X[:, 0].flatten()) \
            + self.range_X[:, 0].flatten()
        return samples

    def _get_samples_halton(self, num_samples: int,
        seed: constants.TYPING_UNION_INT_NONE=None
    ) -> np.ndarray:
        """
        It returns `num_samples` examples sampled by Halton algorithm.

        :param num_samples: the number of samples.
        :type num_samples: int.
        :param seed: None, or random seed.
        :type seed: NoneType or int., optional

        :returns: examples sampled by Halton algorithm. Shape: (`num_samples`, d).
        :rtype: numpy.ndarray

        :raises: AssertionError

        """

        assert isinstance(num_samples, int)
        assert isinstance(seed, (int, type(None)))

        sampler = qmcpy.Halton(self.num_dim, randomize='OWEN', seed=seed)
        samples = sampler.gen_samples(num_samples)

        samples = samples * (self.range_X[:, 1].flatten() - self.range_X[:, 0].flatten()) \
            + self.range_X[:, 0].flatten()
        return samples

    def get_samples(self, str_sampling_method: str,
        fun_objective: constants.TYPING_UNION_CALLABLE_NONE=None,
        num_samples: int=constants.NUM_SAMPLES_AO,
        seed: constants.TYPING_UNION_INT_NONE=None,
    ) -> np.ndarray:
        """
        It returns `num_samples` examples, sampled by a sampling method `str_sampling_method`.

        :param str_sampling_method: the name of sampling method.
        :type str_sampling_method: str.
        :param fun_objective: None, or objective function.
        :type fun_objective: NoneType or callable, optional
        :param num_samples: the number of samples.
        :type num_samples: int., optional
        :param seed: None, or random seed.
        :type seed: NoneType or int., optional

        :returns: sampled examples. Shape: (`num_samples`, d).
        :rtype: numpy.ndarray

        :raises: AssertionError

        """

        assert isinstance(str_sampling_method, str)
        assert callable(fun_objective) or fun_objective is None
        assert isinstance(num_samples, int)
        assert isinstance(seed, (int, type(None)))
        assert str_sampling_method in constants.ALLOWED_SAMPLING_METHOD

        if str_sampling_method == 'grid':
            assert fun_objective is not None
            if self.debug:
                logger.debug('For this option, num_samples is used as num_grids.')
            samples = self._get_samples_grid(num_grids=num_samples)
            samples = utils_bo.get_best_acquisition_by_evaluation(samples, fun_objective)
        elif str_sampling_method == 'uniform':
            samples = self._get_samples_uniform(num_samples, seed=seed)
        elif str_sampling_method == 'gaussian':
            samples = self._get_samples_gaussian(num_samples, seed=seed)
        elif str_sampling_method == 'sobol':
            samples = self._get_samples_sobol(num_samples, seed=seed)
        elif str_sampling_method == 'halton':
            samples = self._get_samples_halton(num_samples, seed=seed)
        else:
            raise NotImplementedError('get_samples: allowed str_sampling_method,\
                but it is not implemented.')

        if self.debug:
            logger.debug('samples:\n%s', utils_logger.get_str_array(samples))

        return samples

    def get_initials(self, str_initial_method: str, num_initials: int,
        seed: constants.TYPING_UNION_INT_NONE=None,
    ) -> np.ndarray:
        """
        It returns `num_initials` examples, sampled by a sampling method `str_initial_method`.

        :param str_initial_method: the name of sampling method.
        :type str_initial_method: str.
        :param num_initials: the number of samples.
        :type num_initials: int.
        :param seed: None, or random seed.
        :type seed: NoneType or int., optional

        :returns: sampled examples. Shape: (`num_samples`, d).
        :rtype: numpy.ndarray

        :raises: AssertionError

        """

        assert isinstance(str_initial_method, str)
        assert isinstance(num_initials, int)
        assert isinstance(seed, (int, type(None)))
        assert str_initial_method in constants.ALLOWED_INITIALIZING_METHOD_BO

        return self.get_samples(str_initial_method, num_samples=num_initials, seed=seed)

    def compute_acquisitions(self, X: np.ndarray,
        X_train: np.ndarray, Y_train: np.ndarray,
        cov_X_X: np.ndarray, inv_cov_X_X: np.ndarray, hyps: dict
    ) -> np.ndarray:
        """
        It computes acquisition function values over 'X',
        where `X_train`, `Y_train`, `cov_X_X`, `inv_cov_X_X`, and `hyps`
        are given.

        :param X: inputs. Shape: (l, d) or (l, m, d).
        :type X: numpy.ndarray
        :param X_train: inputs. Shape: (n, d) or (n, m, d).
        :type X_train: numpy.ndarray
        :param Y_train: outputs. Shape: (n, 1).
        :type Y_train: numpy.ndarray
        :param cov_X_X: kernel matrix over `X_train`. Shape: (n, n).
        :type cov_X_X: numpy.ndarray
        :param inv_cov_X_X: kernel matrix inverse over `X_train`. Shape: (n, n).
        :type inv_cov_X_X: numpy.ndarray
        :param hyps: dictionary of hyperparameters.
        :type hyps: dict.

        :returns: acquisition function values over `X`. Shape: (l, ).
        :rtype: numpy.ndarray

        """

        assert isinstance(X, np.ndarray)
        assert isinstance(X_train, np.ndarray)
        assert isinstance(Y_train, np.ndarray)
        assert len(X.shape) == 2 or len(X.shape) == 3
        assert len(X_train.shape) == 2 or len(X_train.shape) == 3
        assert len(Y_train.shape) == 2
        assert Y_train.shape[1] == 1
        assert X_train.shape[0] == Y_train.shape[0]
        if len(X_train.shape) == 2:
            assert X.shape[1] == X_train.shape[1] == self.num_dim
        else:
            assert X.shape[2] == X_train.shape[2] == self.num_dim

        fun_acquisition = utils_bo.choose_fun_acquisition(self.str_acq, hyps)

        acquisitions = constants.MULTIPLIER_ACQ * self._optimize_objective(
            fun_acquisition, X_train, Y_train,
            X, cov_X_X, inv_cov_X_X, hyps
        )

        assert isinstance(acquisitions, np.ndarray)
        assert len(acquisitions.shape) == 1
        assert X.shape[0] == acquisitions.shape[0]
        return acquisitions

    def _optimize_objective(self, fun_acquisition: constants.TYPING_CALLABLE,
        X_train: np.ndarray, Y_train: np.ndarray,
        X_test: np.ndarray, cov_X_X: np.ndarray,
        inv_cov_X_X: np.ndarray, hyps: dict
    ) -> np.ndarray:
        """
        It returns acquisition function values over `X_test`.

        :param fun_acquisition: acquisition function.
        :type fun_acquisition: callable
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
        pred_mean, pred_std, _ = gp.predict_with_cov(X_train, Y_train, X_test,
            cov_X_X, inv_cov_X_X, hyps, str_cov=self.str_cov,
            prior_mu=self.prior_mu, debug=self.debug)

        acquisitions = fun_acquisition(pred_mean=np.ravel(pred_mean),
            pred_std=np.ravel(pred_std), Y_train=Y_train)
        return acquisitions

    def _get_bounds(self) -> constants.TYPING_LIST:
        """
        It returns list of range tuples, obtained from `self.range_X`.

        :returns: list of range tuples.
        :rtype: list

        """

        list_bounds = []
        for elem in self.range_X:
            list_bounds.append(tuple(elem))
        return list_bounds

    def _optimize(self, fun_negative_acquisition: constants.TYPING_CALLABLE,
        str_sampling_method: str,
        num_samples: int
    ) -> constants.TYPING_TUPLE_TWO_ARRAYS:
        """
        It optimizes `fun_negative_function` with `self.str_optimizer_method_bo`.
        `num_samples` examples are determined by `str_sampling_method`, to
        start acquisition function optimization.

        :param fun_objective: negative acquisition function.
        :type fun_objective: callable
        :param str_sampling_method: the name of sampling method.
        :type str_sampling_method: str.
        :param num_samples: the number of samples.
        :type num_samples: int.

        :returns: tuple of next point to evaluate and all candidates
            determined by acquisition function optimization.
            Shape: ((d, ), (`num_samples`, d)).
        :rtype: (numpy.ndarray, numpy.ndarray)

        """

        list_next_point = []
        if self.str_optimizer_method_bo == 'L-BFGS-B':
            list_bounds = self._get_bounds()
            initials = self.get_samples(str_sampling_method,
                fun_objective=fun_negative_acquisition,
                num_samples=num_samples)

            for arr_initial in initials:
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
                    logger.debug('acquired sample: %s', utils_logger.get_str_array(next_point_x))
        elif self.str_optimizer_method_bo == 'DIRECT': # pragma: no cover
            logger.debug('num_samples is ignored.')

            list_bounds = self._get_bounds()
            next_point = directminimize(
                fun_negative_acquisition,
                bounds=list_bounds,
                maxf=88888,
            )
            next_point_x = next_point.x
            list_next_point.append(next_point_x)
        elif self.str_optimizer_method_bo == 'CMA-ES':
            logger.debug('num_samples is ignored.')

            list_bounds = self._get_bounds()
            list_bounds = np.array(list_bounds)
            def fun_wrapper(f):
                def g(bx):
                    return f(bx)[0]
                return g
            initials = self.get_samples(str_sampling_method,
                fun_objective=fun_negative_acquisition, num_samples=1)
            cur_sigma0 = np.mean(list_bounds[:, 1] - list_bounds[:, 0]) / 4.0
            next_point_x = cma.fmin(fun_wrapper(fun_negative_acquisition),
                initials[0], cur_sigma0,
                options={
                    'bounds': [list_bounds[:, 0], list_bounds[:, 1]],
                    'verbose': -1, 'maxfevals': 1e5
                })[0]
            list_next_point.append(next_point_x)

        next_points = np.array(list_next_point)
        next_point = utils_bo.get_best_acquisition_by_evaluation(
            next_points, fun_negative_acquisition)[0]
        return next_point, next_points

    def optimize(self, X_train: np.ndarray, Y_train: np.ndarray,
        str_sampling_method: str=constants.STR_SAMPLING_METHOD_AO,
        num_samples: int=constants.NUM_SAMPLES_AO,
        str_mlm_method: str=constants.STR_MLM_METHOD,
    ) -> constants.TYPING_TUPLE_ARRAY_DICT:
        """
        It computes acquired example, candidates of acquired examples,
        acquisition function values for the candidates, covariance matrix,
        inverse matrix of the covariance matrix, hyperparameters optimized,
        and execution times.

        :param X_train: inputs. Shape: (n, d) or (n, m, d).
        :type X_train: numpy.ndarray
        :param Y_train: outputs. Shape: (n, 1).
        :type Y_train: numpy.ndarray
        :param str_sampling_method: the name of sampling method for
            acquisition function optimization.
        :type str_sampling_method: str., optional
        :param num_samples: the number of samples.
        :type num_samples: int., optional
        :param str_mlm_method: the name of marginal likelihood maximization
            method for Gaussian process regression.
        :type str_mlm_method: str., optional

        :returns: acquired example and dictionary of information. Shape: ((d, ), dict.).
        :rtype: (numpy.ndarray, dict.)

        :raises: AssertionError

        """

        assert isinstance(X_train, np.ndarray)
        assert isinstance(Y_train, np.ndarray)
        assert isinstance(str_sampling_method, str)
        assert isinstance(num_samples, int)
        assert isinstance(str_mlm_method, str)
        assert len(X_train.shape) == 2
        assert len(Y_train.shape) == 2
        assert Y_train.shape[1] == 1
        assert X_train.shape[0] == Y_train.shape[0]
        assert X_train.shape[1] == self.num_dim
        assert num_samples > 0
        assert str_sampling_method in constants.ALLOWED_SAMPLING_METHOD
        assert str_mlm_method in constants.ALLOWED_MLM_METHOD

        time_start = time.time()

        if self.normalize_Y and np.max(Y_train) != np.min(Y_train):
            Y_train = (Y_train - np.min(Y_train)) / (np.max(Y_train) - np.min(Y_train)) \
                * constants.MULTIPLIER_RESPONSE

        time_start_gp = time.time()
        if str_mlm_method == 'regular':
            cov_X_X, inv_cov_X_X, hyps = gp_kernel.get_optimized_kernel(
                X_train, Y_train,
                self.prior_mu, self.str_cov,
                str_optimizer_method=self.str_optimizer_method_gp,
                str_modelselection_method=self.str_modelselection_method,
                use_ard=self.use_ard,
                debug=self.debug
            )
        elif str_mlm_method == 'combined':
            from bayeso.gp import gp_likelihood
            from bayeso.utils import utils_gp
            from bayeso.utils import utils_covariance

            prior_mu_train = utils_gp.get_prior_mu(self.prior_mu, X_train)

            neg_log_ml_best = np.inf
            cov_X_X_best = None
            inv_cov_X_X_best = None
            hyps_best = None

            for cur_str_optimizer_method in ['BFGS', 'Nelder-Mead']:
                cov_X_X, inv_cov_X_X, hyps = gp_kernel.get_optimized_kernel(
                    X_train, Y_train,
                    self.prior_mu, self.str_cov,
                    str_optimizer_method=cur_str_optimizer_method,
                    str_modelselection_method=self.str_modelselection_method,
                    use_ard=self.use_ard,
                    debug=self.debug
                )
                cur_neg_log_ml_ = gp_likelihood.neg_log_ml(X_train, Y_train,
                    utils_covariance.convert_hyps(self.str_cov, hyps,
                        fix_noise=constants.FIX_GP_NOISE),
                    self.str_cov, prior_mu_train,
                    use_ard=self.use_ard, fix_noise=constants.FIX_GP_NOISE,
                    use_gradient=False, debug=self.debug)

                if cur_neg_log_ml_ < neg_log_ml_best:
                    neg_log_ml_best = cur_neg_log_ml_
                    cov_X_X_best = cov_X_X
                    inv_cov_X_X_best = inv_cov_X_X
                    hyps_best = hyps

            cov_X_X = cov_X_X_best
            inv_cov_X_X = inv_cov_X_X_best
            hyps = hyps_best
        elif str_mlm_method == 'converged':
            fix_noise = constants.FIX_GP_NOISE

            if self.is_optimize_hyps:
                cov_X_X, inv_cov_X_X, hyps = gp_kernel.get_optimized_kernel(
                    X_train, Y_train,
                    self.prior_mu, self.str_cov,
                    str_optimizer_method=self.str_optimizer_method_gp,
                    str_modelselection_method=self.str_modelselection_method,
                    use_ard=self.use_ard,
                    debug=self.debug
                )

                self.is_optimize_hyps = not utils_bo.check_hyps_convergence(self.historical_hyps,
                    hyps, self.str_cov, fix_noise)
            else: # pragma: no cover
                if self.debug:
                    logger.debug('hyps converged.')
                hyps = self.historical_hyps[-1]
                cov_X_X, inv_cov_X_X, _ = covariance.get_kernel_inverse(X_train, hyps,
                    self.str_cov, fix_noise=fix_noise, debug=self.debug)
        else: # pragma: no cover
            raise ValueError('optimize: missing condition for str_mlm_method.')
        time_end_gp = time.time()

        self.historical_hyps.append(hyps)

        fun_acquisition = utils_bo.choose_fun_acquisition(self.str_acq, hyps)

        time_start_acq = time.time()
        fun_negative_acquisition = lambda X_test: -1.0 * constants.MULTIPLIER_ACQ \
            * self._optimize_objective(fun_acquisition, X_train, Y_train,
                X_test, cov_X_X, inv_cov_X_X, hyps)
        next_point, next_points = self._optimize(fun_negative_acquisition,
            str_sampling_method=str_sampling_method, num_samples=num_samples)
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

        if self.debug:
            logger.debug('overall time consumed to acquire: %.4f sec.', time_end - time_start)

        return next_point, dict_info
