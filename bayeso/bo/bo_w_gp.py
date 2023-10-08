#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 4, 2022
#
"""It defines a class of Bayesian optimization
with Gaussian process regression."""

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

from bayeso.bo import base_bo
from bayeso import covariance
from bayeso import constants
from bayeso.gp import gp
from bayeso.gp import gp_kernel
from bayeso.utils import utils_bo
from bayeso.utils import utils_logger


class BOwGP(base_bo.BaseBO):
    """
    It is a Bayesian optimization class with Gaussian process regression.

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
    :param str_optimizer_method_gp: the name of optimization method for
        Gaussian process regression.
    :type str_optimizer_method_gp: str., optional
    :param str_optimizer_method_bo: the name of optimization method for
        Bayesian optimization.
    :type str_optimizer_method_bo: str., optional
    :param str_modelselection_method: the name of model selection method
        for Gaussian process regression.
    :type str_modelselection_method: str., optional
    :param str_exp: the name of experiment.
    :type str_exp: str., optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    """

    def __init__(self, range_X: np.ndarray,
        str_cov: str=constants.STR_COV,
        str_acq: str=constants.STR_BO_ACQ,
        normalize_Y: bool=constants.NORMALIZE_RESPONSE,
        use_ard: bool=constants.USE_ARD,
        prior_mu: constants.TYPING_UNION_CALLABLE_NONE=None,
        str_optimizer_method_gp: str=constants.STR_OPTIMIZER_METHOD_GP,
        str_optimizer_method_bo: str=constants.STR_OPTIMIZER_METHOD_AO,
        str_modelselection_method: str=constants.STR_MODELSELECTION_METHOD,
        str_exp: str=None,
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
        assert isinstance(str_exp, (type(None), str))
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

        str_surrogate = 'gp'
        assert str_surrogate in constants.ALLOWED_SURROGATE

        super().__init__(range_X, str_surrogate, str_acq,
            str_optimizer_method_bo, normalize_Y, str_exp, debug)

        self.str_cov = str_cov
        self.use_ard = use_ard
        self.str_optimizer_method_gp = str_optimizer_method_gp
        self.str_modelselection_method = str_modelselection_method
        self.prior_mu = prior_mu

        self.is_optimize_hyps = True
        self.historical_hyps = []

    def _optimize(self, fun_negative_acquisition: constants.TYPING_CALLABLE,
        str_sampling_method: str,
        num_samples: int,
        seed: int=None,
    ) -> constants.TYPING_TUPLE_TWO_ARRAYS:
        """
        It optimizes `fun_negative_function` with `self.str_optimizer_method_bo`.
        `num_samples` examples are determined by `str_sampling_method`, to
        start acquisition function optimization.

        :param fun_negative_acquisition: negative acquisition function.
        :type fun_negative_acquisition: callable
        :param str_sampling_method: the name of sampling method.
        :type str_sampling_method: str.
        :param num_samples: the number of samples.
        :type num_samples: int.
        :param seed: a random seed.
        :type seed: int., optional

        :returns: tuple of next point to evaluate and all candidates
            determined by acquisition function optimization.
            Shape: ((d, ), (`num_samples`, d)).
        :rtype: (numpy.ndarray, numpy.ndarray)

        """

        list_next_point = []
        if self.str_optimizer_method_bo == 'L-BFGS-B':
            list_bounds = self._get_bounds()
            initials = self.get_samples(str_sampling_method,
                num_samples=num_samples, seed=seed)

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
                    self.logger.debug('acquired sample: %s',
                        utils_logger.get_str_array(next_point_x))
        elif self.str_optimizer_method_bo == 'DIRECT': # pragma: no cover
            self.logger.debug('num_samples is ignored.')

            list_bounds = self._get_bounds()
            next_point = directminimize(
                fun_negative_acquisition,
                bounds=list_bounds,
                maxf=88888,
            )
            next_point_x = next_point.x
            list_next_point.append(next_point_x)
        elif self.str_optimizer_method_bo == 'CMA-ES':
            self.logger.debug('num_samples is ignored.')

            list_bounds = self._get_bounds()
            list_bounds = np.array(list_bounds)
            def fun_wrapper(f):
                def g(bx):
                    return f(bx)[0]
                return g
            initials = self.get_samples(str_sampling_method, num_samples=1, seed=seed)
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

    def compute_posteriors(self,
        X_train: np.ndarray, Y_train: np.ndarray,
        X_test: np.ndarray, cov_X_X: np.ndarray,
        inv_cov_X_X: np.ndarray, hyps: dict
    ) -> np.ndarray:
        """
        It returns posterior mean and standard deviation functions over `X_test`.

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

        :returns: posterior mean and standard deviation functions
            over `X_test`. Shape: ((l, ), (l, )).
        :rtype: (numpy.ndarray, numpy.ndarray)

        :raises: AssertionError

        """

        assert isinstance(X_train, np.ndarray)
        assert isinstance(Y_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(cov_X_X, np.ndarray)
        assert isinstance(inv_cov_X_X, np.ndarray)
        assert isinstance(hyps, dict)
        assert len(X_train.shape) == 2 or len(X_train.shape) == 3
        assert len(Y_train.shape) == 2
        assert len(X_test.shape) == 2 or len(X_test.shape) == 3
        assert len(cov_X_X.shape) == 2
        assert len(inv_cov_X_X.shape) == 2
        assert Y_train.shape[1] == 1
        assert X_train.shape[0] == Y_train.shape[0]
        if len(X_train.shape) == 2:
            assert X_test.shape[1] == X_train.shape[1] == self.num_dim
        else:
            assert X_test.shape[2] == X_train.shape[2] == self.num_dim
        assert cov_X_X.shape[0] == cov_X_X.shape[1] == X_train.shape[0]
        assert inv_cov_X_X.shape[0] == inv_cov_X_X.shape[1] == X_train.shape[0]

        pred_mean, pred_std, _ = gp.predict_with_cov(
            X_train, Y_train, X_test,
            cov_X_X, inv_cov_X_X, hyps, str_cov=self.str_cov,
            prior_mu=self.prior_mu, debug=self.debug
        )

        pred_mean = np.squeeze(pred_mean, axis=1)
        pred_std = np.squeeze(pred_std, axis=1)

        return pred_mean, pred_std

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

        :raises: AssertionError

        """

        assert isinstance(X, np.ndarray)
        assert isinstance(X_train, np.ndarray)
        assert isinstance(Y_train, np.ndarray)
        assert isinstance(cov_X_X, np.ndarray)
        assert isinstance(inv_cov_X_X, np.ndarray)
        assert isinstance(hyps, dict)
        assert len(X.shape) == 1 or len(X.shape) == 2 or len(X.shape) == 3
        assert len(X_train.shape) == 2 or len(X_train.shape) == 3
        assert len(Y_train.shape) == 2
        assert len(cov_X_X.shape) == 2
        assert len(inv_cov_X_X.shape) == 2
        assert Y_train.shape[1] == 1
        assert X_train.shape[0] == Y_train.shape[0]

        if len(X.shape) == 1:
            X = np.atleast_2d(X)

        if len(X_train.shape) == 2:
            assert X.shape[1] == X_train.shape[1] == self.num_dim
        else:
            assert X.shape[2] == X_train.shape[2] == self.num_dim

        assert cov_X_X.shape[0] == cov_X_X.shape[1] == X_train.shape[0]
        assert inv_cov_X_X.shape[0] == inv_cov_X_X.shape[1] == X_train.shape[0]

        fun_acquisition = utils_bo.choose_fun_acquisition(self.str_acq, hyps.get('noise', None))

        pred_mean, pred_std = self.compute_posteriors(
            X_train, Y_train, X,
            cov_X_X, inv_cov_X_X, hyps
        )

        acquisitions = fun_acquisition(
            pred_mean=pred_mean, pred_std=pred_std, Y_train=Y_train
        )
        acquisitions *= constants.MULTIPLIER_ACQ

        return acquisitions

    def optimize(self, X_train: np.ndarray, Y_train: np.ndarray,
        str_sampling_method: str=constants.STR_SAMPLING_METHOD_AO,
        num_samples: int=constants.NUM_SAMPLES_AO,
        str_mlm_method: str=constants.STR_MLM_METHOD,
        seed: int=None,
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
        :param seed: a random seed.
        :type seed: int., optional

        :returns: acquired example and dictionary of information. Shape: ((d, ), dict.).
        :rtype: (numpy.ndarray, dict.)

        :raises: AssertionError

        """

        assert isinstance(X_train, np.ndarray)
        assert isinstance(Y_train, np.ndarray)
        assert isinstance(str_sampling_method, str)
        assert isinstance(num_samples, int)
        assert isinstance(str_mlm_method, str)
        assert isinstance(seed, (type(None), int))
        assert len(X_train.shape) == 2
        assert len(Y_train.shape) == 2
        assert Y_train.shape[1] == 1
        assert X_train.shape[0] == Y_train.shape[0]
        assert X_train.shape[1] == self.num_dim
        assert num_samples > 0
        assert str_sampling_method in constants.ALLOWED_SAMPLING_METHOD
        assert str_mlm_method in constants.ALLOWED_MLM_METHOD

        time_start = time.time()
        Y_train_orig = Y_train

        if self.normalize_Y and str_mlm_method != 'converged':
            if self.debug:
                self.logger.debug('Responses are normalized.')

            Y_train = utils_bo.normalize_min_max(Y_train)

        time_start_surrogate = time.time()

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
                    self.logger.debug('hyps converged.')
                hyps = self.historical_hyps[-1]
                cov_X_X, inv_cov_X_X, _ = covariance.get_kernel_inverse(X_train, hyps,
                    self.str_cov, fix_noise=fix_noise, debug=self.debug)
        else: # pragma: no cover
            raise ValueError('optimize: missing condition for str_mlm_method.')

        self.historical_hyps.append(hyps)

        time_end_surrogate = time.time()

        time_start_acq = time.time()
        fun_negative_acquisition = lambda X_test: -1.0 * self.compute_acquisitions(
            X_test, X_train, Y_train, cov_X_X, inv_cov_X_X, hyps
        )
        next_point, next_points = self._optimize(fun_negative_acquisition,
            str_sampling_method=str_sampling_method,
            num_samples=num_samples,
            seed=seed)

        next_point = utils_bo.check_points_in_bounds(
            next_point[np.newaxis, ...], np.array(self._get_bounds()))[0]
        next_points = utils_bo.check_points_in_bounds(
            next_points, np.array(self._get_bounds()))

        time_end_acq = time.time()

        acquisitions = fun_negative_acquisition(next_points)
        time_end = time.time()

        dict_info = {
            'next_points': next_points,
            'acquisitions': acquisitions,
            'Y_original': Y_train_orig,
            'Y_normalized': Y_train,
            'cov_X_X': cov_X_X,
            'inv_cov_X_X': inv_cov_X_X,
            'hyps': hyps,
            'time_surrogate': time_end_surrogate - time_start_surrogate,
            'time_acq': time_end_acq - time_start_acq,
            'time_overall': time_end - time_start,
        }

        if self.debug:
            self.logger.debug('overall time consumed to acquire: %.4f sec.', time_end - time_start)

        return next_point, dict_info
