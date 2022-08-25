#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 8, 2021
#
"""It defines a wrapper class for Bayesian optimization."""

import time
import numpy as np
from tqdm import tqdm

from bayeso import constants
from bayeso import bo
from bayeso.utils import utils_bo
from bayeso.utils import utils_logger


class BayesianOptimization:
    """
    It is a wrapper class for Bayesian optimization.
    A function for optimizing `fun_target` runs a single round
    of Bayesian optimization with an iteration budget `num_iter`.

    :param range_X: a search space. Shape: (d, 2).
    :type range_X: numpy.ndarray
    :param fun_target: a target function.
    :type fun_target: callable
    :param num_iter: an iteration budget for Bayesian optimization.
    :type num_iter: int.
    :param str_surrogate: the name of surrogate model.
    :type str_surrogate: str., optional
    :param str_cov: the name of covariance function.
    :type str_cov: str., optional
    :param str_acq: the name of acquisition function.
    :type str_acq: str., optional
    :param normalize_Y: a flag for normalizing outputs.
    :type normalize_Y: bool., optional
    :param use_ard: a flag for automatic relevance determination.
    :type use_ard: bool., optional
    :param prior_mu: None, or a prior mean function.
    :type prior_mu: NoneType, or callable, optional
    :param str_initial_method_bo: the name of initialization method for
        sampling initial examples in Bayesian optimization.
    :type str_initial_method_bo: str., optional
    :param str_sampling_method_ao: the name of sampling method for
        acquisition function optimization.
    :type str_sampling_method_ao: str., optional
    :param str_optimizer_method_gp: the name of optimization method for
        Gaussian process regression.
    :type str_optimizer_method_gp: str., optional
    :param str_optimizer_method_bo: the name of optimization method for
        Bayesian optimization.
    :type str_optimizer_method_bo: str., optional
    :param str_mlm_method: the name of marginal likelihood maximization
        method for Gaussian process regression.
    :type str_mlm_method: str., optional
    :param str_modelselection_method: the name of model selection method
        for Gaussian process regression.
    :type str_modelselection_method: str., optional
    :param num_samples_ao: the number of samples for acquisition function
        optimization. If a local search method (e.g., L-BFGS-B) is selected
        for acquisition function optimization, it is employed.
    :type num_samples_ao: int., optional
    :param str_exp: the name of experiment.
    :type str_exp: str., optional
    :param debug: a flag for printing log messages.
    :type debug: bool., optional

    """

    def __init__(self,
        range_X: np.ndarray,
        fun_target: constants.TYPING_CALLABLE,
        num_iter: int,
        str_surrogate: str=constants.STR_SURROGATE,
        str_cov: str=constants.STR_COV,
        str_acq: str=constants.STR_BO_ACQ,
        normalize_Y: bool=constants.NORMALIZE_RESPONSE,
        use_ard: bool=constants.USE_ARD,
        prior_mu: constants.TYPING_UNION_CALLABLE_NONE=None,
        str_initial_method_bo: str=constants.STR_INITIALIZING_METHOD_BO,
        str_sampling_method_ao: str=constants.STR_SAMPLING_METHOD_AO,
        str_optimizer_method_gp: str=constants.STR_OPTIMIZER_METHOD_GP,
        str_optimizer_method_tp: str=constants.STR_OPTIMIZER_METHOD_TP,
        str_optimizer_method_bo: str=constants.STR_OPTIMIZER_METHOD_AO,
        str_mlm_method: str=constants.STR_MLM_METHOD,
        str_modelselection_method: str=constants.STR_MODELSELECTION_METHOD,
        num_samples_ao: int=constants.NUM_SAMPLES_AO,
        str_exp: str=None,
        debug: bool=False,
    ):
        """
        Constructor method

        """

        assert isinstance(range_X, np.ndarray)
        assert callable(fun_target)
        assert isinstance(num_iter, int)
        assert isinstance(str_surrogate, str)
        assert isinstance(str_cov, str)
        assert isinstance(str_acq, str)
        assert isinstance(normalize_Y, bool)
        assert isinstance(use_ard, bool)
        assert callable(prior_mu) or prior_mu is None
        assert isinstance(str_initial_method_bo, str)
        assert isinstance(str_sampling_method_ao, str)
        assert isinstance(str_optimizer_method_gp, str)
        assert isinstance(str_optimizer_method_tp, str)
        assert isinstance(str_optimizer_method_bo, str)
        assert isinstance(str_mlm_method, str)
        assert isinstance(str_modelselection_method, str)
        assert isinstance(num_samples_ao, int)
        assert isinstance(str_exp, (type(None), str))
        assert isinstance(debug, bool)

        assert len(range_X.shape) == 2
        assert range_X.shape[1] == 2
        assert (range_X[:, 0] <= range_X[:, 1]).all()
        assert num_iter > 0
        assert num_samples_ao > 0

        assert str_surrogate in constants.ALLOWED_SURROGATE + constants.ALLOWED_SURROGATE_TREES
        assert str_cov in constants.ALLOWED_COV
        assert str_acq in constants.ALLOWED_BO_ACQ
        assert str_initial_method_bo in constants.ALLOWED_INITIALIZING_METHOD_BO
        assert str_sampling_method_ao in constants.ALLOWED_SAMPLING_METHOD
        assert str_optimizer_method_gp in constants.ALLOWED_OPTIMIZER_METHOD_GP
        assert str_optimizer_method_tp in constants.ALLOWED_OPTIMIZER_METHOD_TP
        assert str_optimizer_method_bo in constants.ALLOWED_OPTIMIZER_METHOD_BO \
            + constants.ALLOWED_OPTIMIZER_METHOD_BO_TREES
        assert str_mlm_method in constants.ALLOWED_MLM_METHOD
        assert str_modelselection_method in constants.ALLOWED_MODELSELECTION_METHOD

        self.range_X = range_X
        self.num_dim = range_X.shape[0]
        self.fun_target = fun_target
        self.num_iter = num_iter
        self.str_surrogate = str_surrogate
        self.str_acq = str_acq
        self.normalize_Y = normalize_Y
        self.str_initial_method_bo = str_initial_method_bo
        self.str_sampling_method_ao = str_sampling_method_ao
        self.str_optimizer_method_bo = utils_bo.check_optimizer_method_bo(
            str_optimizer_method_bo, range_X.shape[0], debug)
        self.num_samples_ao = num_samples_ao
        self.str_exp = str_exp
        self.debug = debug

        if str_surrogate in constants.ALLOWED_SURROGATE_TREES:
            self.model_bo = self._get_model_bo_trees()
        elif str_surrogate == 'gp':
            self.str_cov = str_cov
            self.use_ard = use_ard
            self.prior_mu = prior_mu
            self.str_optimizer_method_gp = str_optimizer_method_gp
            self.str_mlm_method = str_mlm_method
            self.str_modelselection_method = str_modelselection_method

            self.model_bo = self._get_model_bo_gp()
        elif str_surrogate == 'tp':
            self.str_cov = str_cov
            self.use_ard = use_ard
            self.prior_mu = prior_mu
            self.str_optimizer_method_tp = str_optimizer_method_tp

            self.model_bo = self._get_model_bo_tp()
        else:
            raise NotImplementedError('allowed str_surrogate, but it is not implemented.')

    def _get_model_bo_gp(self):
        """
        It returns an object of `bayeso.bo.bo_w_gp.BO_w_GP`.

        :returns: an object of Bayesian optimization.
        :rtype: bayeso.bo.bo_w_gp.BOwGP

        """

        model_bo = bo.BOwGP(
            self.range_X,
            str_cov=self.str_cov,
            str_acq=self.str_acq,
            normalize_Y=self.normalize_Y,
            use_ard=self.use_ard,
            prior_mu=self.prior_mu,
            str_optimizer_method_gp=self.str_optimizer_method_gp,
            str_optimizer_method_bo=self.str_optimizer_method_bo,
            str_modelselection_method=self.str_modelselection_method,
            str_exp=self.str_exp,
            debug=self.debug
        )

        return model_bo

    def _get_model_bo_tp(self):
        """
        It returns an object of `bayeso.bo.bo_w_tp.BO_w_TP`.

        :returns: an object of Bayesian optimization.
        :rtype: bayeso.bo.bo_w_tp.BOwTP

        """

        model_bo = bo.BOwTP(
            self.range_X,
            str_cov=self.str_cov,
            str_acq=self.str_acq,
            normalize_Y=self.normalize_Y,
            use_ard=self.use_ard,
            prior_mu=self.prior_mu,
            str_optimizer_method_tp=self.str_optimizer_method_tp,
            str_optimizer_method_bo=self.str_optimizer_method_bo,
            str_exp=self.str_exp,
            debug=self.debug
        )

        return model_bo

    def _get_model_bo_trees(self):
        """
        It returns an object of `bayeso.bo.bo_w_trees.BO_w_Trees`.

        :returns: an object of Bayesian optimization.
        :rtype: bayeso.bo.bo_w_trees.BOwTrees

        """

        model_bo = bo.BOwTrees(
            self.range_X,
            str_surrogate=self.str_surrogate,
            str_acq=self.str_acq,
            normalize_Y=self.normalize_Y,
            str_optimizer_method_bo=self.str_optimizer_method_bo,
            str_exp=self.str_exp,
            debug=self.debug
        )

        return model_bo

    def _get_next_best_sample(self,
        next_sample: np.ndarray,
        X: np.ndarray,
        next_samples: np.ndarray,
        acq_vals: np.ndarray,
    ) -> np.ndarray:
        """
        It returns the next best sample in terms of acquisition function values.

        :param next_sample: the next sample acquired.
        :type next_sample: np.ndarray
        :param X: the samples evaluated so far.
        :type X: np.ndarray
        :param next_samples: the candidates of the next sample.
        :type next_samples: np.ndarray
        :param acq_vals: the values of acquisition function over `next_samples`.
        :type acq_vals: np.ndarray

        :returns: the next best sample. Shape: (d, ).
        :rtype: numpy.ndarray

        :raises: AssertionError

        """

        assert isinstance(next_sample, np.ndarray)
        assert isinstance(X, np.ndarray)
        assert isinstance(next_samples, np.ndarray)
        assert isinstance(acq_vals, np.ndarray)

        if np.where(np.linalg.norm(next_sample - X, axis=1)\
            < constants.TOLERANCE_DUPLICATED_ACQ)[0].shape[0] > 0: # pragma: no cover
            next_sample = utils_bo.get_next_best_acquisition(
                next_samples, acq_vals, X)

            if self.debug:
                self.model_bo.logger.debug('next_sample is repeated, so next best is selected.\
                    next_sample: %s', utils_logger.get_str_array(next_sample))
        return next_sample

    def optimize_single_iteration(self, X: np.ndarray, Y: np.ndarray
    ) -> constants.TYPING_TUPLE_ARRAY_DICT:
        """
        It returns the optimization result and time consumed
        of single iteration, given `X` and `Y`.

        :param X: inputs. Shape: (n, d) or (n, m, d).
        :type X: numpy.ndarray
        :param Y: outputs. Shape: (n, 1).
        :type Y: numpy.ndarray

        :returns: a tuple of the next sample and information dictionary.
        :rtype: (numpy.ndarray, dict.)

        :raises: AssertionError, NotImplementedError

        """

        if self.str_surrogate in constants.ALLOWED_SURROGATE_TREES:
            next_sample, dict_info = self.model_bo.optimize(X, Y,
                str_sampling_method=self.str_sampling_method_ao,
                num_samples=self.num_samples_ao)
        elif self.str_surrogate == 'gp':
            next_sample, dict_info = self.model_bo.optimize(X, Y,
                str_sampling_method=self.str_sampling_method_ao,
                num_samples=self.num_samples_ao,
                str_mlm_method=self.str_mlm_method)
        elif self.str_surrogate == 'tp':
            next_sample, dict_info = self.model_bo.optimize(X, Y,
                str_sampling_method=self.str_sampling_method_ao,
                num_samples=self.num_samples_ao)
        else:
            raise NotImplementedError('allowed str_surrogate, but it is not implemented.')

        return next_sample, dict_info

    def optimize_with_all_initial_information(self,
        X: np.ndarray, Y: np.ndarray,
    ) -> constants.TYPING_TUPLE_FIVE_ARRAYS:
        """
        It returns the optimization results and times consumed, given
        inital inputs `X` and their corresponding outputs `Y`.

        :param X: initial inputs. Shape: (n, d) or (n, m, d).
        :type X: numpy.ndarray
        :param Y: initial outputs. Shape: (n, 1).
        :type Y: numpy.ndarray

        :returns: a tuple of acquired samples, their function values, overall
            times consumed per iteration, time consumed in modeling Gaussian process
            regression, and time consumed in acquisition function optimization.
            Shape: ((n + `num_iter`, d), (n + `num_iter`, 1),
            (`num_iter`, ), (`num_iter`, ), (`num_iter`, )),
            or ((n + `num_iter`, m, d),
            (n + `num_iter`, m, 1), (`num_iter`, ), (`num_iter`, ),
            (`num_iter`, )).
        :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)

        :raises: AssertionError

        """

        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        assert len(X.shape) == 2
        assert len(Y.shape) == 2
        assert X.shape[0] == Y.shape[0]
        assert Y.shape[1] == 1

        time_start = time.time()

        X_ = X
        Y_ = Y
        time_all_ = []
        time_surrogate_ = []
        time_acq_ = []

        pbar = tqdm(range(0, self.num_iter))
        for ind_iter in pbar:
            self.model_bo.logger.info('Iteration %d', ind_iter + 1)
            time_iter_start = time.time()

            next_sample, dict_info = self.optimize_single_iteration(X_, Y_)

            next_samples = dict_info['next_points']
            acq_vals = dict_info['acquisitions']
            time_surrogate = dict_info['time_surrogate']
            time_acq = dict_info['time_acq']

            if self.debug:
                self.model_bo.logger.debug('next_sample: %s',
                    utils_logger.get_str_array(next_sample))

            next_sample = self._get_next_best_sample(next_sample, X_, next_samples, acq_vals)

            X_ = np.vstack((X_, next_sample))

            time_to_evaluate_start = time.time()
            Y_ = np.vstack((Y_, self.fun_target(next_sample)))
            time_to_evaluate_end = time.time()

            if self.debug:
                self.model_bo.logger.debug('time consumed to evaluate: %.4f sec.',
                    time_to_evaluate_end - time_to_evaluate_start)

            time_iter_end = time.time()
            time_all_.append(time_iter_end - time_iter_start)
            time_surrogate_.append(time_surrogate)
            time_acq_.append(time_acq)

        time_end = time.time()

        if self.debug:
            self.model_bo.logger.debug('overall time consumed in single BO round: %.4f sec.',
                time_end - time_start)

        time_all_ = np.array(time_all_)
        time_surrogate_ = np.array(time_surrogate_)
        time_acq_ = np.array(time_acq_)

        return X_, Y_, time_all_, time_surrogate_, time_acq_

    def optimize_with_initial_inputs(self,
        X: np.ndarray,
    ) -> constants.TYPING_TUPLE_FIVE_ARRAYS:
        """
        It returns the optimization results and times consumed, given
        inital inputs `X`.

        :param X: initial inputs. Shape: (n, d) or (n, m, d).
        :type X: numpy.ndarray

        :returns: a tuple of acquired samples, their function values, overall
            times consumed per iteration, time consumed in modeling Gaussian process
            regression, and time consumed in acquisition function optimization.
            Shape: ((n + `num_iter`, d), (n + `num_iter`, 1),
            (n + `num_iter`, ), (`num_iter`, ), (`num_iter`, )),
            or ((n + `num_iter`, m, d),
            (n + `num_iter`, m, 1), (n + `num_iter`, ), (`num_iter`, ),
            (`num_iter`, )).
        :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)

        :raises: AssertionError

        """

        assert isinstance(X, np.ndarray)
        assert len(X.shape) == 2

        Y = []
        time_initials = []
        for elem in X:
            time_initial_start = time.time()
            Y.append(self.fun_target(elem))
            time_initial_end = time.time()
            time_initials.append(time_initial_end - time_initial_start)
        time_initials = np.array(time_initials)

        Y = np.array(Y)
        Y = np.reshape(Y, (Y.shape[0], 1))

        X_, Y_, time_all_, time_surrogate_, time_acq_ \
            = self.optimize_with_all_initial_information(X, Y)

        time_all_ = np.concatenate((time_initials, time_all_))

        return X_, Y_, time_all_, time_surrogate_, time_acq_

    def print_info(self, num_init, seed):
        """
        It returns the optimization results and times consumed, given
        inital inputs `X`.

        :param num_init: the number of initial points.
        :type num_init: int.
        :param seed: a random seed.
        :type seed: int.

        :returns: None
        :rtype: NoneType

        """

        self.model_bo.logger.info('====================')
        self.model_bo.logger.info('range_X:\n%s', utils_logger.get_str_array(self.range_X))
        self.model_bo.logger.info('num_init: %d', num_init)
        self.model_bo.logger.info('num_iter: %d', self.num_iter)
        self.model_bo.logger.info('str_surrogate: %s', self.str_surrogate)
        if self.str_surrogate in constants.ALLOWED_SURROGATE:
            self.model_bo.logger.info('str_cov: %s', self.str_cov)
        self.model_bo.logger.info('str_acq: %s', self.str_acq)
        self.model_bo.logger.info('normalize_Y: %s', self.normalize_Y)
        if self.str_surrogate in constants.ALLOWED_SURROGATE:
            self.model_bo.logger.info('use_ard: %s', self.use_ard)
        self.model_bo.logger.info('str_initial_method_bo: %s', self.str_initial_method_bo)
        self.model_bo.logger.info('str_sampling_method_ao: %s', self.str_sampling_method_ao)
        if self.str_surrogate in ['gp']:
            self.model_bo.logger.info('str_optimizer_method_gp: %s', self.str_optimizer_method_gp)
        if self.str_surrogate in ['tp']:
            self.model_bo.logger.info('str_optimizer_method_tp: %s', self.str_optimizer_method_tp)
        self.model_bo.logger.info('str_optimizer_method_bo: %s', self.str_optimizer_method_bo)
        if self.str_surrogate in ['gp']:
            self.model_bo.logger.info('str_mlm_method: %s', self.str_mlm_method)
            self.model_bo.logger.info('str_modelselection_method: %s',
                self.str_modelselection_method)
        self.model_bo.logger.info('num_samples_ao: %d', self.num_samples_ao)
        self.model_bo.logger.info('seed: %s', seed)
        self.model_bo.logger.info('debug: %s', self.debug)
        self.model_bo.logger.info('====================')

    def optimize(self,
        num_init: int,
        seed: constants.TYPING_UNION_INT_NONE=None,
    ) -> constants.TYPING_TUPLE_FIVE_ARRAYS:
        """
        It returns the optimization results and times consumed, given
        the number of initial samples `num_init` and a random seed
        `seed`.

        :param num_init: the number of initial samples.
        :type num_init: int.
        :param seed: None, or a random seed.
        :type seed: NoneType or int., optional

        :returns: a tuple of acquired samples, their function values, overall
            times consumed per iteration, time consumed in modeling Gaussian process
            regression, and time consumed in acquisition function optimization.
            Shape: ((`num_init` + `num_iter`, d), (`num_init` + `num_iter`, 1),
            (`num_init` + `num_iter`, ), (`num_iter`, ), (`num_iter`, )),
            or ((`num_init` + `num_iter`, m, d),
            (`num_init` + `num_iter`, m, 1), (`num_init` + `num_iter`, ),
            (`num_iter`, ),
            (`num_iter`, )),
            where d is a dimensionality of the problem we are solving
            and m is a cardinality of sets.
        :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)

        :raises: AssertionError

        """

        assert isinstance(num_init, int)
        assert isinstance(seed, (int, type(None)))
        assert num_init > 0

        self.print_info(num_init, seed)

        time_start = time.time()

        X_init = self.model_bo.get_initials(self.str_initial_method_bo, num_init, seed=seed)
        if self.debug:
            self.model_bo.logger.debug('X_init:\n%s', utils_logger.get_str_array(X_init))

        X, Y, time_all, time_surrogate, time_acq = self.optimize_with_initial_inputs(X_init)

        time_end = time.time()

        if self.debug:
            self.model_bo.logger.debug('overall time consumed including initializations: %.4f sec.',
                time_end - time_start)

        return X, Y, time_all, time_surrogate, time_acq
