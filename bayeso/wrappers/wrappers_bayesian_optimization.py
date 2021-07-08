#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 8, 2021
#
"""It defines a wrapper class for Bayesian optimization."""

import time
import numpy as np

from bayeso import bo
from bayeso import constants
from bayeso.utils import utils_bo
from bayeso.utils import utils_logger

logger = utils_logger.get_logger('wrappers_bayesian_optimization')


class BayesianOptimization:
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
        str_optimizer_method_bo: str=constants.STR_OPTIMIZER_METHOD_AO,
        str_mlm_method: str=constants.STR_MLM_METHOD,
        str_modelselection_method: str=constants.STR_MODELSELECTION_METHOD,
        num_samples_ao: int=constants.NUM_SAMPLES_AO,
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
        assert isinstance(str_optimizer_method_bo, str)
        assert isinstance(str_mlm_method, str)
        assert isinstance(str_modelselection_method, str)
        assert isinstance(num_samples_ao, int)
        assert isinstance(debug, bool)

        assert len(range_X.shape) == 2
        assert range_X.shape[1] == 2
        assert (range_X[:, 0] <= range_X[:, 1]).all()
        assert num_iter > 0
        assert num_samples_ao > 0

        assert str_surrogate in constants.ALLOWED_SURROGATE
        assert str_cov in constants.ALLOWED_COV
        assert str_acq in constants.ALLOWED_BO_ACQ
        assert str_initial_method_bo in constants.ALLOWED_INITIALIZING_METHOD_BO
        assert str_sampling_method_ao in constants.ALLOWED_SAMPLING_METHOD
        assert str_optimizer_method_gp in constants.ALLOWED_OPTIMIZER_METHOD_GP
        assert str_optimizer_method_bo in constants.ALLOWED_OPTIMIZER_METHOD_BO
        assert str_mlm_method in constants.ALLOWED_MLM_METHOD
        assert str_modelselection_method in constants.ALLOWED_MODELSELECTION_METHOD

        self.range_X = range_X
        self.num_dim = range_X.shape[0]
        self.fun_target = fun_target
        self.num_iter = num_iter
        self.str_surrogate = str_surrogate
        self.str_cov = str_cov
        self.str_acq = str_acq
        self.normalize_Y = normalize_Y
        self.use_ard = use_ard
        self.prior_mu = prior_mu
        self.str_initial_method_bo = str_initial_method_bo
        self.str_sampling_method_ao = str_sampling_method_ao
        self.str_optimizer_method_gp = str_optimizer_method_gp
        self.str_optimizer_method_bo = utils_bo.check_optimizer_method_bo(
            str_optimizer_method_bo, range_X.shape[0], debug)
        self.str_mlm_method = str_mlm_method
        self.str_modelselection_method = str_modelselection_method
        self.num_samples_ao = num_samples_ao
        self.debug = debug

        self.model_bo = self._get_model_bo()

    def _get_model_bo(self):
        model_bo = bo.BO(
            self.range_X,
            str_cov=self.str_cov,
            str_acq=self.str_acq,
            normalize_Y=self.normalize_Y,
            use_ard=self.use_ard,
            prior_mu=self.prior_mu,
            str_surrogate=self.str_surrogate,
            str_optimizer_method_gp=self.str_optimizer_method_gp,
            str_optimizer_method_bo=self.str_optimizer_method_bo,
            str_modelselection_method=self.str_modelselection_method,
            debug=self.debug
        )
        return model_bo

    def _get_next_best_sample(self,
        next_sample: np.ndarray,
        X: np.ndarray,
        next_samples: np.ndarray,
        acq_vals: np.ndarray,
    ):
        if np.where(np.linalg.norm(next_sample - X, axis=1)\
            < constants.TOLERANCE_DUPLICATED_ACQ)[0].shape[0] > 0: # pragma: no cover
            next_sample = utils_bo.get_next_best_acquisition(
                next_samples, acq_vals, X)

            if self.debug:
                logger.debug('next_sample is repeated, so next best is selected.\
                    next_sample: %s', utils_logger.get_str_array(next_sample))
        return next_sample

    def optimize_with_all_initial_information(self,
        X: np.ndarray, Y: np.ndarray,
    ) -> constants.TYPING_TUPLE_FIVE_ARRAYS:
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
        time_gp_ = []
        time_acq_ = []

        for ind_iter in range(0, self.num_iter):
            logger.info('Iteration %d', ind_iter + 1)
            time_iter_start = time.time()

            next_sample, dict_info = self.model_bo.optimize(X_, Y_,
                str_sampling_method=self.str_sampling_method_ao,
                num_samples=self.num_samples_ao,
                str_mlm_method=self.str_mlm_method)

            next_samples = dict_info['next_points']
            acq_vals = dict_info['acquisitions']
            time_gp = dict_info['time_gp']
            time_acq = dict_info['time_acq']

            if self.debug:
                logger.debug('next_sample: %s', utils_logger.get_str_array(next_sample))

            next_sample = self._get_next_best_sample(next_sample, X_, next_samples, acq_vals)

            X_ = np.vstack((X_, next_sample))

            time_to_evaluate_start = time.time()
            Y_ = np.vstack((Y_, self.fun_target(next_sample)))
            time_to_evaluate_end = time.time()

            if self.debug:
                logger.debug('time consumed to evaluate: %.4f sec.',
                    time_to_evaluate_end - time_to_evaluate_start)

            time_iter_end = time.time()
            time_all_.append(time_iter_end - time_iter_start)
            time_gp_.append(time_gp)
            time_acq_.append(time_acq)

        time_end = time.time()

        if self.debug:
            logger.debug('overall time consumed in single BO round: %.4f sec.', time_end - time_start)

        time_all_ = np.array(time_all_)
        time_gp_ = np.array(time_gp_)
        time_acq_ = np.array(time_acq_)

        return X_, Y_, time_all_, time_gp_, time_acq_

    def optimize_with_initial_inputs(self,
        X: np.ndarray,
    ) -> constants.TYPING_TUPLE_FIVE_ARRAYS:
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

        X_, Y_, time_all_, time_gp_, time_acq_ \
            = self.optimize_with_all_initial_information(X, Y)

        time_all_ = np.concatenate((time_initials, time_all_))

        return X_, Y_, time_all_, time_gp_, time_acq_

    def optimize(self,
        num_init: int,
        seed: constants.TYPING_UNION_INT_NONE=None,
    ) -> constants.TYPING_TUPLE_FIVE_ARRAYS:
        assert isinstance(num_init, int)
        assert isinstance(seed, (int, type(None)))
        assert num_init > 0

        logger.info('====================')
        logger.info('range_X:\n%s', utils_logger.get_str_array(self.range_X))
        logger.info('num_init: %d', num_init)
        logger.info('num_iter: %d', self.num_iter)
        logger.info('str_surrogate: %s', self.str_surrogate)
        logger.info('str_cov: %s', self.str_cov)
        logger.info('str_acq: %s', self.str_acq)
        logger.info('normalize_Y: %s', self.normalize_Y)
        logger.info('use_ard: %s', self.use_ard)
        logger.info('str_initial_method_bo: %s', self.str_initial_method_bo)
        logger.info('str_sampling_method_ao: %s', self.str_sampling_method_ao)
        logger.info('str_optimizer_method_gp: %s', self.str_optimizer_method_gp)
        logger.info('str_optimizer_method_bo: %s', self.str_optimizer_method_bo)
        logger.info('str_mlm_method: %s', self.str_mlm_method)
        logger.info('str_modelselection_method: %s', self.str_modelselection_method)
        logger.info('num_samples_ao: %d', self.num_samples_ao)
        logger.info('seed: %s', seed)
        logger.info('debug: %s', self.debug)
        logger.info('====================')

        time_start = time.time()

        X_init = self.model_bo.get_initials(self.str_initial_method_bo, num_init, seed=seed)
        if self.debug:
            logger.debug('X_init:\n%s', utils_logger.get_str_array(X_init))

        X, Y, time_all, time_gp, time_acq = self.optimize_with_initial_inputs(X_init)

        time_end = time.time()

        if self.debug:
            logger.debug('overall time consumed including initializations: %.4f sec.',
                time_end - time_start)

        return X, Y, time_all, time_gp, time_acq
