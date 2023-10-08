#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 4, 2022
#
"""It defines a class of Bayesian optimization
with tree-based surrogate models."""

import time
import numpy as np

from bayeso.bo import base_bo
from bayeso.trees import trees_common
from bayeso import constants
from bayeso.utils import utils_bo


class BOwTrees(base_bo.BaseBO):
    """
    It is a Bayesian optimization class with tree-based surrogate models.

    :param range_X: a search space. Shape: (d, 2).
    :type range_X: numpy.ndarray
    :param str_surrogate: the name of surrogate model.
    :type str_surrogate: str., optional
    :param str_acq: the name of acquisition function.
    :type str_acq: str., optional
    :param normalize_Y: flag for normalizing outputs.
    :type normalize_Y: bool., optional
    :param str_optimizer_method_bo: the name of optimization method for
        Bayesian optimization.
    :type str_optimizer_method_bo: str., optional
    :param str_exp: the name of experiment.
    :type str_exp: str., optional
    :param debug: flag for printing log messages.
    :type debug: bool., optional

    """

    def __init__(self, range_X: np.ndarray,
        str_surrogate: str=constants.STR_SURROGATE_TREES,
        str_acq: str=constants.STR_BO_ACQ,
        normalize_Y: bool=constants.NORMALIZE_RESPONSE,
        str_optimizer_method_bo: str=constants.STR_OPTIMIZER_METHOD_AO_TREES,
        str_exp: str=None,
        debug: bool=False
    ):
        """
        Constructor method

        """

        assert isinstance(range_X, np.ndarray)
        assert isinstance(str_surrogate, str)
        assert isinstance(str_acq, str)
        assert isinstance(normalize_Y, bool)
        assert isinstance(str_optimizer_method_bo, str)
        assert isinstance(str_exp, (type(None), str))
        assert isinstance(debug, bool)
        assert len(range_X.shape) == 2
        assert range_X.shape[1] == 2
        assert (range_X[:, 0] <= range_X[:, 1]).all()
        assert str_surrogate in constants.ALLOWED_SURROGATE_TREES
        assert str_acq in constants.ALLOWED_BO_ACQ
        assert str_optimizer_method_bo in constants.ALLOWED_OPTIMIZER_METHOD_BO_TREES

        super().__init__(range_X, str_surrogate, str_acq,
            str_optimizer_method_bo, normalize_Y, str_exp, debug)

    def get_trees(self, X_train, Y_train,
        num_trees=100,
        depth_max=5,
        size_min_leaf=1,
    ):
        """
        It returns a list of trees.

        :param X_train: inputs. Shape: (n, d).
        :type X_train: numpy.ndarray
        :param Y_train: outputs. Shape: (n, 1).
        :type Y_train: numpy.ndarray
        :param num_trees: the number of trees.
        :type num_trees: int., optional
        :param depth_max: maximum depth.
        :type depth_max: int., optional
        :param size_min_leaf: minimum size of leaves.
        :type size_min_leaf: int., optional

        :returns: list of trees.
        :rtype: list

        :raises: AssertionError

        """

        assert isinstance(X_train, np.ndarray)
        assert isinstance(Y_train, np.ndarray)
        assert isinstance(num_trees, int)
        assert isinstance(depth_max, int)
        assert isinstance(size_min_leaf, int)
        assert len(X_train.shape) == 2
        assert len(Y_train.shape) == 2
        assert X_train.shape[0] == Y_train.shape[0]
        assert Y_train.shape[1] == 1

        num_features = int(np.sqrt(self.num_dim))

        if self.str_surrogate == 'rf':
            from bayeso.trees import trees_random_forest
            trees = trees_random_forest.get_random_forest(
                X_train, Y_train, num_trees, depth_max, size_min_leaf, num_features
            )
        else:
            raise NotImplementedError('allowed str_surrogate, but it is not implemented.')

        return trees

    def compute_posteriors(self,
        X: np.ndarray, trees: constants.TYPING_LIST
    ) -> np.ndarray:
        """
        It returns posterior mean and standard deviation functions over `X`.

        :param X: inputs to test. Shape: (l, d).
        :type X: numpy.ndarray
        :param trees: list of trees.
        :type trees: list

        :returns: posterior mean and standard deviation functions
            over `X`. Shape: ((l, ), (l, )).
        :rtype: (numpy.ndarray, numpy.ndarray)

        :raises: AssertionError

        """

        assert isinstance(X, np.ndarray)
        assert isinstance(trees, list)
        assert len(X.shape) == 2
        assert X.shape[1] == self.num_dim

        pred_mean, pred_std = trees_common.predict_by_trees(X, trees)

        pred_mean = np.squeeze(pred_mean, axis=1)
        pred_std = np.squeeze(pred_std, axis=1)

        return pred_mean, pred_std

    def compute_acquisitions(self, X: np.ndarray,
        X_train: np.ndarray, Y_train: np.ndarray,
        trees: constants.TYPING_LIST
    ) -> np.ndarray:
        """
        It computes acquisition function values over 'X',
        where `X_train` and `Y_train` are given.

        :param X: inputs. Shape: (l, d).
        :type X: numpy.ndarray
        :param X_train: inputs. Shape: (n, d).
        :type X_train: numpy.ndarray
        :param Y_train: outputs. Shape: (n, 1).
        :type Y_train: numpy.ndarray
        :param trees: list of trees.
        :type trees: list

        :returns: acquisition function values over `X`. Shape: (l, ).
        :rtype: numpy.ndarray

        :raises: AssertionError

        """

        assert isinstance(X, np.ndarray)
        assert isinstance(X_train, np.ndarray)
        assert isinstance(Y_train, np.ndarray)
        assert len(X.shape) == 1 or len(X.shape) == 2 or len(X.shape) == 3
        assert len(X_train.shape) == 2 or len(X_train.shape) == 3
        assert len(Y_train.shape) == 2
        assert Y_train.shape[1] == 1
        assert X_train.shape[0] == Y_train.shape[0]

        if len(X.shape) == 1:
            X = np.atleast_2d(X)

        assert X.shape[1] == X_train.shape[1] == self.num_dim

        fun_acquisition = utils_bo.choose_fun_acquisition(self.str_acq, constants.GP_NOISE)

        pred_mean, pred_std = self.compute_posteriors(X, trees)

        acquisitions = fun_acquisition(
            pred_mean=pred_mean, pred_std=pred_std, Y_train=Y_train
        )
        acquisitions *= constants.MULTIPLIER_ACQ

        return acquisitions

    def optimize(self, X_train: np.ndarray, Y_train: np.ndarray,
        str_sampling_method: str=constants.STR_SAMPLING_METHOD_AO_TREES,
        num_samples: int=constants.NUM_SAMPLES_AO_TREES,
        seed: int=None,
    ) -> constants.TYPING_TUPLE_ARRAY_DICT:
        """
        It computes acquired example, candidates of acquired examples,
        acquisition function values for the candidates, covariance matrix,
        inverse matrix of the covariance matrix, hyperparameters optimized,
        and execution times.

        :param X_train: inputs. Shape: (n, d).
        :type X_train: numpy.ndarray
        :param Y_train: outputs. Shape: (n, 1).
        :type Y_train: numpy.ndarray
        :param str_sampling_method: the name of sampling method for
            acquisition function optimization.
        :type str_sampling_method: str., optional
        :param num_samples: the number of samples.
        :type num_samples: int., optional
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
        assert isinstance(seed, (type(None), int))
        assert len(X_train.shape) == 2
        assert len(Y_train.shape) == 2
        assert Y_train.shape[1] == 1
        assert X_train.shape[0] == Y_train.shape[0]
        assert X_train.shape[1] == self.num_dim
        assert num_samples > 0
        assert str_sampling_method in constants.ALLOWED_SAMPLING_METHOD

        time_start = time.time()
        Y_train_orig = Y_train

        if self.normalize_Y:
            if self.debug:
                self.logger.debug('Responses are normalized.')

            Y_train = utils_bo.normalize_min_max(Y_train)

        time_start_surrogate = time.time()
        trees = self.get_trees(X_train, Y_train)
        time_end_surrogate = time.time()

        time_start_acq = time.time()
        next_points = self.get_samples(str_sampling_method, num_samples=num_samples, seed=seed)

        next_points = utils_bo.check_points_in_bounds(next_points, np.array(self._get_bounds()))

        fun_negative_acquisition = lambda X_test: -1.0 * self.compute_acquisitions(
            X_test, X_train, Y_train, trees
        )
        acquisitions = fun_negative_acquisition(next_points)
        ind_next_point = np.argmin(acquisitions)
        next_point = next_points[ind_next_point]

        time_end_acq = time.time()

        time_end = time.time()

        dict_info = {
            'next_points': next_points,
            'acquisitions': acquisitions,
            'Y_original': Y_train_orig,
            'Y_normalized': Y_train,
            'trees': trees,
            'time_surrogate': time_end_surrogate - time_start_surrogate,
            'time_acq': time_end_acq - time_start_acq,
            'time_overall': time_end - time_start,
        }

        if self.debug:
            self.logger.debug('overall time consumed to acquire: %.4f sec.', time_end - time_start)

        return next_point, dict_info
