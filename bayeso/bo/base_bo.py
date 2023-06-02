#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 22, 2023
#
"""It defines an abstract class of Bayesian optimization."""

import abc
import numpy as np
import scipy.stats.qmc as scsqmc

from bayeso import constants
from bayeso.utils import utils_bo
from bayeso.utils import utils_common
from bayeso.utils import utils_logger


class BaseBO(abc.ABC):
    """
    It is a Bayesian optimization class.

    :param range_X: a search space. Shape: (d, 2).
    :type range_X: numpy.ndarray
    :param str_surrogate: the name of surrogate model.
    :type str_surrogate: str.
    :param str_acq: the name of acquisition function.
    :type str_acq: str.
    :param str_optimizer_method_bo: the name of optimization method for
        Bayesian optimization.
    :type str_optimizer_method_bo: str.
    :param normalize_Y: flag for normalizing outputs.
    :type normalize_Y: bool.
    :param str_exp: the name of experiment.
    :type str_exp: str.
    :param debug: flag for printing log messages.
    :type debug: bool.

    """

    def __init__(self,
        range_X: np.ndarray,
        str_surrogate: str,
        str_acq: str,
        str_optimizer_method_bo: str,
        normalize_Y: bool,
        str_exp: str,
        debug: bool
    ):
        """
        Constructor method

        """

        assert isinstance(range_X, np.ndarray)
        assert isinstance(str_surrogate, str)
        assert isinstance(str_acq, str)
        assert isinstance(str_optimizer_method_bo, str)
        assert isinstance(normalize_Y, bool)
        assert isinstance(debug, bool)
        assert isinstance(str_exp, (type(None), str))
        assert len(range_X.shape) == 2
        assert range_X.shape[1] == 2
        assert (range_X[:, 0] <= range_X[:, 1]).all()
        assert str_surrogate in constants.ALLOWED_SURROGATE \
            + constants.ALLOWED_SURROGATE_TREES
        assert str_acq in constants.ALLOWED_BO_ACQ
        assert str_optimizer_method_bo in constants.ALLOWED_OPTIMIZER_METHOD_BO \
            + constants.ALLOWED_OPTIMIZER_METHOD_BO_TREES

        self.range_X = range_X
        self.num_dim = range_X.shape[0]
        self.str_surrogate = str_surrogate
        self.str_acq = str_acq
        self.str_optimizer_method_bo = utils_bo.check_optimizer_method_bo(
            str_optimizer_method_bo, range_X.shape[0], debug)
        self.normalize_Y = normalize_Y
        self.str_exp = str_exp
        self.debug = debug

        if str_exp is not None:
            self.logger = utils_logger.get_logger(f'bo_w_{str_surrogate}_{str_exp}')
        else:
            self.logger = utils_logger.get_logger(f'bo_w_{str_surrogate}')

    def _get_random_state(self, seed: constants.TYPING_UNION_INT_NONE):
        """
        It returns a random state, defined by `seed`.

        :param seed: None, or a random seed.
        :type seed: NoneType or int.

        :returns: a random state.
        :rtype: numpy.random.RandomState

        :raises: AssertionError

        """

        assert isinstance(seed, (int, constants.TYPE_NONE))

        if seed is not None:
            state_random = np.random.RandomState(seed)
        else:
            state_random = np.random.RandomState()

        return state_random

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
        assert isinstance(seed, (int, constants.TYPE_NONE))

        state_random = self._get_random_state(seed)

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
        assert isinstance(seed, (int, constants.TYPE_NONE))

        state_random = self._get_random_state(seed)

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
        assert isinstance(seed, (int, constants.TYPE_NONE))

        sampler = scsqmc.Sobol(self.num_dim, scramble=True, seed=seed)
        samples = sampler.random(num_samples)

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
        assert isinstance(seed, (int, constants.TYPE_NONE))

        sampler = scsqmc.Halton(self.num_dim, scramble=True, seed=seed)
        samples = sampler.random(num_samples)

        samples = samples * (self.range_X[:, 1].flatten() - self.range_X[:, 0].flatten()) \
            + self.range_X[:, 0].flatten()
        return samples

    def get_samples(self, str_sampling_method: str,
        num_samples: int=constants.NUM_SAMPLES_AO,
        seed: constants.TYPING_UNION_INT_NONE=None,
    ) -> np.ndarray:
        """
        It returns `num_samples` examples, sampled by a sampling method `str_sampling_method`.

        :param str_sampling_method: the name of sampling method.
        :type str_sampling_method: str.
        :param num_samples: the number of samples.
        :type num_samples: int., optional
        :param seed: None, or random seed.
        :type seed: NoneType or int., optional

        :returns: sampled examples. Shape: (`num_samples`, d).
        :rtype: numpy.ndarray

        :raises: AssertionError

        """

        assert isinstance(str_sampling_method, str)
        assert isinstance(num_samples, int)
        assert isinstance(seed, (int, constants.TYPE_NONE))
        assert str_sampling_method in constants.ALLOWED_SAMPLING_METHOD

        if str_sampling_method == 'grid':
            if self.debug:
                self.logger.debug('For this option, num_samples is used as num_grids.')
            samples = self._get_samples_grid(num_grids=num_samples)
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
            self.logger.debug('samples:\n%s', utils_logger.get_str_array(samples))

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
        assert isinstance(seed, (int, constants.TYPE_NONE))
        assert str_initial_method in constants.ALLOWED_INITIALIZING_METHOD_BO

        return self.get_samples(str_initial_method, num_samples=num_initials, seed=seed)

    @abc.abstractmethod
    def compute_posteriors(self): # pragma: no cover
        """
        It is an abstract method.

        """

    @abc.abstractmethod
    def compute_acquisitions(self): # pragma: no cover
        """
        It is an abstract method.

        """

    @abc.abstractmethod
    def optimize(self): # pragma: no cover
        """
        It is an abstract method.

        """
