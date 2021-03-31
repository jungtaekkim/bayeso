#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: September 24, 2020
#
"""It is utilities for common features."""

import functools
import numpy as np

from bayeso import constants


def validate_types(func: constants.TYPING_CALLABLE) -> constants.TYPING_CALLABLE:
    """
    It is a decorator for validating the number of types, which are declared for typing.

    :param func: an original function.
    :type func: callable

    :returns: a callable decorator.
    :rtype: callable

    :raises: AssertionError

    """

    annos = func.__annotations__
    assert len(annos) == func.__code__.co_argcount + 1
#    arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]

    @functools.wraps(func)
    def _validate_types(*args, **kwargs):
        return func(*args, **kwargs)

    return _validate_types

@validate_types
def get_grids(ranges: np.ndarray, num_grids: int) -> np.ndarray:
    """
    It returns grids of given `ranges`, where each of dimension has `num_grids` partitions.

    :param ranges: ranges. Shape: (d, 2).
    :type ranges: numpy.ndarray
    :param num_grids: the number of partitions per dimension.
    :type num_grids: int.

    :returns: grids of given `ranges`. Shape: (`num_grids`:math:`^{\\text{d}}`, d).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(ranges, np.ndarray)
    assert isinstance(num_grids, int)
    assert len(ranges.shape) == 2
    assert ranges.shape[1] == 2
    assert (ranges[:, 0] <= ranges[:, 1]).all()

    list_grids = []
    for range_ in ranges:
        list_grids.append(np.linspace(range_[0], range_[1], num_grids))
    list_grids_mesh = list(np.meshgrid(*list_grids))
    list_grids = []
    for elem in list_grids_mesh:
        list_grids.append(elem.flatten(order='C'))
    arr_grids = np.vstack(tuple(list_grids))
    arr_grids = arr_grids.T
    return arr_grids

@validate_types
def get_minimum(Y_all: np.ndarray, num_init: int) -> constants.TYPING_TUPLE_THREE_ARRAYS:
    """
    It returns accumulated minima at each iteration, their arithmetic means
    over rounds, and their standard deviations over rounds, which is widely
    used in Bayesian optimization community.

    :param Y_all: historical function values. Shape: (r, t) where r is the
        number of Bayesian optimization rounds and t is the number of
        iterations including initial points for each round. For example,
        if we run 50 iterations with 5 initial examples and repeat this
        procedure 3 times, r would be 3 and t would be 55 (= 50 + 5).
    :type Y_all: numpy.ndarray
    :param num_init: the number of initial points.
    :type num_init: int.

    :returns: tuple of accumulated minima, their arithmetic means over
        rounds, and their standard deviations over rounds.
        Shape: ((r, t - `num_init` + 1), (t - `num_init` + 1, ), (t - `num_init` + 1, )).
    :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray)

    :raises: AssertionError

    """

    assert isinstance(Y_all, np.ndarray)
    assert isinstance(num_init, int)
    assert len(Y_all.shape) == 2
    assert Y_all.shape[1] > num_init

    list_minima = []

    for by in Y_all:
        minimum_best = np.inf
        list_minima_ = []
        for y in by[:num_init]:
            if minimum_best > y:
                minimum_best = y
        list_minima_.append(minimum_best)
        for y in by[num_init:]:
            if minimum_best > y:
                minimum_best = y
            list_minima_.append(minimum_best)
        list_minima.append(list_minima_)

    minima = np.array(list_minima)
    mean_minima = np.mean(minima, axis=0)
    std_minima = np.std(minima, axis=0)

    return minima, mean_minima, std_minima

@validate_types
def get_time(time_all: np.ndarray, num_init: int, include_init: bool) -> np.ndarray:
    """
    It returns the means of accumulated execution times over rounds.

    :param time_all: execution times for all Bayesian optimization rounds.
        Shape: (r, t) where r is the number of Bayesian optimization rounds
        and t is the number of iterations (including initial points if
        `include_init` is True, or excluding them if `include_init` is
        False) for each round.

    :type time_all: numpy.ndarray
    :param num_init: the number of initial points. If `include_init` is
        False, it is ignored even if it is provided.
    :type num_init: int.
    :param include_init: flag for describing whether execution times to
        observe initial examples have been included or not.
    :type include_init: bool.

    :returns: arithmetic means of accumulated execution times over rounds.
        Shape: (t - `num_init`, ) if `include_init` is True. (t, ), otherwise.
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(time_all, np.ndarray)
    assert isinstance(num_init, int)
    assert isinstance(include_init, bool)
    assert len(time_all.shape) == 2
    if include_init:
        assert time_all.shape[1] > num_init

    list_time = []
    for time_ in time_all:
        list_time_ = np.array([0.0])

        if include_init:
            time_ = time_[num_init:]
        list_time_ = np.concatenate((list_time_, np.cumsum(time_)))
        list_time.append(list_time_)
    list_time = np.array(list_time)

    return np.mean(list_time, axis=0)
