#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: September 24, 2020
#
"""It is utilities for covariance functions."""

import numpy as np

from bayeso.utils import utils_common
from bayeso import constants


@utils_common.validate_types
def _get_list_first() -> list:
    """
    It provides list of strings.
    The strings in that list require two hyperparameters, `signal` and `lengthscales`.
    We simply call it as `list_first`.

    :returns: list of strings, which satisfy some requirements we mentioned above.
    :rtype: list

    """

    list_first = ['eq', 'se', 'matern32', 'matern52']
    list_first += ['set_' + str_ for str_ in list_first]
    return list_first

@utils_common.validate_types
def get_hyps(str_cov: str, dim: int,
    use_ard: bool=True
) -> dict:
    """
    It returns a dictionary of default hyperparameters for covariance
    function, where `str_cov` and `dim` are given. If `use_ard` is True,
    the length scales would be `dim`-dimensional vector.

    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param dim: dimensionality of the problem we are solving.
    :type dim: int.
    :param use_ard: flag for automatic relevance determination.
    :type use_ard: bool., optional

    :returns: dictionary of default hyperparameters for covariance function.
    :rtype: dict.

    :raises: AssertionError

    """

    assert isinstance(str_cov, str)
    assert isinstance(dim, int)
    assert isinstance(use_ard, bool)
    assert str_cov in constants.ALLOWED_GP_COV

    hyps = dict()
    hyps['noise'] = constants.GP_NOISE

    list_first = _get_list_first()

    if str_cov in list_first:
        hyps['signal'] = 1.0
        if use_ard:
            hyps['lengthscales'] = np.ones(dim)
        else:
            # TODO: It makes bunch of erros. I should fix it.
            hyps['lengthscales'] = 1.0
    else:
        raise NotImplementedError('get_hyps: allowed str_cov, but it is not implemented.')
    return hyps

@utils_common.validate_types
def get_range_hyps(str_cov: str, dim: int,
    use_ard: bool=True,
    fix_noise: bool=False
) -> list:
    """
    It returns default optimization ranges of hyperparameters for Gaussian process regression.

    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param dim: dimensionality of the problem we are solving.
    :type dim: int.
    :param use_ard: flag for automatic relevance determination.
    :type use_ard: bool., optional
    :param fix_noise: flag for fixing a noise.
    :type fix_noise: bool., optional

    :returns: list of default optimization ranges for hyperparameters.
    :rtype: list

    :raises: AssertionError

    """

    assert isinstance(str_cov, str)
    assert isinstance(dim, int)
    assert isinstance(use_ard, bool)
    assert isinstance(fix_noise, bool)
    assert str_cov in constants.ALLOWED_GP_COV

    range_hyps = []

    list_first = _get_list_first()

    if not fix_noise:
        range_hyps += constants.RANGE_NOISE

    if str_cov in list_first:
        range_hyps += constants.RANGE_SIGNAL # for signal scale
        if use_ard: # for lengthscales
            for _ in range(0, dim):
                range_hyps += constants.RANGE_LENGTHSCALES
        else:
            range_hyps += constants.RANGE_LENGTHSCALES
    else:
        raise NotImplementedError('get_hyps: allowed str_cov, but it is not implemented.')

    return range_hyps

@utils_common.validate_types
def convert_hyps(str_cov: str, hyps: dict,
    fix_noise: bool=False
) -> np.ndarray:
    """
    It converts hyperparameters dictionary, `hyps` to numpy array.

    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param hyps: dictionary of hyperparameters for covariance function.
    :type hyps: dict.
    :param fix_noise: flag for fixing a noise.
    :type fix_noise: bool., optional

    :returns: converted array of the hyperparameters given by `hyps`.
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(str_cov, str)
    assert isinstance(hyps, dict)
    assert isinstance(fix_noise, bool)
    assert str_cov in constants.ALLOWED_GP_COV

    list_hyps = []
    if not fix_noise:
        list_hyps.append(hyps['noise'])

    list_first = _get_list_first()

    if str_cov in list_first:
        list_hyps.append(hyps['signal'])
        for elem_lengthscale in hyps['lengthscales']:
            list_hyps.append(elem_lengthscale)
    else:
        raise NotImplementedError('convert_hyps: allowed str_cov, but it is not implemented.')
    return np.array(list_hyps)

@utils_common.validate_types
def restore_hyps(str_cov: str, hyps: np.ndarray,
    fix_noise: bool=False,
    noise: float=constants.GP_NOISE
) -> dict:
    """
    It restores hyperparameters array, `hyps` to dictionary.

    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param hyps: array of hyperparameters for covariance function.
    :type hyps: numpy.ndarray
    :param fix_noise: flag for fixing a noise.
    :type fix_noise: bool., optional
    :param noise: fixed noise value.
    :type noise: float, optional

    :returns: restored dictionary of the hyperparameters given by `hyps`.
    :rtype: dict.

    :raises: AssertionError

    """

    assert isinstance(str_cov, str)
    assert isinstance(hyps, np.ndarray)
    assert isinstance(fix_noise, bool)
    assert isinstance(noise, float)
    assert len(hyps.shape) == 1
    assert str_cov in constants.ALLOWED_GP_COV

    dict_hyps = dict()
    if not fix_noise:
        dict_hyps['noise'] = hyps[0]
        ind_start = 1
    else:
        dict_hyps['noise'] = noise
        ind_start = 0

    list_first = _get_list_first()

    if str_cov in list_first:
        dict_hyps['signal'] = hyps[ind_start]
        list_lengthscales = []
        for ind_elem in range(ind_start+1, len(hyps)):
            list_lengthscales.append(hyps[ind_elem])
        dict_hyps['lengthscales'] = np.array(list_lengthscales)
    else:
        raise NotImplementedError('restore_hyps: allowed str_cov, but it is not implemented.')
    return dict_hyps

@utils_common.validate_types
def validate_hyps_dict(hyps: dict, str_cov: str, dim: int) -> constants.TYPING_TUPLE_DICT_BOOL:
    """
    It validates hyperparameters dictionary, `hyps`.

    :param hyps: dictionary of hyperparameters for covariance function.
    :type hyps: dict.
    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param dim: dimensionality of the problem we are solving.
    :type dim: int.

    :returns: a tuple of valid hyperparameters and validity flag.
    :rtype: (dict., bool.)

    :raises: AssertionError

    """

    assert isinstance(hyps, dict)
    assert isinstance(str_cov, str)
    assert isinstance(dim, int)
    assert str_cov in constants.ALLOWED_GP_COV

    is_valid = True

    if 'noise' not in hyps:
        is_valid = False
    else:
        if not isinstance(hyps['noise'], float):
            is_valid = False
        else:
            if np.abs(hyps['noise']) >= constants.BOUND_UPPER_GP_NOISE:
                hyps['noise'] = constants.BOUND_UPPER_GP_NOISE

    if str_cov in ('eq', 'se', 'matern32', 'matern52'):
        if 'lengthscales' not in hyps:
            is_valid = False
        else:
            if isinstance(hyps['lengthscales'], np.ndarray) \
                and hyps['lengthscales'].shape[0] != dim:
                is_valid = False
            if not isinstance(hyps['lengthscales'], np.ndarray) \
                and not isinstance(hyps['lengthscales'], float):
                is_valid = False
        if 'signal' not in hyps:
            is_valid = False
        else:
            if not isinstance(hyps['signal'], float):
                is_valid = False
    else:
        is_valid = False

    return hyps, is_valid

@utils_common.validate_types
def validate_hyps_arr(hyps: np.ndarray, str_cov: str, dim: int
) -> constants.TYPING_TUPLE_ARRAY_BOOL:
    """
    It validates hyperparameters array, `hyps`.

    :param hyps: array of hyperparameters for covariance function.
    :type hyps: numpy.ndarray
    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param dim: dimensionality of the problem we are solving.
    :type dim: int.

    :returns: a tuple of valid hyperparameters and validity flag.
    :rtype: (numpy.ndarray, bool.)

    :raises: AssertionError

    """

    assert isinstance(hyps, np.ndarray)
    assert isinstance(str_cov, str)
    assert isinstance(dim, int)
    assert str_cov in constants.ALLOWED_GP_COV

#    is_valid = True

    raise NotImplementedError('validate_hyps_arr in utils_covariance.py')
