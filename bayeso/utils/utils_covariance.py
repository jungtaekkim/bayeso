#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: December 29, 2020
#
"""It is utilities for covariance functions."""

import numpy as np

from bayeso.utils import utils_common
from bayeso import constants


@utils_common.validate_types
def _get_list_first() -> constants.TYPING_LIST[str]:
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
    use_gp: bool=True,
    use_ard: bool=True,
) -> dict:
    """
    It returns a dictionary of default hyperparameters for covariance
    function, where `str_cov` and `dim` are given. If `use_ard` is True,
    the length scales would be `dim`-dimensional vector.

    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param dim: dimensionality of the problem we are solving.
    :type dim: int.
    :param use_gp: flag for Gaussian process or Student-$t$ process.
    :type use_gp: bool., optional
    :param use_ard: flag for automatic relevance determination.
    :type use_ard: bool., optional

    :returns: dictionary of default hyperparameters for covariance function.
    :rtype: dict.

    :raises: AssertionError

    """

    assert isinstance(str_cov, str)
    assert isinstance(dim, int)
    assert isinstance(use_gp, bool)
    assert isinstance(use_ard, bool)
    assert str_cov in constants.ALLOWED_COV

    hyps = dict()
    hyps['noise'] = constants.GP_NOISE

    if not use_gp:
        hyps['dof'] = 5.0

    list_first = _get_list_first()

    if str_cov in list_first:
        hyps['signal'] = 1.0
        if use_ard:
            hyps['lengthscales'] = np.ones(dim)
        else:
            hyps['lengthscales'] = 1.0
    else:
        raise NotImplementedError('get_hyps: allowed str_cov, but it is not implemented.')
    return hyps

@utils_common.validate_types
def get_range_hyps(str_cov: str, dim: int,
    use_gp: bool=True,
    use_ard: bool=True,
    fix_noise: bool=False
) -> constants.TYPING_LIST[list]:
    """
    It returns default optimization ranges of hyperparameters for Gaussian process regression.

    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param dim: dimensionality of the problem we are solving.
    :type dim: int.
    :param use_gp: flag for Gaussian process or Student-$t$ process.
    :type use_gp: bool., optional
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
    assert isinstance(use_gp, bool)
    assert isinstance(use_ard, bool)
    assert isinstance(fix_noise, bool)
    assert str_cov in constants.ALLOWED_COV

    range_hyps = []

    list_first = _get_list_first()

    if not fix_noise:
        range_hyps += constants.RANGE_NOISE

    if not use_gp:
        range_hyps += constants.RANGE_DOF

    if str_cov in list_first:
        range_hyps += constants.RANGE_SIGNAL # for signal scale
        if use_ard: # for lengthscales
            for _ in range(0, dim):
                range_hyps += constants.RANGE_LENGTHSCALES
        else:
            # INFO: dim is ignored.
            range_hyps += constants.RANGE_LENGTHSCALES
    else:
        raise NotImplementedError('get_hyps: allowed str_cov, but it is not implemented.')

    return range_hyps

@utils_common.validate_types
def convert_hyps(str_cov: str, hyps: dict,
    use_gp: bool=True,
    fix_noise: bool=False
) -> np.ndarray:
    """
    It converts hyperparameters dictionary, `hyps` to numpy array.

    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param hyps: dictionary of hyperparameters for covariance function.
    :type hyps: dict.
    :param use_gp: flag for Gaussian process or Student-$t$ process.
    :type use_gp: bool., optional
    :param fix_noise: flag for fixing a noise.
    :type fix_noise: bool., optional

    :returns: converted array of the hyperparameters given by `hyps`.
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(str_cov, str)
    assert isinstance(hyps, dict)
    assert isinstance(use_gp, bool)
    assert isinstance(fix_noise, bool)
    assert str_cov in constants.ALLOWED_COV

    list_hyps = []
    if not fix_noise:
        list_hyps.append(hyps['noise'])

    if not use_gp:
        list_hyps.append(hyps['dof'])

    list_first = _get_list_first()

    if str_cov in list_first:
        list_hyps.append(hyps['signal'])
        if isinstance(hyps['lengthscales'], np.ndarray):
            for elem_lengthscale in hyps['lengthscales']:
                list_hyps.append(elem_lengthscale)
        elif isinstance(hyps['lengthscales'], float):
            list_hyps.append(hyps['lengthscales'])
        else: # pragma: no cover
            raise ValueError('covert_hyps: not allowed type for lengthscales.')
    else:
        raise NotImplementedError('convert_hyps: allowed str_cov, but it is not implemented.')

    return np.array(list_hyps)

@utils_common.validate_types
def restore_hyps(str_cov: str, hyps: np.ndarray,
    use_gp: bool=True,
    use_ard: bool=True,
    fix_noise: bool=False,
    noise: float=constants.GP_NOISE
) -> dict:
    """
    It restores hyperparameters array, `hyps` to dictionary.

    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param hyps: array of hyperparameters for covariance function.
    :type hyps: numpy.ndarray
    :param use_gp: flag for Gaussian process or Student-$t$ process.
    :type use_gp: bool., optional
    :param use_ard: flag for using automatic relevance determination.
    :type use_ard: bool., optional
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
    assert isinstance(use_gp, bool)
    assert isinstance(use_ard, bool)
    assert isinstance(fix_noise, bool)
    assert isinstance(noise, float)
    assert len(hyps.shape) == 1
    assert str_cov in constants.ALLOWED_COV

    dict_hyps = dict()
    if not fix_noise:
        dict_hyps['noise'] = hyps[0]
        ind_start = 1
    else:
        dict_hyps['noise'] = noise
        ind_start = 0

    if not use_gp:
        dict_hyps['dof'] = hyps[ind_start]
        ind_start += 1

    list_first = _get_list_first()

    if str_cov in list_first:
        dict_hyps['signal'] = hyps[ind_start]

        if use_ard:
            list_lengthscales = []
            for ind_elem in range(ind_start + 1, len(hyps)):
                list_lengthscales.append(hyps[ind_elem])
            dict_hyps['lengthscales'] = np.array(list_lengthscales)
        else:
            assert hyps.shape[0] == ind_start + 2
            dict_hyps['lengthscales'] = hyps[ind_start + 1]
    else:
        raise NotImplementedError('restore_hyps: allowed str_cov, but it is not implemented.')
    return dict_hyps

@utils_common.validate_types
def validate_hyps_dict(hyps: dict, str_cov: str, dim: int,
    use_gp: bool=True
) -> constants.TYPING_TUPLE_DICT_BOOL:
    """
    It validates hyperparameters dictionary, `hyps`.

    :param hyps: dictionary of hyperparameters for covariance function.
    :type hyps: dict.
    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param dim: dimensionality of the problem we are solving.
    :type dim: int.
    :param use_gp: flag for Gaussian process or Student-$t$ process.
    :type use_gp: bool., optional

    :returns: a tuple of valid hyperparameters and validity flag.
    :rtype: (dict., bool.)

    :raises: AssertionError

    """

    assert isinstance(hyps, dict)
    assert isinstance(str_cov, str)
    assert isinstance(dim, int)
    assert isinstance(use_gp, bool)
    assert str_cov in constants.ALLOWED_COV

    if 'noise' not in hyps:
        raise ValueError('validate_hyps_dict: invalid noise.')

    if not isinstance(hyps['noise'], float):
        raise ValueError('validate_hyps_dict: invalid noise.')

    if np.abs(hyps['noise']) >= constants.BOUND_UPPER_GP_NOISE:
        hyps['noise'] = constants.BOUND_UPPER_GP_NOISE

    if not use_gp:
        if 'dof' not in hyps:
            raise ValueError('validate_hyps_dict: invalid dof.')

        if not isinstance(hyps['dof'], float):
            raise ValueError('validate_hyps_dict: invalid dof.')

        if isinstance(hyps['dof'], float) and hyps['dof'] <= 2.0:
            hyps['dof'] = 2.00001

    if 'lengthscales' not in hyps:
        raise ValueError('validate_hyps_dict: invalid lengthscales.')

    if isinstance(hyps['lengthscales'], np.ndarray) \
        and hyps['lengthscales'].shape[0] != dim:
        raise ValueError('validate_hyps_dict: invalid lengthscales.')
    if not isinstance(hyps['lengthscales'], np.ndarray) \
        and not isinstance(hyps['lengthscales'], float):
        raise ValueError('validate_hyps_dict: invalid lengthscales.')

    if 'signal' not in hyps:
        raise ValueError('validate_hyps_dict: invalid signal.')

    if not isinstance(hyps['signal'], float):
        raise ValueError('validate_hyps_dict: invalid signal.')

    return hyps

@utils_common.validate_types
def validate_hyps_arr(hyps: np.ndarray, str_cov: str, dim: int,
    use_gp: bool=True
) -> constants.TYPING_TUPLE_ARRAY_BOOL:
    """
    It validates hyperparameters array, `hyps`.

    :param hyps: array of hyperparameters for covariance function.
    :type hyps: numpy.ndarray
    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param dim: dimensionality of the problem we are solving.
    :type dim: int.
    :param use_gp: flag for Gaussian process or Student-$t$ process.
    :type use_gp: bool., optional

    :returns: a tuple of valid hyperparameters and validity flag.
    :rtype: (numpy.ndarray, bool.)

    :raises: AssertionError

    """

    assert isinstance(hyps, np.ndarray)
    assert isinstance(str_cov, str)
    assert isinstance(dim, int)
    assert isinstance(use_gp, bool)
    assert str_cov in constants.ALLOWED_COV

#    is_valid = True

    raise NotImplementedError('validate_hyps_arr in utils_covariance.py')

@utils_common.validate_types
def check_str_cov(str_fun: str, str_cov: str, shape_X1: tuple,
    shape_X2: tuple=None
) -> constants.TYPE_NONE:
    """
    It is for validating the shape of X1 (and optionally the shape of X2).

    :param str_fun: the name of function.
    :type str_fun: str.
    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param shape_X1: the shape of X1.
    :type shape_X1: tuple
    :param shape_X2: None, or the shape of X2.
    :type shape_X2: NoneType or tuple, optional

    :returns: None, if it is valid. Raise an error, otherwise.
    :rtype: NoneType

    :raises: AssertionError, ValueError

    """

    assert isinstance(str_fun, str)
    assert isinstance(str_cov, str)
    assert isinstance(shape_X1, tuple)
    assert shape_X2 is None or isinstance(shape_X2, tuple)

    if str_cov in constants.ALLOWED_COV_BASE:
        assert len(shape_X1) == 2
        if shape_X2 is not None:
            assert len(shape_X2) == 2
    elif str_cov in constants.ALLOWED_COV_SET:
        assert len(shape_X1) == 3
        if shape_X2 is not None:
            assert len(shape_X2) == 3
    elif str_cov in constants.ALLOWED_COV: # pragma: no cover
        raise ValueError('{}: missing conditions for str_cov.'.format(str_fun))
    else:
        raise ValueError('{}: invalid str_cov.'.format(str_fun))
