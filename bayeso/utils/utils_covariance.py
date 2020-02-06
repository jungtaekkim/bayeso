# utils_covariance
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: April 15, 2019

import numpy as np

from bayeso import constants


def _get_list_first():
    """
    It provides list of strings.
    The strings in that list require two hyperparameters, `signal` and `lengthscales`.
    We simply call it as `list_first`.

    :returns: list of strings, which satisfy some requirements we mentioned above.
    :rtype: list

    """

    list_first = ['se', 'matern32', 'matern52']
    list_first += ['set_' + str_ for str_ in list_first]
    return list_first

def get_hyps(str_cov, int_dim, is_ard=True):
    """
    It returns a dictionary of default hyperparameters for covariance function, where `str_cov` and `int_dim` are given. If `is_ard` is True, the length scales would be `int_dim`-dimensional vector.

    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param int_dim: dimensionality of the problem we are solving.
    :type int_dim: int.
    :param is_ard: flag for automatic relevance determination.
    :type is_ard: bool., optional

    :returns: dictionary of default hyperparameters for covariance function.
    :rtype: dict.

    :raises: AssertionError

    """

    assert isinstance(str_cov, str)
    assert isinstance(int_dim, int)
    assert isinstance(is_ard, bool)
    assert str_cov in constants.ALLOWED_GP_COV

    hyps = dict()
    hyps['noise'] = constants.GP_NOISE

    list_first = _get_list_first()

    if str_cov in list_first:
        hyps['signal'] = 1.0
        if is_ard:
            hyps['lengthscales'] = np.ones(int_dim)
        else:
            # TODO: It makes bunch of erros. I should fix it.
            hyps['lengthscales'] = 1.0
    else:
        raise NotImplementedError('get_hyps: allowed str_cov, but it is not implemented.')
    return hyps

def get_range_hyps(str_cov, int_dim,
    is_ard=True,
    is_fixed_noise=False
):
    """
    It returns default optimization ranges of hyperparameters for Gaussian process regression.

    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param int_dim: dimensionality of the problem we are solving.
    :type int_dim: int.
    :param is_ard: flag for automatic relevance determination.
    :type is_ard: bool., optional
    :param is_fixed_noise: flag for fixing a noise.
    :type is_fixed_noise: bool., optional

    :returns: list of default optimization ranges for hyperparameters.
    :rtype: list

    :raises: AssertionError

    """

    assert isinstance(str_cov, str)
    assert isinstance(int_dim, int)
    assert isinstance(is_ard, bool)
    assert isinstance(is_fixed_noise, bool)
    assert str_cov in constants.ALLOWED_GP_COV

    range_hyps = []

    list_first = _get_list_first()

    if not is_fixed_noise:
        range_hyps += constants.RANGE_NOISE

    if str_cov in list_first:
        range_hyps += constants.RANGE_SIGNAL # for signal scale
        if is_ard: # for lengthscales
            for _ in range(0, int_dim):
                range_hyps += constants.RANGE_LENGTHSCALES
        else:
            range_hyps += constants.RANGE_LENGTHSCALES
    else:
        raise NotImplementedError('get_hyps: allowed str_cov, but it is not implemented.')

    return range_hyps

def convert_hyps(str_cov, hyps, is_fixed_noise=False):
    """
    It converts hyperparameters dictionary, `hyps` to numpy array.

    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param hyps: dictionary of hyperparameters for covariance function.
    :type hyps: dict.
    :param is_fixed_noise: flag for fixing a noise.
    :type is_fixed_noise: bool., optional

    :returns: converted array of the hyperparameters given by `hyps`.
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(str_cov, str)
    assert isinstance(hyps, dict)
    assert isinstance(is_fixed_noise, bool)
    assert str_cov in constants.ALLOWED_GP_COV

    list_hyps = []
    if not is_fixed_noise:
        list_hyps.append(hyps['noise'])

    list_first = _get_list_first()

    if str_cov in list_first:
        list_hyps.append(hyps['signal'])
        for elem_lengthscale in hyps['lengthscales']:
            list_hyps.append(elem_lengthscale)
    else:
        raise NotImplementedError('convert_hyps: allowed str_cov, but it is not implemented.')
    return np.array(list_hyps)

def restore_hyps(str_cov, hyps, is_fixed_noise=False, fixed_noise=constants.GP_NOISE):
    """
    It restores hyperparameters array, `hyps` to dictionary.

    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param hyps: array of hyperparameters for covariance function.
    :type hyps: numpy.ndarray
    :param is_fixed_noise: flag for fixing a noise.
    :type is_fixed_noise: bool., optional
    :param fixed_noise: fixed noise value.
    :type fixed_noise: float, optional

    :returns: restored dictionary of the hyperparameters given by `hyps`.
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(str_cov, str)
    assert isinstance(hyps, np.ndarray)
    assert isinstance(is_fixed_noise, bool)
    assert isinstance(fixed_noise, float)
    assert len(hyps.shape) == 1
    assert str_cov in constants.ALLOWED_GP_COV

    dict_hyps = dict()
    if not is_fixed_noise:
        dict_hyps['noise'] = hyps[0]
        ind_start = 1
    else:
        dict_hyps['noise'] = fixed_noise
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

def validate_hyps_dict(dict_hyps, str_cov, int_dim):
    """
    It validates hyperparameters dictionary, `dict_hyps`.

    :param dict_hyps: dictionary of hyperparameters for covariance function.
    :type dict_hyps: dict.
    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param int_dim: dimensionality of the problem we are solving.
    :type int_dim: int.

    :returns: a tuple of valid hyperparameters and validity flag.
    :rtype: (dict., bool.)

    :raises: AssertionError

    """

    assert isinstance(dict_hyps, dict)
    assert isinstance(str_cov, str)
    assert isinstance(int_dim, int)
    assert str_cov in constants.ALLOWED_GP_COV

    is_valid = True

    if 'noise' not in dict_hyps:
        is_valid = False
    else:
        if not isinstance(dict_hyps['noise'], float):
            is_valid = False
        else:
            if np.abs(dict_hyps['noise']) >= constants.BOUND_UPPER_GP_NOISE:
                dict_hyps['noise'] = constants.BOUND_UPPER_GP_NOISE

    if str_cov == 'se' or str_cov == 'matern32' or str_cov == 'matern52':
        if 'lengthscales' not in dict_hyps:
            is_valid = False
        else:
            if isinstance(dict_hyps['lengthscales'], np.ndarray) and dict_hyps['lengthscales'].shape[0] != int_dim:
                is_valid = False
            if not isinstance(dict_hyps['lengthscales'], np.ndarray) and not isinstance(dict_hyps['lengthscales'], float):
                is_valid = False
        if 'signal' not in dict_hyps:
            is_valid = False
        else:
            if not isinstance(dict_hyps['signal'], float):
                is_valid = False
    else:
        is_valid = False
    return dict_hyps, is_valid

def validate_hyps_arr(arr_hyps, str_cov, int_dim):
    """
    It validates hyperparameters array, `arr_hyps`.

    :param arr_hyps: array of hyperparameters for covariance function.
    :type arr_hyps: numpy.ndarray
    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param int_dim: dimensionality of the problem we are solving.
    :type int_dim: int.

    :returns: a tuple of valid hyperparameters and validity flag.
    :rtype: (numpy.ndarray, bool.)

    :raises: AssertionError

    """

    assert isinstance(arr_hyps, np.ndarray)
    assert isinstance(str_cov, str)
    assert isinstance(int_dim, int)
    assert str_cov in constants.ALLOWED_GP_COV

#    is_valid = True

    raise NotImplementedError('validate_hyps_arr in utils_covariance.py')
