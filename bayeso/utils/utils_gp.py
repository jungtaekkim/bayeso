#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: September 24, 2020
#
"""It is utilities for Gaussian process regression."""

import numpy as np

from bayeso.utils import utils_common
from bayeso import constants


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

    if str_cov in constants.ALLOWED_GP_COV_BASE:
        assert len(shape_X1) == 2
        if shape_X2 is not None:
            assert len(shape_X2) == 2
    elif str_cov in constants.ALLOWED_GP_COV_SET:
        assert len(shape_X1) == 3
        if shape_X2 is not None:
            assert len(shape_X2) == 3
    elif str_cov in constants.ALLOWED_GP_COV: # pragma: no cover
        raise ValueError('{}: missing conditions for str_cov.'.format(str_fun))
    else:
        raise ValueError('{}: invalid str_cov.'.format(str_fun))

@utils_common.validate_types
def get_prior_mu(prior_mu: constants.TYPING_UNION_CALLABLE_NONE, X: np.ndarray) -> np.ndarray:
    """
    It computes the prior mean function values over inputs X.

    :param prior_mu: prior mean function or None.
    :type prior_mu: function or NoneType
    :param X: inputs for prior mean function. Shape: (n, d) or (n, m, d).
    :type X: numpy.ndarray

    :returns: zero array, or array of prior mean function values. Shape: (n, 1).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(X, np.ndarray)
    assert callable(prior_mu) or prior_mu is None
    assert len(X.shape) == 2 or len(X.shape) == 3

    if prior_mu is None:
        prior_mu_X = np.zeros((X.shape[0], 1))
    else:
        prior_mu_X = prior_mu(X)
        assert len(prior_mu_X.shape) == 2
        assert X.shape[0] == prior_mu_X.shape[0]
    return prior_mu_X
