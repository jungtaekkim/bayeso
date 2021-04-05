#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: December 29, 2020
#
"""It is utilities for Gaussian process regression and
Student-:math:`t` process regression."""

import numpy as np

from bayeso.utils import utils_common
from bayeso import constants


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

@utils_common.validate_types
def validate_common_args(X_train: np.ndarray, Y_train: np.ndarray,
    str_cov: str, prior_mu: constants.TYPING_UNION_CALLABLE_NONE,
    debug: bool,
    X_test: constants.TYPING_UNION_ARRAY_NONE=None,
) -> constants.TYPE_NONE:
    """
    It validates the common arguments for various functions.

    :param X_train: inputs. Shape: (n, d) or (n, m, d).
    :type X_train: numpy.ndarray
    :param Y_train: outputs. Shape: (n, 1).
    :type Y_train: numpy.ndarray
    :param str_cov: the name of covariance function.
    :type str_cov: str.
    :param prior_mu: None, or prior mean function.
    :type prior_mu: NoneType, or function
    :param debug: flag for printing log messages.
    :type debug: bool.
    :param X_test: inputs or None. Shape: (l, d) or (l, m, d).
    :type X_test: numpy.ndarray, or NoneType, optional

    :returns: None.
    :rtype: NoneType

    :raises: AssertionError

    """

    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(str_cov, str)
    assert callable(prior_mu) or prior_mu is None
    assert isinstance(debug, bool)
    assert len(Y_train.shape) == 2
    assert X_train.shape[0] == Y_train.shape[0]
    assert isinstance(X_test, (np.ndarray, type(None)))

    if X_test is not None:
        assert X_train.shape[1] == X_test.shape[1]
