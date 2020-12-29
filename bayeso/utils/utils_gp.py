#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: December 29, 2020
#
"""It is utilities for Gaussian process regression."""

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
