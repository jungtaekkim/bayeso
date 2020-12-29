#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: December 29, 2020
#
"""test_utils_gp"""

import typing
import pytest
import numpy as np

from bayeso.utils import utils_gp

TEST_EPSILON = 1e-7


def test_get_prior_mu_typing():
    annos = utils_gp.get_prior_mu.__annotations__

    assert annos['prior_mu'] == typing.Union[callable, type(None)]
    assert annos['X'] == np.ndarray
    assert annos['return'] == np.ndarray

def test_get_prior_mu():
    fun_prior = lambda X: np.expand_dims(np.linalg.norm(X, axis=1), axis=1)
    fun_prior_1d = lambda X: np.linalg.norm(X, axis=1)
    X = np.reshape(np.arange(0, 90), (30, 3))

    with pytest.raises(AssertionError) as error:
        utils_gp.get_prior_mu(1, X)
    with pytest.raises(AssertionError) as error:
        utils_gp.get_prior_mu(fun_prior, 1)
    with pytest.raises(AssertionError) as error:
        utils_gp.get_prior_mu(fun_prior, np.arange(0, 100))
    with pytest.raises(AssertionError) as error:
        utils_gp.get_prior_mu(None, np.arange(0, 100))
    with pytest.raises(AssertionError) as error:
        utils_gp.get_prior_mu(fun_prior_1d, X)

    assert (utils_gp.get_prior_mu(None, X) == np.zeros((X.shape[0], 1))).all()
    assert (utils_gp.get_prior_mu(fun_prior, X) == fun_prior(X)).all()
