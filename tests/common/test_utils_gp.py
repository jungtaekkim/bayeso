# test_utils_gp
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 12, 2020

import numpy as np
import pytest

from bayeso.utils import utils_gp

TEST_EPSILON = 1e-7


def test_check_str_cov():
    with pytest.raises(AssertionError) as error:
        utils_gp.check_str_cov(1, 'se', (2, 1))
    with pytest.raises(AssertionError) as error:
        utils_gp.check_str_cov('test', 1, (2, 1))
    with pytest.raises(AssertionError) as error:
        utils_gp.check_str_cov('test', 'se', 1)
    with pytest.raises(AssertionError) as error:
        utils_gp.check_str_cov('test', 'se', (2, 100, 100))
    with pytest.raises(AssertionError) as error:
        utils_gp.check_str_cov('test', 'se', (2, 100), shape_X2=(2, 100, 100))
    with pytest.raises(AssertionError) as error:
        utils_gp.check_str_cov('test', 'set_se', (2, 100), shape_X2=(2, 100, 100))
    with pytest.raises(AssertionError) as error:
        utils_gp.check_str_cov('test', 'set_se', (2, 100, 100), shape_X2=(2, 100))
    with pytest.raises(AssertionError) as error:
        utils_gp.check_str_cov('test', 'se', (2, 1), shape_X2=1)

    with pytest.raises(ValueError) as error:
        utils_gp.check_str_cov('test', 'abc', (2, 1))

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
