#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: December 29, 2020
#
"""test_utils_gp"""

import typing
import pytest
import numpy as np

from bayeso.utils import utils_gp as package_target

TEST_EPSILON = 1e-7


def test_get_prior_mu_typing():
    annos = package_target.get_prior_mu.__annotations__

    assert annos['prior_mu'] == typing.Union[typing.Callable, type(None)]
    assert annos['X'] == np.ndarray
    assert annos['return'] == np.ndarray

def test_get_prior_mu():
    fun_prior = lambda X: np.expand_dims(np.linalg.norm(X, axis=1), axis=1)
    fun_prior_1d = lambda X: np.linalg.norm(X, axis=1)
    X = np.reshape(np.arange(0, 90), (30, 3))

    with pytest.raises(AssertionError) as error:
        package_target.get_prior_mu(1, X)
    with pytest.raises(AssertionError) as error:
        package_target.get_prior_mu(fun_prior, 1)
    with pytest.raises(AssertionError) as error:
        package_target.get_prior_mu(fun_prior, np.arange(0, 100))
    with pytest.raises(AssertionError) as error:
        package_target.get_prior_mu(None, np.arange(0, 100))
    with pytest.raises(AssertionError) as error:
        package_target.get_prior_mu(fun_prior_1d, X)

    assert (package_target.get_prior_mu(None, X) == np.zeros((X.shape[0], 1))).all()
    assert (package_target.get_prior_mu(fun_prior, X) == fun_prior(X)).all()

def test_validate_common_args_typing():
    annos = package_target.validate_common_args.__annotations__

    assert annos['X_train'] == np.ndarray
    assert annos['Y_train'] == np.ndarray
    assert annos['str_cov'] == str
    assert annos['prior_mu'] == typing.Union[typing.Callable, type(None)]
    assert annos['debug'] == bool
    assert annos['X_test'] == typing.Union[np.ndarray, type(None)]
    assert annos['return'] == type(None)

def test_validate_common_args():
    X_train = np.ones((10, 4))
    Y_train = np.zeros((10, 1))
    X_test = np.ones((5, 4))
    str_cov = 'matern32'
    prior_mu = lambda x: x + 1
    debug = True

    with pytest.raises(AssertionError) as error:
        package_target.validate_common_args(1, Y_train, X_test, str_cov, prior_mu, debug)
    with pytest.raises(AssertionError) as error:
        package_target.validate_common_args(X_train, 1, X_test, str_cov, prior_mu, debug)
    with pytest.raises(AssertionError) as error:
        package_target.validate_common_args(X_train, Y_train, 1, str_cov, prior_mu, debug)
    with pytest.raises(AssertionError) as error:
        package_target.validate_common_args(X_train, Y_train, X_test, 1, prior_mu, debug)
    with pytest.raises(AssertionError) as error:
        package_target.validate_common_args(X_train, Y_train, X_test, str_cov, 1, debug)
    with pytest.raises(AssertionError) as error:
        package_target.validate_common_args(X_train, Y_train, X_test, str_cov, prior_mu, 1)
    with pytest.raises(AssertionError) as error:
        package_target.validate_common_args(X_train, np.zeros(10), X_test, str_cov, prior_mu, debug)
    with pytest.raises(AssertionError) as error:
        package_target.validate_common_args(X_train, np.zeros((3, 1)), X_test, str_cov, prior_mu, debug)
    with pytest.raises(AssertionError) as error:
        package_target.validate_common_args(X_train, Y_train, np.zeros((3, 2)), str_cov, prior_mu, debug)
