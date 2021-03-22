#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: March 22, 2021
#
"""test_tp"""

import typing
import pytest
import numpy as np

from bayeso import constants
from bayeso.tp import tp as package_target
from bayeso.tp import tp_kernel
from bayeso.utils import utils_covariance


TEST_EPSILON = 1e-7

def test_sample_functions_typing():
    annos = package_target.sample_functions.__annotations__

    assert annos['nu'] == float
    assert annos['mu'] == np.ndarray
    assert annos['Sigma'] == np.ndarray
    assert annos['num_samples'] == int
    assert annos['return'] == np.ndarray

def test_sample_functions():
    num_points = 10
    nu = 4.0
    mu = np.zeros(num_points)
    Sigma = np.eye(num_points)
    num_samples = 20

    with pytest.raises(AssertionError) as error:
        package_target.sample_functions(nu, mu, 'abc')
    with pytest.raises(AssertionError) as error:
        package_target.sample_functions(nu, 'abc', Sigma)
    with pytest.raises(AssertionError) as error:
        package_target.sample_functions('abc', mu, Sigma)
    with pytest.raises(AssertionError) as error:
        package_target.sample_functions(4, mu, Sigma)
    with pytest.raises(AssertionError) as error:
        package_target.sample_functions(nu, mu, np.eye(20))
    with pytest.raises(AssertionError) as error:
        package_target.sample_functions(nu, mu, np.ones(num_points))
    with pytest.raises(AssertionError) as error:
        package_target.sample_functions(nu, np.zeros(20), Sigma)
    with pytest.raises(AssertionError) as error:
        package_target.sample_functions(nu, np.eye(10), Sigma)
    with pytest.raises(AssertionError) as error:
        package_target.sample_functions(nu, mu, Sigma, num_samples='abc')
    with pytest.raises(AssertionError) as error:
        package_target.sample_functions(nu, mu, Sigma, num_samples=1.2)


    functions = package_target.sample_functions(nu, mu, Sigma, num_samples=num_samples)
    assert functions.shape[1] == num_points
    assert functions.shape[0] == num_samples

    functions = package_target.sample_functions(np.inf, mu, Sigma, num_samples=num_samples)
    assert functions.shape[1] == num_points
    assert functions.shape[0] == num_samples

def test_predict_with_cov_typing():
    annos = package_target.predict_with_cov.__annotations__

    assert annos['X_train'] == np.ndarray
    assert annos['Y_train'] == np.ndarray
    assert annos['X_test'] == np.ndarray
    assert annos['cov_X_X'] == np.ndarray
    assert annos['inv_cov_X_X'] == np.ndarray
    assert annos['hyps'] == dict
    assert annos['str_cov'] == str
    assert annos['prior_mu'] == typing.Union[callable, type(None)]
    assert annos['debug'] == bool
    assert annos['return'] == typing.Tuple[float, np.ndarray, np.ndarray, np.ndarray]

def test_predict_with_cov():
    np.random.seed(42)
    dim_X = 2
    num_X = 5
    num_X_test = 20
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)
    X_test = np.random.randn(num_X_test, dim_X)
    prior_mu = None
    cov_X_X, inv_cov_X_X, hyps = tp_kernel.get_optimized_kernel(X, Y, prior_mu, 'se')

    with pytest.raises(AssertionError) as error:
        package_target.predict_with_cov(X, Y, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov='se', prior_mu='abc')
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_cov(X, Y, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov=1, prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_cov(X, Y, X_test, cov_X_X, inv_cov_X_X, 1, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_cov(X, Y, X_test, cov_X_X, 1, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_cov(X, Y, X_test, 1, inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)

    with pytest.raises(AssertionError) as error:
        package_target.predict_with_cov(X, Y, 1, cov_X_X, inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_cov(X, 1, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_cov(1, Y, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_cov(np.random.randn(num_X, 1), Y, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_cov(np.random.randn(10, dim_X), Y, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)

    with pytest.raises(AssertionError) as error:
        package_target.predict_with_cov(X, np.random.randn(10, 1), X_test, cov_X_X, inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_cov(X, Y, X_test, np.random.randn(3, 3), inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_cov(X, Y, X_test, np.random.randn(10), inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_cov(X, Y, X_test, cov_X_X, np.random.randn(10), hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_cov(X, Y, X_test, np.random.randn(10), np.random.randn(10), hyps, str_cov='se', prior_mu=prior_mu)

    nu_test, mu_test, sigma_test, Sigma_test = package_target.predict_with_cov(X, Y, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    print(nu_test)
    print(mu_test)
    print(sigma_test)
    print(Sigma_test)

def test_predict_with_hyps_typing():
    annos = package_target.predict_with_hyps.__annotations__

    assert annos['X_train'] == np.ndarray
    assert annos['Y_train'] == np.ndarray
    assert annos['X_test'] == np.ndarray
    assert annos['hyps'] == dict
    assert annos['str_cov'] == str
    assert annos['prior_mu'] == typing.Union[callable, type(None)]
    assert annos['debug'] == bool
    assert annos['return'] == typing.Tuple[float, np.ndarray, np.ndarray, np.ndarray]

def test_predict_with_hyps():
    np.random.seed(42)
    dim_X = 2
    num_X = 5
    num_X_test = 20
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)
    X_test = np.random.randn(num_X_test, dim_X)
    prior_mu = None
    cov_X_X, inv_cov_X_X, hyps = tp_kernel.get_optimized_kernel(X, Y, prior_mu, 'se')

    with pytest.raises(AssertionError) as error:
        package_target.predict_with_hyps(X, Y, X_test, hyps, str_cov='se', prior_mu='abc')
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_hyps(X, Y, X_test, hyps, str_cov=1, prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_hyps(X, Y, X_test, 1, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_hyps(X, Y, 1, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_hyps(X, 1, X_test, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_hyps(1, Y, X_test, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_hyps(np.random.randn(num_X, 1), Y, X_test, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_hyps(np.random.randn(10, dim_X), Y, X_test, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_hyps(X, np.random.randn(10, 1), X_test, hyps, str_cov='se', prior_mu=prior_mu)

    nu_test, mu_test, sigma_test, Sigma_test = package_target.predict_with_hyps(X, Y, X_test, hyps, str_cov='se', prior_mu=prior_mu)
    print(nu_test)
    print(mu_test)
    print(sigma_test)
    print(Sigma_test)

def test_predict_with_optimized_hyps_typing():
    annos = package_target.predict_with_optimized_hyps.__annotations__

    assert annos['X_train'] == np.ndarray
    assert annos['Y_train'] == np.ndarray
    assert annos['X_test'] == np.ndarray
    assert annos['str_cov'] == str
    assert annos['str_optimizer_method'] == str
    assert annos['prior_mu'] == typing.Union[callable, type(None)]
    assert annos['fix_noise'] == float
    assert annos['debug'] == bool
    assert annos['return'] == typing.Tuple[float, np.ndarray, np.ndarray, np.ndarray]

def test_predict_with_optimized_hyps():
    np.random.seed(42)
    dim_X = 2
    num_X = 5
    num_X_test = 20
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)
    X_test = np.random.randn(num_X_test, dim_X)
    prior_mu = None

    with pytest.raises(AssertionError) as error:
        package_target.predict_with_optimized_hyps(X, Y, X_test, str_cov='se', prior_mu='abc')
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_optimized_hyps(X, Y, X_test, str_cov=1, prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_optimized_hyps(X, Y, 1, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_optimized_hyps(X, 1, X_test, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_optimized_hyps(1, Y, X_test, str_cov='se', prior_mu=prior_mu)

    with pytest.raises(AssertionError) as error:
        package_target.predict_with_optimized_hyps(np.random.randn(num_X, 1), Y, X_test, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_optimized_hyps(np.random.randn(10, dim_X), Y, X_test, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_optimized_hyps(X, np.random.randn(10, 1), X_test, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_optimized_hyps(X, Y, X_test, str_optimizer_method=1)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_optimized_hyps(X, Y, X_test, fix_noise=1)
    with pytest.raises(AssertionError) as error:
        package_target.predict_with_optimized_hyps(X, Y, X_test, debug=1)

    nu_test, mu_test, sigma_test, Sigma_test = package_target.predict_with_optimized_hyps(X, Y, X_test, debug=True)
    print(nu_test)
    print(mu_test)
    print(sigma_test)
    print(Sigma_test)
