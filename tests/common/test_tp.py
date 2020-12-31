#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: December 31, 2020
#
"""test_tp"""

import typing
import pytest
import numpy as np

from bayeso import constants
from bayeso.tp import tp as package_target
from bayeso.utils import utils_covariance


TEST_EPSILON = 1e-7

def test_neg_log_ml_typing():
    annos = package_target.neg_log_ml.__annotations__

    assert annos['X_train'] == np.ndarray
    assert annos['Y_train'] == np.ndarray
    assert annos['hyps'] == np.ndarray
    assert annos['str_cov'] == str
    assert annos['prior_mu_train'] == np.ndarray
    assert annos['fix_noise'] == bool
    assert annos['use_gradient'] == bool
    assert annos['debug'] == bool
    assert annos['return'] == typing.Union[float, typing.Tuple[float, np.ndarray]]

def test_neg_log_ml():
    dim_X = 3
    str_cov = 'se'
    X = np.reshape(np.arange(0, 9), (3, dim_X))
    Y = np.expand_dims(np.arange(3, 10, 3), axis=1)
    fix_noise = False
    use_gp = False

    dict_hyps = utils_covariance.get_hyps(str_cov, dim_X, use_gp=use_gp)
    arr_hyps = utils_covariance.convert_hyps(str_cov, dict_hyps, fix_noise=fix_noise, use_gp=use_gp)
    prior_mu_X = np.zeros((3, 1))

    with pytest.raises(AssertionError) as error:
        package_target.neg_log_ml(np.arange(0, 3), Y, arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        package_target.neg_log_ml(X, np.arange(0, 3), arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        package_target.neg_log_ml(X, Y, dict_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        package_target.neg_log_ml(X, Y, arr_hyps, 1, prior_mu_X)
    with pytest.raises(ValueError) as error:
        package_target.neg_log_ml(X, Y, arr_hyps, 'abc', prior_mu_X)
    with pytest.raises(AssertionError) as error:
        package_target.neg_log_ml(X, Y, arr_hyps, str_cov, np.arange(0, 3))
    with pytest.raises(AssertionError) as error:
        package_target.neg_log_ml(np.reshape(np.arange(0, 12), (4, dim_X)), Y, arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        package_target.neg_log_ml(X, np.expand_dims(np.arange(0, 4), axis=1), arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        package_target.neg_log_ml(X, Y, arr_hyps, str_cov, np.expand_dims(np.arange(0, 4), axis=1))
    with pytest.raises(AssertionError) as error:
        package_target.neg_log_ml(X, Y, arr_hyps, str_cov, prior_mu_X, fix_noise=1)
    with pytest.raises(AssertionError) as error:
        package_target.neg_log_ml(X, Y, arr_hyps, str_cov, prior_mu_X, debug=1)

    neg_log_ml_ = package_target.neg_log_ml(X, Y, arr_hyps, str_cov, prior_mu_X, fix_noise=fix_noise, use_gradient=False)
    print(neg_log_ml_)
    truth_log_ml_ = 5.634155417555853
    assert np.abs(neg_log_ml_ - truth_log_ml_) < TEST_EPSILON

    neg_log_ml_, neg_grad_log_ml_ = package_target.neg_log_ml(X, Y, arr_hyps, str_cov, prior_mu_X, fix_noise=fix_noise, use_gradient=True)
    print(neg_log_ml_)
    print(neg_grad_log_ml_)

    truth_log_ml_ = 5.634155417555853
    truth_grad_log_ml_ = np.array([
        -1.60446383e-02,
        1.75087448e-01,
        -1.60448396e+00,
        -5.50871167e-05,
        -5.50871167e-05,
        -5.50871167e-05,
    ])
    assert np.abs(neg_log_ml_ - truth_log_ml_) < TEST_EPSILON
    assert np.all(np.abs(neg_grad_log_ml_ - truth_grad_log_ml_) < TEST_EPSILON)

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

def test_get_optimized_kernel_typing():
    annos = package_target.get_optimized_kernel.__annotations__

    assert annos['X_train'] == np.ndarray
    assert annos['Y_train'] == np.ndarray
    assert annos['prior_mu'] == typing.Union[callable, type(None)]
    assert annos['str_cov'] == str
    assert annos['str_optimizer_method'] == str
    assert annos['fix_noise'] == bool
    assert annos['debug'] == bool
    assert annos['return'] == typing.Tuple[np.ndarray, np.ndarray, dict]

def test_get_optimized_kernel():
    np.random.seed(42)
    dim_X = 3
    num_X = 10
    num_instances = 5
    X = np.random.randn(num_X, dim_X)
    X_set = np.random.randn(num_X, num_instances, dim_X)
    Y = np.random.randn(num_X, 1)
    prior_mu = None

    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(X, Y, prior_mu, 1)
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(X, Y, 1, 'se')
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(X, 1, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(1, Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(np.ones(num_X), Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(X, np.ones(num_X), prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(np.ones((50, 3)), Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(X, np.ones((50, 1)), prior_mu, 'se')
    with pytest.raises(ValueError) as error:
        package_target.get_optimized_kernel(X, Y, prior_mu, 'abc')
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(X, Y, prior_mu, 'se', str_optimizer_method=1)
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(X, Y, prior_mu, 'se', fix_noise=1)
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(X, Y, prior_mu, 'se', debug=1)

    # INFO: tests for set inputs
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(X_set, Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(X, Y, prior_mu, 'set_se')
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(X_set, Y, prior_mu, 'set_se', debug=1)

    cov_X_X, inv_cov_X_X, hyps = package_target.get_optimized_kernel(X, Y, prior_mu, 'se')
    print(hyps)
    cov_X_X, inv_cov_X_X, hyps = package_target.get_optimized_kernel(X, Y, prior_mu, 'eq')
    print(hyps)
    cov_X_X, inv_cov_X_X, hyps = package_target.get_optimized_kernel(X, Y, prior_mu, 'matern32')
    print(hyps)
    cov_X_X, inv_cov_X_X, hyps = package_target.get_optimized_kernel(X, Y, prior_mu, 'matern52')
    print(hyps)

    cov_X_X, inv_cov_X_X, hyps = package_target.get_optimized_kernel(X, Y, prior_mu, 'se', str_optimizer_method='L-BFGS-B')
    print(hyps)

    cov_X_X, inv_cov_X_X, hyps = package_target.get_optimized_kernel(X, Y, prior_mu, 'se', str_optimizer_method='SLSQP')
    print(hyps)

#    cov_X_X, inv_cov_X_X, hyps = package_target.get_optimized_kernel(X_set, Y, prior_mu, 'set_se')
#    print(hyps)
    cov_X_X, inv_cov_X_X, hyps = package_target.get_optimized_kernel(X_set, Y, prior_mu, 'set_se', str_optimizer_method='L-BFGS-B')
    print(hyps)

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
    cov_X_X, inv_cov_X_X, hyps = package_target.get_optimized_kernel(X, Y, prior_mu, 'se')

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
    cov_X_X, inv_cov_X_X, hyps = package_target.get_optimized_kernel(X, Y, prior_mu, 'se')

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
