# test_gp
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 07, 2020

import numpy as np
import pytest

from bayeso import constants
from bayeso.gp import gp
from bayeso.utils import utils_covariance


TEST_EPSILON = 1e-7

def test_sample_functions():
    num_points = 10
    mu = np.zeros(num_points)
    Sigma = np.eye(num_points)
    num_samples = 5

    with pytest.raises(AssertionError) as error:
        gp.sample_functions(mu, 'abc')
    with pytest.raises(AssertionError) as error:
        gp.sample_functions('abc', Sigma)
    with pytest.raises(AssertionError) as error:
        gp.sample_functions(mu, np.eye(20))
    with pytest.raises(AssertionError) as error:
        gp.sample_functions(mu, np.ones(num_points))
    with pytest.raises(AssertionError) as error:
        gp.sample_functions(np.zeros(20), Sigma)
    with pytest.raises(AssertionError) as error:
        gp.sample_functions(np.eye(10), Sigma)
    with pytest.raises(AssertionError) as error:
        gp.sample_functions(mu, Sigma, num_samples='abc')
    with pytest.raises(AssertionError) as error:
        gp.sample_functions(mu, Sigma, num_samples=1.2)


    functions = gp.sample_functions(mu, Sigma, num_samples=num_samples)
    assert functions.shape[1] == num_points
    assert functions.shape[0] == num_samples

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
        gp.get_optimized_kernel(X, Y, prior_mu, 1)
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(X, Y, 1, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(X, 1, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(1, Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(np.ones(num_X), Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(X, np.ones(num_X), prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(np.ones((50, 3)), Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(X, np.ones((50, 1)), prior_mu, 'se')
    with pytest.raises(ValueError) as error:
        gp.get_optimized_kernel(X, Y, prior_mu, 'abc')
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(X, Y, prior_mu, 'se', str_framework=1)
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(X, Y, prior_mu, 'se', str_optimizer_method=1)
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(X, Y, prior_mu, 'se', str_modelselection_method=1)
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(X, Y, prior_mu, 'se', is_fixed_noise=1)
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(X, Y, prior_mu, 'se', debug=1)

    # INFO: tests for set inputs
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(X_set, Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(X, Y, prior_mu, 'set_se')
    with pytest.raises(AssertionError) as error:
        gp.get_optimized_kernel(X_set, Y, prior_mu, 'set_se', debug=1)

    cov_X_X, inv_cov_X_X, hyps = gp.get_optimized_kernel(X, Y, prior_mu, 'se')
    print(hyps)
    cov_X_X, inv_cov_X_X, hyps = gp.get_optimized_kernel(X, Y, prior_mu, 'se', str_optimizer_method='BFGS')
    print(hyps)
    cov_X_X, inv_cov_X_X, hyps = gp.get_optimized_kernel(X, Y, prior_mu, 'se', str_optimizer_method='L-BFGS-B')
    print(hyps)
    cov_X_X, inv_cov_X_X, hyps = gp.get_optimized_kernel(X, Y, prior_mu, 'se', str_modelselection_method='loocv')
    print(hyps)
    cov_X_X, inv_cov_X_X, hyps = gp.get_optimized_kernel(X_set, Y, prior_mu, 'set_se')
    print(hyps)
    cov_X_X, inv_cov_X_X, hyps = gp.get_optimized_kernel(X_set, Y, prior_mu, 'set_se', str_optimizer_method='L-BFGS-B')
    print(hyps)
    cov_X_X, inv_cov_X_X, hyps = gp.get_optimized_kernel(X_set, Y, prior_mu, 'set_se', str_modelselection_method='loocv')
    print(hyps)

def test_predict_test_():
    np.random.seed(42)
    dim_X = 2
    num_X = 5
    num_X_test = 20
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)
    X_test = np.random.randn(num_X_test, dim_X)
    prior_mu = None
    cov_X_X, inv_cov_X_X, hyps = gp.get_optimized_kernel(X, Y, prior_mu, 'se')
    
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(X, Y, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov='se', prior_mu='abc')
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(X, Y, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov=1, prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(X, Y, X_test, cov_X_X, inv_cov_X_X, 1, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(X, Y, X_test, cov_X_X, 1, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(X, Y, X_test, 1, inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(X, Y, 1, cov_X_X, inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(X, 1, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(1, Y, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(np.random.randn(num_X, 1), Y, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(np.random.randn(10, dim_X), Y, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(X, np.random.randn(10, 1), X_test, cov_X_X, inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(X, Y, X_test, np.random.randn(3, 3), inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(X, Y, X_test, np.random.randn(10), inv_cov_X_X, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(X, Y, X_test, cov_X_X, np.random.randn(10), hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test_(X, Y, X_test, np.random.randn(10), np.random.randn(10), hyps, str_cov='se', prior_mu=prior_mu)

def test_predict_test():
    np.random.seed(42)
    dim_X = 2
    num_X = 5
    num_X_test = 20
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)
    X_test = np.random.randn(num_X_test, dim_X)
    prior_mu = None
    cov_X_X, inv_cov_X_X, hyps = gp.get_optimized_kernel(X, Y, prior_mu, 'se')
    
    with pytest.raises(AssertionError) as error:
        gp.predict_test(X, Y, X_test, hyps, str_cov='se', prior_mu='abc')
    with pytest.raises(AssertionError) as error:
        gp.predict_test(X, Y, X_test, hyps, str_cov=1, prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test(X, Y, X_test, 1, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test(X, Y, 1, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test(X, 1, X_test, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test(1, Y, X_test, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test(np.random.randn(num_X, 1), Y, X_test, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test(np.random.randn(10, dim_X), Y, X_test, hyps, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_test(X, np.random.randn(10, 1), X_test, hyps, str_cov='se', prior_mu=prior_mu)
    
    mu_Xs, sigma_Xs, Sigma_Xs = gp.predict_test(X, Y, X_test, hyps, str_cov='se', prior_mu=prior_mu)
    print(mu_Xs)
    print(sigma_Xs)
    print(Sigma_Xs)

def test_predict_optimized():
    np.random.seed(42)
    dim_X = 2
    num_X = 5
    num_X_test = 20
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)
    X_test = np.random.randn(num_X_test, dim_X)
    prior_mu = None
    
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(X, Y, X_test, str_cov='se', prior_mu='abc')
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(X, Y, X_test, str_cov=1, prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(X, Y, 1, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(X, 1, X_test, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(1, Y, X_test, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(np.random.randn(num_X, 1), Y, X_test, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(np.random.randn(10, dim_X), Y, X_test, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(X, np.random.randn(10, 1), X_test, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(X, Y, X_test, is_fixed_noise=1)
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(X, Y, X_test, debug=1)
    
    mu_Xs, sigma_Xs, Sigma_Xs = gp.predict_optimized(X, Y, X_test)
    print(mu_Xs)
    print(sigma_Xs)
    print(Sigma_Xs)
