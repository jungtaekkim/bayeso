# test_gp
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 07, 2020

import numpy as np
import pytest

from bayeso import constants
from bayeso.gp import gp_common
from bayeso.utils import utils_covariance


TEST_EPSILON = 1e-7

def test_check_str_cov():
    with pytest.raises(AssertionError) as error:
        gp_common._check_str_cov(1, 'se', (2, 1))
    with pytest.raises(AssertionError) as error:
        gp_common._check_str_cov('test', 1, (2, 1))
    with pytest.raises(AssertionError) as error:
        gp_common._check_str_cov('test', 'se', 1)
    with pytest.raises(AssertionError) as error:
        gp_common._check_str_cov('test', 'se', (2, 100, 100))
    with pytest.raises(AssertionError) as error:
        gp_common._check_str_cov('test', 'se', (2, 100), shape_X2=(2, 100, 100))
    with pytest.raises(AssertionError) as error:
        gp_common._check_str_cov('test', 'set_se', (2, 100), shape_X2=(2, 100, 100))
    with pytest.raises(AssertionError) as error:
        gp_common._check_str_cov('test', 'set_se', (2, 100, 100), shape_X2=(2, 100))
    with pytest.raises(AssertionError) as error:
        gp_common._check_str_cov('test', 'se', (2, 1), shape_X2=1)

    with pytest.raises(ValueError) as error:
        gp_common._check_str_cov('test', 'abc', (2, 1))

def test_get_prior_mu():
    fun_prior = lambda X: np.expand_dims(np.linalg.norm(X, axis=1), axis=1)
    fun_prior_1d = lambda X: np.linalg.norm(X, axis=1)
    X = np.reshape(np.arange(0, 90), (30, 3))

    with pytest.raises(AssertionError) as error:
        gp_common.get_prior_mu(1, X)
    with pytest.raises(AssertionError) as error:
        gp_common.get_prior_mu(fun_prior, 1)
    with pytest.raises(AssertionError) as error:
        gp_common.get_prior_mu(fun_prior, np.arange(0, 100))
    with pytest.raises(AssertionError) as error:
        gp_common.get_prior_mu(None, np.arange(0, 100))
    with pytest.raises(AssertionError) as error:
        gp_common.get_prior_mu(fun_prior_1d, X)

    assert (gp_common.get_prior_mu(None, X) == np.zeros((X.shape[0], 1))).all()
    assert (gp_common.get_prior_mu(fun_prior, X) == fun_prior(X)).all()

def test_get_kernel_inverse():
    dim_X = 3
    X = np.reshape(np.arange(0, 9), (3, dim_X))
    hyps = utils_covariance.get_hyps('se', dim_X)

    with pytest.raises(AssertionError) as error:
        gp_common.get_kernel_inverse(1, hyps, 'se')
    with pytest.raises(AssertionError) as error:
        gp_common.get_kernel_inverse(np.arange(0, 100), hyps, 'se')
    with pytest.raises(AssertionError) as error:
        gp_common.get_kernel_inverse(X, 1, 'se')
    with pytest.raises(AssertionError) as error:
        gp_common.get_kernel_inverse(X, hyps, 1)
    with pytest.raises(ValueError) as error:
        gp_common.get_kernel_inverse(X, hyps, 'abc')
    with pytest.raises(AssertionError) as error:
        gp_common.get_kernel_inverse(X, hyps, 'se', debug=1)
    with pytest.raises(AssertionError) as error:
        gp_common.get_kernel_inverse(X, hyps, 'se', is_gradient='abc')
    with pytest.raises(AssertionError) as error:
        gp_common.get_kernel_inverse(X, hyps, 'se', is_fixed_noise='abc')

    cov_X_X, inv_cov_X_X, grad_cov_X_X = gp_common.get_kernel_inverse(X, hyps, 'se')
    print(cov_X_X)
    print(inv_cov_X_X)
    truth_cov_X_X = np.array([
        [1.00011000e+00, 1.37095909e-06, 3.53262857e-24],
        [1.37095909e-06, 1.00011000e+00, 1.37095909e-06],
        [3.53262857e-24, 1.37095909e-06, 1.00011000e+00]
    ])
    truth_inv_cov_X_X = np.array([
        [9.99890012e-01, -1.37065753e-06, 1.87890871e-12],
        [-1.37065753e-06, 9.99890012e-01, -1.37065753e-06],
        [1.87890871e-12, -1.37065753e-06, 9.99890012e-01]
    ])
    assert (np.abs(cov_X_X - truth_cov_X_X) < TEST_EPSILON).all()
    assert (np.abs(inv_cov_X_X - truth_inv_cov_X_X) < TEST_EPSILON).all()
    assert cov_X_X.shape == inv_cov_X_X.shape

    cov_X_X, inv_cov_X_X, grad_cov_X_X = gp_common.get_kernel_inverse(X, hyps, 'se', is_gradient=True, is_fixed_noise=True)
    print(grad_cov_X_X)
    print(grad_cov_X_X.shape)

    truth_grad_cov_X_X = np.array([
        [
            [2.00002000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [2.74191817e-06, 3.70158953e-05, 3.70158953e-05, 3.70158953e-05],
            [7.06525714e-24, 3.81523886e-22, 3.81523886e-22, 3.81523886e-22]
        ], [
            [2.74191817e-06, 3.70158953e-05, 3.70158953e-05, 3.70158953e-05],
            [2.00002000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [2.74191817e-06, 3.70158953e-05, 3.70158953e-05, 3.70158953e-05]
        ], [
            [7.06525714e-24, 3.81523886e-22, 3.81523886e-22, 3.81523886e-22],
            [2.74191817e-06, 3.70158953e-05, 3.70158953e-05, 3.70158953e-05],
            [2.00002000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]
        ]
    ])
    assert (np.abs(cov_X_X - truth_cov_X_X) < TEST_EPSILON).all()
    assert (np.abs(inv_cov_X_X - truth_inv_cov_X_X) < TEST_EPSILON).all()
    assert (np.abs(grad_cov_X_X - truth_grad_cov_X_X) < TEST_EPSILON).all()
    assert cov_X_X.shape == inv_cov_X_X.shape == grad_cov_X_X.shape[:2]

def test_get_kernel_cholesky():
    dim_X = 3
    X = np.reshape(np.arange(0, 9), (3, dim_X))
    hyps = utils_covariance.get_hyps('se', dim_X)

    with pytest.raises(AssertionError) as error:
        gp_common.get_kernel_cholesky(1, hyps, 'se')
    with pytest.raises(AssertionError) as error:
        gp_common.get_kernel_cholesky(np.arange(0, 10), hyps, 'se')
    with pytest.raises(AssertionError) as error:
        gp_common.get_kernel_cholesky(X, 1, 'se')
    with pytest.raises(AssertionError) as error:
        gp_common.get_kernel_cholesky(X, hyps, 1)
    with pytest.raises(ValueError) as error:
        gp_common.get_kernel_cholesky(X, hyps, 'abc')
    with pytest.raises(AssertionError) as error:
        gp_common.get_kernel_cholesky(X, hyps, 'se', debug=1)

    cov_X_X, lower, _ = gp_common.get_kernel_cholesky(X, hyps, 'se')
    print(cov_X_X)
    print(lower)
    truth_cov_X_X = [
        [1.00011000e+00, 1.37095909e-06, 3.53262857e-24],
        [1.37095909e-06, 1.00011000e+00, 1.37095909e-06],
        [3.53262857e-24, 1.37095909e-06, 1.00011000e+00],
    ]
    truth_lower = [
        [1.00005500e+00, 0.00000000e+00, 0.00000000e+00],
        [1.37088369e-06, 1.00005500e+00, 0.00000000e+00],
        [3.53243429e-24, 1.37088369e-06, 1.00005500e+00],
    ]
    assert (np.abs(cov_X_X - truth_cov_X_X) < TEST_EPSILON).all()
    assert (np.abs(lower - truth_lower) < TEST_EPSILON).all()
    assert cov_X_X.shape == lower.shape
