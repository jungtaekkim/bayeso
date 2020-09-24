#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: September 24, 2020
#
"""test_gp_common"""

import typing
import pytest
import numpy as np

from bayeso.gp import gp_common
from bayeso.utils import utils_covariance

TEST_EPSILON = 1e-7


def test_get_kernel_inverse_typing():
    annos = gp_common.get_kernel_inverse.__annotations__

    assert annos['X_train'] == np.ndarray
    assert annos['hyps'] == dict
    assert annos['str_cov'] == str
    assert annos['fix_noise'] == bool
    assert annos['use_gradient'] == bool
    assert annos['debug'] == bool
    assert annos['return'] == typing.Tuple[np.ndarray, np.ndarray, np.ndarray]

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
        gp_common.get_kernel_inverse(X, hyps, 'se', use_gradient='abc')
    with pytest.raises(AssertionError) as error:
        gp_common.get_kernel_inverse(X, hyps, 'se', fix_noise='abc')

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

    cov_X_X, inv_cov_X_X, grad_cov_X_X = gp_common.get_kernel_inverse(X, hyps, 'se', use_gradient=True, fix_noise=True)
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

def test_get_kernel_cholesky_typing():
    annos = gp_common.get_kernel_cholesky.__annotations__

    assert annos['X_train'] == np.ndarray
    assert annos['hyps'] == dict
    assert annos['str_cov'] == str
    assert annos['fix_noise'] == bool
    assert annos['use_gradient'] == bool
    assert annos['debug'] == bool
    assert annos['return'] == typing.Tuple[np.ndarray, np.ndarray, np.ndarray]

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
