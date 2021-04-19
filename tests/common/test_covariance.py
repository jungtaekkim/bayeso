#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: December 29, 2020
#
"""test_covariance"""

import typing
import pytest
import numpy as np

from bayeso import covariance as package_target
from bayeso.utils import utils_covariance


TEST_EPSILON = 1e-7

def test_choose_fun_cov_typing():
    annos = package_target.choose_fun_cov.__annotations__

    assert annos['str_cov'] == str
    assert annos['return'] == typing.Callable

def test_choose_fun_cov():
    with pytest.raises(AssertionError) as error:
        package_target.choose_fun_cov(123)
    with pytest.raises(NotImplementedError) as error:
        package_target.choose_fun_cov('abc')

    assert package_target.choose_fun_cov('se') == package_target.cov_se
    assert package_target.choose_fun_cov('matern32') == package_target.cov_matern32
    assert package_target.choose_fun_cov('matern52') == package_target.cov_matern52

def test_choose_fun_grad_cov_typing():
    annos = package_target.choose_fun_grad_cov.__annotations__

    assert annos['str_cov'] == str
    assert annos['return'] == typing.Callable

def test_choose_fun_grad_cov():
    with pytest.raises(AssertionError) as error:
        package_target.choose_fun_grad_cov(123)
    with pytest.raises(NotImplementedError) as error:
        package_target.choose_fun_grad_cov('abc')

    assert package_target.choose_fun_grad_cov('se') == package_target.grad_cov_se
    assert package_target.choose_fun_grad_cov('matern32') == package_target.grad_cov_matern32
    assert package_target.choose_fun_grad_cov('matern52') == package_target.grad_cov_matern52

def test_get_kernel_inverse_typing():
    annos = package_target.get_kernel_inverse.__annotations__

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
        package_target.get_kernel_inverse(1, hyps, 'se')
    with pytest.raises(AssertionError) as error:
        package_target.get_kernel_inverse(np.arange(0, 100), hyps, 'se')
    with pytest.raises(AssertionError) as error:
        package_target.get_kernel_inverse(X, 1, 'se')
    with pytest.raises(AssertionError) as error:
        package_target.get_kernel_inverse(X, hyps, 1)
    with pytest.raises(ValueError) as error:
        package_target.get_kernel_inverse(X, hyps, 'abc')
    with pytest.raises(AssertionError) as error:
        package_target.get_kernel_inverse(X, hyps, 'se', debug=1)
    with pytest.raises(AssertionError) as error:
        package_target.get_kernel_inverse(X, hyps, 'se', use_gradient='abc')
    with pytest.raises(AssertionError) as error:
        package_target.get_kernel_inverse(X, hyps, 'se', fix_noise='abc')

    cov_X_X, inv_cov_X_X, grad_cov_X_X = package_target.get_kernel_inverse(X, hyps, 'se')
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

    cov_X_X, inv_cov_X_X, grad_cov_X_X = package_target.get_kernel_inverse(X, hyps, 'se', use_gradient=True, fix_noise=True)
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
    annos = package_target.get_kernel_cholesky.__annotations__

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
        package_target.get_kernel_cholesky(1, hyps, 'se')
    with pytest.raises(AssertionError) as error:
        package_target.get_kernel_cholesky(np.arange(0, 10), hyps, 'se')
    with pytest.raises(AssertionError) as error:
        package_target.get_kernel_cholesky(X, 1, 'se')
    with pytest.raises(AssertionError) as error:
        package_target.get_kernel_cholesky(X, hyps, 1)
    with pytest.raises(ValueError) as error:
        package_target.get_kernel_cholesky(X, hyps, 'abc')
    with pytest.raises(AssertionError) as error:
        package_target.get_kernel_cholesky(X, hyps, 'se', debug=1)

    cov_X_X, lower, _ = package_target.get_kernel_cholesky(X, hyps, 'se')
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

def test_cov_se_typing():
    annos = package_target.cov_se.__annotations__

    assert annos['X'] == np.ndarray
    assert annos['Xp'] == np.ndarray
    assert annos['lengthscales'] == typing.Union[np.ndarray, float]
    assert annos['signal'] == float
    assert annos['return'] == np.ndarray

def test_cov_se():
    with pytest.raises(AssertionError) as error:
        package_target.cov_se(np.zeros((1, 2)), np.zeros((1, 2)), np.array([1.0, 1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        package_target.cov_se(np.zeros((1, 2)), np.zeros((1, 3)), np.array([1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        package_target.cov_se(np.zeros((1, 3)), np.zeros((1, 2)), np.array([1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        package_target.cov_se(np.zeros((1, 2)), np.zeros((1, 2)), np.array([1.0, 1.0]), 1)
    assert np.abs(package_target.cov_se(np.zeros((1, 2)), np.zeros((1, 2)), 1.0, 0.1)[0] - 0.01) < TEST_EPSILON

    X = np.array([[1.0, 2.0, 0.0]])
    Xp = np.array([[2.0, 1.0, 1.0]])
    cur_hyps = utils_covariance.get_hyps('se', 3)
    cov_ = package_target.cov_se(X, Xp, cur_hyps['lengthscales'], cur_hyps['signal'])
    print(cov_)
    truth_cov_ = 0.22313016014842987
    assert np.abs(cov_[0] - truth_cov_) < TEST_EPSILON

    X = np.array([[1.0, 2.0, 0.0]])
    Xp = np.array([[2.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    cur_hyps = utils_covariance.get_hyps('se', 3)
    cur_hyps['lengthscales'] = 1.0
    cov_ = package_target.cov_se(X, Xp, cur_hyps['lengthscales'], cur_hyps['signal'])
    print(cov_)
    truth_cov_ = np.array([[0.22313016, 0.082085]])
    assert np.all(np.abs(cov_[0] - truth_cov_) < TEST_EPSILON)

def test_grad_cov_se_typing():
    annos = package_target.grad_cov_se.__annotations__

    assert annos['cov_X_Xp'] == np.ndarray
    assert annos['X'] == np.ndarray
    assert annos['Xp'] == np.ndarray
    assert annos['hyps'] == dict
    assert annos['num_hyps'] == int
    assert annos['fix_noise'] == bool
    assert annos['return'] == np.ndarray

def test_grad_cov_se():
    str_cov = 'se'
    cur_hyps = utils_covariance.get_hyps(str_cov, 2)
    X_train = np.array([
        [2.0, 1.0],
        [1.0, 1.0],
    ])
    num_hyps = X_train.shape[1] + 1
    cov_ = package_target.cov_main(str_cov, X_train, X_train, cur_hyps, True)
    print(cov_)

    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_se('abc', X_train, X_train, cur_hyps, num_hyps, True)
    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_se(cov_, 'abc', X_train, cur_hyps, num_hyps, True)
    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_se(cov_, X_train, 'abc', cur_hyps, num_hyps, True)
    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_se(cov_, X_train, X_train, 'abc', num_hyps, True)
    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_se(cov_, X_train, X_train, cur_hyps, 'abc', True)
    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_se(cov_, X_train, X_train, cur_hyps, num_hyps, 'abc')

    num_hyps = X_train.shape[1] + 2
    grad_cov_ = package_target.grad_cov_se(cov_, X_train, X_train, cur_hyps, num_hyps, False)
    print(grad_cov_)

    truth_grad_cov_ = np.array([
        [
            [0.02, 2.00002, 0., 0.],
            [0., 1.21306132, 0.60653066, 0.60653066],
        ], [
            [0., 1.21306132, 0.60653066, 0.60653066],
            [0.02, 2.00002, 0., 0.]
        ]
    ])

    assert np.all(np.abs(truth_grad_cov_ - grad_cov_) < TEST_EPSILON)

def test_cov_matern32_typing():
    annos = package_target.cov_matern32.__annotations__

    assert annos['X'] == np.ndarray
    assert annos['Xp'] == np.ndarray
    assert annos['lengthscales'] == typing.Union[np.ndarray, float]
    assert annos['signal'] == float
    assert annos['return'] == np.ndarray

def test_cov_matern32():
    with pytest.raises(AssertionError) as error:
        package_target.cov_matern32(np.zeros((1, 2)), np.zeros((1, 2)), np.array([1.0, 1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        package_target.cov_matern32(np.zeros((1, 2)), np.zeros((1, 3)), np.array([1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        package_target.cov_matern32(np.zeros((1, 3)), np.zeros((1, 2)), np.array([1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        package_target.cov_matern32(np.zeros((1, 2)), np.zeros((1, 2)), np.array([1.0, 1.0]), 1)
    assert np.abs(package_target.cov_matern32(np.zeros((1, 2)), np.zeros((1, 2)), 1.0, 0.1)[0] - 0.01) < TEST_EPSILON

    X = np.array([[1.0, 2.0, 0.0]])
    Xp = np.array([[2.0, 1.0, 1.0]])
    cur_hyps = utils_covariance.get_hyps('matern32', 3)
    cov_ = package_target.cov_matern32(X, Xp, cur_hyps['lengthscales'], cur_hyps['signal'])
    print(cov_)
    truth_cov_ = 0.19914827347145583
    assert np.abs(cov_[0] - truth_cov_) < TEST_EPSILON

    X = np.array([[1.0, 2.0, 0.0]])
    Xp = np.array([[2.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    cur_hyps = utils_covariance.get_hyps('matern32', 3)
    cur_hyps['lengthscales'] = 1.0
    cov_ = package_target.cov_matern32(X, Xp, cur_hyps['lengthscales'], cur_hyps['signal'])
    print(cov_)
    truth_cov_ = np.array([[0.19914827, 0.1013397]])
    assert np.all(np.abs(cov_[0] - truth_cov_) < TEST_EPSILON)

def test_grad_cov_matern32_typing():
    annos = package_target.grad_cov_matern32.__annotations__

    assert annos['cov_X_Xp'] == np.ndarray
    assert annos['X'] == np.ndarray
    assert annos['Xp'] == np.ndarray
    assert annos['hyps'] == dict
    assert annos['num_hyps'] == int
    assert annos['fix_noise'] == bool
    assert annos['return'] == np.ndarray

def test_grad_cov_matern32():
    str_cov = 'matern32'
    cur_hyps = utils_covariance.get_hyps(str_cov, 2)
    X_train = np.array([
        [2.0, 1.0],
        [1.0, 1.0],
    ])
    num_hyps = X_train.shape[1] + 1
    cov_ = package_target.cov_main(str_cov, X_train, X_train, cur_hyps, True)
    print(cov_)

    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_matern32('abc', X_train, X_train, cur_hyps, num_hyps, True)
    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_matern32(cov_, 'abc', X_train, cur_hyps, num_hyps, True)
    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_matern32(cov_, X_train, 'abc', cur_hyps, num_hyps, True)
    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_matern32(cov_, X_train, X_train, 'abc', num_hyps, True)
    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_matern32(cov_, X_train, X_train, cur_hyps, 'abc', True)
    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_matern32(cov_, X_train, X_train, cur_hyps, num_hyps, 'abc')

    num_hyps = X_train.shape[1] + 2
    grad_cov_ = package_target.grad_cov_matern32(cov_, X_train, X_train, cur_hyps, num_hyps, False)
    print(grad_cov_)

    truth_grad_cov_ = np.array([
        [
            [0.02, 2.00002, 0., 0.],
            [0., 0.96671545, 0.53076362, 0.53076362],
        ], [
            [0., 0.96671545, 0.53076362, 0.53076362],
            [0.02, 2.00002, 0., 0.]
        ]
    ])

    assert np.all(np.abs(truth_grad_cov_ - grad_cov_) < TEST_EPSILON)

    num_hyps = X_train.shape[1] + 1
    grad_cov_ = package_target.grad_cov_matern32(cov_, X_train, X_train, cur_hyps, num_hyps, True)
    print(grad_cov_)

    truth_grad_cov_ = np.array([
        [
            [2.00002, 0., 0.],
            [0.96671545, 0.53076362, 0.53076362],
        ], [
            [0.96671545, 0.53076362, 0.53076362],
            [2.00002, 0., 0.]
        ]
    ])

    assert np.all(np.abs(truth_grad_cov_ - grad_cov_) < TEST_EPSILON)

def test_cov_matern52_typing():
    annos = package_target.cov_matern52.__annotations__

    assert annos['X'] == np.ndarray
    assert annos['Xp'] == np.ndarray
    assert annos['lengthscales'] == typing.Union[np.ndarray, float]
    assert annos['signal'] == float
    assert annos['return'] == np.ndarray

def test_cov_matern52():
    with pytest.raises(AssertionError) as error:
        package_target.cov_matern52(np.zeros((1, 2)), np.zeros((1, 2)), np.array([1.0, 1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        package_target.cov_matern52(np.zeros((1, 2)), np.zeros((1, 3)), np.array([1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        package_target.cov_matern52(np.zeros((1, 3)), np.zeros((1, 2)), np.array([1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        package_target.cov_matern52(np.zeros((1, 2)), np.zeros((1, 2)), np.array([1.0, 1.0]), 1)
    assert np.abs(package_target.cov_matern52(np.zeros((1, 2)), np.zeros((1, 2)), 1.0, 0.1)[0] - 0.01) < TEST_EPSILON

    X = np.array([[1.0, 2.0, 0.0]])
    Xp = np.array([[2.0, 1.0, 1.0]])
    cur_hyps = utils_covariance.get_hyps('matern52', 3)
    cov_ = package_target.cov_matern52(X, Xp, cur_hyps['lengthscales'], cur_hyps['signal'])
    print(cov_)
    truth_cov_ = 0.20532087608359792
    assert np.abs(cov_[0] - truth_cov_) < TEST_EPSILON

    X = np.array([[1.0, 2.0, 0.0]])
    Xp = np.array([[2.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    cur_hyps = utils_covariance.get_hyps('matern52', 3)
    cur_hyps['lengthscales'] = 1.0
    cov_ = package_target.cov_matern52(X, Xp, cur_hyps['lengthscales'], cur_hyps['signal'])
    print(cov_)
    truth_cov_ = np.array([[0.20532088, 0.09657724]])
    assert np.all(np.abs(cov_[0] - truth_cov_) < TEST_EPSILON)

def test_grad_cov_matern52_typing():
    annos = package_target.grad_cov_matern52.__annotations__

    assert annos['cov_X_Xp'] == np.ndarray
    assert annos['X'] == np.ndarray
    assert annos['Xp'] == np.ndarray
    assert annos['hyps'] == dict
    assert annos['num_hyps'] == int
    assert annos['fix_noise'] == bool
    assert annos['return'] == np.ndarray

def test_grad_cov_matern52():
    str_cov = 'matern52'
    cur_hyps = utils_covariance.get_hyps(str_cov, 2)
    X_train = np.array([
        [2.0, 1.0],
        [1.0, 1.0],
    ])
    num_hyps = X_train.shape[1] + 1
    cov_ = package_target.cov_main(str_cov, X_train, X_train, cur_hyps, True)
    print(cov_)

    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_matern52('abc', X_train, X_train, cur_hyps, num_hyps, True)
    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_matern52(cov_, 'abc', X_train, cur_hyps, num_hyps, True)
    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_matern52(cov_, X_train, 'abc', cur_hyps, num_hyps, True)
    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_matern52(cov_, X_train, X_train, 'abc', num_hyps, True)
    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_matern52(cov_, X_train, X_train, cur_hyps, 'abc', True)
    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_matern52(cov_, X_train, X_train, cur_hyps, num_hyps, 'abc')

    num_hyps = X_train.shape[1] + 2
    grad_cov_ = package_target.grad_cov_matern52(cov_, X_train, X_train, cur_hyps, num_hyps, False)
    print(grad_cov_)

    truth_grad_cov_ = np.array([
        [
            [0.02, 2.00002, 0., 0.],
            [0., 1.04798822, 0.57644039, 0.57644039],
        ], [
            [0., 1.04798822, 0.57644039, 0.57644039],
            [0.02, 2.00002, 0., 0.]
        ]
    ])

    assert np.all(np.abs(truth_grad_cov_ - grad_cov_) < TEST_EPSILON)

    num_hyps = X_train.shape[1] + 1
    grad_cov_ = package_target.grad_cov_matern52(cov_, X_train, X_train, cur_hyps, num_hyps, True)
    print(grad_cov_)

    truth_grad_cov_ = np.array([
        [
            [2.00002, 0., 0.],
            [1.04798822, 0.57644039, 0.57644039],
        ], [
            [1.04798822, 0.57644039, 0.57644039],
            [2.00002, 0., 0.]
        ]
    ])

    assert np.all(np.abs(truth_grad_cov_ - grad_cov_) < TEST_EPSILON)

def test_cov_set_typing():
    annos = package_target.cov_set.__annotations__

    assert annos['str_cov'] == str
    assert annos['X'] == np.ndarray
    assert annos['Xp'] == np.ndarray
    assert annos['lengthscales'] == typing.Union[np.ndarray, float]
    assert annos['signal'] == float
    assert annos['return'] == np.ndarray

def test_cov_set():
    num_instances = 5
    num_dim = 3
    str_cov = 'matern52'
    with pytest.raises(AssertionError) as error:
        package_target.cov_set(1, np.zeros((num_instances, num_dim)), np.zeros((num_instances, num_dim)), np.array([1.0, 1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        package_target.cov_set('abc', np.zeros((num_instances, num_dim)), np.zeros((num_instances, num_dim)), np.array([1.0, 1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        package_target.cov_set(str_cov, np.zeros((num_instances, num_dim)), np.zeros((num_instances, num_dim)), np.array([1.0, 1.0, 1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        package_target.cov_set(str_cov, np.zeros((num_instances, num_dim+1)), np.zeros((num_instances, num_dim)), np.array([1.0, 1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        package_target.cov_set(str_cov, np.zeros((num_instances, num_dim)), np.zeros((num_instances, num_dim+1)), np.array([1.0, 1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        package_target.cov_set(str_cov, np.zeros((num_instances, num_dim)), np.zeros((num_instances, num_dim)), np.array([1.0, 1.0, 1.0]), 1)
    assert np.abs(package_target.cov_set(str_cov, np.zeros((num_instances, num_dim)), np.zeros((num_instances, num_dim)), 1.0, 0.1) - 0.01) < TEST_EPSILON

    bx = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0]])
    bxp = np.array([[2.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    cur_hyps = utils_covariance.get_hyps('matern52', 3)
    cov_ = package_target.cov_set(str_cov, bx, bxp, cur_hyps['lengthscales'], cur_hyps['signal'])
    print(cov_)
    truth_cov_ = 0.23061736638896702
    assert np.abs(cov_ - truth_cov_) < TEST_EPSILON

def test_cov_main_typing():
    annos = package_target.cov_main.__annotations__

    assert annos['str_cov'] == str
    assert annos['X'] == np.ndarray
    assert annos['Xp'] == np.ndarray
    assert annos['hyps'] == dict
    assert annos['same_X_Xp'] == bool
    assert annos['jitter'] == float
    assert annos['return'] == np.ndarray

def test_cov_main():
    cur_hyps = utils_covariance.get_hyps('se', 3)
    with pytest.raises(AssertionError) as error:
        package_target.cov_main('se', np.zeros((10, 2)), np.zeros((20, 3)), cur_hyps, False, jitter=0.001)
    with pytest.raises(AssertionError) as error:
        package_target.cov_main('se', np.zeros((10, 3)), np.zeros((20, 2)), cur_hyps, False, jitter=0.001)
    with pytest.raises(ValueError) as error:
        package_target.cov_main('se', np.zeros((10, 2)), np.zeros((20, 2)), cur_hyps, False, jitter=0.001)
    with pytest.raises(AssertionError) as error:
        package_target.cov_main('se', 1.0, np.zeros((20, 3)), cur_hyps, False, jitter=0.001)
    with pytest.raises(AssertionError) as error:
        package_target.cov_main('se', np.zeros((10, 2)), 1.0, cur_hyps, False, jitter=0.001)
    with pytest.raises(AssertionError) as error:
        package_target.cov_main(1.0, np.zeros((10, 3)), np.zeros((20, 3)), cur_hyps, False, jitter=0.001)
    with pytest.raises(AssertionError) as error:
        package_target.cov_main('se', np.zeros((10, 3)), np.zeros((20, 3)), 2.1, False, jitter=0.001)
    with pytest.raises(AssertionError) as error:
        package_target.cov_main('se', np.zeros((10, 3)), np.zeros((20, 3)), cur_hyps, 'abc', jitter=0.001)
    with pytest.raises(AssertionError) as error:
        package_target.cov_main('se', np.zeros((10, 3)), np.zeros((12, 3)), cur_hyps, True, jitter=0.001)
    with pytest.raises(AssertionError) as error:
        package_target.cov_main('se', np.zeros((10, 3)), np.zeros((20, 3)), cur_hyps, False, jitter=1)
    with pytest.raises(AssertionError) as error:
        package_target.cov_main('abc', np.zeros((10, 3)), np.zeros((20, 3)), cur_hyps, False, jitter=0.001)

    cur_hyps.pop('signal', None)
    with pytest.raises(ValueError) as error:
        package_target.cov_main('se', np.zeros((10, 3)), np.zeros((20, 3)), cur_hyps, False)
    with pytest.raises(ValueError) as error:
        package_target.cov_main('set_se', np.zeros((10, 5, 3)), np.zeros((20, 5, 3)), cur_hyps, False)

    cur_hyps = utils_covariance.get_hyps('se', 3)
    cov_ = package_target.cov_main('se', np.zeros((10, 3)), np.zeros((20, 3)), cur_hyps, False, jitter=0.001)
    assert np.all(cov_ == np.ones((10, 20)))

    cov_ = package_target.cov_main('set_se', np.zeros((10, 5, 3)), np.zeros((20, 5, 3)), cur_hyps, False, jitter=0.001)
    assert np.all(cov_ == np.ones((10, 20)))

    cur_hyps = utils_covariance.get_hyps('matern32', 3)
    cov_ = package_target.cov_main('matern32', np.zeros((10, 3)), np.zeros((20, 3)), cur_hyps, False, jitter=0.001)
    assert np.all(cov_ == np.ones((10, 20)))

    cov_ = package_target.cov_main('set_matern32', np.zeros((10, 5, 3)), np.zeros((20, 5, 3)), cur_hyps, False, jitter=0.001)
    assert np.all(cov_ == np.ones((10, 20)))

    cov_ = package_target.cov_main('set_matern32', np.zeros((10, 5, 3)), np.zeros((10, 5, 3)), cur_hyps, False, jitter=0.001)
    assert np.all(cov_ == np.ones((10, 10)))

    cov_ = package_target.cov_main('set_matern32', np.zeros((10, 5, 3)), np.zeros((10, 5, 3)), cur_hyps, True, jitter=0.001)
    assert np.all(cov_ == np.ones((10, 10)) + np.eye(10) * 1e-3)

    cur_hyps = utils_covariance.get_hyps('matern52', 3)
    cov_ = package_target.cov_main('matern52', np.zeros((10, 3)), np.zeros((20, 3)), cur_hyps, False, jitter=0.001)
    assert np.all(cov_ == np.ones((10, 20)))

    cov_ = package_target.cov_main('set_matern52', np.zeros((10, 5, 3)), np.zeros((20, 5, 3)), cur_hyps, False, jitter=0.001)
    assert np.all(cov_ == np.ones((10, 20)))

    cov_ = package_target.cov_main('set_matern52', np.zeros((10, 5, 3)), np.zeros((10, 5, 3)), cur_hyps, False, jitter=0.001)
    assert np.all(cov_ == np.ones((10, 10)))

    cov_ = package_target.cov_main('set_matern52', np.zeros((10, 5, 3)), np.zeros((10, 5, 3)), cur_hyps, True, jitter=0.001)
    assert np.all(cov_ == np.ones((10, 10)) + np.eye(10) * 1e-3)

def test_grad_cov_main_typing():
    annos = package_target.grad_cov_main.__annotations__

    assert annos['str_cov'] == str
    assert annos['X'] == np.ndarray
    assert annos['Xp'] == np.ndarray
    assert annos['hyps'] == dict
    assert annos['fix_noise'] == bool
    assert annos['same_X_Xp'] == bool
    assert annos['jitter'] == float
    assert annos['return'] == np.ndarray

def test_grad_cov_main():
    cur_hyps = utils_covariance.get_hyps('se', 2)

    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_main(123, np.zeros((10, 2)), np.zeros((10, 2)), cur_hyps, True)
    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_main('se', 123, np.zeros((10, 2)), cur_hyps, True)
    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_main('se', np.zeros((10, 2)), 123, cur_hyps, True)
    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_main('se', np.zeros((10, 2)), np.zeros((10, 2)), 123, True)
    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_main('se', np.zeros((10, 2)), np.zeros((10, 2)), cur_hyps, 'abc')
    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_main('abc', np.zeros((10, 2)), np.zeros((10, 2)), cur_hyps, True)
    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_main('se', np.zeros((10, 2)), np.zeros((10, 2)), cur_hyps, True, same_X_Xp='abc')
    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_main('se', np.zeros((10, 2)), np.zeros((10, 2)), cur_hyps, True, same_X_Xp=False)
    with pytest.raises(AssertionError) as error:
        package_target.grad_cov_main('se', np.zeros((10, 2)), np.zeros((10, 2)), cur_hyps, True, jitter='abc')

    grad_cov_ = package_target.grad_cov_main('se', np.ones((1, 2)), np.ones((1, 2)), cur_hyps, True)

    print(grad_cov_)
    truth_grad_cov_ = np.array([[[2.00002, 0.0, 0.0]]])
    assert np.all(np.abs(grad_cov_ - truth_grad_cov_) < TEST_EPSILON)

    grad_cov_ = package_target.grad_cov_main('se', np.ones((1, 2)), np.ones((1, 2)), cur_hyps, False)

    print(grad_cov_)
    truth_grad_cov_ = np.array([[[0.02, 2.00002, 0.0, 0.0]]])
    assert np.all(np.abs(grad_cov_ - truth_grad_cov_) < TEST_EPSILON)

    cur_hyps['lengthscales'] = 1.0
    grad_cov_ = package_target.grad_cov_main('se', np.ones((1, 2)), np.zeros((1, 2)), cur_hyps, False)

    print(grad_cov_)
    truth_grad_cov_ = np.array([[[0.02, 0.73577888, 0.73577888]]])
    assert np.all(np.abs(grad_cov_ - truth_grad_cov_) < TEST_EPSILON)

    grad_cov_ = package_target.grad_cov_main('matern32', np.ones((1, 2)), np.zeros((1, 2)), cur_hyps, False)

    print(grad_cov_)
    truth_grad_cov_ = np.array([[[0.02, 0.59566154, 0.51802578]]])
    assert np.all(np.abs(grad_cov_ - truth_grad_cov_) < TEST_EPSILON)

    grad_cov_ = package_target.grad_cov_main('matern32', np.ones((1, 2)), np.zeros((1, 2)), cur_hyps, True)

    print(grad_cov_)
    truth_grad_cov_ = np.array([[[0.59566154, 0.51802578]]])
    assert np.all(np.abs(grad_cov_ - truth_grad_cov_) < TEST_EPSILON)

    grad_cov_ = package_target.grad_cov_main('matern52', np.ones((1, 2)), np.zeros((1, 2)), cur_hyps, False)

    print(grad_cov_)
    truth_grad_cov_ = np.array([[[0.02, 0.63458673, 0.8305486]]])
    assert np.all(np.abs(grad_cov_ - truth_grad_cov_) < TEST_EPSILON)
