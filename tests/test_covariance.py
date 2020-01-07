# test_covariance
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: November 25, 2019

import pytest
import numpy as np

from bayeso import covariance
from bayeso.utils import utils_covariance


TEST_EPSILON = 1e-5

def test_choose_fun_cov():
    with pytest.raises(AssertionError) as error:
        covariance.choose_fun_cov(123)
    with pytest.raises(AssertionError) as error:
        covariance.choose_fun_cov('se', 'abc')
    with pytest.raises(NotImplementedError) as error:
        covariance.choose_fun_cov('abc')

    assert covariance.choose_fun_cov('se') == covariance.cov_se
    assert covariance.choose_fun_cov('matern32') == covariance.cov_matern32
    assert covariance.choose_fun_cov('matern52') == covariance.cov_matern52
    assert covariance.choose_fun_cov('se', is_grad=True) == covariance.grad_cov_se
    assert covariance.choose_fun_cov('matern32', is_grad=True) == covariance.grad_cov_matern32
    assert covariance.choose_fun_cov('matern52', is_grad=True) == covariance.grad_cov_matern52

def test_cov_se():
    with pytest.raises(AssertionError) as error:
        covariance.cov_se(np.zeros(2), np.zeros(2), np.array([1.0, 1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        covariance.cov_se(np.zeros(2), np.zeros(3), np.array([1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        covariance.cov_se(np.zeros(3), np.zeros(2), np.array([1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        covariance.cov_se(np.zeros(2), np.zeros(2), np.array([1.0, 1.0]), 1)
    assert np.abs(covariance.cov_se(np.zeros(2), np.zeros(2), 1.0, 0.1) - 0.01) < TEST_EPSILON

    bx = np.array([1.0, 2.0, 0.0])
    bxp = np.array([2.0, 1.0, 1.0])
    cur_hyps = utils_covariance.get_hyps('se', 3)
    cov_ = covariance.cov_se(bx, bxp, cur_hyps['lengthscales'], cur_hyps['signal'])
    print(cov_)
    truth_cov_ = 0.22313016014842987
    assert np.abs(cov_ - truth_cov_) < TEST_EPSILON

def test_grad_cov_se():
    str_cov = 'se'
    cur_hyps = utils_covariance.get_hyps(str_cov, 2)
    X_train = np.array([
        [2.0, 1.0],
        [1.0, 1.0],
    ])
    num_hyps = X_train.shape[1] + 1
    cov_ = covariance.cov_main(str_cov, X_train, X_train, cur_hyps, True)
    print(cov_)

    with pytest.raises(AssertionError) as error:
        covariance.grad_cov_se('abc', X_train, X_train, cur_hyps, num_hyps, True)
    with pytest.raises(AssertionError) as error:
        covariance.grad_cov_se(cov_, 'abc', X_train, cur_hyps, num_hyps, True)
    with pytest.raises(AssertionError) as error:
        covariance.grad_cov_se(cov_, X_train, 'abc', cur_hyps, num_hyps, True)
    with pytest.raises(AssertionError) as error:
        covariance.grad_cov_se(cov_, X_train, X_train, 'abc', num_hyps, True)
    with pytest.raises(AssertionError) as error:
        covariance.grad_cov_se(cov_, X_train, X_train, cur_hyps, 'abc', True)
    with pytest.raises(AssertionError) as error:
        covariance.grad_cov_se(cov_, X_train, X_train, cur_hyps, num_hyps, 'abc')

    num_hyps = X_train.shape[1] + 2
    grad_cov_ = covariance.grad_cov_se(cov_, X_train, X_train, cur_hyps, num_hyps, False)
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

def test_cov_matern32():
    with pytest.raises(AssertionError) as error:
        covariance.cov_matern32(np.zeros(2), np.zeros(2), np.array([1.0, 1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        covariance.cov_matern32(np.zeros(2), np.zeros(3), np.array([1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        covariance.cov_matern32(np.zeros(3), np.zeros(2), np.array([1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        covariance.cov_matern32(np.zeros(2), np.zeros(2), np.array([1.0, 1.0]), 1)
    assert np.abs(covariance.cov_matern32(np.zeros(2), np.zeros(2), 1.0, 0.1) - 0.01) < TEST_EPSILON

    bx = np.array([1.0, 2.0, 0.0])
    bxp = np.array([2.0, 1.0, 1.0])
    cur_hyps = utils_covariance.get_hyps('matern32', 3)
    cov_ = covariance.cov_matern32(bx, bxp, cur_hyps['lengthscales'], cur_hyps['signal'])
    print(cov_)
    truth_cov_ = 0.19914827347145583
    assert np.abs(cov_ - truth_cov_) < TEST_EPSILON

def test_grad_cov_matern32():
    str_cov = 'matern32'
    cur_hyps = utils_covariance.get_hyps(str_cov, 2)
    X_train = np.array([
        [2.0, 1.0],
        [1.0, 1.0],
    ])
    num_hyps = X_train.shape[1] + 1
    cov_ = covariance.cov_main(str_cov, X_train, X_train, cur_hyps, True)
    print(cov_)

    with pytest.raises(AssertionError) as error:
        covariance.grad_cov_matern32('abc', X_train, X_train, cur_hyps, num_hyps, True)
    with pytest.raises(AssertionError) as error:
        covariance.grad_cov_matern32(cov_, 'abc', X_train, cur_hyps, num_hyps, True)
    with pytest.raises(AssertionError) as error:
        covariance.grad_cov_matern32(cov_, X_train, 'abc', cur_hyps, num_hyps, True)
    with pytest.raises(AssertionError) as error:
        covariance.grad_cov_matern32(cov_, X_train, X_train, 'abc', num_hyps, True)
    with pytest.raises(AssertionError) as error:
        covariance.grad_cov_matern32(cov_, X_train, X_train, cur_hyps, 'abc', True)
    with pytest.raises(AssertionError) as error:
        covariance.grad_cov_matern32(cov_, X_train, X_train, cur_hyps, num_hyps, 'abc')

    num_hyps = X_train.shape[1] + 2
    grad_cov_ = covariance.grad_cov_matern32(cov_, X_train, X_train, cur_hyps, num_hyps, False)
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

def test_cov_matern52():
    with pytest.raises(AssertionError) as error:
        covariance.cov_matern52(np.zeros(2), np.zeros(2), np.array([1.0, 1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        covariance.cov_matern52(np.zeros(2), np.zeros(3), np.array([1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        covariance.cov_matern52(np.zeros(3), np.zeros(2), np.array([1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        covariance.cov_matern52(np.zeros(2), np.zeros(2), np.array([1.0, 1.0]), 1)
    assert np.abs(covariance.cov_matern52(np.zeros(2), np.zeros(2), 1.0, 0.1) - 0.01) < TEST_EPSILON

    bx = np.array([1.0, 2.0, 0.0])
    bxp = np.array([2.0, 1.0, 1.0])
    cur_hyps = utils_covariance.get_hyps('matern52', 3)
    cov_ = covariance.cov_matern52(bx, bxp, cur_hyps['lengthscales'], cur_hyps['signal'])
    print(cov_)
    truth_cov_ = 0.20532087608359792
    assert np.abs(cov_ - truth_cov_) < TEST_EPSILON

def test_grad_cov_matern52():
    str_cov = 'matern52'
    cur_hyps = utils_covariance.get_hyps(str_cov, 2)
    X_train = np.array([
        [2.0, 1.0],
        [1.0, 1.0],
    ])
    num_hyps = X_train.shape[1] + 1
    cov_ = covariance.cov_main(str_cov, X_train, X_train, cur_hyps, True)
    print(cov_)

    with pytest.raises(AssertionError) as error:
        covariance.grad_cov_matern52('abc', X_train, X_train, cur_hyps, num_hyps, True)
    with pytest.raises(AssertionError) as error:
        covariance.grad_cov_matern52(cov_, 'abc', X_train, cur_hyps, num_hyps, True)
    with pytest.raises(AssertionError) as error:
        covariance.grad_cov_matern52(cov_, X_train, 'abc', cur_hyps, num_hyps, True)
    with pytest.raises(AssertionError) as error:
        covariance.grad_cov_matern52(cov_, X_train, X_train, 'abc', num_hyps, True)
    with pytest.raises(AssertionError) as error:
        covariance.grad_cov_matern52(cov_, X_train, X_train, cur_hyps, 'abc', True)
    with pytest.raises(AssertionError) as error:
        covariance.grad_cov_matern52(cov_, X_train, X_train, cur_hyps, num_hyps, 'abc')

    num_hyps = X_train.shape[1] + 2
    grad_cov_ = covariance.grad_cov_matern52(cov_, X_train, X_train, cur_hyps, num_hyps, False)
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

def test_cov_set():
    num_instances = 5
    num_dim = 3
    str_cov = 'matern52'
    with pytest.raises(AssertionError) as error:
        covariance.cov_set(1, np.zeros((num_instances, num_dim)), np.zeros((num_instances, num_dim)), np.array([1.0, 1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        covariance.cov_set('abc', np.zeros((num_instances, num_dim)), np.zeros((num_instances, num_dim)), np.array([1.0, 1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        covariance.cov_set(str_cov, np.zeros((num_instances, num_dim)), np.zeros((num_instances, num_dim)), np.array([1.0, 1.0, 1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        covariance.cov_set(str_cov, np.zeros((num_instances, num_dim+1)), np.zeros((num_instances, num_dim)), np.array([1.0, 1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        covariance.cov_set(str_cov, np.zeros((num_instances, num_dim)), np.zeros((num_instances, num_dim+1)), np.array([1.0, 1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        covariance.cov_set(str_cov, np.zeros((num_instances, num_dim)), np.zeros((num_instances, num_dim)), np.array([1.0, 1.0, 1.0]), 1)
    assert np.abs(covariance.cov_set(str_cov, np.zeros((num_instances, num_dim)), np.zeros((num_instances, num_dim)), 1.0, 0.1) - 0.01) < TEST_EPSILON

    bx = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0]])
    bxp = np.array([[2.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    cur_hyps = utils_covariance.get_hyps('matern52', 3)
    cov_ = covariance.cov_set(str_cov, bx, bxp, cur_hyps['lengthscales'], cur_hyps['signal'])
    print(cov_)
    truth_cov_ = 0.23061736638896702
    assert np.abs(cov_ - truth_cov_) < TEST_EPSILON

def test_cov_main():
    cur_hyps = utils_covariance.get_hyps('se', 3)
    with pytest.raises(AssertionError) as error:
        covariance.cov_main('se', np.zeros((10, 2)), np.zeros((20, 3)), cur_hyps, False, jitter=0.001)
    with pytest.raises(AssertionError) as error:
        covariance.cov_main('se', np.zeros((10, 3)), np.zeros((20, 2)), cur_hyps, False, jitter=0.001)
    with pytest.raises(ValueError) as error:
        covariance.cov_main('se', np.zeros((10, 2)), np.zeros((20, 2)), cur_hyps, False, jitter=0.001)
    with pytest.raises(AssertionError) as error:
        covariance.cov_main('se', 1.0, np.zeros((20, 3)), cur_hyps, False, jitter=0.001)
    with pytest.raises(AssertionError) as error:
        covariance.cov_main('se', np.zeros((10, 2)), 1.0, cur_hyps, False, jitter=0.001)
    with pytest.raises(AssertionError) as error:
        covariance.cov_main(1.0, np.zeros((10, 3)), np.zeros((20, 3)), cur_hyps, False, jitter=0.001)
    with pytest.raises(AssertionError) as error:
        covariance.cov_main('se', np.zeros((10, 3)), np.zeros((20, 3)), 2.1, False, jitter=0.001)
    with pytest.raises(AssertionError) as error:
        covariance.cov_main('se', np.zeros((10, 3)), np.zeros((20, 3)), cur_hyps, 'abc', jitter=0.001)
    with pytest.raises(AssertionError) as error:
        covariance.cov_main('se', np.zeros((10, 3)), np.zeros((12, 3)), cur_hyps, True, jitter=0.001)
    with pytest.raises(AssertionError) as error:
        covariance.cov_main('se', np.zeros((10, 3)), np.zeros((20, 3)), cur_hyps, False, jitter=1)
    with pytest.raises(AssertionError) as error:
        covariance.cov_main('abc', np.zeros((10, 3)), np.zeros((20, 3)), cur_hyps, False, jitter=0.001)

    cur_hyps.pop('signal', None)
    with pytest.raises(ValueError) as error:
        covariance.cov_main('se', np.zeros((10, 3)), np.zeros((20, 3)), cur_hyps, False)
    cur_hyps = utils_covariance.get_hyps('se', 3)
    cov_ = covariance.cov_main('se', np.zeros((10, 3)), np.zeros((20, 3)), cur_hyps, False, jitter=0.001)

    cur_hyps = utils_covariance.get_hyps('matern32', 3)
    cov_ = covariance.cov_main('matern32', np.zeros((10, 3)), np.zeros((20, 3)), cur_hyps, False, jitter=0.001)

    cur_hyps = utils_covariance.get_hyps('matern52', 3)
    cov_ = covariance.cov_main('matern52', np.zeros((10, 3)), np.zeros((20, 3)), cur_hyps, False, jitter=0.001)

def test_grad_cov_main():
    cur_hyps = utils_covariance.get_hyps('se', 2)

    with pytest.raises(AssertionError) as error:
        covariance.grad_cov_main(123, np.zeros((10, 2)), np.zeros((10, 2)), cur_hyps, True)
    with pytest.raises(AssertionError) as error:
        covariance.grad_cov_main('se', 123, np.zeros((10, 2)), cur_hyps, True)
    with pytest.raises(AssertionError) as error:
        covariance.grad_cov_main('se', np.zeros((10, 2)), 123, cur_hyps, True)
    with pytest.raises(AssertionError) as error:
        covariance.grad_cov_main('se', np.zeros((10, 2)), np.zeros((10, 2)), 123, True)
    with pytest.raises(AssertionError) as error:
        covariance.grad_cov_main('se', np.zeros((10, 2)), np.zeros((10, 2)), cur_hyps, 'abc')
    with pytest.raises(AssertionError) as error:
        covariance.grad_cov_main('abc', np.zeros((10, 2)), np.zeros((10, 2)), cur_hyps, True)
    with pytest.raises(AssertionError) as error:
        covariance.grad_cov_main('se', np.zeros((10, 2)), np.zeros((10, 2)), cur_hyps, True, jitter='abc')

    grad_cov_ = covariance.grad_cov_main('se', np.ones((1, 2)), np.ones((1, 2)), cur_hyps, True)

    print(grad_cov_)
    truth_grad_cov_ = np.array([[[2.00002, 0.0, 0.0]]])
    assert np.all(np.abs(grad_cov_ - truth_grad_cov_) < TEST_EPSILON)

    grad_cov_ = covariance.grad_cov_main('se', np.ones((1, 2)), np.ones((1, 2)), cur_hyps, False)

    print(grad_cov_)
    truth_grad_cov_ = np.array([[[0.02, 2.00002, 0.0, 0.0]]])
    assert np.all(np.abs(grad_cov_ - truth_grad_cov_) < TEST_EPSILON)

    cur_hyps['lengthscales'] = 1.0
    grad_cov_ = covariance.grad_cov_main('se', np.ones((1, 2)), np.ones((1, 2)), cur_hyps, False)

    print(grad_cov_)
    truth_grad_cov_ = np.array([[[0.02, 2.00002, 0.0]]])
    assert np.all(np.abs(grad_cov_ - truth_grad_cov_) < TEST_EPSILON)
