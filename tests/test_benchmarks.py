# test_acquisition
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 03, 2018

import numpy as np
import pytest

from bayeso import benchmarks


TEST_EPSILON = 1e-5

def test_branin():
    fun_target = benchmarks.branin
    with pytest.raises(AssertionError) as error:
        fun_target(1)
    with pytest.raises(AssertionError) as error:
        fun_target(np.arange(0, 10))
    with pytest.raises(AssertionError) as error:
        fun_target(np.zeros((10, 3)))
    with pytest.raises(AssertionError) as error:
        fun_target(np.zeros((10, 2, 1)))
    with pytest.raises(AssertionError) as error:
        fun_target(np.zeros((10, 2)), a='abc')
    with pytest.raises(AssertionError) as error:
        fun_target(np.zeros((10, 2)), b='abc')
    with pytest.raises(AssertionError) as error:
        fun_target(np.zeros((10, 2)), c='abc')
    with pytest.raises(AssertionError) as error:
        fun_target(np.zeros((10, 2)), r='abc')
    with pytest.raises(AssertionError) as error:
        fun_target(np.zeros((10, 2)), s='abc')
    with pytest.raises(AssertionError) as error:
        fun_target(np.zeros((10, 2)), t='abc')

    X = np.array([0.0, 0.0])
    val_fun = fun_target(X)
    truth_val_fun = np.array([55.60211264])
    assert (np.abs(val_fun - truth_val_fun) < TEST_EPSILON).all()

    X = np.array([[0.0, 0.0]])
    val_fun = fun_target(X)
    truth_val_fun = np.array([55.60211264])
    assert (np.abs(val_fun - truth_val_fun) < TEST_EPSILON).all()

    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    val_fun = fun_target(X)
    assert len(val_fun.shape) == 1
    assert val_fun.shape[0] == X.shape[0]

def test_ackley():
    with pytest.raises(AssertionError) as error:
        benchmarks.ackley(1)
    with pytest.raises(AssertionError) as error:
        benchmarks.ackley(np.zeros((10, 2, 1)))
    with pytest.raises(AssertionError) as error:
        benchmarks.ackley(np.zeros((10, 2)), a='abc')
    with pytest.raises(AssertionError) as error:
        benchmarks.ackley(np.zeros((10, 2)), b='abc')
    with pytest.raises(AssertionError) as error:
        benchmarks.ackley(np.zeros((10, 2)), c='abc')

    X = np.array([0.0, 0.0])
    val_fun = benchmarks.ackley(X)
    truth_val_fun = np.array([0.0])
    assert (np.abs(val_fun - truth_val_fun) < TEST_EPSILON).all()

    X = np.array([[0.0, 0.0]])
    val_fun = benchmarks.ackley(X)
    truth_val_fun = np.array([0.0])
    assert (np.abs(val_fun - truth_val_fun) < TEST_EPSILON).all()

    X = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    val_fun = benchmarks.ackley(X)
    assert len(val_fun.shape) == 1
    assert val_fun.shape[0] == X.shape[0]

def test_eggholder():
    with pytest.raises(AssertionError) as error:
        benchmarks.eggholder(1)
    with pytest.raises(AssertionError) as error:
        benchmarks.eggholder(np.arange(0, 10))
    with pytest.raises(AssertionError) as error:
        benchmarks.eggholder(np.zeros((10, 3)))
    with pytest.raises(AssertionError) as error:
        benchmarks.eggholder(np.zeros((10, 2, 1)))

    X = np.array([0.0, 0.0])
    val_fun = benchmarks.eggholder(X)
    truth_val_fun = np.array([-25.46033719])
    assert (np.abs(val_fun - truth_val_fun) < TEST_EPSILON).all()

    X = np.array([[0.0, 0.0]])
    val_fun = benchmarks.eggholder(X)
    truth_val_fun = np.array([-25.46033719])
    assert (np.abs(val_fun - truth_val_fun) < TEST_EPSILON).all()

    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    val_fun = benchmarks.eggholder(X)
    assert len(val_fun.shape) == 1
    assert val_fun.shape[0] == X.shape[0]

def test_sixhumpcamel():
    fun_target = benchmarks.sixhumpcamel
    with pytest.raises(AssertionError) as error:
        fun_target(1)
    with pytest.raises(AssertionError) as error:
        fun_target(np.arange(0, 10))
    with pytest.raises(AssertionError) as error:
        fun_target(np.zeros((10, 3)))
    with pytest.raises(AssertionError) as error:
        fun_target(np.zeros((10, 2, 1)))

    X = np.array([0.0, 0.0])
    val_fun = fun_target(X)
    print(val_fun)
    truth_val_fun = np.array([0.0])
    assert (np.abs(val_fun - truth_val_fun) < TEST_EPSILON).all()

    X = np.array([1.0, 1.0])
    val_fun = fun_target(X)
    print(val_fun)
    truth_val_fun = np.array([3.23333333])
    assert (np.abs(val_fun - truth_val_fun) < TEST_EPSILON).all()

    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    val_fun = fun_target(X)
    assert len(val_fun.shape) == 1
    assert val_fun.shape[0] == X.shape[0]

def test_beale():
    fun_target = benchmarks.beale
    with pytest.raises(AssertionError) as error:
        fun_target(1)
    with pytest.raises(AssertionError) as error:
        fun_target(np.arange(0, 10))
    with pytest.raises(AssertionError) as error:
        fun_target(np.zeros((10, 3)))
    with pytest.raises(AssertionError) as error:
        fun_target(np.zeros((10, 2, 1)))

    X = np.array([0.0, 0.0])
    val_fun = fun_target(X)
    print(val_fun)
    truth_val_fun = np.array([14.203125])
    assert (np.abs(val_fun - truth_val_fun) < TEST_EPSILON).all()

    X = np.array([1.0, 2.0])
    val_fun = fun_target(X)
    print(val_fun)
    truth_val_fun = np.array([126.453125])
    assert (np.abs(val_fun - truth_val_fun) < TEST_EPSILON).all()

    X = np.array([[0.0, 0.0], [1.0, 2.0]])
    val_fun = fun_target(X)
    assert len(val_fun.shape) == 1
    assert val_fun.shape[0] == X.shape[0]

def test_goldsteinprice():
    fun_target = benchmarks.goldsteinprice
    with pytest.raises(AssertionError) as error:
        fun_target(1)
    with pytest.raises(AssertionError) as error:
        fun_target(np.arange(0, 10))
    with pytest.raises(AssertionError) as error:
        fun_target(np.zeros((10, 3)))
    with pytest.raises(AssertionError) as error:
        fun_target(np.zeros((10, 2, 1)))

    X = np.array([0.0, 0.0])
    val_fun = fun_target(X)
    print(val_fun)
    truth_val_fun = np.array([600.0])
    assert (np.abs(val_fun - truth_val_fun) < TEST_EPSILON).all()

    X = np.array([1.0, 1.0])
    val_fun = fun_target(X)
    print(val_fun)
    truth_val_fun = np.array([1876.0])
    assert (np.abs(val_fun - truth_val_fun) < TEST_EPSILON).all()

    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    val_fun = fun_target(X)
    assert len(val_fun.shape) == 1
    assert val_fun.shape[0] == X.shape[0]

def test_bohachevsky():
    fun_target = benchmarks.bohachevsky
    with pytest.raises(AssertionError) as error:
        fun_target(1)
    with pytest.raises(AssertionError) as error:
        fun_target(np.arange(0, 10))
    with pytest.raises(AssertionError) as error:
        fun_target(np.zeros((10, 3)))
    with pytest.raises(AssertionError) as error:
        fun_target(np.zeros((10, 2, 1)))

    X = np.array([0.0, 0.0])
    val_fun = fun_target(X)
    print(val_fun)
    truth_val_fun = np.array([0.0])
    assert (np.abs(val_fun - truth_val_fun) < TEST_EPSILON).all()

    X = np.array([1.0, 1.0])
    val_fun = fun_target(X)
    print(val_fun)
    truth_val_fun = np.array([3.6])
    assert (np.abs(val_fun - truth_val_fun) < TEST_EPSILON).all()

    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    val_fun = fun_target(X)
    assert len(val_fun.shape) == 1
    assert val_fun.shape[0] == X.shape[0]

def test_hartmann6d():
    fun_target = benchmarks.hartmann6d
    with pytest.raises(AssertionError) as error:
        fun_target(1)
    with pytest.raises(AssertionError) as error:
        fun_target(np.arange(0, 10))
    with pytest.raises(AssertionError) as error:
        fun_target(np.zeros((10, 3)))
    with pytest.raises(AssertionError) as error:
        fun_target(np.zeros((10, 2, 1)))

    X = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    val_fun = fun_target(X)
    print(val_fun)
    truth_val_fun = np.array([-0.00508911])
    assert (np.abs(val_fun - truth_val_fun) < TEST_EPSILON).all()

    X = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    val_fun = fun_target(X)
    print(val_fun)
    truth_val_fun = np.array([-3.40853927e-5])
    assert (np.abs(val_fun - truth_val_fun) < TEST_EPSILON).all()

    X = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    val_fun = fun_target(X)
    assert len(val_fun.shape) == 1
    assert val_fun.shape[0] == X.shape[0]

def test_holdertable():
    fun_target = benchmarks.holdertable
    with pytest.raises(AssertionError) as error:
        fun_target(1)
    with pytest.raises(AssertionError) as error:
        fun_target(np.arange(0, 10))
    with pytest.raises(AssertionError) as error:
        fun_target(np.zeros((10, 3)))
    with pytest.raises(AssertionError) as error:
        fun_target(np.zeros((10, 2, 1)))

    X = np.array([0.0, 0.0])
    val_fun = fun_target(X)
    print(val_fun)
    truth_val_fun = np.array([0.0])
    assert (np.abs(val_fun - truth_val_fun) < TEST_EPSILON).all()

    X = np.array([1.0, 1.0])
    val_fun = fun_target(X)
    print(val_fun)
    truth_val_fun = np.array([-0.78789663])
    assert (np.abs(val_fun - truth_val_fun) < TEST_EPSILON).all()

    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    val_fun = fun_target(X)
    assert len(val_fun.shape) == 1
    assert val_fun.shape[0] == X.shape[0]

