# test_acquisition
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 03, 2018

import numpy as np
import pytest

from bayeso import benchmarks


TEST_EPSILON = 1e-5

def test_branin():
    with pytest.raises(AssertionError) as error:
        benchmarks.branin(1)
    with pytest.raises(AssertionError) as error:
        benchmarks.branin(np.arange(0, 10))
    with pytest.raises(AssertionError) as error:
        benchmarks.branin(np.zeros((10, 3)))
    with pytest.raises(AssertionError) as error:
        benchmarks.branin(np.zeros((10, 2, 1)))
    with pytest.raises(AssertionError) as error:
        benchmarks.branin(np.zeros((10, 2)), a='abc')
    with pytest.raises(AssertionError) as error:
        benchmarks.branin(np.zeros((10, 2)), b='abc')
    with pytest.raises(AssertionError) as error:
        benchmarks.branin(np.zeros((10, 2)), c='abc')
    with pytest.raises(AssertionError) as error:
        benchmarks.branin(np.zeros((10, 2)), r='abc')
    with pytest.raises(AssertionError) as error:
        benchmarks.branin(np.zeros((10, 2)), s='abc')
    with pytest.raises(AssertionError) as error:
        benchmarks.branin(np.zeros((10, 2)), t='abc')

    X = np.array([[0.0, 0.0]])
    val_fun = benchmarks.branin(X)
    truth_val_fun = np.array([55.60211264])
    assert (val_fun - truth_val_fun < TEST_EPSILON).all()

    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    val_fun = benchmarks.branin(X)
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

    X = np.array([[0.0, 0.0]])
    val_fun = benchmarks.ackley(X)
    truth_val_fun = np.array([0.0])
    assert (val_fun - truth_val_fun < TEST_EPSILON).all()

    X = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    val_fun = benchmarks.ackley(X)
    assert len(val_fun.shape) == 1
    assert val_fun.shape[0] == X.shape[0]
