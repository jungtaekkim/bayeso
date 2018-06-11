# test_acquisition
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: June 12, 2018

import numpy as np
import pytest

from bayeso import acquisition


TEST_EPSILON = 1e-5

def test_pi():
    with pytest.raises(AssertionError) as error:
        acquisition.pi('abc', np.ones(10), np.zeros((5, 1)))
    with pytest.raises(AssertionError) as error:
        acquisition.pi(np.ones(10), 'abc', np.zeros((5, 1)))
    with pytest.raises(AssertionError) as error:
        acquisition.pi(np.ones(10), np.ones(10), 'abc')
    with pytest.raises(AssertionError) as error:
        acquisition.pi(np.ones(10), np.ones(10), np.zeros((5, 1)), 1)
    with pytest.raises(AssertionError) as error:
        acquisition.pi(np.ones(5), np.ones(10), np.zeros((5, 1)))
    with pytest.raises(AssertionError) as error:
        acquisition.pi(np.ones(10), np.ones(10), np.zeros(5))
    with pytest.raises(AssertionError) as error:
        acquisition.pi(np.ones(10), np.ones((10, 1)), np.zeros((5, 1)))
    with pytest.raises(AssertionError) as error:
        acquisition.pi(np.ones((10, 1)), np.ones(10), np.zeros((5, 1)))

def test_ei():
    with pytest.raises(AssertionError) as error:
        acquisition.ei('abc', np.ones(10), np.zeros((5, 1)))
    with pytest.raises(AssertionError) as error:
        acquisition.ei(np.ones(10), 'abc', np.zeros((5, 1)))
    with pytest.raises(AssertionError) as error:
        acquisition.ei(np.ones(10), np.ones(10), 'abc')
    with pytest.raises(AssertionError) as error:
        acquisition.ei(np.ones(10), np.ones(10), np.zeros((5, 1)), 1)
    with pytest.raises(AssertionError) as error:
        acquisition.ei(np.ones(5), np.ones(10), np.zeros((5, 1)))
    with pytest.raises(AssertionError) as error:
        acquisition.ei(np.ones(10), np.ones(10), np.zeros(5))
    with pytest.raises(AssertionError) as error:
        acquisition.ei(np.ones(10), np.ones((10, 1)), np.zeros((5, 1)))
    with pytest.raises(AssertionError) as error:
        acquisition.ei(np.ones((10, 1)), np.ones(10), np.zeros((5, 1)))

def test_ucb():
    with pytest.raises(AssertionError) as error:
        acquisition.ucb('abc', np.ones(10))
    with pytest.raises(AssertionError) as error:
        acquisition.ucb(np.ones(10), 'abc')
    with pytest.raises(AssertionError) as error:
        acquisition.ucb(np.ones(10), np.ones(10), kappa='abc')
    with pytest.raises(AssertionError) as error:
        acquisition.ucb(np.ones(5), np.ones(10))
    with pytest.raises(AssertionError) as error:
        acquisition.ucb(np.ones(10), np.ones(10), Y_train='abc')
    with pytest.raises(AssertionError) as error:
        acquisition.ucb(np.ones(10), np.ones(10), Y_train=np.zeros(5))
    with pytest.raises(AssertionError) as error:
        acquisition.ucb(np.ones(10), np.ones((10, 1)))
    with pytest.raises(AssertionError) as error:
        acquisition.ucb(np.ones((10, 1)), np.ones(10))
    with pytest.raises(AssertionError) as error:
        acquisition.ucb(np.ones(10), np.ones(10), is_increased='abc')
