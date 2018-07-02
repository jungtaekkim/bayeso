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

