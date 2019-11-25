# test_bohachevsky
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: November 25, 2019

import numpy as np
import copy
import pytest

from bayeso import benchmarks
from bayeso.utils import utils_benchmarks


TEST_EPSILON = 1e-5
FUN_TARGET = benchmarks.bohachevsky
INFO_FUN = benchmarks.INFO_BOHACHEVSKY
INT_DIM = 2

def test_validate_info():
    info_fun = copy.deepcopy(INFO_FUN)
    assert utils_benchmarks.validate_info(info_fun)

def test_global_minimum():
    for elem_X in INFO_FUN.get('global_minimum_X', np.array([])):
        cur_X = utils_benchmarks.get_covariate(INFO_FUN, elem_X, INT_DIM)
        val_fun = FUN_TARGET(np.expand_dims(elem_X, axis=0))
        assert (np.abs(val_fun - INFO_FUN.get('global_minimum_y', np.inf)) < TEST_EPSILON).all()

def test_function():
    with pytest.raises(AssertionError) as error:
        FUN_TARGET('abc')
    with pytest.raises(AssertionError) as error:
        FUN_TARGET(np.ones((2, 2, 2)))
    with pytest.raises(AssertionError) as error:
        FUN_TARGET(np.ones((2, 3)))
    with pytest.raises(AssertionError) as error:
        FUN_TARGET(np.ones(3))

    assert (np.abs(FUN_TARGET(np.ones(2)) - 3.6) < TEST_EPSILON).all()
