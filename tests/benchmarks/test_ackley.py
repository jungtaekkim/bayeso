# test_ackley
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 03, 2018

import numpy as np
import pytest

from bayeso import benchmarks
from bayeso.utils import utils_benchmarks


TEST_EPSILON = 1e-5
FUN_TARGET = benchmarks.ackley
INFO_FUN = benchmarks.INFO_ACKLEY
INT_DIM = 3

def test_validate_info():
    assert utils_benchmarks.validate_info(INFO_FUN)

def test_global_minimum():
    for elem_X in INFO_FUN.get('global_minimum_X', np.array([])):
        cur_X = utils_benchmarks.get_covariate(INFO_FUN, elem_X, INT_DIM)
        val_fun = FUN_TARGET(np.expand_dims(elem_X, axis=0))
        assert (val_fun - INFO_FUN.get('global_minimum_y', np.inf) < TEST_EPSILON).all()
