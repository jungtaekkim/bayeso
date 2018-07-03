# test_branin
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 03, 2018

import numpy as np
import pytest

from bayeso import benchmarks


TEST_EPSILON = 1e-5
FUN_TARGET = benchmarks.branin
INFO_FUN = benchmarks.INFO_BRANIN

def test_global_minimum():
    for elem_X in INFO_FUN.get('global_minimum_X', np.array([])):
        val_fun = benchmarks.branin(np.expand_dims(elem_X, axis=0))
        assert (val_fun - INFO_FUN.get('global_minimum_y', 0.0) < TEST_EPSILON).all()
