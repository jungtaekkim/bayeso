# test_utils_benchmarks
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 09, 2018

import pytest
import numpy as np
import copy

from bayeso import benchmarks
from bayeso.utils import utils_benchmarks


def test_validate_info():
    with pytest.raises(AssertionError) as error:
        utils_benchmarks.validate_info(1.0)

    info_branin = copy.deepcopy(benchmarks.INFO_BRANIN)
    info_branin.pop('dim_fun')
    assert not utils_benchmarks.validate_info(info_branin)

    info_branin = copy.deepcopy(benchmarks.INFO_BRANIN)
    info_branin.pop('bounds')
    assert not utils_benchmarks.validate_info(info_branin)

    info_branin = copy.deepcopy(benchmarks.INFO_BRANIN)
    info_branin.pop('global_minimum_X')
    assert not utils_benchmarks.validate_info(info_branin)

    info_branin = copy.deepcopy(benchmarks.INFO_BRANIN)
    info_branin.pop('global_minimum_y')
    assert not utils_benchmarks.validate_info(info_branin)

    info_branin = copy.deepcopy(benchmarks.INFO_BRANIN)
    info_branin['dim_fun'] = 'abc'
    assert not utils_benchmarks.validate_info(info_branin)

    info_branin = copy.deepcopy(benchmarks.INFO_BRANIN)
    info_branin['dim_fun'] = 4
    assert not utils_benchmarks.validate_info(info_branin)

    info_branin = copy.deepcopy(benchmarks.INFO_BRANIN)
    info_branin['bounds'] = 'abc'
    assert not utils_benchmarks.validate_info(info_branin)

    info_branin = copy.deepcopy(benchmarks.INFO_BRANIN)
    info_branin['bounds'] = np.arange(0, 10)
    assert not utils_benchmarks.validate_info(info_branin)

    info_branin = copy.deepcopy(benchmarks.INFO_BRANIN)
    info_branin['global_minimum_X'] = 'abc'
    assert not utils_benchmarks.validate_info(info_branin)

    info_branin = copy.deepcopy(benchmarks.INFO_BRANIN)
    info_branin['global_minimum_X'] = np.arange(0, 10)
    assert not utils_benchmarks.validate_info(info_branin)

    info_branin = copy.deepcopy(benchmarks.INFO_BRANIN)
    info_branin['bounds'] = np.zeros((3, 2))
    assert not utils_benchmarks.validate_info(info_branin)

    info_branin = copy.deepcopy(benchmarks.INFO_BRANIN)
    info_branin['global_minimum_X'] = np.zeros((10, 3))
    assert not utils_benchmarks.validate_info(info_branin)

    info_branin = copy.deepcopy(benchmarks.INFO_BRANIN)
    info_branin['bounds'] = np.zeros((2, 4))
    assert not utils_benchmarks.validate_info(info_branin)

    info_ackley = copy.deepcopy(benchmarks.INFO_ACKLEY)
    info_ackley['dim_fun'] = 4
    assert not utils_benchmarks.validate_info(info_ackley)

    info_ackley = copy.deepcopy(benchmarks.INFO_ACKLEY)
    info_ackley['bounds'] = np.zeros((3, 2))
    assert not utils_benchmarks.validate_info(info_ackley)

    info_ackley = copy.deepcopy(benchmarks.INFO_ACKLEY)
    info_ackley['global_minimum_X'] = np.zeros((2, 5))
    assert not utils_benchmarks.validate_info(info_ackley)

def test_get_bounds():
    info_branin = copy.deepcopy(benchmarks.INFO_BRANIN)
    info_ackley = copy.deepcopy(benchmarks.INFO_ACKLEY)
    with pytest.raises(AssertionError) as error:
        utils_benchmarks.get_bounds(1, 2)
    with pytest.raises(AssertionError) as error:
        utils_benchmarks.get_bounds(info_branin, 1.2)
    with pytest.raises(ValueError) as error:
        utils_benchmarks.get_bounds(info_branin, 3)
    with pytest.raises(AssertionError) as error:
        utils_benchmarks.get_bounds(info_branin, np.inf)
    with pytest.raises(AssertionError) as error:
        utils_benchmarks.get_bounds(info_branin, -2)
    with pytest.raises(AssertionError) as error:
        utils_benchmarks.get_bounds(info_ackley, np.inf)
    with pytest.raises(AssertionError) as error:
        utils_benchmarks.get_bounds(info_ackley, -2)
    
    bounds = utils_benchmarks.get_bounds(info_branin, 2)
    assert (bounds == info_branin['bounds']).all()

    bounds = utils_benchmarks.get_bounds(info_ackley, 4)
    assert (bounds == np.repeat(info_ackley['bounds'], 4, axis=0)).all()

def test_get_covariate():
    info_branin = copy.deepcopy(benchmarks.INFO_BRANIN)
    info_ackley = copy.deepcopy(benchmarks.INFO_ACKLEY)
    with pytest.raises(AssertionError) as error:
        utils_benchmarks.get_covariate(info_branin, np.array([0.0, 0.0]), 2.1)
    with pytest.raises(AssertionError) as error:
        utils_benchmarks.get_covariate(info_branin, 'abc', 2)
    with pytest.raises(AssertionError) as error:
        utils_benchmarks.get_covariate(1, np.array([0.0, 0.0]), 2)
    with pytest.raises(AssertionError) as error:
        utils_benchmarks.get_covariate(info_branin, np.array([[0.0, 0.0]]), 2)
    with pytest.raises(ValueError) as error:
        utils_benchmarks.get_covariate(info_branin, np.array([0.0, 0.0, 0.0]), 2)
    with pytest.raises(ValueError) as error:
        utils_benchmarks.get_covariate(info_ackley, np.array([0.0, 0.0]), 3)

    covariate = utils_benchmarks.get_covariate(info_branin, np.array([0.0, 0.0]), 2)
    assert (covariate == np.array([0.0, 0.0])).all()

    covariate = utils_benchmarks.get_covariate(info_ackley, np.array([0.0]), 5)
    assert (covariate == np.array([0.0, 0.0, 0.0, 0.0, 0.0])).all()
