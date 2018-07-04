# test_utils_covariance
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 04, 2018

import pytest
import numpy as np

from bayeso.utils import utils_covariance
from bayeso import constants


def test_get_hyps():
    with pytest.raises(AssertionError) as error:
        utils_covariance.get_hyps(1.2, 2.1)
    with pytest.raises(AssertionError) as error:
        utils_covariance.get_hyps(1.2, 2)
    with pytest.raises(AssertionError) as error:
        utils_covariance.get_hyps('se', 2.1)
    with pytest.raises(ValueError) as error:
        utils_covariance.get_hyps('abc', 2)
    
    cur_hyps =  utils_covariance.get_hyps('se', 2)
    assert cur_hyps['noise'] == constants.GP_NOISE
    assert cur_hyps['signal'] == 1.0
    assert len(cur_hyps['lengthscales'].shape) == 1
    assert (cur_hyps['lengthscales'] == np.array([1.0, 1.0])).all()

def test_convert_hyps():
    with pytest.raises(AssertionError) as error:
        utils_covariance.convert_hyps(1.2, 2.1)
    with pytest.raises(AssertionError) as error:
        utils_covariance.convert_hyps(1.2, dict())
    with pytest.raises(AssertionError) as error:
        utils_covariance.convert_hyps('se', 2.1)
    with pytest.raises(AssertionError) as error:
        utils_covariance.convert_hyps('abc', 2.1)
    with pytest.raises(AssertionError) as error:
        utils_covariance.convert_hyps('se', dict(), is_fixed_noise=1)

    cur_hyps = {'noise': 0.1, 'signal': 1.0, 'lengthscales': np.array([1.0, 1.0])}
    converted_hyps = utils_covariance.convert_hyps('se', cur_hyps)
    assert converted_hyps[0] == cur_hyps['noise']
    assert converted_hyps[1] == cur_hyps['signal']
    assert (converted_hyps[2:] == cur_hyps['lengthscales']).all()

def test_restore_hyps():
    with pytest.raises(AssertionError) as error:
        utils_covariance.restore_hyps(1.2, 2.1)
    with pytest.raises(AssertionError) as error:
        utils_covariance.restore_hyps(1.2, np.array([1.0, 1.0]))
    with pytest.raises(AssertionError) as error:
        utils_covariance.restore_hyps('se', 2.1)
    with pytest.raises(AssertionError) as error:
        utils_covariance.restore_hyps('abc', 2.1)
    with pytest.raises(AssertionError) as error:
        utils_covariance.restore_hyps('se', np.array([[1.0, 1.0], [1.0, 1.0]]))
    with pytest.raises(AssertionError) as error:
        utils_covariance.restore_hyps('se', np.array([1.0, 1.0, 1.0]), is_fixed_noise=1)
    with pytest.raises(AssertionError) as error:
        utils_covariance.restore_hyps('se', np.array([1.0, 1.0, 1.0]), fixed_noise='abc')

    cur_hyps = np.array([0.1, 1.0, 1.0, 1.0, 1.0])
    restored_hyps = utils_covariance.restore_hyps('se', cur_hyps)
    assert restored_hyps['noise'] == cur_hyps[0]
    assert restored_hyps['signal'] == cur_hyps[1]
    assert (restored_hyps['lengthscales'] == cur_hyps[2:]).all()

    restored_hyps = utils_covariance.restore_hyps('se', cur_hyps, is_fixed_noise=True)
    assert restored_hyps['noise'] == constants.GP_NOISE
    assert restored_hyps['signal'] == cur_hyps[0]
    assert (restored_hyps['lengthscales'] == cur_hyps[1:]).all()
