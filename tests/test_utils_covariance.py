# test_utils_covariance
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 18, 2018

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
    with pytest.raises(AssertionError) as error:
        utils_covariance.get_hyps('abc', 2)
    with pytest.raises(AssertionError) as error:
        utils_covariance.get_hyps('se', 2, is_ard='abc')
   
    cur_hyps =  utils_covariance.get_hyps('se', 2)
    assert cur_hyps['noise'] == constants.GP_NOISE
    assert cur_hyps['signal'] == 1.0
    assert len(cur_hyps['lengthscales'].shape) == 1
    assert (cur_hyps['lengthscales'] == np.array([1.0, 1.0])).all()

    cur_hyps =  utils_covariance.get_hyps('se', 2, is_ard=False)
    assert cur_hyps['noise'] == constants.GP_NOISE
    assert cur_hyps['signal'] == 1.0
    assert cur_hyps['lengthscales'] == 1.0

    cur_hyps =  utils_covariance.get_hyps('matern32', 2, is_ard=False)
    assert cur_hyps['noise'] == constants.GP_NOISE
    assert cur_hyps['signal'] == 1.0
    assert cur_hyps['lengthscales'] == 1.0

    cur_hyps =  utils_covariance.get_hyps('matern52', 2, is_ard=False)
    assert cur_hyps['noise'] == constants.GP_NOISE
    assert cur_hyps['signal'] == 1.0
    assert cur_hyps['lengthscales'] == 1.0

def test_get_range_hyps():
    with pytest.raises(AssertionError) as error:
        utils_covariance.get_range_hyps(1.0, 2)
    with pytest.raises(AssertionError) as error:
        utils_covariance.get_range_hyps('abc', 2)
    with pytest.raises(AssertionError) as error:
        utils_covariance.get_range_hyps('se', 1.2)
    with pytest.raises(AssertionError) as error:
        utils_covariance.get_range_hyps('se', 2, is_ard='abc')
    with pytest.raises(AssertionError) as error:
        utils_covariance.get_range_hyps('se', 2, is_ard=1)
    with pytest.raises(AssertionError) as error:
        utils_covariance.get_range_hyps('se', 2, is_fixed_noise=1)
    with pytest.raises(AssertionError) as error:
        utils_covariance.get_range_hyps('se', 2, is_fixed_noise='abc')
    
    cur_range = utils_covariance.get_range_hyps('se', 2, is_ard=False, is_fixed_noise=False)
    print(type(cur_range))
    print(cur_range)
    assert isinstance(cur_range, list)
    assert cur_range == [[0.001, 10.0], [0.01, 1000.0], [0.01, 1000.0]]

def test_convert_hyps():
    cur_hyps = {'noise': 0.1, 'signal': 1.0, 'lengthscales': np.array([1.0, 1.0])}

    with pytest.raises(AssertionError) as error:
        utils_covariance.convert_hyps(1.2, 2.1)
    with pytest.raises(AssertionError) as error:
        utils_covariance.convert_hyps(1.2, dict())
    with pytest.raises(AssertionError) as error:
        utils_covariance.convert_hyps('se', 2.1)
    with pytest.raises(AssertionError) as error:
        utils_covariance.convert_hyps('abc', 2.1)
    with pytest.raises(AssertionError) as error:
        utils_covariance.convert_hyps('abc', cur_hyps)
    with pytest.raises(AssertionError) as error:
        utils_covariance.convert_hyps('se', dict(), is_fixed_noise=1)

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

def test_validate_hyps_dict():
    num_dim = 2
    str_cov = 'matern32'
    cur_hyps =  utils_covariance.get_hyps(str_cov, num_dim)
    cur_hyps.pop('noise')
    with pytest.raises(AssertionError) as error:
        _, is_valid = utils_covariance.validate_hyps_dict(cur_hyps, str_cov, num_dim)
        assert is_valid == True

    cur_hyps =  utils_covariance.get_hyps(str_cov, num_dim)
    with pytest.raises(AssertionError) as error:
        _, is_valid = utils_covariance.validate_hyps_dict(cur_hyps, 'abc', num_dim)
        assert is_valid == True

    cur_hyps =  utils_covariance.get_hyps(str_cov, num_dim)
    cur_hyps.pop('lengthscales')
    with pytest.raises(AssertionError) as error:
        _, is_valid = utils_covariance.validate_hyps_dict(cur_hyps, str_cov, num_dim)
        assert is_valid == True

    cur_hyps =  utils_covariance.get_hyps(str_cov, num_dim)
    cur_hyps.pop('signal')
    with pytest.raises(AssertionError) as error:
        _, is_valid = utils_covariance.validate_hyps_dict(cur_hyps, str_cov, num_dim)
        assert is_valid == True

    cur_hyps =  utils_covariance.get_hyps(str_cov, num_dim)
    cur_hyps['noise'] = 'abc'
    with pytest.raises(AssertionError) as error:
        _, is_valid = utils_covariance.validate_hyps_dict(cur_hyps, str_cov, num_dim)
        assert is_valid == True

    cur_hyps =  utils_covariance.get_hyps(str_cov, num_dim)
    cur_hyps['noise'] = np.inf
    cur_hyps, is_valid = utils_covariance.validate_hyps_dict(cur_hyps, str_cov, num_dim)
    assert cur_hyps['noise'] == constants.BOUND_UPPER_GP_NOISE

    cur_hyps =  utils_covariance.get_hyps(str_cov, num_dim)
    with pytest.raises(AssertionError) as error:
        _, is_valid = utils_covariance.validate_hyps_dict(cur_hyps, str_cov, 123)
        assert is_valid == True

    cur_hyps =  utils_covariance.get_hyps(str_cov, num_dim)
    cur_hyps['lengthscales'] = 'abc'
    with pytest.raises(AssertionError) as error:
        _, is_valid = utils_covariance.validate_hyps_dict(cur_hyps, str_cov, num_dim)
        assert is_valid == True

    cur_hyps =  utils_covariance.get_hyps(str_cov, num_dim)
    cur_hyps['signal'] = 'abc'
    with pytest.raises(AssertionError) as error:
        _, is_valid = utils_covariance.validate_hyps_dict(cur_hyps, str_cov, num_dim)
        assert is_valid == True
