# test_utils_covariance
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: September 17, 2020

import pytest
import numpy as np
import typing

from bayeso.utils import utils_covariance
from bayeso import constants


def test_get_list_first_typing():
    annos = utils_covariance._get_list_first.__annotations__

    assert annos['return'] == list

def test_get_hyps_typing():
    annos = utils_covariance.get_hyps.__annotations__

    assert annos['str_cov'] == str
    assert annos['dim'] == int
    assert annos['use_ard'] == bool
    assert annos['return'] == dict

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
        utils_covariance.get_hyps('se', 2, use_ard='abc')
   
    cur_hyps =  utils_covariance.get_hyps('se', 2)
    assert cur_hyps['noise'] == constants.GP_NOISE
    assert cur_hyps['signal'] == 1.0
    assert len(cur_hyps['lengthscales'].shape) == 1
    assert (cur_hyps['lengthscales'] == np.array([1.0, 1.0])).all()

    cur_hyps =  utils_covariance.get_hyps('se', 2, use_ard=False)
    assert cur_hyps['noise'] == constants.GP_NOISE
    assert cur_hyps['signal'] == 1.0
    assert cur_hyps['lengthscales'] == 1.0

    cur_hyps =  utils_covariance.get_hyps('matern32', 2, use_ard=False)
    assert cur_hyps['noise'] == constants.GP_NOISE
    assert cur_hyps['signal'] == 1.0
    assert cur_hyps['lengthscales'] == 1.0

    cur_hyps =  utils_covariance.get_hyps('matern52', 2, use_ard=False)
    assert cur_hyps['noise'] == constants.GP_NOISE
    assert cur_hyps['signal'] == 1.0
    assert cur_hyps['lengthscales'] == 1.0

def test_get_range_hyps_typing():
    annos = utils_covariance.get_range_hyps.__annotations__

    assert annos['str_cov'] == str
    assert annos['dim'] == int
    assert annos['use_ard'] == bool
    assert annos['fix_noise'] == bool
    assert annos['return'] == list

def test_get_range_hyps():
    with pytest.raises(AssertionError) as error:
        utils_covariance.get_range_hyps(1.0, 2)
    with pytest.raises(AssertionError) as error:
        utils_covariance.get_range_hyps('abc', 2)
    with pytest.raises(AssertionError) as error:
        utils_covariance.get_range_hyps('se', 1.2)
    with pytest.raises(AssertionError) as error:
        utils_covariance.get_range_hyps('se', 2, use_ard='abc')
    with pytest.raises(AssertionError) as error:
        utils_covariance.get_range_hyps('se', 2, use_ard=1)
    with pytest.raises(AssertionError) as error:
        utils_covariance.get_range_hyps('se', 2, fix_noise=1)
    with pytest.raises(AssertionError) as error:
        utils_covariance.get_range_hyps('se', 2, fix_noise='abc')
    
    cur_range = utils_covariance.get_range_hyps('se', 2, use_ard=False, fix_noise=False)
    print(type(cur_range))
    print(cur_range)
    assert isinstance(cur_range, list)
    assert cur_range == [[0.001, 10.0], [0.01, 1000.0], [0.01, 1000.0]]

def test_convert_hyps_typing():
    annos = utils_covariance.convert_hyps.__annotations__

    assert annos['str_cov'] == str
    assert annos['hyps'] == dict
    assert annos['fix_noise'] == bool
    assert annos['return'] == np.ndarray

def test_convert_hyps():
    cur_hyps = {'noise': 0.1, 'signal': 1.0, 'lengthscales': np.array([2.0, 2.0])}

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
        utils_covariance.convert_hyps('se', dict(), fix_noise=1)

    converted_hyps = utils_covariance.convert_hyps('se', cur_hyps, fix_noise=False)
    assert len(converted_hyps.shape) == 1
    assert converted_hyps.shape[0] == 4
    assert converted_hyps[0] == cur_hyps['noise']
    assert converted_hyps[1] == cur_hyps['signal']
    assert (converted_hyps[2:] == cur_hyps['lengthscales']).all()

    converted_hyps = utils_covariance.convert_hyps('se', cur_hyps, fix_noise=True)
    assert len(converted_hyps.shape) == 1
    assert converted_hyps.shape[0] == 3
    assert converted_hyps[0] == cur_hyps['signal']
    assert (converted_hyps[1:] == cur_hyps['lengthscales']).all()

def test_restore_hyps_typing():
    annos = utils_covariance.restore_hyps.__annotations__

    assert annos['str_cov'] == str
    assert annos['hyps'] == np.ndarray
    assert annos['fix_noise'] == bool
    assert annos['noise'] == float
    assert annos['return'] == dict

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
        utils_covariance.restore_hyps('se', np.array([1.0, 1.0, 1.0]), fix_noise=1)
    with pytest.raises(AssertionError) as error:
        utils_covariance.restore_hyps('se', np.array([1.0, 1.0, 1.0]), noise='abc')

    cur_hyps = np.array([0.1, 1.0, 1.0, 1.0, 1.0])
    restored_hyps = utils_covariance.restore_hyps('se', cur_hyps, fix_noise=False)
    assert restored_hyps['noise'] == cur_hyps[0]
    assert restored_hyps['signal'] == cur_hyps[1]
    assert (restored_hyps['lengthscales'] == cur_hyps[2:]).all()

    restored_hyps = utils_covariance.restore_hyps('se', cur_hyps, fix_noise=True)
    assert restored_hyps['noise'] == constants.GP_NOISE
    assert restored_hyps['signal'] == cur_hyps[0]
    assert (restored_hyps['lengthscales'] == cur_hyps[1:]).all()

def test_validate_hyps_dict_typing():
    annos = utils_covariance.validate_hyps_dict.__annotations__

    assert annos['hyps'] == dict
    assert annos['str_cov'] == str
    assert annos['dim'] == int
    assert annos['return'] == typing.Tuple[dict, bool]

def test_validate_hyps_dict():
    num_dim = 2
    str_cov = 'matern32'

    cur_hyps = utils_covariance.get_hyps(str_cov, num_dim)
    with pytest.raises(AssertionError) as error:
        _, is_valid = utils_covariance.validate_hyps_dict(123, str_cov, num_dim)
    with pytest.raises(AssertionError) as error:
        _, is_valid = utils_covariance.validate_hyps_dict(cur_hyps, 'abc', num_dim)
    with pytest.raises(AssertionError) as error:
        _, is_valid = utils_covariance.validate_hyps_dict(cur_hyps, str_cov, 'abc')

    cur_hyps = utils_covariance.get_hyps(str_cov, num_dim)
    cur_hyps.pop('noise')
    with pytest.raises(AssertionError) as error:
        _, is_valid = utils_covariance.validate_hyps_dict(cur_hyps, str_cov, num_dim)
        assert is_valid == True

    cur_hyps = utils_covariance.get_hyps(str_cov, num_dim)
    cur_hyps.pop('lengthscales')
    with pytest.raises(AssertionError) as error:
        _, is_valid = utils_covariance.validate_hyps_dict(cur_hyps, str_cov, num_dim)
        assert is_valid == True

    cur_hyps = utils_covariance.get_hyps(str_cov, num_dim)
    cur_hyps.pop('signal')
    with pytest.raises(AssertionError) as error:
        _, is_valid = utils_covariance.validate_hyps_dict(cur_hyps, str_cov, num_dim)
        assert is_valid == True

    cur_hyps = utils_covariance.get_hyps(str_cov, num_dim)
    cur_hyps['noise'] = 'abc'
    with pytest.raises(AssertionError) as error:
        _, is_valid = utils_covariance.validate_hyps_dict(cur_hyps, str_cov, num_dim)
        assert is_valid == True

    cur_hyps = utils_covariance.get_hyps(str_cov, num_dim)
    cur_hyps['noise'] = np.inf
    cur_hyps, is_valid = utils_covariance.validate_hyps_dict(cur_hyps, str_cov, num_dim)
    assert cur_hyps['noise'] == constants.BOUND_UPPER_GP_NOISE

    cur_hyps = utils_covariance.get_hyps(str_cov, num_dim)
    with pytest.raises(AssertionError) as error:
        _, is_valid = utils_covariance.validate_hyps_dict(cur_hyps, str_cov, 123)
        assert is_valid == True

    cur_hyps = utils_covariance.get_hyps(str_cov, num_dim)
    cur_hyps['lengthscales'] = 'abc'
    with pytest.raises(AssertionError) as error:
        _, is_valid = utils_covariance.validate_hyps_dict(cur_hyps, str_cov, num_dim)
        assert is_valid == True

    cur_hyps = utils_covariance.get_hyps(str_cov, num_dim)
    cur_hyps['signal'] = 'abc'
    with pytest.raises(AssertionError) as error:
        _, is_valid = utils_covariance.validate_hyps_dict(cur_hyps, str_cov, num_dim)
        assert is_valid == True

def test_validate_hyps_arr_typing():
    annos = utils_covariance.validate_hyps_arr.__annotations__

    assert annos['hyps'] == np.ndarray
    assert annos['str_cov'] == str
    assert annos['dim'] == int
    assert annos['return'] == typing.Tuple[np.ndarray, bool]

def test_validate_hyps_arr():
    num_dim = 2
    str_cov = 'matern32'

    cur_hyps = utils_covariance.get_hyps(str_cov, num_dim)
    cur_hyps = utils_covariance.convert_hyps(str_cov, cur_hyps, fix_noise=False)
    with pytest.raises(AssertionError) as error:
        _, is_valid = utils_covariance.validate_hyps_arr(123, str_cov, num_dim)
    with pytest.raises(AssertionError) as error:
        _, is_valid = utils_covariance.validate_hyps_arr(cur_hyps, 'abc', num_dim)
    with pytest.raises(AssertionError) as error:
        _, is_valid = utils_covariance.validate_hyps_arr(cur_hyps, str_cov, 'abc')

