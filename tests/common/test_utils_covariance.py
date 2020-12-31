#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: December 29, 2020
#
"""test_utils_covariance"""

import typing
import pytest
import numpy as np

from bayeso import constants
from bayeso.utils import utils_covariance as package_target


def test_get_list_first_typing():
    annos = package_target._get_list_first.__annotations__

    assert annos['return'] == list

def test_get_hyps_typing():
    annos = package_target.get_hyps.__annotations__

    assert annos['str_cov'] == str
    assert annos['dim'] == int
    assert annos['use_ard'] == bool
    assert annos['use_gp'] == bool
    assert annos['return'] == dict

def test_get_hyps():
    with pytest.raises(AssertionError) as error:
        package_target.get_hyps(1.2, 2.1)
    with pytest.raises(AssertionError) as error:
        package_target.get_hyps(1.2, 2)
    with pytest.raises(AssertionError) as error:
        package_target.get_hyps('se', 2.1)
    with pytest.raises(AssertionError) as error:
        package_target.get_hyps('abc', 2)
    with pytest.raises(AssertionError) as error:
        package_target.get_hyps('se', 2, use_ard='abc')
    with pytest.raises(AssertionError) as error:
        package_target.get_hyps('se', 2, use_gp='abc')

    cur_hyps =  package_target.get_hyps('se', 2)
    assert cur_hyps['noise'] == constants.GP_NOISE
    assert cur_hyps['signal'] == 1.0
    assert len(cur_hyps['lengthscales'].shape) == 1
    assert (cur_hyps['lengthscales'] == np.array([1.0, 1.0])).all()

    cur_hyps =  package_target.get_hyps('se', 2, use_ard=False)
    assert cur_hyps['noise'] == constants.GP_NOISE
    assert cur_hyps['signal'] == 1.0
    assert cur_hyps['lengthscales'] == 1.0

    cur_hyps =  package_target.get_hyps('matern32', 2, use_ard=False)
    assert cur_hyps['noise'] == constants.GP_NOISE
    assert cur_hyps['signal'] == 1.0
    assert cur_hyps['lengthscales'] == 1.0

    cur_hyps =  package_target.get_hyps('matern52', 2, use_ard=False)
    assert cur_hyps['noise'] == constants.GP_NOISE
    assert cur_hyps['signal'] == 1.0
    assert cur_hyps['lengthscales'] == 1.0

    cur_hyps =  package_target.get_hyps('matern32', 2, use_ard=False, use_gp=False)
    assert cur_hyps['noise'] == constants.GP_NOISE
    assert cur_hyps['signal'] == 1.0
    assert cur_hyps['lengthscales'] == 1.0
    assert cur_hyps['dof'] == 5.0

    cur_hyps =  package_target.get_hyps('matern32', 2, use_ard=True, use_gp=False)
    assert cur_hyps['noise'] == constants.GP_NOISE
    assert cur_hyps['signal'] == 1.0
    assert np.all(cur_hyps['lengthscales'] == np.array([1.0, 1.0]))
    assert cur_hyps['dof'] == 5.0

def test_get_range_hyps_typing():
    annos = package_target.get_range_hyps.__annotations__

    assert annos['str_cov'] == str
    assert annos['dim'] == int
    assert annos['use_ard'] == bool
    assert annos['use_gp'] == bool
    assert annos['fix_noise'] == bool
    assert annos['return'] == list

def test_get_range_hyps():
    with pytest.raises(AssertionError) as error:
        package_target.get_range_hyps(1.0, 2)
    with pytest.raises(AssertionError) as error:
        package_target.get_range_hyps('abc', 2)
    with pytest.raises(AssertionError) as error:
        package_target.get_range_hyps('se', 1.2)
    with pytest.raises(AssertionError) as error:
        package_target.get_range_hyps('se', 2, use_ard='abc')
    with pytest.raises(AssertionError) as error:
        package_target.get_range_hyps('se', 2, use_ard=1)
    with pytest.raises(AssertionError) as error:
        package_target.get_range_hyps('se', 2, use_gp='abc')
    with pytest.raises(AssertionError) as error:
        package_target.get_range_hyps('se', 2, use_gp=1)
    with pytest.raises(AssertionError) as error:
        package_target.get_range_hyps('se', 2, fix_noise=1)
    with pytest.raises(AssertionError) as error:
        package_target.get_range_hyps('se', 2, fix_noise='abc')

    cur_range = package_target.get_range_hyps('se', 2, use_ard=False, fix_noise=False)
    print(type(cur_range))
    print(cur_range)
    assert isinstance(cur_range, list)
    assert cur_range == [[0.001, 10.0], [0.01, 1000.0], [0.01, 1000.0]]

    cur_range = package_target.get_range_hyps('se', 2, use_ard=False, fix_noise=False, use_gp=False)
    print(type(cur_range))
    print(cur_range)
    assert isinstance(cur_range, list)
    assert cur_range == [[0.001, 10.0], [2.00001, 200.0], [0.01, 1000.0], [0.01, 1000.0]]

def test_convert_hyps_typing():
    annos = package_target.convert_hyps.__annotations__

    assert annos['str_cov'] == str
    assert annos['hyps'] == dict
    assert annos['fix_noise'] == bool
    assert annos['use_gp'] == bool
    assert annos['return'] == np.ndarray

def test_convert_hyps():
    cur_hyps = {'noise': 0.1, 'signal': 1.0, 'lengthscales': np.array([2.0, 2.0])}

    with pytest.raises(AssertionError) as error:
        package_target.convert_hyps(1.2, 2.1)
    with pytest.raises(AssertionError) as error:
        package_target.convert_hyps(1.2, dict())
    with pytest.raises(AssertionError) as error:
        package_target.convert_hyps('se', 2.1)
    with pytest.raises(AssertionError) as error:
        package_target.convert_hyps('abc', 2.1)
    with pytest.raises(AssertionError) as error:
        package_target.convert_hyps('abc', cur_hyps)
    with pytest.raises(AssertionError) as error:
        package_target.convert_hyps('se', dict(), fix_noise=1)
    with pytest.raises(AssertionError) as error:
        package_target.convert_hyps('se', cur_hyps, use_gp=1)
    with pytest.raises(AssertionError) as error:
        package_target.convert_hyps('se', cur_hyps, use_gp='abc')

    converted_hyps = package_target.convert_hyps('se', cur_hyps, fix_noise=False)
    assert len(converted_hyps.shape) == 1
    assert converted_hyps.shape[0] == 4
    assert converted_hyps[0] == cur_hyps['noise']
    assert converted_hyps[1] == cur_hyps['signal']
    assert (converted_hyps[2:] == cur_hyps['lengthscales']).all()

    converted_hyps = package_target.convert_hyps('se', cur_hyps, fix_noise=True)
    assert len(converted_hyps.shape) == 1
    assert converted_hyps.shape[0] == 3
    assert converted_hyps[0] == cur_hyps['signal']
    assert (converted_hyps[1:] == cur_hyps['lengthscales']).all()

    cur_hyps = {'noise': 0.1, 'signal': 1.0, 'lengthscales': np.array([2.0, 2.0]), 'dof': 100.0}
    converted_hyps = package_target.convert_hyps('se', cur_hyps, fix_noise=False, use_gp=False)

    assert len(converted_hyps.shape) == 1
    assert converted_hyps.shape[0] == 5
    assert converted_hyps[0] == cur_hyps['noise']
    assert converted_hyps[1] == cur_hyps['dof']
    assert converted_hyps[2] == cur_hyps['signal']
    assert (converted_hyps[3:] == cur_hyps['lengthscales']).all()

    converted_hyps = package_target.convert_hyps('se', cur_hyps, fix_noise=True, use_gp=False)
    assert len(converted_hyps.shape) == 1
    assert converted_hyps.shape[0] == 4
    assert converted_hyps[0] == cur_hyps['dof']
    assert converted_hyps[1] == cur_hyps['signal']
    assert (converted_hyps[2:] == cur_hyps['lengthscales']).all()

def test_restore_hyps_typing():
    annos = package_target.restore_hyps.__annotations__

    assert annos['str_cov'] == str
    assert annos['hyps'] == np.ndarray
    assert annos['use_gp'] == bool
    assert annos['fix_noise'] == bool
    assert annos['noise'] == float
    assert annos['return'] == dict

def test_restore_hyps():
    with pytest.raises(AssertionError) as error:
        package_target.restore_hyps(1.2, 2.1)
    with pytest.raises(AssertionError) as error:
        package_target.restore_hyps(1.2, np.array([1.0, 1.0]))
    with pytest.raises(AssertionError) as error:
        package_target.restore_hyps('se', 2.1)
    with pytest.raises(AssertionError) as error:
        package_target.restore_hyps('abc', 2.1)
    with pytest.raises(AssertionError) as error:
        package_target.restore_hyps('se', np.array([[1.0, 1.0], [1.0, 1.0]]))
    with pytest.raises(AssertionError) as error:
        package_target.restore_hyps('se', np.array([1.0, 1.0, 1.0]), fix_noise=1)
    with pytest.raises(AssertionError) as error:
        package_target.restore_hyps('se', np.array([1.0, 1.0, 1.0]), noise='abc')
    with pytest.raises(AssertionError) as error:
        package_target.restore_hyps('se', np.array([1.0, 1.0, 1.0]), use_gp=1)
    with pytest.raises(AssertionError) as error:
        package_target.restore_hyps('se', np.array([1.0, 1.0, 1.0]), use_gp='abc')

    cur_hyps = np.array([0.1, 1.0, 1.0, 1.0, 1.0])
    restored_hyps = package_target.restore_hyps('se', cur_hyps, fix_noise=False)
    assert restored_hyps['noise'] == cur_hyps[0]
    assert restored_hyps['signal'] == cur_hyps[1]
    assert (restored_hyps['lengthscales'] == cur_hyps[2:]).all()

    restored_hyps = package_target.restore_hyps('se', cur_hyps, fix_noise=True)
    assert restored_hyps['noise'] == constants.GP_NOISE
    assert restored_hyps['signal'] == cur_hyps[0]
    assert (restored_hyps['lengthscales'] == cur_hyps[1:]).all()

    cur_hyps = np.array([0.1, 100.0, 20.0, 1.0, 1.0, 1.0])
    restored_hyps = package_target.restore_hyps('se', cur_hyps, fix_noise=False, use_gp=False)
    assert restored_hyps['noise'] == cur_hyps[0]
    assert restored_hyps['dof'] == cur_hyps[1]
    assert restored_hyps['signal'] == cur_hyps[2]
    assert (restored_hyps['lengthscales'] == cur_hyps[3:]).all()

    cur_hyps = np.array([100.0, 20.0, 1.0, 1.0, 1.0])
    restored_hyps = package_target.restore_hyps('se', cur_hyps, fix_noise=True, use_gp=False)
    assert restored_hyps['noise'] == constants.GP_NOISE
    assert restored_hyps['dof'] == cur_hyps[0]
    assert restored_hyps['signal'] == cur_hyps[1]
    assert (restored_hyps['lengthscales'] == cur_hyps[2:]).all()

def test_validate_hyps_dict_typing():
    annos = package_target.validate_hyps_dict.__annotations__

    assert annos['hyps'] == dict
    assert annos['str_cov'] == str
    assert annos['dim'] == int
    assert annos['use_gp'] == bool
    assert annos['return'] == typing.Tuple[dict, bool]

def test_validate_hyps_dict():
    num_dim = 2
    str_cov = 'matern32'

    cur_hyps = package_target.get_hyps(str_cov, num_dim)
    with pytest.raises(AssertionError) as error:
        _, is_valid = package_target.validate_hyps_dict(123, str_cov, num_dim)
    with pytest.raises(AssertionError) as error:
        _, is_valid = package_target.validate_hyps_dict(cur_hyps, 'abc', num_dim)
    with pytest.raises(AssertionError) as error:
        _, is_valid = package_target.validate_hyps_dict(cur_hyps, str_cov, 'abc')
    with pytest.raises(AssertionError) as error:
        _, is_valid = package_target.validate_hyps_dict(cur_hyps, str_cov, num_dim, use_gp=1)
    with pytest.raises(AssertionError) as error:
        _, is_valid = package_target.validate_hyps_dict(cur_hyps, str_cov, num_dim, use_gp='abc')

    cur_hyps = package_target.get_hyps(str_cov, num_dim)
    cur_hyps.pop('noise')
    with pytest.raises(AssertionError) as error:
        _, is_valid = package_target.validate_hyps_dict(cur_hyps, str_cov, num_dim)
        assert is_valid == True

    cur_hyps = package_target.get_hyps(str_cov, num_dim)
    cur_hyps.pop('lengthscales')
    with pytest.raises(AssertionError) as error:
        _, is_valid = package_target.validate_hyps_dict(cur_hyps, str_cov, num_dim)
        assert is_valid == True

    cur_hyps = package_target.get_hyps(str_cov, num_dim)
    cur_hyps.pop('signal')
    with pytest.raises(AssertionError) as error:
        _, is_valid = package_target.validate_hyps_dict(cur_hyps, str_cov, num_dim)
        assert is_valid == True

    cur_hyps = package_target.get_hyps(str_cov, num_dim)
    cur_hyps['noise'] = 'abc'
    with pytest.raises(AssertionError) as error:
        _, is_valid = package_target.validate_hyps_dict(cur_hyps, str_cov, num_dim)
        assert is_valid == True

    cur_hyps = package_target.get_hyps(str_cov, num_dim)
    cur_hyps['noise'] = np.inf
    cur_hyps, is_valid = package_target.validate_hyps_dict(cur_hyps, str_cov, num_dim)
    assert cur_hyps['noise'] == constants.BOUND_UPPER_GP_NOISE

    cur_hyps = package_target.get_hyps(str_cov, num_dim)
    with pytest.raises(AssertionError) as error:
        _, is_valid = package_target.validate_hyps_dict(cur_hyps, str_cov, 123)
        assert is_valid == True

    cur_hyps = package_target.get_hyps(str_cov, num_dim)
    cur_hyps['lengthscales'] = 'abc'
    with pytest.raises(AssertionError) as error:
        _, is_valid = package_target.validate_hyps_dict(cur_hyps, str_cov, num_dim)
        assert is_valid == True

    cur_hyps = package_target.get_hyps(str_cov, num_dim)
    cur_hyps['signal'] = 'abc'
    with pytest.raises(AssertionError) as error:
        _, is_valid = package_target.validate_hyps_dict(cur_hyps, str_cov, num_dim)
        assert is_valid == True

    cur_hyps = package_target.get_hyps(str_cov, num_dim, use_gp=False)
    cur_hyps['signal'] = 'abc'
    with pytest.raises(AssertionError) as error:
        _, is_valid = package_target.validate_hyps_dict(cur_hyps, str_cov, num_dim)
        assert is_valid == True

    cur_hyps = package_target.get_hyps(str_cov, num_dim, use_gp=False)
    cur_hyps['dof'] = 'abc'
    with pytest.raises(AssertionError) as error:
        _, is_valid = package_target.validate_hyps_dict(cur_hyps, str_cov, num_dim, use_gp=False)
        assert is_valid == True

    cur_hyps = package_target.get_hyps(str_cov, num_dim, use_ard=False, use_gp=False)
    cur_hyps['lengthscales'] = 'abc'
    with pytest.raises(AssertionError) as error:
        _, is_valid = package_target.validate_hyps_dict(cur_hyps, str_cov, num_dim, use_gp=False)
        assert is_valid == True

def test_validate_hyps_arr_typing():
    annos = package_target.validate_hyps_arr.__annotations__

    assert annos['hyps'] == np.ndarray
    assert annos['str_cov'] == str
    assert annos['dim'] == int
    assert annos['use_gp'] == bool
    assert annos['return'] == typing.Tuple[np.ndarray, bool]

def test_validate_hyps_arr():
    num_dim = 2
    str_cov = 'matern32'

    cur_hyps = package_target.get_hyps(str_cov, num_dim)
    cur_hyps = package_target.convert_hyps(str_cov, cur_hyps, fix_noise=False)
    with pytest.raises(AssertionError) as error:
        _, is_valid = package_target.validate_hyps_arr(123, str_cov, num_dim)
    with pytest.raises(AssertionError) as error:
        _, is_valid = package_target.validate_hyps_arr(cur_hyps, 'abc', num_dim)
    with pytest.raises(AssertionError) as error:
        _, is_valid = package_target.validate_hyps_arr(cur_hyps, str_cov, 'abc')
    with pytest.raises(AssertionError) as error:
        _, is_valid = package_target.validate_hyps_arr(cur_hyps, str_cov, num_dim, use_gp='abc')

def test_check_str_cov_typing():
    annos = package_target.check_str_cov.__annotations__

    assert annos['str_fun'] == str
    assert annos['str_cov'] == str
    assert annos['shape_X1'] == tuple
    assert annos['shape_X2'] == tuple
    assert annos['return'] == type(None)

def test_check_str_cov():
    with pytest.raises(AssertionError) as error:
        package_target.check_str_cov(1, 'se', (2, 1))
    with pytest.raises(AssertionError) as error:
        package_target.check_str_cov('test', 1, (2, 1))
    with pytest.raises(AssertionError) as error:
        package_target.check_str_cov('test', 'se', 1)
    with pytest.raises(AssertionError) as error:
        package_target.check_str_cov('test', 'se', (2, 100, 100))
    with pytest.raises(AssertionError) as error:
        package_target.check_str_cov('test', 'se', (2, 100), shape_X2=(2, 100, 100))
    with pytest.raises(AssertionError) as error:
        package_target.check_str_cov('test', 'set_se', (2, 100), shape_X2=(2, 100, 100))
    with pytest.raises(AssertionError) as error:
        package_target.check_str_cov('test', 'set_se', (2, 100, 100), shape_X2=(2, 100))
    with pytest.raises(AssertionError) as error:
        package_target.check_str_cov('test', 'se', (2, 1), shape_X2=1)

    with pytest.raises(ValueError) as error:
        package_target.check_str_cov('test', 'abc', (2, 1))
