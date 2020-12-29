#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: September 24, 2020
#
"""test_utils_logger"""

import logging
import pytest
import numpy as np

from bayeso.utils import utils_logger as package_target


def test_get_logger_typing():
    annos = package_target.get_logger.__annotations__

    assert annos['str_name'] == str
    assert annos['return'] == logging.Logger

def test_get_logger():
    with pytest.raises(AssertionError) as error:
        package_target.get_logger(123)
    with pytest.raises(AssertionError) as error:
        package_target.get_logger(12.3)

    logger = package_target.get_logger('abc')

    assert type(logger) == logging.Logger

def test_get_str_array_1d_typing():
    annos = package_target.get_str_array_1d.__annotations__

    assert annos['arr'] == np.ndarray
    assert annos['return'] == str

def test_get_str_array_1d():
    with pytest.raises(AssertionError) as error:
        package_target.get_str_array_1d(123)
    with pytest.raises(AssertionError) as error:
        package_target.get_str_array_1d(12.3)
    with pytest.raises(AssertionError) as error:
        package_target.get_str_array_1d(np.zeros((10, 2)))
    with pytest.raises(AssertionError) as error:
        package_target.get_str_array_1d(np.zeros((10, 2, 2)))

    str_ = package_target.get_str_array_1d(np.array([1, 2, 3]))
    print(str_)

    assert str_ == '[1, 2, 3]'

    str_ = package_target.get_str_array_1d(np.array([1.1, 2.5, 3.0]))
    print(str_)

    assert str_ == '[1.100, 2.500, 3.000]'

def test_get_str_array_2d_typing():
    annos = package_target.get_str_array_2d.__annotations__

    assert annos['arr'] == np.ndarray
    assert annos['return'] == str

def test_get_str_array_2d():
    with pytest.raises(AssertionError) as error:
        package_target.get_str_array_2d(123)
    with pytest.raises(AssertionError) as error:
        package_target.get_str_array_2d(12.3)
    with pytest.raises(AssertionError) as error:
        package_target.get_str_array_2d(np.zeros(10))
    with pytest.raises(AssertionError) as error:
        package_target.get_str_array_2d(np.zeros((10, 2, 2)))

    str_ = package_target.get_str_array_2d(np.array([[1, 2, 3], [2, 2, 2]]))
    print(str_)

    assert str_ == '[[1, 2, 3],\n[2, 2, 2]]'

    str_ = package_target.get_str_array_2d(np.array([[1.1, 2.2, 3.33], [2.2, 2.4, 2.9]]))
    print(str_)

    assert str_ == '[[1.100, 2.200, 3.330],\n[2.200, 2.400, 2.900]]'

def test_get_str_array_3d_typing():
    annos = package_target.get_str_array_3d.__annotations__

    assert annos['arr'] == np.ndarray
    assert annos['return'] == str

def test_get_str_array_3d():
    with pytest.raises(AssertionError) as error:
        package_target.get_str_array_3d(123)
    with pytest.raises(AssertionError) as error:
        package_target.get_str_array_3d(12.3)
    with pytest.raises(AssertionError) as error:
        package_target.get_str_array_3d(np.zeros(10))
    with pytest.raises(AssertionError) as error:
        package_target.get_str_array_3d(np.zeros((10, 2)))

    str_ = package_target.get_str_array_3d(np.array([[[1, 2, 3], [2, 2, 2]], [[1, 2, 3], [2, 2, 2]]]))
    print(str_)

    assert str_ == '[[[1, 2, 3],\n[2, 2, 2]],\n[[1, 2, 3],\n[2, 2, 2]]]'

    str_ = package_target.get_str_array_3d(np.array([[[1.1, 2.2, 3.33], [2.2, 2.4, 2.9]], [[1.1, 2.2, 3.33], [2.2, 2.4, 2.9]]]))
    print(str_)

    assert str_ == '[[[1.100, 2.200, 3.330],\n[2.200, 2.400, 2.900]],\n[[1.100, 2.200, 3.330],\n[2.200, 2.400, 2.900]]]'

def test_get_str_array_typing():
    annos = package_target.get_str_array.__annotations__

    assert annos['arr'] == np.ndarray
    assert annos['return'] == str

def test_get_str_array():
    with pytest.raises(AssertionError) as error:
        package_target.get_str_array(123)
    with pytest.raises(AssertionError) as error:
        package_target.get_str_array(12.3)

    str_ = package_target.get_str_array(np.array([1, 2, 3]))
    print(str_)
    assert str_ == '[1, 2, 3]'

    str_ = package_target.get_str_array(np.array([1.1, 2.5, 3.0]))
    print(str_)
    assert str_ == '[1.100, 2.500, 3.000]'

    str_ = package_target.get_str_array(np.array([[1, 2, 3], [2, 2, 2]]))
    print(str_)
    assert str_ == '[[1, 2, 3],\n[2, 2, 2]]'

    str_ = package_target.get_str_array(np.array([[1.1, 2.2, 3.33], [2.2, 2.4, 2.9]]))
    print(str_)
    assert str_ == '[[1.100, 2.200, 3.330],\n[2.200, 2.400, 2.900]]'

    str_ = package_target.get_str_array(np.array([[[1, 2, 3], [2, 2, 2]], [[1, 2, 3], [2, 2, 2]]]))
    print(str_)
    assert str_ == '[[[1, 2, 3],\n[2, 2, 2]],\n[[1, 2, 3],\n[2, 2, 2]]]'

    str_ = package_target.get_str_array(np.array([[[1.1, 2.2, 3.33], [2.2, 2.4, 2.9]], [[1.1, 2.2, 3.33], [2.2, 2.4, 2.9]]]))
    print(str_)
    assert str_ == '[[[1.100, 2.200, 3.330],\n[2.200, 2.400, 2.900]],\n[[1.100, 2.200, 3.330],\n[2.200, 2.400, 2.900]]]'

def test_get_str_hyps_typing():
    annos = package_target.get_str_hyps.__annotations__

    assert annos['hyps'] == dict
    assert annos['return'] == str

def test_get_str_hyps():
    with pytest.raises(AssertionError) as error:
        package_target.get_str_hyps(123)
    with pytest.raises(AssertionError) as error:
        package_target.get_str_hyps(12.3)
    with pytest.raises(AssertionError) as error:
        package_target.get_str_hyps('abc')
    with pytest.raises(AssertionError) as error:
        package_target.get_str_hyps(np.zeros(3))

    hyps = {'signal': 1.0, 'noise': 1e-4, 'lengthscales': np.array([1.0, 2.0])}
    str_ = package_target.get_str_hyps(hyps)
    print(str_)
    list_truths = [
        "{'signal': 1.000, 'noise': 0.000, 'lengthscales': [1.000, 2.000]}",
        "{'signal': 1.000, 'lengthscales': [1.000, 2.000], 'noise': 0.000}",
        "{'lengthscales': [1.000, 2.000], 'signal': 1.000, 'noise': 0.000}",
        "{'lengthscales': [1.000, 2.000], 'noise': 0.000, 'signal': 1.000}",
        "{'noise': 0.000, 'signal': 1.000, 'lengthscales': [1.000, 2.000]}",
        "{'noise': 0.000, 'lengthscales': [1.000, 2.000], 'signal': 1.000}",
    ]
    assert str_ in list_truths

    hyps = {'signal': 1, 'noise': 1e-3, 'lengthscales': np.array([1.0, 2.0])}
    str_ = package_target.get_str_hyps(hyps)
    print(str_)
    list_truths = [
        "{'signal': 1, 'noise': 0.001, 'lengthscales': [1.000, 2.000]}",
        "{'signal': 1, 'lengthscales': [1.000, 2.000], 'noise': 0.001}",
        "{'lengthscales': [1.000, 2.000], 'signal': 1, 'noise': 0.001}",
        "{'lengthscales': [1.000, 2.000], 'noise': 0.001, 'signal': 1}",
        "{'noise': 0.001, 'signal': 1, 'lengthscales': [1.000, 2.000]}",
        "{'noise': 0.001, 'lengthscales': [1.000, 2.000], 'signal': 1}",
    ]
    assert str_ in list_truths
