#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: March 22, 2021
#
"""test_gp_kernel"""

import typing
import pytest
import numpy as np

from bayeso import constants
from bayeso.gp import gp_kernel as package_target
from bayeso.utils import utils_covariance


TEST_EPSILON = 1e-7

def test_get_optimized_kernel_typing():
    annos = package_target.get_optimized_kernel.__annotations__

    assert annos['X_train'] == np.ndarray
    assert annos['Y_train'] == np.ndarray
    assert annos['prior_mu'] == typing.Union[callable, type(None)]
    assert annos['str_cov'] == str
    assert annos['str_optimizer_method'] == str
    assert annos['str_modelselection_method'] == str
    assert annos['fix_noise'] == bool
    assert annos['debug'] == bool
    assert annos['return'] == typing.Tuple[np.ndarray, np.ndarray, dict]

def test_get_optimized_kernel():
    np.random.seed(42)
    dim_X = 3
    num_X = 10
    num_instances = 5
    X = np.random.randn(num_X, dim_X)
    X_set = np.random.randn(num_X, num_instances, dim_X)
    Y = np.random.randn(num_X, 1)
    prior_mu = None

    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(X, Y, prior_mu, 1)
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(X, Y, 1, 'se')
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(X, 1, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(1, Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(np.ones(num_X), Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(X, np.ones(num_X), prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(np.ones((50, 3)), Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(X, np.ones((50, 1)), prior_mu, 'se')
    with pytest.raises(ValueError) as error:
        package_target.get_optimized_kernel(X, Y, prior_mu, 'abc')
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(X, Y, prior_mu, 'se', str_optimizer_method=1)
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(X, Y, prior_mu, 'se', str_modelselection_method=1)
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(X, Y, prior_mu, 'se', fix_noise=1)
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(X, Y, prior_mu, 'se', debug=1)

    # INFO: tests for set inputs
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(X_set, Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(X, Y, prior_mu, 'set_se')
    with pytest.raises(AssertionError) as error:
        package_target.get_optimized_kernel(X_set, Y, prior_mu, 'set_se', debug=1)

    cov_X_X, inv_cov_X_X, hyps = package_target.get_optimized_kernel(X, Y, prior_mu, 'se')
    print(hyps)
    cov_X_X, inv_cov_X_X, hyps = package_target.get_optimized_kernel(X, Y, prior_mu, 'eq')
    print(hyps)
    cov_X_X, inv_cov_X_X, hyps = package_target.get_optimized_kernel(X, Y, prior_mu, 'matern32')
    print(hyps)
    cov_X_X, inv_cov_X_X, hyps = package_target.get_optimized_kernel(X, Y, prior_mu, 'matern52')
    print(hyps)

    cov_X_X, inv_cov_X_X, hyps = package_target.get_optimized_kernel(X, Y, prior_mu, 'se', str_optimizer_method='BFGS')
    print(hyps)
    cov_X_X, inv_cov_X_X, hyps = package_target.get_optimized_kernel(X, Y, prior_mu, 'se', str_optimizer_method='L-BFGS-B')
    print(hyps)

    cov_X_X, inv_cov_X_X, hyps = package_target.get_optimized_kernel(X, Y, prior_mu, 'se', str_optimizer_method='Nelder-Mead', debug=True)
    print(hyps)
    cov_X_X, inv_cov_X_X, hyps = package_target.get_optimized_kernel(X, Y, prior_mu, 'se', str_optimizer_method='SLSQP', debug=True)
    print(hyps)
    cov_X_X, inv_cov_X_X, hyps = package_target.get_optimized_kernel(X, Y, prior_mu, 'se', str_optimizer_method='SLSQP-Bounded', debug=True)
    print(hyps)

    cov_X_X, inv_cov_X_X, hyps = package_target.get_optimized_kernel(X, Y, prior_mu, 'se', str_modelselection_method='loocv')
    print(hyps)

    cov_X_X, inv_cov_X_X, hyps = package_target.get_optimized_kernel(X, Y, prior_mu, 'se')
    print(hyps)

    cov_X_X, inv_cov_X_X, hyps = package_target.get_optimized_kernel(X_set, Y, prior_mu, 'set_se')
    print(hyps)
    cov_X_X, inv_cov_X_X, hyps = package_target.get_optimized_kernel(X_set, Y, prior_mu, 'set_se', str_optimizer_method='L-BFGS-B')
    print(hyps)
    cov_X_X, inv_cov_X_X, hyps = package_target.get_optimized_kernel(X_set, Y, prior_mu, 'set_se', str_modelselection_method='loocv')
    print(hyps)
