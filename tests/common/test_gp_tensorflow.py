# test_gp_tensorflow
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 11, 2020

import numpy as np
import pytest

from bayeso import constants
try:
    from bayeso.gp import gp_tensorflow
except:
    gp_tensorflow = None

from bayeso.utils import utils_covariance

TEST_EPSILON = 1e-7


def test_get_optimized_kernel():
    np.random.seed(42)
    dim_X = 3
    num_X = 10
    num_instances = 5
    X = np.random.randn(num_X, dim_X)
    X_set = np.random.randn(num_X, num_instances, dim_X)
    Y = np.random.randn(num_X, 1)
    prior_mu = None

    if gp_tensorflow is None:
        pytest.skip('TensorFlow is not installed.')

    with pytest.raises(AssertionError) as error:
        gp_tensorflow.get_optimized_kernel(X, Y, prior_mu, 1)
    with pytest.raises(AssertionError) as error:
        gp_tensorflow.get_optimized_kernel(X, Y, 1, 'se')
    with pytest.raises(AssertionError) as error:
        gp_tensorflow.get_optimized_kernel(X, 1, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp_tensorflow.get_optimized_kernel(1, Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp_tensorflow.get_optimized_kernel(np.ones(num_X), Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp_tensorflow.get_optimized_kernel(X, np.ones(num_X), prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp_tensorflow.get_optimized_kernel(np.ones((50, 3)), Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp_tensorflow.get_optimized_kernel(X, np.ones((50, 1)), prior_mu, 'se')
    with pytest.raises(ValueError) as error:
        gp_tensorflow.get_optimized_kernel(X, Y, prior_mu, 'abc')
    with pytest.raises(AssertionError) as error:
        gp_tensorflow.get_optimized_kernel(X, Y, prior_mu, 'se', is_fixed_noise=1)
    with pytest.raises(AssertionError) as error:
        gp_tensorflow.get_optimized_kernel(X, Y, prior_mu, 'se', num_iters='abc')
    with pytest.raises(AssertionError) as error:
        gp_tensorflow.get_optimized_kernel(X, Y, prior_mu, 'se', debug=1)

    # INFO: tests for set inputs
    with pytest.raises(AssertionError) as error:
        gp_tensorflow.get_optimized_kernel(X_set, Y, prior_mu, 'se')
    with pytest.raises(AssertionError) as error:
        gp_tensorflow.get_optimized_kernel(X, Y, prior_mu, 'set_se')
    with pytest.raises(AssertionError) as error:
        gp_tensorflow.get_optimized_kernel(X_set, Y, prior_mu, 'set_se', debug=1)

    cov_X_X, inv_cov_X_X, hyps = gp_tensorflow.get_optimized_kernel(X, Y, prior_mu, 'eq', num_iters=0)
    print(hyps)

    cov_X_X, inv_cov_X_X, hyps = gp_tensorflow.get_optimized_kernel(X, Y, prior_mu, 'se', num_iters=0)
    print(hyps)

    cov_X_X, inv_cov_X_X, hyps = gp_tensorflow.get_optimized_kernel(X, Y, prior_mu, 'matern32', num_iters=0)
    print(hyps)

    cov_X_X, inv_cov_X_X, hyps = gp_tensorflow.get_optimized_kernel(X, Y, prior_mu, 'matern52', num_iters=0)
    print(hyps)

    with pytest.raises(NotImplementedError) as error:
        cov_X_X, inv_cov_X_X, hyps = gp_tensorflow.get_optimized_kernel(X_set, Y, prior_mu, 'set_se')
        print(hyps)
