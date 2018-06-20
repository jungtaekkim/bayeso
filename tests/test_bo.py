# test_bo
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: June 20, 2018

import numpy as np
import pytest

from bayeso import bo
from bayeso import gp
from bayeso.utils import utils_covariance


TEST_EPSILON = 1e-5

def test_predict_optimized():
    np.random.seed(42)
    dim_X = 2
    num_X = 5
    num_X_test = 20
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)
    X_test = np.random.randn(num_X_test, dim_X)
    prior_mu = None
    
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(X, Y, X_test, str_cov='se', prior_mu='abc')
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(X, Y, X_test, str_cov=1, prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(X, Y, 1, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(X, 1, X_test, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(1, Y, X_test, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(np.random.randn(num_X, 1), Y, X_test, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(np.random.randn(10, dim_X), Y, X_test, str_cov='se', prior_mu=prior_mu)
    with pytest.raises(AssertionError) as error:
        gp.predict_optimized(X, np.random.randn(10, 1), X_test, str_cov='se', prior_mu=prior_mu)
