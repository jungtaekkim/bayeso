# test_gp
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: May 30, 2018

import numpy as np
import pytest

from bayeso import gp
from bayeso.utils import utils_covariance
from bayeso import benchmarks


TEST_EPSILON = 1e-5

def test_get_prior_mu():
    fun_prior = lambda X: np.expand_dims(np.linalg.norm(X, axis=1), axis=1)
    fun_prior_1d = lambda X: np.linalg.norm(X, axis=1)
    X = np.reshape(np.arange(0, 90), (30, 3))

    with pytest.raises(AssertionError) as error:
        gp.get_prior_mu(1, X)
    with pytest.raises(AssertionError) as error:
        gp.get_prior_mu(fun_prior, 1)
    with pytest.raises(AssertionError) as error:
        gp.get_prior_mu(fun_prior, np.arange(0, 100))
    with pytest.raises(AssertionError) as error:
        gp.get_prior_mu(None, np.arange(0, 100))
    with pytest.raises(AssertionError) as error:
        gp.get_prior_mu(fun_prior_1d, X)

    assert (gp.get_prior_mu(None, X) == np.zeros((X.shape[0], 1))).all()
    assert (gp.get_prior_mu(fun_prior, X) == fun_prior(X)).all()

def test_get_kernels():
    dim_X = 3
    X = np.reshape(np.arange(0, 9), (3, dim_X))
    hyps = utils_covariance.get_hyps('se', dim_X)

    with pytest.raises(AssertionError) as error:
        gp.get_kernels(1, hyps, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_kernels(np.arange(0, 100), hyps, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_kernels(X, 1, 'se')
    with pytest.raises(AssertionError) as error:
        gp.get_kernels(X, hyps, 1)
    with pytest.raises(ValueError) as error:
        gp.get_kernels(X, hyps, 'abc')

    cov_X_X, inv_cov_X_X = gp.get_kernels(X, hyps, 'se')
    print(cov_X_X)
    print(inv_cov_X_X)
    truth_cov_X_X = [
        [1.01001000e+00, 1.37095909e-06, 3.53262857e-24],
        [1.37095909e-06, 1.01001000e+00, 1.37095909e-06],
        [3.53262857e-24, 1.37095909e-06, 1.01001000e+00]
    ]
    truth_inv_cov_X_X = [
        [9.90089207e-01, -1.34391916e-06, 1.82419797e-12],
        [-1.34391916e-06, 9.90089207e-01, -1.34391916e-06],
        [1.82419797e-12, -1.34391916e-06, 9.90089207e-01]
    ]
    assert (cov_X_X - truth_cov_X_X < TEST_EPSILON).all()
    assert (inv_cov_X_X - truth_inv_cov_X_X < TEST_EPSILON).all()

def test_log_ml():
    dim_X = 3
    str_cov = 'se'
    X = np.reshape(np.arange(0, 9), (3, dim_X))
    Y = np.expand_dims(np.arange(3, 10, 3), axis=1)
    dict_hyps = utils_covariance.get_hyps(str_cov, dim_X)
    arr_hyps = utils_covariance.convert_hyps(str_cov, dict_hyps)
    prior_mu_X = np.zeros((3, 1))

    with pytest.raises(AssertionError) as error:
        gp.log_ml(np.arange(0, 3), Y, arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp.log_ml(X, np.arange(0, 3), arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp.log_ml(X, Y, dict_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp.log_ml(X, Y, arr_hyps, 1, prior_mu_X)
    with pytest.raises(ValueError) as error:
        gp.log_ml(X, Y, arr_hyps, 'abc', prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp.log_ml(X, Y, arr_hyps, str_cov, np.arange(0, 3))
    with pytest.raises(AssertionError) as error:
        gp.log_ml(np.reshape(np.arange(0, 12), (4, dim_X)), Y, arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp.log_ml(X, np.expand_dims(np.arange(0, 4), axis=1), arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        gp.log_ml(X, Y, arr_hyps, str_cov, np.expand_dims(np.arange(0, 4), axis=1))

    log_ml = gp.log_ml(X, Y, arr_hyps, str_cov, prior_mu_X)
    print(log_ml)
    truth_log_ml = 65.14727922868668
    assert log_ml - truth_log_ml < TEST_EPSILON

def test_get_optimized_kernels():
    pass
