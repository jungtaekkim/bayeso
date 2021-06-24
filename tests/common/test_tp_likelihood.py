#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: March 22, 2021
#
"""test_tp_likelihood"""

import typing
import pytest
import numpy as np

from bayeso import constants
from bayeso.tp import tp_likelihood as package_target
from bayeso.utils import utils_covariance


TEST_EPSILON = 1e-7

def test_neg_log_ml_typing():
    annos = package_target.neg_log_ml.__annotations__

    assert annos['X_train'] == np.ndarray
    assert annos['Y_train'] == np.ndarray
    assert annos['hyps'] == np.ndarray
    assert annos['str_cov'] == str
    assert annos['prior_mu_train'] == np.ndarray
    assert annos['fix_noise'] == bool
    assert annos['use_gradient'] == bool
    assert annos['debug'] == bool
    assert annos['return'] == typing.Union[float, typing.Tuple[float, np.ndarray]]

def test_neg_log_ml():
    dim_X = 3
    str_cov = 'se'
    X = np.reshape(np.arange(0, 9), (3, dim_X))
    Y = np.expand_dims(np.arange(3, 10, 3), axis=1)
    fix_noise = False
    use_gp = False

    dict_hyps = utils_covariance.get_hyps(str_cov, dim_X, use_gp=use_gp)
    arr_hyps = utils_covariance.convert_hyps(str_cov, dict_hyps, fix_noise=fix_noise, use_gp=use_gp)
    prior_mu_X = np.zeros((3, 1))

    with pytest.raises(AssertionError) as error:
        package_target.neg_log_ml(np.arange(0, 3), Y, arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        package_target.neg_log_ml(X, np.arange(0, 3), arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        package_target.neg_log_ml(X, Y, dict_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        package_target.neg_log_ml(X, Y, arr_hyps, 1, prior_mu_X)
    with pytest.raises(ValueError) as error:
        package_target.neg_log_ml(X, Y, arr_hyps, 'abc', prior_mu_X)
    with pytest.raises(AssertionError) as error:
        package_target.neg_log_ml(X, Y, arr_hyps, str_cov, np.arange(0, 3))
    with pytest.raises(AssertionError) as error:
        package_target.neg_log_ml(np.reshape(np.arange(0, 12), (4, dim_X)), Y, arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        package_target.neg_log_ml(X, np.expand_dims(np.arange(0, 4), axis=1), arr_hyps, str_cov, prior_mu_X)
    with pytest.raises(AssertionError) as error:
        package_target.neg_log_ml(X, Y, arr_hyps, str_cov, np.expand_dims(np.arange(0, 4), axis=1))
    with pytest.raises(AssertionError) as error:
        package_target.neg_log_ml(X, Y, arr_hyps, str_cov, prior_mu_X, fix_noise=1)
    with pytest.raises(AssertionError) as error:
        package_target.neg_log_ml(X, Y, arr_hyps, str_cov, prior_mu_X, debug=1)

    neg_log_ml_ = package_target.neg_log_ml(X, Y, arr_hyps, str_cov, prior_mu_X, fix_noise=fix_noise, use_gradient=False)
    print(neg_log_ml_)
    truth_log_ml_ = 5.634155417555853
    assert np.abs(neg_log_ml_ - truth_log_ml_) < TEST_EPSILON

    neg_log_ml_, neg_grad_log_ml_ = package_target.neg_log_ml(X, Y, arr_hyps, str_cov, prior_mu_X, fix_noise=fix_noise, use_gradient=True)
    print(neg_log_ml_)
    print(neg_grad_log_ml_)
    print(neg_grad_log_ml_[2])
    print(neg_grad_log_ml_[3])
    print(neg_grad_log_ml_[4])

    truth_log_ml_ = 5.634155417555853
    truth_grad_log_ml_ = np.array([
        -1.60446383e-02,
        1.75087448e-01,
        -1.60448396e+00,
        -1.836237221888097e-05,
        -1.836237221888097e-05,
        -1.836237221888097e-05,
    ])
    assert np.abs(neg_log_ml_ - truth_log_ml_) < TEST_EPSILON
    assert np.all(np.abs(neg_grad_log_ml_ - truth_grad_log_ml_) < TEST_EPSILON)

    dict_hyps = utils_covariance.get_hyps(str_cov, dim_X, use_gp=use_gp)
    arr_hyps = utils_covariance.convert_hyps(str_cov, dict_hyps, fix_noise=True, use_gp=use_gp)

    neg_log_ml_, neg_grad_log_ml_ = package_target.neg_log_ml(X, Y, arr_hyps, str_cov, prior_mu_X, fix_noise=True, use_gradient=True)
    print(neg_log_ml_)
    print(neg_grad_log_ml_)
