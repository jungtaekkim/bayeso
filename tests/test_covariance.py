# test_covariance
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: Jun 01, 2018

import pytest
import numpy as np

from bayeso import covariance
from bayeso.utils import utils_covariance


def test_cov_se():
    with pytest.raises(AssertionError) as error:
        covariance.cov_se(np.zeros(2), np.zeros(2), 1.0, 0.1)
    with pytest.raises(AssertionError) as error:
        covariance.cov_se(np.zeros(2), np.zeros(3), np.array([1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        covariance.cov_se(np.zeros(3), np.zeros(2), np.array([1.0, 1.0]), 0.1)
    with pytest.raises(AssertionError) as error:
        covariance.cov_se(np.zeros(2), np.zeros(2), np.array([1.0, 1.0]), 1)

    bx = np.array([1.0, 2.0, 0.0])
    bxp = np.array([2.0, 1.0, 1.0])
    cur_hyps = utils_covariance.get_hyps('se', 3)
    cov_ = covariance.cov_se(bx, bxp, cur_hyps['lengthscales'], cur_hyps['signal'])
    assert cov_ == cur_hyps['signal']**2 * np.exp(-0.5 * np.linalg.norm((bx - bxp) / cur_hyps['lengthscales'], ord=2)**2)

def test_cov_main():
    cur_hyps = utils_covariance.get_hyps('se', 3)
    with pytest.raises(AssertionError) as error:
        covariance.cov_main('se', np.zeros((10, 2)), np.zeros((20, 3)), cur_hyps, 0.001)
    with pytest.raises(AssertionError) as error:
        covariance.cov_main('se', np.zeros((10, 3)), np.zeros((20, 2)), cur_hyps, 0.001)
    with pytest.raises(AssertionError) as error:
        covariance.cov_main('se', np.zeros((10, 2)), np.zeros((20, 2)), cur_hyps, 0.001)
    with pytest.raises(AssertionError) as error:
        covariance.cov_main('se', 1.0, np.zeros((20, 3)), cur_hyps, 0.001)
    with pytest.raises(AssertionError) as error:
        covariance.cov_main('se', np.zeros((10, 2)), 1.0, cur_hyps, 0.001)
    with pytest.raises(AssertionError) as error:
        covariance.cov_main(1.0, np.zeros((10, 3)), np.zeros((20, 3)), cur_hyps, 0.001)
    with pytest.raises(AssertionError) as error:
        covariance.cov_main('se', np.zeros((10, 3)), np.zeros((20, 3)), 2.1, 0.001)
    with pytest.raises(AssertionError) as error:
        covariance.cov_main('se', np.zeros((10, 3)), np.zeros((20, 3)), cur_hyps, 1)

    with pytest.raises(ValueError) as error:
        covariance.cov_main('abc', np.zeros((10, 3)), np.zeros((20, 3)), cur_hyps, 0.001)

    cur_hyps.pop('signal', None)
    with pytest.raises(ValueError) as error:
        covariance.cov_main('se', np.zeros((10, 3)), np.zeros((20, 3)), cur_hyps)
   
