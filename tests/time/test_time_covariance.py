import numpy as np
import time
import pytest

from bayeso import covariance
from bayeso.utils import utils_covariance


num_rounds = 100


def _cov_main_vector(str_cov, num_dim, num_X, num_Xs):
    cur_hyps = utils_covariance.get_hyps(str_cov, num_dim)
    cov_ = covariance.cov_main(str_cov, np.zeros((num_X, num_dim)), np.zeros((num_Xs, num_dim)), cur_hyps, False, jitter=0.001)
    return cov_

def _cov_main_set(str_cov, num_dim, num_inst_X, num_inst_Xs, num_set_X, num_set_Xs):
    cur_hyps = utils_covariance.get_hyps(str_cov, num_dim)
    cov_ = covariance.cov_main(str_cov, np.zeros((num_set_X, num_inst_X, num_dim)), np.zeros((num_set_Xs, num_inst_Xs, num_dim)), cur_hyps, False, jitter=0.001)
    return cov_

@pytest.mark.timeout(50)
def test_cov_se(benchmark):
    cov_ = benchmark.pedantic(_cov_main_vector, args=('se', 3, 1000, 2000), rounds=num_rounds, iterations=1)
    print(cov_)

@pytest.mark.timeout(200)
def test_cov_set_se(benchmark):
    cov_ = benchmark.pedantic(_cov_main_set, args=('set_se', 3, 100, 200, 10, 40), rounds=num_rounds, iterations=1)
    print(cov_)

@pytest.mark.timeout(50)
def test_cov_matern32(benchmark):
    cov_ = benchmark.pedantic(_cov_main_vector, args=('matern32', 3, 1000, 2000), rounds=num_rounds, iterations=1)
    print(cov_)

@pytest.mark.timeout(200)
def test_cov_set_matern32(benchmark):
    cov_ = benchmark.pedantic(_cov_main_set, args=('set_matern32', 3, 100, 200, 10, 40), rounds=num_rounds, iterations=1)
    print(cov_)

@pytest.mark.timeout(50)
def test_cov_matern52(benchmark):
    cov_ = benchmark.pedantic(_cov_main_vector, args=('matern52', 3, 1000, 2000), rounds=num_rounds, iterations=1)
    print(cov_)

@pytest.mark.timeout(200)
def test_cov_set_matern52(benchmark):
    cov_ = benchmark.pedantic(_cov_main_set, args=('set_matern52', 3, 100, 200, 10, 40), rounds=num_rounds, iterations=1)
    print(cov_)
