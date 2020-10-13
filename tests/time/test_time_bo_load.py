import numpy as np
import time
import pytest

from bayeso import bo


num_rounds = 100

ranges = np.array([
    [0.0, 1.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [0.0, 1.0],
])


def _get_model_bo(str_optimizer_method_bo):
    model_bo = bo.BO(ranges, str_cov='matern52', str_acq='ei', str_optimizer_method_bo=str_optimizer_method_bo)
    return model_bo

@pytest.mark.timeout(10)
def test_get_model_bo_direct(benchmark):
    model_bo = benchmark.pedantic(_get_model_bo, args=('DIRECT', ), rounds=num_rounds, iterations=1)
    print(model_bo)

@pytest.mark.timeout(10)
def test_get_model_bo_lbfgsb(benchmark):
    model_bo = benchmark.pedantic(_get_model_bo, args=('L-BFGS-B', ), rounds=num_rounds, iterations=1)
    print(model_bo)

@pytest.mark.timeout(10)
def test_get_model_bo_cmaes(benchmark):
    model_bo = benchmark.pedantic(_get_model_bo, args=('CMA-ES', ), rounds=num_rounds, iterations=1)
    print(model_bo)
