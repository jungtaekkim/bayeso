import numpy as np
import time
import pytest

from bayeso import bo


num_rounds = 10

ranges = np.array([
    [0.0, 1.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [0.0, 1.0],
])

num_data = 10
dim_data = ranges.shape[0]
seed = 42

X_train = np.random.RandomState(seed).randn(num_data, dim_data)
Y_train = np.random.RandomState(seed).randn(num_data, 1)


def _acquire_next_point(str_optimizer_method_bo, num_samples):
    model_bo = bo.BO(ranges, str_cov='matern52', str_acq='ei', str_optimizer_method_bo=str_optimizer_method_bo)
    next_point, _ = model_bo.optimize(X_train, Y_train, num_samples=num_samples)

    return next_point

@pytest.mark.timeout(500)
def test_acquire_next_point_direct(benchmark):
    next_point = benchmark.pedantic(_acquire_next_point, args=('DIRECT', 100), rounds=num_rounds, iterations=1)
    print(next_point)

@pytest.mark.timeout(10)
def test_acquire_next_point_cmaes(benchmark):
    next_point = benchmark.pedantic(_acquire_next_point, args=('CMA-ES', 100), rounds=num_rounds, iterations=1)
    print(next_point)

@pytest.mark.timeout(1)
def test_acquire_next_point_lbfgsb_1(benchmark):
    next_point = benchmark.pedantic(_acquire_next_point, args=('L-BFGS-B', 1), rounds=num_rounds, iterations=1)
    print(next_point)

@pytest.mark.timeout(2)
def test_acquire_next_point_lbfgsb_10(benchmark):
    next_point = benchmark.pedantic(_acquire_next_point, args=('L-BFGS-B', 10), rounds=num_rounds, iterations=1)
    print(next_point)

@pytest.mark.timeout(15)
def test_acquire_next_point_lbfgsb_100(benchmark):
    next_point = benchmark.pedantic(_acquire_next_point, args=('L-BFGS-B', 100), rounds=num_rounds, iterations=1)
    print(next_point)

@pytest.mark.timeout(150)
def test_acquire_next_point_lbfgsb_1000(benchmark):
    next_point = benchmark.pedantic(_acquire_next_point, args=('L-BFGS-B', 1000), rounds=num_rounds, iterations=1)
    print(next_point)
