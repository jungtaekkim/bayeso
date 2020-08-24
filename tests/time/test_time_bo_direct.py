import numpy as np
import time
import pytest

from bayeso import bo


str_optimizer_method_bo = 'DIRECT'
arr_range = np.array([
    [0.0, 1.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [0.0, 1.0],
])

def _get_model_bo(str_optimizer_method_bo):
    model_bo = bo.BO(arr_range, str_cov='matern52', str_acq='ei', str_optimizer_method_bo=str_optimizer_method_bo)
    return model_bo

@pytest.mark.timeout(10)
def test_get_model_bo(benchmark):
    model_bo = benchmark.pedantic(_get_model_bo, args=(str_optimizer_method_bo, ), rounds=100, iterations=1)
    print(model_bo)

