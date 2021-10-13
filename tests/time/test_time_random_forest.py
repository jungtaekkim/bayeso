import numpy as np
import time
import pytest

from bayeso.trees import trees_random_forest
from bayeso.trees import trees_common
from bayeso.utils import utils_common


num_rounds = 10

np.random.seed(42)
X_train = np.array([
    [-3.0],
    [-1.0],
    [0.0],
    [1.0],
    [2.0],
    [4.0],
])
Y_train = np.cos(X_train) + np.random.randn(X_train.shape[0], 1) * 0.2

num_trees = 100
depth_max = 5
size_min_leaf = 2
num_features = 1

trees = trees_random_forest.get_random_forest(
    X_train, Y_train, num_trees, depth_max, size_min_leaf, num_features
)

def _predict_by_trees(num_test):
    X_test = np.linspace(-5, 5, num_test)
    X_test = X_test.reshape((num_test, 1))

    mu, sigma = trees_common.predict_by_trees(X_test, trees)

    return mu, sigma

@pytest.mark.timeout(500)
def test_predict_by_trees_10(benchmark):
    num_test = 10

    mu, sigma = benchmark.pedantic(_predict_by_trees, args=(num_test, ), rounds=num_rounds, iterations=1)

    assert mu.shape[0] == sigma.shape[0] == num_test

@pytest.mark.timeout(500)
def test_predict_by_trees_100(benchmark):
    num_test = 100

    mu, sigma = benchmark.pedantic(_predict_by_trees, args=(num_test, ), rounds=num_rounds, iterations=1)

    assert mu.shape[0] == sigma.shape[0] == num_test

@pytest.mark.timeout(500)
def test_predict_by_trees_111(benchmark):
    num_test = 111

    mu, sigma = benchmark.pedantic(_predict_by_trees, args=(num_test, ), rounds=num_rounds, iterations=1)

    assert mu.shape[0] == sigma.shape[0] == num_test

@pytest.mark.timeout(500)
def test_predict_by_trees_1000(benchmark):
    num_test = 1000

    mu, sigma = benchmark.pedantic(_predict_by_trees, args=(num_test, ), rounds=num_rounds, iterations=1)

    assert mu.shape[0] == sigma.shape[0] == num_test

@pytest.mark.timeout(500)
def test_predict_by_trees_1111(benchmark):
    num_test = 1111

    mu, sigma = benchmark.pedantic(_predict_by_trees, args=(num_test, ), rounds=num_rounds, iterations=1)

    assert mu.shape[0] == sigma.shape[0] == num_test

@pytest.mark.timeout(500)
def test_predict_by_trees_4444(benchmark):
    num_test = 4444

    mu, sigma = benchmark.pedantic(_predict_by_trees, args=(num_test, ), rounds=num_rounds, iterations=1)

    assert mu.shape[0] == sigma.shape[0] == num_test

@pytest.mark.timeout(500)
def test_predict_by_trees_10000(benchmark):
    num_test = 10000

    mu, sigma = benchmark.pedantic(_predict_by_trees, args=(num_test, ), rounds=num_rounds, iterations=1)

    assert mu.shape[0] == sigma.shape[0] == num_test
