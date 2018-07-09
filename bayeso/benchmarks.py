# benchmarks
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 09, 2018

import numpy as np

INFO_BRANIN = {
    'dim_fun': 2,
    'bounds': np.array([
        [-5, 10],
        [0, 15]
    ]),
    'global_minimum_X': np.array([
        [-np.pi, 12.275],
        [np.pi, 2.275],
        [9.42478, 2.475],
    ]),
    'global_minimum_y': 0.397887,
}

INFO_ACKLEY = {
    'dim_fun': np.inf,
    'bounds': np.array([
        [-32.768, 32.768],
    ]),
    'global_minimum_X': np.array([
        [0.0],
    ]),
    'global_minimum_y': 0.0,
}

INFO_EGGHOLDER = {
    'dim_fun': 2,
    'bounds': np.array([
        [-512.0, 512.0],
        [-512.0, 512.0],
    ]),
    'global_minimum_X': np.array([
        [512.0, 404.2319],
    ]),
    'global_minimum_y': -959.6407,
}

def branin(X,
    a=1.0,
    b=5.1 / (4.0 * np.pi**2),
    c=5 / np.pi,
    r=6.0,
    s=10.0,
    t=1 / (8 * np.pi),
):
    assert isinstance(X, np.ndarray)
    assert isinstance(a, float)
    assert isinstance(b, float)
    assert isinstance(c, float)
    assert isinstance(r, float)
    assert isinstance(s, float)
    assert isinstance(t, float)
    assert len(X.shape) == 1 or len(X.shape) == 2
    if len(X.shape) == 1:
        assert X.shape[0] == 2
        X = np.expand_dims(X, axis=0)
    elif len(X.shape) == 2:
        assert X.shape[1] == 2

    Y = a * (X[:, 1] - b * X[:, 0]**2 + c * X[:, 0] - r)**2 + s * (1 - t) * np.cos(X[:, 0]) + s
    return Y

def ackley(X,
    a=20.0,
    b=0.2,
    c=2.0*np.pi,
):
    assert isinstance(X, np.ndarray)
    assert isinstance(a, float)
    assert isinstance(b, float)
    assert isinstance(c, float)
    assert len(X.shape) == 1 or len(X.shape) == 2
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis=0)

    dim_X = X.shape[1]
    Y = -a * np.exp(-b * np.linalg.norm(X, ord=2, axis=1) * np.sqrt(1.0 / dim_X)) - np.exp(1.0/dim_X * np.sum(np.cos(c * X), axis=1)) + a + np.exp(1.0)
    return Y

def eggholder(X):
    assert isinstance(X, np.ndarray)
    assert len(X.shape) == 1 or len(X.shape) == 2
    if len(X.shape) == 1:
        assert X.shape[0] == 2
        X = np.expand_dims(X, axis=0)
    elif len(X.shape) == 2:
        assert X.shape[1] == 2

    Y = -1.0 * (X[:, 1] + 47.0) * np.sin(np.sqrt(np.abs(X[:, 1] + X[:, 0] / 2.0 + 47.0))) - X[:, 0] * np.sin(np.sqrt(np.abs(X[:, 0] - (X[:, 1] + 47.0))))
    return Y

