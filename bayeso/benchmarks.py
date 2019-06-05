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

INFO_SIXHUMPCAMEL = {
    'dim_fun': 2,
    'bounds': np.array([
        [-3.0, 3.0],
        [-2.0, 2.0],
    ]),
    'global_minimum_X': np.array([
        [0.0898, -0.7126],
        [-0.0898, 0.7126],
    ]),
    'global_minimum_y': -1.0316,
}

INFO_BEALE = {
    'dim_fun': 2,
    'bounds': np.array([
        [-4.5, 4.5],
        [-4.5, 4.5],
    ]),
    'global_minimum_X': np.array([
        [3.0, 0.5],
    ]),
    'global_minimum_y': 0.0,
}

INFO_GOLDSTEINPRICE = {
    'dim_fun': 2,
    'bounds': np.array([
        [-2.0, 2.0],
        [-2.0, 2.0],
    ]),
    'global_minimum_X': np.array([
        [0.0, -1.0],
    ]),
    'global_minimum_y': 3.0,
}

INFO_BOHACHEVSKY = {
    'dim_fun': 2,
    'bounds': np.array([
        [-100.0, 100.0],
        [-100.0, 100.0],
    ]),
    'global_minimum_X': np.array([
        [0.0, 0.0],
    ]),
    'global_minimum_y': 0.0,
}

INFO_HARTMANN6D = {
    'dim_fun': 6,
    'bounds': np.array([
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]),
    'global_minimum_X': np.array([
        [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573],
    ]),
    'global_minimum_y': -3.32237,
}

INFO_HOLDERTABLE = {
    'dim_fun': 2,
    'bounds': np.array([
        [-10.0, 10.0],
        [-10.0, 10.0],
    ]),
    'global_minimum_X': np.array([
        [8.05502, 9.66459],
        [8.05502, -9.66459],
        [-8.05502, 9.66459],
        [-8.05502, -9.66459],
    ]),
    'global_minimum_y': -19.2085,
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

def sixhumpcamel(X):
    assert isinstance(X, np.ndarray)
    assert len(X.shape) == 1 or len(X.shape) == 2
    if len(X.shape) == 1:
        assert X.shape[0] == 2
        X = np.expand_dims(X, axis=0)
    elif len(X.shape) == 2:
        assert X.shape[1] == 2

    Y = (4.0 - 2.1 * X[:, 0]**2 + X[:, 0]**4 / 3.0) * X[:, 0]**2 + X[:, 0] * X[:, 1] + (-4.0 + 4.0 * X[:, 1]**2) * X[:, 1]**2
    return Y

def beale(X):
    assert isinstance(X, np.ndarray)
    assert len(X.shape) == 1 or len(X.shape) == 2
    if len(X.shape) == 1:
        assert X.shape[0] == 2
        X = np.expand_dims(X, axis=0)
    elif len(X.shape) == 2:
        assert X.shape[1] == 2

    Y = (1.5 - X[:, 0] + X[:, 0] * X[:, 1])**2 + (2.25 - X[:, 0] + X[:, 0] * X[:, 1]**2)**2 + (2.625 - X[:, 0] + X[:, 0] * X[:, 1]**3)**2
    return Y

def goldsteinprice(X):
    assert isinstance(X, np.ndarray)
    assert len(X.shape) == 1 or len(X.shape) == 2
    if len(X.shape) == 1:
        assert X.shape[0] == 2
        X = np.expand_dims(X, axis=0)
    elif len(X.shape) == 2:
        assert X.shape[1] == 2

    term_1a = (X[:, 0] + X[:, 1] + 1.0)**2
    term_1b = 19.0 - 14.0 * X[:, 0] + 3.0 * X[:, 0]**2 - 14.0 * X[:, 1] + 6.0 * X[:, 0] * X[:, 1] + 3.0 * X[:, 1]**2
    term_1 = 1.0 + term_1a * term_1b

    term_2a = (2.0 * X[:, 0] - 3.0 * X[:, 1])**2
    term_2b = 18.0 - 32.0 * X[:, 0] + 12.0 * X[:, 0]**2 + 48.0 * X[:, 1] - 36.0 * X[:, 0] * X[:, 1] + 27.0 * X[:, 1]**2
    term_2 = 30.0 + term_2a * term_2b

    Y = term_1 * term_2
    return Y

def bohachevsky(X):
    assert isinstance(X, np.ndarray)
    assert len(X.shape) == 1 or len(X.shape) == 2
    if len(X.shape) == 1:
        assert X.shape[0] == 2
        X = np.expand_dims(X, axis=0)
    elif len(X.shape) == 2:
        assert X.shape[1] == 2

    Y = X[:, 0]**2 + 2.0 * X[:, 1]**2 - 0.3 * np.cos(3.0 * np.pi * X[:, 0]) - 0.4 * np.cos(4.0 * np.pi * X[:, 1]) + 0.7
    return Y

def hartmann6d(X):
    assert isinstance(X, np.ndarray)
    assert len(X.shape) == 1 or len(X.shape) == 2
    if len(X.shape) == 1:
        assert X.shape[0] == 6
        X = np.expand_dims(X, axis=0)
    elif len(X.shape) == 2:
        assert X.shape[1] == 6

    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ])

    Y = np.zeros(X.shape[0])
    for ind_ in range(0, X.shape[0]):
        outer = 0.0
        for i_ in range(0, 4):
            inner = 0.0
            for j_ in range(0, 6):
                inner += A[i_, j_] * (X[ind_, j_] - P[i_, j_])**2
            outer += alpha[i_] * np.exp(-1.0 * inner)
        Y[ind_] = -1.0 * outer
    return Y

def holdertable(X):
    assert isinstance(X, np.ndarray)
    assert len(X.shape) == 1 or len(X.shape) == 2
    if len(X.shape) == 1:
        assert X.shape[0] == 2
        X = np.expand_dims(X, axis=0)
    elif len(X.shape) == 2:
        assert X.shape[1] == 2

    Y = -1.0 * np.abs(np.sin(X[:, 0]) * np.cos(X[:, 1]) * np.exp(np.abs(1.0 - np.sqrt(X[:, 0]**2 + X[:, 1]**2) / np.pi)))
    return Y

