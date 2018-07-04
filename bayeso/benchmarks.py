# benchmarks
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 03, 2018

import numpy as np

INFO_BRANIN = {
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

