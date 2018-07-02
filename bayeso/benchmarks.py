# benchmarks
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 03, 2018

import numpy as np

BOUND_BRANIN = np.array([[-5, 10], [0, 15]])


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
    assert len(X.shape) == 2
    assert X.shape[1] == 2

    Y = a * (X[:, 1] - b * X[:, 0]**2 + c * X[:, 0] - r)**2 + s * (1 - t) * np.cos(X[:, 0]) + s
    Y = np.expand_dims(Y, axis=1)
    return Y

