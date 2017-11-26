import numpy as np

BOUND_BRANIN = np.array([[-5, 10], [0, 15]])

def branin(X):
    # TODO: raise error for checking whether or not 2 dimensions
    # TODO: raise error for more than third-order tensor
    if len(X.shape) == 1:
        X = np.reshape(X, (1, X.shape[0]))

    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5 / np.pi
    r = 6.0
    s = 10.0
    t = 1 / (8 * np.pi)

    Y = a * (X[:, 1] - b * X[:, 0]**2 + c * X[:, 0] - r)**2 + s * (1 - t) * np.cos(X[:, 0]) + s
    Y = np.reshape(Y, (Y.shape[0], 1))
    return Y



