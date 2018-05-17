import numpy as np
import sys
sys.path.append('../')

from bayeso import gp
from bayeso import utils

def main():
    X_train = np.array([
        [-3.0],
        [-2.0],
        [-1.0],
        [2.0],
        [1.2],
        [1.1],
    ])
    Y_train = np.cos(X_train) + 10.0
    num_test = 200
    X_test = np.linspace(-3, 3, num_test)
    X_test = X_test.reshape((num_test, 1))
    mu, sigma = gp.predict_optimized(X_train, Y_train, X_test)
    utils.plot_gp(X_train, Y_train, X_test, mu, sigma, '../results/gp/', 'test_optimized')

if __name__ == '__main__':
    main()

