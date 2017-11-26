import numpy as np
import sys
sys.path.append('../')

from bayeso import gp
from bayeso import utils

def main():
    X_train = np.array([
        [-3],
        [-1],
        [1],
        [2],
    ])
    Y_train = np.cos(X_train)
    num_test = 200
    X_test = np.linspace(-3, 3, num_test)
    X_test = X_test.reshape((num_test, 1))
    hyps = {
        'signal': 0.5,
        'lengthscales': 0.5,
        'noise': 0.02,
    }
    mu, sigma = gp.predict_test(X_train, Y_train, X_test, hyps)
    utils.plot_gp(X_train, Y_train, X_test, mu, sigma, '../results/gp/', 'test')

if __name__ == '__main__':
    main()

