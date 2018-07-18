# example_gp_mml_kernel_matern32
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 18, 2018

import numpy as np
import os

from bayeso import gp
from bayeso.utils import utils_common
from bayeso.utils import utils_plotting


PATH_SAVE = './figures/gp/'
STR_COV = 'matern32'

def main():
    np.random.seed(42)
    X_train = np.array([
        [-3],
        [-1],
        [1],
        [2],
    ])
    Y_train = np.cos(X_train) + np.random.randn(X_train.shape[0], 1) * 0.2
    num_test = 200
    X_test = np.linspace(-3, 3, num_test)
    X_test = X_test.reshape((num_test, 1))
    Y_test_truth = np.cos(X_test)
    hyps = {
        'signal': 0.5,
        'lengthscales': 0.5,
        'noise': 0.02,
    }
    mu, sigma = gp.predict_optimized(X_train, Y_train, X_test, str_cov=STR_COV, debug=True)
    utils_plotting.plot_gp(X_train, Y_train, X_test, mu, sigma, Y_test_truth, path_save=PATH_SAVE, str_postfix='cos_' + STR_COV)

if __name__ == '__main__':
    if not os.path.isdir(PATH_SAVE):
        os.makedirs(PATH_SAVE)
    main()

