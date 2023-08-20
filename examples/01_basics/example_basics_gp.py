#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 21, 2023
#

import numpy as np
import os

from bayeso.gp import gp
from bayeso.utils import utils_common
from bayeso.utils import utils_plotting


path_save = None

if path_save is not None and not os.path.isdir(path_save):
    os.makedirs(path_save)

np.random.seed(42)

X_train = np.array([
    [-3],
    [-1],
    [1],
    [2],
])
Y_train = np.cos(X_train) + np.random.randn(X_train.shape[0], 1) * 0.1
num_test = 200

X_test = np.linspace(-3, 3, num_test)
X_test = X_test.reshape((num_test, 1))
Y_test = np.cos(X_test)
hyps = {
    'signal': 0.5,
    'lengthscales': 0.5,
    'noise': 0.02,
}
mu, sigma, Sigma = gp.predict_with_hyps(X_train, Y_train, X_test, hyps)

str_postfix = f'cos' if path_save is not None else None

utils_plotting.plot_gp_via_distribution(X_train, Y_train, X_test, mu, sigma, Y_test, path_save=path_save, str_postfix=str_postfix)
