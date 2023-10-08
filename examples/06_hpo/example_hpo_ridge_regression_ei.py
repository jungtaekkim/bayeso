#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 17, 2023
#

import numpy as np
import os
import sklearn
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model

from bayeso import gp
from bayeso import bo
from bayeso import acquisition
from bayeso.wrappers import wrappers_bo_function
from bayeso.utils import utils_plotting


HOUSING = sklearn.datasets.fetch_california_housing()
HOUSING_DATA = HOUSING.data
HOUSING_LABELS = HOUSING.target
DATA_TRAIN, DATA_TEST, LABELS_TRAIN, LABELS_TEST = sklearn.model_selection.train_test_split(HOUSING_DATA, HOUSING_LABELS, test_size=0.3)

def fun_target(X):
    print(X)
    ridge_model = sklearn.linear_model.Ridge(alpha=X[0])
    ridge_model.fit(DATA_TRAIN, LABELS_TRAIN)
    preds = ridge_model.predict(DATA_TEST)
    mse = sklearn.metrics.mean_squared_error(LABELS_TEST, preds)
    return mse

path_save = None

if path_save is not None and not os.path.isdir(path_save):
    os.makedirs(path_save)

# (alpha, )
num_init = 1

model_bo = bo.BO(np.array([[0.1, 2]]), debug=True)

list_Y = []
list_time = []

for _ in range(0, 10):
    X_final, Y_final, time_final, _, _ = wrappers_bo_function.run_single_round(model_bo, fun_target, num_init, 10, str_initial_method_bo='sobol', str_sampling_method_ao='sobol', num_samples_ao=100)

    list_Y.append(Y_final)
    list_time.append(time_final)

arr_Y = np.array(list_Y)
arr_Y = np.expand_dims(np.squeeze(arr_Y), axis=0)
arr_time = np.array(list_time)
arr_time = np.expand_dims(arr_time, axis=0)

utils_plotting.plot_minimum_vs_iter(arr_Y, ['ridge'], num_init, True, path_save=path_save, str_postfix='ridge')
utils_plotting.plot_minimum_vs_time(arr_time, arr_Y, ['ridge'], num_init, True, path_save=path_save, str_postfix='ridge')
