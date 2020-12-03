# example_hpo_ridge_regression_ei
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: January 06, 2020

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
from bayeso.wrappers import wrappers_bo
from bayeso.utils import utils_plotting

BOSTON = sklearn.datasets.load_boston()
BOSTON_DATA = BOSTON.data
BOSTON_LABELS = BOSTON.target
DATA_TRAIN, DATA_TEST, LABELS_TRAIN, LABELS_TEST = sklearn.model_selection.train_test_split(BOSTON_DATA, BOSTON_LABELS, test_size=0.3)
PATH_SAVE = '../figures/hpo/'


def fun_target(X):
    print(X)
    ridge_model = sklearn.linear_model.Ridge(alpha=X[0])
    ridge_model.fit(DATA_TRAIN, LABELS_TRAIN)
    preds = ridge_model.predict(DATA_TEST)
    mse = sklearn.metrics.mean_squared_error(LABELS_TEST, preds)
    return mse

def main():
    # (max_depth, n_estimators)
    num_init = 5

    model_bo = bo.BO(np.array([[0.1, 2]]), debug=True)
    list_Y = []
    list_time = []
    for _ in range(0, 10):
        X_final, Y_final, time_final, _, _ = wrappers_bo.run_single_round(model_bo, fun_target, num_init, 10, str_initial_method_bo='uniform', str_sampling_method_ao='uniform', num_samples_ao=100)
        list_Y.append(Y_final)
        list_time.append(time_final)
    arr_Y = np.array(list_Y)
    arr_Y = np.expand_dims(np.squeeze(arr_Y), axis=0)
    arr_time = np.array(list_time)
    arr_time = np.expand_dims(arr_time, axis=0)

    utils_plotting.plot_minimum_vs_iter(arr_Y, ['ridge'], num_init, True, path_save=PATH_SAVE, str_postfix='ridge')
    utils_plotting.plot_minimum_vs_time(arr_time, arr_Y, ['ridge'], num_init, True, path_save=PATH_SAVE, str_postfix='ridge')


if __name__ == '__main__':
    if not os.path.isdir(PATH_SAVE):
        os.makedirs(PATH_SAVE)
    main()
