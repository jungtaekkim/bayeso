# example_hpo_ridge_regression_ei
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: June 24, 2018

import numpy as np
import sklearn
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model

from bayeso import gp
from bayeso import bo
from bayeso import acquisition
from bayeso.utils import utils_bo
from bayeso.utils import utils_plotting

BOSTON = sklearn.datasets.load_boston()
BOSTON_DATA = BOSTON.data
BOSTON_LABELS = BOSTON.target
DATA_TRAIN, DATA_TEST, LABELS_TRAIN, LABELS_TEST = sklearn.model_selection.train_test_split(BOSTON_DATA, BOSTON_LABELS, test_size=0.3)

def fun_target(X):
    print(X)
    ridge_model = sklearn.linear_model.Ridge(alpha=X[0])
    ridge_model.fit(DATA_TRAIN, LABELS_TRAIN)
    preds = ridge_model.predict(DATA_TEST)
    return sklearn.metrics.mean_squared_error(LABELS_TEST, preds)

def main():
    # (max_depth, n_estimators)
    int_init = 3

    model_bo = bo.BO(np.array([[0.1, 2]]), debug=True)
    list_Y = []
    for _ in range(0, 10):
        X_final, Y_final = utils_bo.optimize_many_with_random_init(model_bo, fun_target, int_init, 10, str_initial_method_bo='uniform', str_initial_method_ao='grid', int_samples_ao=100)
        list_Y.append(Y_final)
    arr_Y = np.array(list_Y)
    arr_Y = np.expand_dims(np.squeeze(arr_Y), axis=0)
    utils_plotting.plot_minimum(arr_Y, ['ridge'], int_init, True, path_save='../results/hpo/', str_postfix='ridge')

if __name__ == '__main__':
    main()

