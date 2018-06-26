# example_hpo_xgboost_ei
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: June 24, 2018

import numpy as np
import xgboost as xgb
import sklearn
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

from bayeso import gp
from bayeso import bo
from bayeso import acquisition
from bayeso.utils import utils_bo
from bayeso.utils import utils_plot


DIGITS = sklearn.datasets.load_digits()
DIGITS_DATA = DIGITS.images
DIGITS_DATA = np.reshape(DIGITS_DATA, (DIGITS_DATA.shape[0], DIGITS_DATA.shape[1] * DIGITS_DATA.shape[2]))
DIGITS_LABELS = DIGITS.target
DATA_TRAIN, DATA_TEST, LABELS_TRAIN, LABELS_TEST = sklearn.model_selection.train_test_split(DIGITS_DATA, DIGITS_LABELS, test_size=0.3)

def fun_target(X):
    print(X)
    xgb_model = xgb.XGBClassifier(max_depth=int(X[0]), n_estimators=int(X[1])).fit(DATA_TRAIN, LABELS_TRAIN)
    preds = xgb_model.predict(DATA_TEST)
    return 1.0 - sklearn.metrics.accuracy_score(LABELS_TEST, preds)

def main():
    # (max_depth, n_estimators)
    int_init = 3
    model_bo = bo.BO(np.array([[1, 10], [100, 500]]), debug=True)
    list_Y = []
    for _ in range(0, 5):
        X_final, Y_final = utils_bo.optimize_many_with_random_init(model_bo, fun_target, int_init, 50, str_initial_method_bo='uniform', str_initial_method_optimizer='grid', int_samples_ao=100)
        list_Y.append(Y_final)
    arr_Y = np.array(list_Y)
    arr_Y = np.expand_dims(np.squeeze(arr_Y), axis=0)
    utils_plot.plot_minimum(arr_Y, ['xgboost'], int_init, True, path_save='../results/hpo/', str_postfix='xgboost')


if __name__ == '__main__':
    main()
