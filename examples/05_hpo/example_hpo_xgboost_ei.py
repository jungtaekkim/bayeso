# example_hpo_xgboost_ei
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: December 29, 2020

import numpy as np
import os
import xgboost as xgb
import sklearn
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

from bayeso import gp
from bayeso import bo
from bayeso import acquisition
from bayeso.wrappers import wrappers_bo
from bayeso.utils import utils_plotting

DIGITS = sklearn.datasets.load_digits()
DIGITS_DATA = DIGITS.images
DIGITS_DATA = np.reshape(DIGITS_DATA, (DIGITS_DATA.shape[0], DIGITS_DATA.shape[1] * DIGITS_DATA.shape[2]))
DIGITS_LABELS = DIGITS.target
DATA_TRAIN, DATA_TEST, LABELS_TRAIN, LABELS_TEST = sklearn.model_selection.train_test_split(DIGITS_DATA, DIGITS_LABELS, test_size=0.3, stratify=DIGITS_LABELS)
PATH_SAVE = '../figures/hpo/'


def fun_target(X):
    print(X)
    xgb_model = xgb.XGBClassifier(max_depth=int(X[0]), n_estimators=int(X[1]), use_label_encoder=False).fit(DATA_TRAIN, LABELS_TRAIN, eval_metric='mlogloss')
    preds = xgb_model.predict(DATA_TEST)
    return 1.0 - sklearn.metrics.accuracy_score(LABELS_TEST, preds)

def main():
    # (max_depth, n_estimators)
    num_init = 1

    model_bo = bo.BO(np.array([[1, 10], [100, 500]]), debug=True)
    list_Y = []
    list_time = []
    for _ in range(0, 5):
        X_final, Y_final, time_final, _, _ = wrappers_bo.run_single_round(model_bo, fun_target, num_init, 10, str_initial_method_bo='sobol', str_sampling_method_ao='sobol', num_samples_ao=100)
        list_Y.append(Y_final)
        list_time.append(time_final)
    arr_Y = np.array(list_Y)
    arr_Y = np.expand_dims(np.squeeze(arr_Y), axis=0)
    arr_time = np.array(list_time)
    arr_time = np.expand_dims(arr_time, axis=0)

    utils_plotting.plot_minimum_vs_iter(arr_Y, ['xgboost'], num_init, True, path_save=PATH_SAVE, str_postfix='xgboost')
    utils_plotting.plot_minimum_vs_time(arr_time, arr_Y, ['xgboost'], num_init, True, path_save=PATH_SAVE, str_postfix='xgboost')


if __name__ == '__main__':
    if not os.path.isdir(PATH_SAVE):
        os.makedirs(PATH_SAVE)
    main()
