import numpy as np
import sys
import xgboost as xgb
import sklearn
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

sys.path.append('../bayeso')
import gp
import bo
import acquisition
import utils

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
    model_bo = bo.BO(np.array([[1, 10], [100, 500]]))
    list_Y = []
    for _ in range(0, 5):
        X_final, Y_final = bo.optimize_many_with_random_init(model_bo, fun_target, 3, 50)
        list_Y.append(Y_final)
    utils.plot_minimum([list_Y], ['xgboost'], '../results/aitrics_workshop/', 'xgboost', 3, True)

if __name__ == '__main__':
    main()

