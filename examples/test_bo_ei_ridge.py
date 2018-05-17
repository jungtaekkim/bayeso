import numpy as np
import sys
import xgboost as xgb
import sklearn
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model

sys.path.append('../bayeso')
import gp
import bo
import acquisition
import utils

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
    model_bo = bo.BO(np.array([[0.1, 2]]))
    list_Y = []
    for _ in range(0, 10):
        X_final, Y_final = bo.optimize_many_with_random_init(model_bo, fun_target, 3, 10)
        list_Y.append(Y_final)
    utils.plot_minimum([list_Y], ['xgboost'], '../results/aitrics_workshop/', 'ridge', 3, True)

if __name__ == '__main__':
    main()

