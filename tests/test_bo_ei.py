import numpy as np
import sys
sys.path.append('../bayeso')

import gp
import bo
import acquisition
import utils

def fun_target(X):
    return 4.0 * np.cos(X) + 0.1 * X + 2.0 * np.sin(X) + 0.4 * (X - 0.5)**2

def main():
    X_train = np.array([
        [-5],
        [-1],
        [1],
        [2],
    ])
    model_bo = bo.BO(np.array([[-6., 6.]]))
    X_test = np.linspace(-6, 6, 400)
    X_test = np.reshape(X_test, (400, 1))
    for ind_ in range(1, 20+1):
        Y_train = fun_target(X_train)
        next_x, cov_X_X, inv_cov_X_X, hyps = model_bo.optimize(X_train, fun_target(X_train))
        mu_test, sigma_test = gp.predict_test_(X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps)
        acq_test = acquisition.ei(mu_test.flatten(), sigma_test.flatten(), Y_train.flatten())
        acq_test = np.reshape(acq_test, (acq_test.shape[0], 1))
        X_train = np.vstack((X_train, next_x))
        Y_train = fun_target(X_train)
        utils.plot_bo_step(X_train, Y_train, X_test, fun_target(X_test), mu_test, sigma_test, '../results/gp/', 'test_cos_' + str(ind_), 4)
        utils.plot_bo_step_acq(X_train, Y_train, X_test, fun_target(X_test), mu_test, sigma_test, acq_test, '../results/gp/', 'test_cos_' + str(ind_), 4)

if __name__ == '__main__':
    main()

