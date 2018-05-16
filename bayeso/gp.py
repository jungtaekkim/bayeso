import numpy as np
import scipy 
import scipy.optimize

def cov_se(x, xp, lengthscales, signal):
    return signal**2 * np.exp(-0.5 * np.linalg.norm((x - xp)/lengthscales)**2)

def cov_main(str_cov, X, Xs, hyps, jitter=1e-5):
    num_X = X.shape[0]
    num_d_X = X.shape[1]
    num_Xs = Xs.shape[0]
    num_d_Xs = Xs.shape[1]
    if num_d_X != num_d_Xs:
        print('ERROR: matrix dimensions: ', num_d_X, num_d_Xs)
        raise ValueError('matrix dimensions are different.')

    cov_ = np.zeros((num_X, num_Xs))
    if num_X == num_Xs:
        cov_ += np.eye(num_X) * jitter
    if str_cov == 'se':
        if hyps.get('lengthscales') is None or hyps.get('signal') is None:
            raise ValueError('hyperparameters are insufficient.')
        for ind_X in range(0, num_X):
            for ind_Xs in range(0, num_Xs):
                cov_[ind_X, ind_Xs] += cov_se(X[ind_X], Xs[ind_Xs], hyps['lengthscales'], hyps['signal'])
    else:
        raise ValueError('kernel is inappropriate.')
    return cov_

def get_hyps(str_cov, num_dim):
    hyps = dict()
    hyps['noise'] = 0.1
    if str_cov == 'se':
        hyps['signal'] = 1.0
        hyps['lengthscales'] = np.zeros(num_dim) + 1.0
    else:
        raise ValueError('kernel is inappropriate.')
    return hyps

def convert_hyps(str_cov, hyps):
    list_hyps = []
    list_hyps.append(hyps['noise'])
    if str_cov == 'se':
        list_hyps.append(hyps['signal'])
        for elem_lengthscale in hyps['lengthscales']:
            list_hyps.append(elem_lengthscale)
    else:
        raise ValueError('kernel is inappropriate.')
    return np.array(list_hyps)

def restore_hyps(str_cov, hyps):
    hyps = hyps.flatten()
    dict_hyps = dict()
    dict_hyps['noise'] = hyps[0]
    if str_cov == 'se':
        dict_hyps['signal'] = hyps[1]
        list_lengthscales = []
        for ind_elem in range(2, len(hyps)):
            list_lengthscales.append(hyps[ind_elem])
        dict_hyps['lengthscales'] = np.array(list_lengthscales)
    else:
        raise ValueError('kernel is inappropriate.')
    return dict_hyps

def get_prior_mu(prior_mu, X):
    if prior_mu is None:
        prior_mu_X = np.zeros((X.shape[0], 1))
    else:
        prior_mu_X = prior_mu(X)
    return prior_mu_X

def get_kernels(X_train, hyps, str_cov):
    cov_X_X = cov_main(str_cov, X_train, X_train, hyps) + hyps['noise']**2 * np.eye(X_train.shape[0])
    cov_X_X = (cov_X_X + cov_X_X.T) / 2.0
    inv_cov_X_X = np.linalg.inv(cov_X_X)
    return cov_X_X, inv_cov_X_X

def log_ml(X_train, Y_train, hyps, str_cov, prior_mu_train):
    hyps = restore_hyps(str_cov, hyps)
    cov_X_X, inv_cov_X_X = get_kernels(X_train, hyps, str_cov)
    new_Y_train = Y_train - prior_mu_train

    first_term = -0.5 * np.dot(np.dot(new_Y_train.T, inv_cov_X_X), new_Y_train)
    second_term = -0.5 * np.log(np.linalg.det(cov_X_X))
    third_term = -float(X_train.shape[1]) / 2.0 * np.log(2.0 * np.pi)
    return first_term + second_term + third_term

def get_optimized_kernels(X_train, Y_train, prior_mu, str_cov, verbose=False):
    prior_mu_train = get_prior_mu(prior_mu, X_train)
    num_dim = X_train.shape[1]
    neg_log_ml = lambda hyps: -1.0 * log_ml(X_train, Y_train, hyps, str_cov, prior_mu_train)
    result_optimized = scipy.optimize.minimize(neg_log_ml, convert_hyps(str_cov, get_hyps(str_cov, num_dim)), method='L-BFGS-B')
    hyps = restore_hyps(str_cov, result_optimized.x)
    if verbose:
        print('INFORM: optimized result for gpr ', hyps)
    cov_X_X, inv_cov_X_X = get_kernels(X_train, hyps, str_cov)
    return cov_X_X, inv_cov_X_X, hyps

def predict_test_(X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov='se', prior_mu=None):
    prior_mu_train = get_prior_mu(prior_mu, X_train)
    prior_mu_test = get_prior_mu(prior_mu, X_test)
    cov_X_Xs = cov_main(str_cov, X_train, X_test, hyps)
    cov_Xs_Xs = cov_main(str_cov, X_test, X_test, hyps) + hyps['noise']**2 * np.eye(X_test.shape[0])
    cov_Xs_Xs = (cov_Xs_Xs + cov_Xs_Xs.T) / 2.0

    mu_Xs = np.dot(np.dot(cov_X_Xs.T, inv_cov_X_X), Y_train - prior_mu_train) + prior_mu_test
    Sigma_Xs = cov_Xs_Xs - np.dot(np.dot(cov_X_Xs.T, inv_cov_X_X), cov_X_Xs)
    return mu_Xs, np.sqrt(np.maximum(np.diag(Sigma_Xs), 0.0))

def predict_test(X_train, Y_train, X_test, hyps, str_cov='se', prior_mu=None):
    cov_X_X, inv_cov_X_X = get_kernels(X_train, hyps, str_cov)
    mu_Xs, sigma_Xs = predict_test_(X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov, prior_mu)
    return mu_Xs, sigma_Xs

def predict_optimized(X_train, Y_train, X_test, str_cov='se', prior_mu=None, verbose=False):
    cov_X_X, inv_cov_X_X, hyps = get_optimized_kernels(X_train, Y_train, prior_mu, str_cov, verbose=verbose)
    mu_Xs, sigma_Xs = predict_test_(X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps, str_cov, prior_mu)
    return mu_Xs, sigma_Xs



