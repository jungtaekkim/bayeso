import bayeso
from bayeso.wrappers import BayesianOptimization
from bayeso_benchmarks import Branin

import numpy as np
import time
import argparse
import wandb


parser = argparse.ArgumentParser()
parser.add_argument('--num_iter', type=int, required=True)
parser.add_argument('--cov', type=str, required=True)
parser.add_argument('--acq', type=str, required=True)
parser.add_argument('--initial', type=str, required=True)
parser.add_argument('--ind_bo', type=int, required=True)

str_fun = 'branin'
str_surrogate = 'gp'
str_optimizer = 'L-BFGS-B'
num_samples_ao = 100
normalize_Y = True
num_init = 1

args = parser.parse_args()
num_iter = args.num_iter
str_cov = args.cov
str_acq = args.acq
str_initial_method = args.initial
ind_bo = args.ind_bo

config = {
    'str_fun': str_fun,
    'str_surrogate': str_surrogate,
    'str_optimizer': str_optimizer,
    'num_samples_ao': num_samples_ao,
    'normalize_Y': normalize_Y,
    'num_iter': num_iter,
    'str_cov': str_cov,
    'str_acq': str_acq,
    'str_initial_method': str_initial_method,
    'num_init': num_init,
    'ind_bo': ind_bo,
}

wandb.init(project='wandb-branin', entity='jungtaekkim', config=config)


def get_bo(str_fun, range_X, fun_target, num_iter, str_surrogate, str_cov, str_acq, str_initial_method_bo, str_optimizer_method_bo, num_samples_ao, normalize_Y):
    model_bo = BayesianOptimization(range_X, fun_target, num_iter,
        str_surrogate=str_surrogate,
        str_cov=str_cov,
        str_acq=str_acq,
        normalize_Y=normalize_Y,
        use_ard=True,
        str_initial_method_bo=str_initial_method_bo,
        str_sampling_method_ao='sobol',
        str_optimizer_method_gp='BFGS',
        str_optimizer_method_tp='SLSQP',
        str_optimizer_method_bo=str_optimizer_method_bo,
        str_mlm_method='regular',
        str_modelselection_method='ml',
        num_samples_ao=num_samples_ao,
        str_exp=str_fun,
        debug=False,
    )

    return model_bo

if __name__ == '__main__':
    seed = 42

    if str_fun == 'branin':
        obj_fun = Branin()
    else:
        raise ValueError()

    fun_target = obj_fun.output
    range_X = obj_fun.get_bounds()

    seed_ = seed * ind_bo + 101

    model_bo = get_bo(
        str_fun, 
        range_X, fun_target, num_iter,
        str_surrogate, str_cov, str_acq,
        str_initial_method, str_optimizer,
        num_samples_ao, normalize_Y
    )

    X = model_bo.model_bo.get_initials(str_initial_method, num_init, seed=seed_)
    Y = fun_target(X)
    times_overall = []
    times_surrogate = []
    times_acq = []

    ind_min = np.argmin(Y)

    wandb.log({
        'bx': X[ind_min],
        'y': Y[ind_min],
        'min_y': Y[ind_min],
        'time_overall': 0.0,
        'time_surrogate': 0.0,
        'time_acq': 0.0,
    })

    for ind_iter in range(0, num_iter):
        print('Iteration {}'.format(ind_iter + 1))

        time_start = time.time()
        next_sample, dict_info = model_bo.optimize_single_iteration(X, Y)

        X = np.concatenate((X, [next_sample]), axis=0)
        Y = np.concatenate((Y, fun_target(next_sample)), axis=0)

        time_surrogate = dict_info['time_surrogate']
        time_acq = dict_info['time_acq']

        time_end = time.time()

        times_overall.append(time_end - time_start)
        times_surrogate.append(time_surrogate)
        times_acq.append(time_acq)

        ind_min = np.argmin(Y)

        wandb.log({
            'bx': X[-1],
            'y': Y[-1],
            'min_y': Y[ind_min],
            'time_overall': times_overall[-1],
            'time_surrogate': times_surrogate[-1],
            'time_acq': times_acq[-1],
        })

    dict_all = {
        'X': X,
        'Y': Y,
        'times_overall': times_overall,
        'times_surrogate': times_surrogate,
        'times_acq': times_acq,
    }
