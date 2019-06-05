# constants
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: April 26, 2019

import numpy as np


JITTER_ACQ = 1e-5
JITTER_COV = 1e-5
JITTER_LOG = 1e-10

STR_OPTIMIZER_METHOD_GP = 'BFGS'
STR_OPTIMIZER_METHOD_BO = 'L-BFGS-B'
STR_GP_COV = 'matern52'
STR_BO_ACQ = 'ei'
STR_BO_INITIALIZATION = 'uniform'
STR_AO_INITIALIZATION = 'uniform'
STR_MLM_METHOD = 'regular'
STR_MODELSELECTION_METHOD = 'ml'

NUM_BO_GRID = 50
NUM_BO_RANDOM = 1000
NUM_ACQ_SAMPLES = 100

MULTIPLIER_ACQ = 10.0
MULTIPLIER_RESPONSE = 10.0

IS_NORMALIZED_RESPONSE = True

GP_NOISE = 1e-2
IS_FIXED_GP_NOISE = True
BOUND_UPPER_GP_NOISE = np.inf
RANGE_SIGNAL = [[1e-2, 1e3]]
RANGE_LENGTHSCALES = [[1e-2, 1e3]]
RANGE_NOISE = [[1e-3, 1e1]]

TIME_PAUSE = 2.0
RANGE_SHADE = 1.96

ALLOWED_OPTIMIZER_METHOD_GP = ['BFGS', 'L-BFGS-B', 'DIRECT', 'CMA-ES']
ALLOWED_OPTIMIZER_METHOD_BO = ['L-BFGS-B', 'DIRECT', 'CMA-ES']
# INFO: Do not use _ (underscore) in base str_cov.
ALLOWED_GP_COV_BASE = ['se', 'matern32', 'matern52']
ALLOWED_GP_COV_SET = ['set_' + str_cov for str_cov in ALLOWED_GP_COV_BASE]
ALLOWED_GP_COV = ALLOWED_GP_COV_BASE + ALLOWED_GP_COV_SET
ALLOWED_BO_ACQ = ['pi', 'ei', 'ucb', 'aei', 'pure_exploit', 'pure_explore']
ALLOWED_INITIALIZATIONS_BO = ['sobol', 'uniform', 'latin']
ALLOWED_INITIALIZATIONS_AO = ALLOWED_INITIALIZATIONS_BO + ['grid']
ALLOWED_MLM_METHOD = ['regular', 'converged']
ALLOWED_MODELSELECTION_METHOD = ['ml', 'loocv']

KEYS_INFO_BENCHMARK = ['dim_fun', 'bounds', 'global_minimum_X', 'global_minimum_y']

COLORS = [
    'red',
    'green',
    'blue',
    'orange',
    'olive',
    'purple',
    'darkred',
    'limegreen',
    'deepskyblue',
    'lightsalmon',
    'aquamarine',
    'navy',
    'rosybrown',
    'darkkhaki',
    'darkslategray',
]

MARKERS = [
    '.',
    'x',
    '*',
    '+',
    '^',
    'v',
    '<',
    '>',
    'd',
    ',',
    '8',
    'h',
    '1',
    '2',
    '3',
]
