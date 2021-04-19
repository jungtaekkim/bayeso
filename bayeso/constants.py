#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: March 22, 2021
#
"""This file declares various default constants.
If you would like to see the details, check out
the Python script in the repository directly."""

import typing
import numpy as np


JITTER_ACQ = 1e-5
JITTER_COV = 1e-5
JITTER_LOG = 1e-7

TOLERANCE_DUPLICATED_ACQ = 1e-4

STR_SURROGATE = 'gp'
STR_OPTIMIZER_METHOD_GP = 'BFGS'
STR_OPTIMIZER_METHOD_TP = 'SLSQP'
STR_COV = 'matern52'
STR_BO_ACQ = 'ei'
STR_INITIALIZING_METHOD_BO = 'sobol'
STR_OPTIMIZER_METHOD_AO = 'L-BFGS-B'
STR_SAMPLING_METHOD_AO = 'sobol'
STR_MLM_METHOD = 'regular'
STR_MODELSELECTION_METHOD = 'ml'

NUM_GRIDS_AO = 50
NUM_SAMPLES_AO = 100

MULTIPLIER_ACQ = 10.0
MULTIPLIER_RESPONSE = 10.0

NORMALIZE_RESPONSE = True

USE_ARD = True
GP_NOISE = 1e-2
FIX_GP_NOISE = True
BOUND_UPPER_GP_NOISE = np.inf
RANGE_SIGNAL = [[1e-2, 1e3]]
RANGE_LENGTHSCALES = [[1e-2, 1e3]]
RANGE_NOISE = [[1e-3, 1e1]]
RANGE_DOF = [[2.00001, 200.0]]

TIME_PAUSE = 2.0
RANGE_SHADE = 1.96

ALLOWED_OPTIMIZER_METHOD_GP = [
    'BFGS',
    'L-BFGS-B',
    'Nelder-Mead',
    'SLSQP',
    'SLSQP-Bounded',
]
ALLOWED_OPTIMIZER_METHOD_TP = ['L-BFGS-B', 'SLSQP']
ALLOWED_OPTIMIZER_METHOD_BO = ['L-BFGS-B', 'DIRECT', 'CMA-ES']

# INFO: Do not use _ (underscore) in base str_cov.
ALLOWED_COV_BASE = ['eq', 'se', 'matern32', 'matern52']
ALLOWED_COV_SET = ['set_' + str_cov for str_cov in ALLOWED_COV_BASE]
ALLOWED_COV = ALLOWED_COV_BASE + ALLOWED_COV_SET
ALLOWED_BO_ACQ = ['pi', 'ei', 'ucb', 'aei', 'pure_exploit', 'pure_explore']
ALLOWED_INITIALIZING_METHOD_BO = ['uniform', 'gaussian', 'sobol', 'halton']
ALLOWED_SAMPLING_METHOD = ALLOWED_INITIALIZING_METHOD_BO + ['grid']
ALLOWED_MLM_METHOD = ['regular', 'combined', 'converged']
ALLOWED_MODELSELECTION_METHOD = ['ml', 'loocv']
ALLOWED_SURROGATE = ['gp', 'tp']

KEYS_INFO_BENCHMARK = ['dim_fun', 'bounds', 'global_minimum_X', 'global_minimum_y']

COLORS = np.array([
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
])

MARKERS = np.array([
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
])

TYPE_NONE = type(None)
TYPE_ARR = np.ndarray

TYPING_CALLABLE = typing.Callable
TYPING_LIST = typing.List
TYPING_TUPLE_DICT_BOOL = typing.Tuple[dict, bool]
TYPING_TUPLE_ARRAY_BOOL = typing.Tuple[TYPE_ARR, bool]
TYPING_TUPLE_ARRAY_DICT = typing.Tuple[TYPE_ARR, dict]
TYPING_TUPLE_ARRAY_FLOAT = typing.Tuple[TYPE_ARR, float]
TYPING_TUPLE_TWO_ARRAYS = typing.Tuple[TYPE_ARR, TYPE_ARR]
TYPING_TUPLE_TWO_ARRAYS_DICT = typing.Tuple[TYPE_ARR, TYPE_ARR, dict]
TYPING_TUPLE_THREE_ARRAYS = typing.Tuple[TYPE_ARR, TYPE_ARR, TYPE_ARR]
TYPING_TUPLE_FIVE_ARRAYS = typing.Tuple[TYPE_ARR, TYPE_ARR, TYPE_ARR, TYPE_ARR, TYPE_ARR]
TYPING_TUPLE_FLOAT_THREE_ARRAYS = typing.Tuple[float, TYPE_ARR, TYPE_ARR, TYPE_ARR]
TYPING_TUPLE_FLOAT_ARRAY = typing.Tuple[float, TYPE_ARR]

TYPING_UNION_INT_NONE = typing.Union[int, TYPE_NONE]
TYPING_UNION_INT_FLOAT = typing.Union[int, float]
TYPING_UNION_FLOAT_NONE = typing.Union[float, TYPE_NONE]
TYPING_UNION_FLOAT_TWO_FLOATS = typing.Union[float, TYPING_TUPLE_TWO_ARRAYS]
TYPING_UNION_FLOAT_FA = typing.Union[float, TYPING_TUPLE_FLOAT_ARRAY]
TYPING_UNION_ARRAY_NONE = typing.Union[TYPE_ARR, TYPE_NONE]
TYPING_UNION_ARRAY_FLOAT = typing.Union[TYPE_ARR, float]
TYPING_UNION_CALLABLE_NONE = typing.Union[TYPING_CALLABLE, TYPE_NONE]
TYPING_UNION_STR_NONE = typing.Union[str, TYPE_NONE]
