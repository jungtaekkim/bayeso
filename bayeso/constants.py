#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: September 24, 2020
#
"""This file is for declaring various default constants. If you would like to see the details, check out the repository."""

import typing
import numpy as np


JITTER_ACQ = 1e-5
JITTER_COV = 1e-5
JITTER_LOG = 1e-7

STR_OPTIMIZER_METHOD_GP = 'BFGS'
STR_GP_COV = 'matern52'
STR_BO_ACQ = 'ei'
STR_INITIALIZING_METHOD_BO = 'uniform'
STR_OPTIMIZER_METHOD_AO = 'L-BFGS-B'
STR_SAMPLING_METHOD_AO = 'uniform'
STR_MLM_METHOD = 'regular'
STR_MODELSELECTION_METHOD = 'ml'
STR_FRAMEWORK_GP = 'scipy'

NUM_GRIDS_AO = 50
NUM_SAMPLES_AO = 100

MULTIPLIER_ACQ = 10.0
MULTIPLIER_RESPONSE = 10.0

NORMALIZE_RESPONSE = True

GP_NOISE = 1e-2
FIX_GP_NOISE = True
BOUND_UPPER_GP_NOISE = np.inf
RANGE_SIGNAL = [[1e-2, 1e3]]
RANGE_LENGTHSCALES = [[1e-2, 1e3]]
RANGE_NOISE = [[1e-3, 1e1]]

TIME_PAUSE = 2.0
RANGE_SHADE = 1.96

ALLOWED_OPTIMIZER_METHOD_GP = ['BFGS', 'L-BFGS-B', 'Nelder-Mead', 'DIRECT']
ALLOWED_OPTIMIZER_METHOD_BO = ['L-BFGS-B', 'DIRECT', 'CMA-ES']
# INFO: Do not use _ (underscore) in base str_cov.
ALLOWED_GP_COV_BASE = ['eq', 'se', 'matern32', 'matern52']
ALLOWED_GP_COV_SET = ['set_' + str_cov for str_cov in ALLOWED_GP_COV_BASE]
ALLOWED_GP_COV = ALLOWED_GP_COV_BASE + ALLOWED_GP_COV_SET
ALLOWED_BO_ACQ = ['pi', 'ei', 'ucb', 'aei', 'pure_exploit', 'pure_explore']
ALLOWED_INITIALIZING_METHOD_BO = ['sobol', 'uniform', 'latin']
ALLOWED_SAMPLING_METHOD = ALLOWED_INITIALIZING_METHOD_BO + ['grid']
ALLOWED_MLM_METHOD = ['regular', 'converged']
ALLOWED_MODELSELECTION_METHOD = ['ml', 'loocv']
ALLOWED_FRAMEWORK_GP = ['scipy', 'tensorflow', 'gpytorch']

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

TYPE_NONE = type(None)
TYPING_TUPLE_DICT_BOOL = typing.Tuple[dict, bool]
TYPING_TUPLE_ARRAY_BOOL = typing.Tuple[np.ndarray, bool]
TYPING_TUPLE_ARRAY_DICT = typing.Tuple[np.ndarray, dict]
TYPING_TUPLE_ARRAY_FLOAT = typing.Tuple[np.ndarray, float]
TYPING_TUPLE_TWO_ARRAYS = typing.Tuple[np.ndarray, np.ndarray]
TYPING_TUPLE_TWO_ARRAYS_DICT = typing.Tuple[np.ndarray, np.ndarray, dict]
TYPING_TUPLE_THREE_ARRAYS = typing.Tuple[np.ndarray, np.ndarray, np.ndarray]
TYPING_TUPLE_FIVE_ARRAYS = typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]

TYPING_UNION_INT_NONE = typing.Union[int, TYPE_NONE]
TYPING_UNION_INT_FLOAT = typing.Union[int, float]
TYPING_UNION_FLOAT_NONE = typing.Union[float, TYPE_NONE]
TYPING_UNION_FLOAT_TWO_FLOATS = typing.Union[float, typing.Tuple[float, float]]
TYPING_UNION_ARRAY_NONE = typing.Union[np.ndarray, TYPE_NONE]
TYPING_UNION_ARRAY_FLOAT = typing.Union[np.ndarray, float]
TYPING_UNION_CALLABLE_NONE = typing.Union[callable, TYPE_NONE]
TYPING_UNION_STR_NONE = typing.Union[str, TYPE_NONE]
