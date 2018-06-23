# constants
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: June 23, 2018

JITTER_ACQ = 1e-5
JITTER_COV = 1e-5

STR_OPTIMIZER_METHOD_BO = 'L-BFGS-B'
STR_OPTIMIZER_METHOD_GP = 'L-BFGS-B'
STR_GP_COV = 'se'
STR_BO_ACQ = 'ei'
STR_BO_INITIALIZATION = 'sobol'
STR_OPTIMIZER_INITIALIZATION = 'sobol'

NUM_BO_GRID = 50
NUM_BO_RANDOM = 1000
NUM_ACQ_SAMPLES = 50

MULTIPLIER_ACQ = 10.0

TIME_PAUSE = 2.0
RANGE_SHADE = 1.96

ALLOWED_INITIALIZATIONS_BO = ['sobol', 'uniform', 'latin']
ALLOWED_INITIALIZATIONS_OPTIMIZER = ALLOWED_INITIALIZATIONS_BO + ['grid']

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
