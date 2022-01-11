#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: September 24, 2020
#
"""These files are for implementing various wrappers."""

from bayeso.wrappers import wrappers_bo_function
from bayeso.wrappers import wrappers_bo_class


run_single_round_with_all_initial_information \
    = wrappers_bo_function.run_single_round_with_all_initial_information
run_single_round_with_initial_inputs = wrappers_bo_function.run_single_round_with_initial_inputs
run_single_round = wrappers_bo_function.run_single_round

BayesianOptimization = wrappers_bo_class.BayesianOptimization
