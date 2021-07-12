#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 8, 2021
#
"""test_wrappers"""


def test_import_run_single_round_with_all_initial_information():
    from bayeso.wrappers import run_single_round_with_all_initial_information
    import bayeso.wrappers

    print(run_single_round_with_all_initial_information)
    print(bayeso.wrappers.run_single_round_with_all_initial_information)

def test_import_run_single_round_with_initial_inputs():
    from bayeso.wrappers import run_single_round_with_initial_inputs
    import bayeso.wrappers

    print(run_single_round_with_initial_inputs)
    print(bayeso.wrappers.run_single_round_with_initial_inputs)

def test_import_run_single_round():
    from bayeso.wrappers import run_single_round
    import bayeso.wrappers

    print(run_single_round)
    print(bayeso.wrappers.run_single_round)

def test_import_bayesian_optimization():
    from bayeso.wrappers import BayesianOptimization
    import bayeso.wrappers

    print(BayesianOptimization)
    print(bayeso.wrappers.BayesianOptimization)
