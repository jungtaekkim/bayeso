#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 8, 2021
#
"""test_wrappers"""

def test_import_bayesian_optimization():
    from bayeso.wrappers import BayesianOptimization
    import bayeso.wrappers

    print(BayesianOptimization)
    print(bayeso.wrappers.BayesianOptimization)
