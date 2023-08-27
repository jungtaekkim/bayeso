About BayesO
############

Simple, but essential Bayesian optimization package.
It is designed to run advanced Bayesian optimization with implementation-specific and application-specific modifications as well as to run Bayesian optimization in various applications simply.
This package contains the codes for Gaussian process regression and Gaussian process-based Bayesian optimization.
Some famous benchmark and custom benchmark functions for Bayesian optimization are included in `bayeso-benchmarks <https://github.com/jungtaekkim/bayeso-benchmarks>`_, which can be used to test the Bayesian optimization strategy. If you are interested in this package, please refer to that repository.

Supported Python Version
========================

We test our package in the following versions.

- Python 3.7
- Python 3.8
- Python 3.9
- Python 3.10
- Python 3.11

Examples
========

We provide a list of examples.

.. code-block:: text

    examples/
    ├── 01_basics
    │   ├── example_basics_bo.py: a basic example of Bayesian optimization
    │   └── example_basics_gp.py: a basic example of Gaussian processes
    ├── 02_surrogates
    │   ├── example_generic_trees.py: an example of modeling generic trees
    │   ├── example_gp_mml_comparisons.py: an example of Gaussian processes with different optimization methods for marginal likelihood maximization
    │   ├── example_gp_mml_kernels.py: an example of Gaussian processes with different kernels
    │   ├── example_gp_mml_many_points.py: an example of Gaussian processes with many data points
    │   ├── example_gp_mml_y_scales.py: an example of Gaussian processes with different scales of function evaluations
    │   ├── example_gp_priors.py: an example of Gaussian processes with differnt prior functions
    │   ├── example_random_forest.py: an example of modeling random forests
    │   └── example_tp_mml_kernels.py: an example of student-$t$ processes with different kernels
    ├── 03_bo
    │   ├── example_bo_aei.py: an example of Bayesian optimization with augmented expected improvement
    │   ├── example_bo_ei.py: an example of Bayesian optimization with expected improvement
    │   ├── example_bo_pi.py: an example of Bayesian optimization with the probability of improvement
    │   ├── example_bo_pure_exploit.py: an example of Bayesian optimization with pure exploitation
    │   ├── example_bo_pure_explore.py: an example of Bayesian optimization with pure exploration
    │   └── example_bo_ucb.py: an example of Bayesian optimization with Gaussain process upper confidence bound
    ├── 04_bo_with_surrogates
    │   ├── example_bo_w_gp.py: an example of Bayesian optimization with Gaussian process surrogates
    │   └── example_bo_w_tp.py: an example of Bayesian optimization with student-$t$ process surrogates
    ├── 05_benchmarks
    │   ├── example_benchmarks_ackley_bo_ei.py: an example of Bayesian optimization for the Ackley function
    │   ├── example_benchmarks_bohachevsky_bo_ei.py: an example of Bayesian optimization for the Bohachevsky function
    │   ├── example_benchmarks_branin_bo_ei.py: an example of Bayesian optimization for the Branin function
    │   ├── example_benchmarks_branin_gp.py: an example of Gaussian processes for the Branin function
    │   ├── example_benchmarks_branin_ts.py: an example of Thompson sampling for the Branin function
    │   └── example_benchmarks_hartmann6d_bo_ei.py: an example of Bayesian optimization for the Hartmann 6D function
    ├── 06_hpo
    │   ├── example_hpo_ridge_regression_ei.py: an example of hyperparameter optimization for ridge regression
    │   └── example_hpo_xgboost_ei.py: an example of hyperparameter optimization for XGBoost
    ├── 07_wandb
    │   ├── script_wandb_branin.sh: a script for optimizing the Branin function with WandB
    │   └── wandb_branin.py: an example for optimizing the Branin function with WandB
    └── 99_notebooks
        ├── example_bo_branin.ipynb: a notebook of Bayesian optimization for the Branin function
        ├── example_gp.ipynb: a notebook of Gaussian processes
        ├── example_hpo_xgboost.ipynb: a notebook of hyperparameter optimization for XGBoost
        ├── example_tp.ipynb: a notebook of student-$t$ processes
        └── example_ts_gp_prior.ipynb: a notebook of Thompson sampling

Tests
=====

We provide a list of tests.

.. code-block:: text

    tests/
    ├── common
    │   ├── test_acquisition.py: tests for acquisition.py
    │   ├── test_bo_bo_w_gp.py: tests for bo_w_gp.py
    │   ├── test_bo_bo_w_tp.py: tests for bo_w_tp.py
    │   ├── test_bo_bo_w_trees.py: tests for bo_w_trees.py
    │   ├── test_bo.py: tests for the bo subpackage
    │   ├── test_covariance.py: tests for covariance.py
    │   ├── test_gp_gp.py: tests for gp.py
    │   ├── test_gp_kernel.py: tests for gp_kernel.py
    │   ├── test_gp_likelihood.py: tests for gp_likelihood.py
    │   ├── test_import.py: tests for importing BayesO
    │   ├── test_thompson_sampling.py: tests for thompson_sampling.py
    │   ├── test_tp_kernel.py: tests for tp_kernel.py
    │   ├── test_tp_likelihood.py: tests for tp_likelihood.py
    │   ├── test_tp_tp.py: tests for tp.py
    │   ├── test_trees.py: tests for the trees subpackage
    │   ├── test_trees_trees_common.py: tests for trees_common.py
    │   ├── test_trees_trees_generic_trees.py: tests for trees_generic_trees.py
    │   ├── test_trees_trees_random_forest.py: tests for trees_random_forest.py
    │   ├── test_utils_bo.py: tests for utils_bo.py
    │   ├── test_utils_common.py: tests for utils_common.py
    │   ├── test_utils_covariance.py: tests for utils_covariance.py
    │   ├── test_utils_gp.py: tests for utils_gp.py
    │   ├── test_utils_logger.py: tests for utils_logger.py
    │   ├── test_utils_plotting.py: tests for utils_plotting.py
    │   ├── test_version.py: tests for checking the version of BayesO
    │   ├── test_wrappers_bo_class.py: tests for wrappers_bo_class.py
    │   ├── test_wrappers_bo_function.py: tests for wrappers_bo_function.py
    │   └── test_wrappers.py: tests for the wrappers subpackage
    ├── integration_test.py: end-to-end tests for BayesO
    └── time
        ├── test_time_bo_load.py: time tests for loading the BO class
        ├── test_time_bo_optimize.py: time tests for running Bayesian optimization
        ├── test_time_covariance.py: time tests for calculating covariance functions
        └── test_time_random_forest.py: time tests for modeling random forests

Related Package for Benchmark Functions
=======================================

The related package **bayeso-benchmarks**, which contains some famous benchmark functions and custom benchmark functions is hosted in `this repository <https://github.com/jungtaekkim/bayeso-benchmarks>`_. It can be used to test a Bayesian optimization strategy.

The details of benchmark functions implemented in **bayeso-benchmarks** are described in `these notes <https://jungtaek.github.io/notes/benchmarks_bo.pdf>`_.

Citation
========

.. code-block:: latex

    @misc{KimJ2017bayeso,
        author={Kim, Jungtaek and Choi, Seungjin},
        title={{BayesO}: A {Bayesian} optimization framework in {Python}},
        howpublished={\url{https://bayeso.org}},
        year={2017}
    }

License
=======

`MIT License <https://github.com/jungtaekkim/bayeso/blob/main/LICENSE>`_
