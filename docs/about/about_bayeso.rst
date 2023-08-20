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

.. code-block:: bash

    examples/
    ├── 01_basics
    │   ├── example_basics_bo.py
    │   └── example_basics_gp.py
    ├── 02_surrogates
    │   ├── example_generic_trees.py
    │   ├── example_gp_mml_comparisons.py
    │   ├── example_gp_mml_kernels.py
    │   ├── example_gp_mml_many_points.py
    │   ├── example_gp_mml_y_scales.py
    │   ├── example_gp_priors.py
    │   ├── example_random_forest.py
    │   └── example_tp_mml_kernels.py
    ├── 03_bo
    │   ├── example_bo_aei.py
    │   ├── example_bo_ei.py
    │   ├── example_bo_pi.py
    │   ├── example_bo_pure_exploit.py
    │   ├── example_bo_pure_explore.py
    │   └── example_bo_ucb.py
    ├── 04_bo_with_surrogates
    │   ├── example_bo_w_gp.py
    │   └── example_bo_w_tp.py
    ├── 05_benchmarks
    │   ├── example_benchmarks_ackley_bo_ei.py
    │   ├── example_benchmarks_bohachevsky_bo_ei.py
    │   ├── example_benchmarks_branin_bo_ei.py
    │   ├── example_benchmarks_branin_gp.py
    │   ├── example_benchmarks_branin_ts.py
    │   └── example_benchmarks_hartmann6d_bo_ei.py
    ├── 06_hpo
    │   ├── example_hpo_ridge_regression_ei.py
    │   └── example_hpo_xgboost_ei.py
    ├── 07_wandb
    │   ├── script_wandb_branin.sh
    │   └── wandb_branin.py
    └── 99_notebooks
        ├── example_bo_branin.ipynb
        ├── example_gp.ipynb
        ├── example_hpo_xgboost.ipynb
        ├── example_tp.ipynb
        └── example_ts_gp_prior.ipynb

Tests
=====

We provide a list of tests.

.. code-block:: bash

    tests/
    ├── common
    │   ├── test_acquisition.py
    │   ├── test_bo_bo_w_gp.py
    │   ├── test_bo_bo_w_tp.py
    │   ├── test_bo_bo_w_trees.py
    │   ├── test_bo.py
    │   ├── test_covariance.py
    │   ├── test_gp_gp.py
    │   ├── test_gp_kernel.py
    │   ├── test_gp_likelihood.py
    │   ├── test_import.py
    │   ├── test_thompson_sampling.py
    │   ├── test_tp_kernel.py
    │   ├── test_tp_likelihood.py
    │   ├── test_tp_tp.py
    │   ├── test_trees.py
    │   ├── test_trees_trees_common.py
    │   ├── test_trees_trees_generic_trees.py
    │   ├── test_trees_trees_random_forest.py
    │   ├── test_utils_bo.py
    │   ├── test_utils_common.py
    │   ├── test_utils_covariance.py
    │   ├── test_utils_gp.py
    │   ├── test_utils_logger.py
    │   ├── test_utils_plotting.py
    │   ├── test_version.py
    │   ├── test_wrappers_bo_class.py
    │   ├── test_wrappers_bo_function.py
    │   └── test_wrappers.py
    ├── integration_test.py
    └── time
        ├── test_time_bo_load.py
        ├── test_time_bo_optimize.py
        ├── test_time_covariance.py
        └── test_time_random_forest.py

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
