.. image:: _static/assets/logo_bayeso_capitalized.*
    :alt: BayesO logo
    :width: 400px
    :align: center

----------------------------------

BayesO: A Bayesian optimization framework in Python
===================================================

`BayesO <http://bayeso.org>`_ (pronounced "bayes-o") is a simple, but essential Bayesian optimization package, written in Python.
It is developed by `machine learning group <http://mlg.postech.ac.kr>`_ at POSTECH.
This project is licensed under `the MIT license <https://opensource.org/licenses/MIT>`_.

This documentation describes the details of implementation, getting started guides, some examples with BayesO, and Python API specifications.
The code can be found in `our GitHub repository <https://github.com/jungtaekkim/bayeso>`_.


.. toctree::
   :maxdepth: 1

   about/about_bayeso
   about/about_bo

.. toctree::
   :maxdepth: 1
   :caption: Getting Started:

   getting_started/installation

.. toctree::
   :maxdepth: 1
   :caption: Example:

   example/gp
   example/ts_gp_prior
   example/branin
   example/hpo

.. toctree::
   :maxdepth: 2
   :caption: Python API:

   python_api/bayeso
   python_api/bayeso.gp
   python_api/bayeso.tp
   python_api/bayeso.utils
   python_api/bayeso.wrappers
