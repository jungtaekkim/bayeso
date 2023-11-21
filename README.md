<p align="center">
<img src="https://raw.githubusercontent.com/jungtaekkim/bayeso/main/docs/_static/assets/logo_bayeso_capitalized.svg" width="400" />
</p>

# BayesO: A Bayesian Optimization Framework in Python
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05320/status.svg)](https://doi.org/10.21105/joss.05320)
[![Build Status](https://github.com/jungtaekkim/bayeso/actions/workflows/pytest.yml/badge.svg)](https://github.com/jungtaekkim/bayeso/actions/workflows/pytest.yml)
[![Coverage Status](https://coveralls.io/repos/github/jungtaekkim/bayeso/badge.svg?branch=main)](https://coveralls.io/github/jungtaekkim/bayeso?branch=main)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/bayeso)](https://pypi.org/project/bayeso/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/bayeso/badge/?version=main)](https://bayeso.readthedocs.io/en/main/?badge=main)

<p align="center">
<img src="https://raw.githubusercontent.com/jungtaekkim/bayeso/main/docs/_static/steps/ei.gif" width="600" />
</p>

Simple, but essential Bayesian optimization package.

* [https://bayeso.org](https://bayeso.org)
* [Online documentation](https://bayeso.readthedocs.io)

## Installation
We recommend installing it with `virtualenv`.
You can choose one of three installation options.

* Using PyPI repository (for user installation)

To install the released version in PyPI repository, command it.

```shell
$ pip install bayeso
```

* Using source code (for developer installation)

To install `bayeso` from source code, command

```shell
$ pip install .
```
in the `bayeso` root.

* Using source code (for editable development mode)

To use editable development mode, command

```shell
$ pip install -r requirements.txt
$ python setup.py develop
```
in the `bayeso` root.

* Uninstallation

If you would like to uninstall `bayeso`, command it.

```shell
$ pip uninstall bayeso
```

## Required Packages
Mandatory pacakges are inlcuded in `requirements.txt`.
The following `requirements` files include the package list, the purpose of which is described as follows.

* `requirements-optional.txt`: It is an optional package list, but it needs to be installed to execute some features of `bayeso`.
* `requirements-dev.txt`: It is for developing the `bayeso` package.
* `requirements-examples.txt`: It needs to be installed to execute the examples included in the `bayeso` repository.

## Supported Python Version
We test our package in the following versions.

* Python 3.7
* Python 3.8
* Python 3.9
* Python 3.10
* Python 3.11

## Examples and Tests
We provide a [list of examples](EXAMPLES.md) and a [list of tests](TESTS.md).

## Citation
```
@article{KimJ2023joss,
    author={Kim, Jungtaek and Choi, Seungjin},
    title={{BayesO}: A {Bayesian} optimization framework in {Python}},
    journal={Journal of Open Source Software},
    volume={8},
    number={90},
    pages={5320},
    year={2023}
}
```

## License
[MIT License](LICENSE)
