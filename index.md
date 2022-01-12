[![Build Status](https://github.com/jungtaekkim/bayeso/actions/workflows/pytest.yml/badge.svg)](https://github.com/jungtaekkim/bayeso/actions/workflows/pytest.yml)
[![Coverage Status](https://coveralls.io/repos/github/jungtaekkim/bayeso/badge.svg?branch=main)](https://coveralls.io/github/jungtaekkim/bayeso?branch=main)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/bayeso)](https://pypi.org/project/bayeso/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/bayeso/badge/?version=main)](https://bayeso.readthedocs.io/en/main/?badge=main)

<p align="center">
<img src="logo_bayeso_capitalized.svg" width="400" />
</p>

<p align="center">
<img src="ei.gif" width="600" />
</p>


Simple, but essential Bayesian optimization package.

* [BayesO: GitHub repository](https://github.com/jungtaekkim/bayeso)
* [BayesO Benchmarks: GitHub repository](https://github.com/jungtaekkim/bayeso-benchmarks)
* [Online documentation](https://bayeso.readthedocs.io)

## Installation
We recommend installing it with virtualenv.
You can choose one of three installation options.

* Using PyPI repository (for user installation)

To install the released version in PyPI repository, command it.

```shell
$ pip install bayeso
```

* Using source code (for developer installation)

To install BayesO from source code, command

```shell
$ pip install .
```
in the BayesO root.

* Using source code (for editable development mode)

To use editable development mode, command

```shell
$ pip install -r requirements.txt
$ python setup.py develop
```
in the BayesO root.

* Uninstallation

If you would like to uninstall BayesO, command it.

```shell
$ pip uninstall bayeso
```

## Required Packages
Mandatory pacakges are inlcuded in requirements.txt.
The following requirements files include the package list, the purpose of which is described as follows.

* requirements-optional.txt: It is an optional package list, but it needs to be installed to execute some features of BayesO.
* requirements-dev.txt: It is for developing the BayesO package.
* requirements-examples.txt: It needs to be installed to execute the examples included in the BayesO repository.

## Related Package
* [BayesO Benchmarks](https://github.com/jungtaekkim/bayeso-benchmarks): We implement benchmark functions for Bayesian optimization. This package is included in requirements-optional.txt.

## Supported Python Version
We test our package in the following versions.

* Python 3.6
* Python 3.7
* Python 3.8
* Python 3.9

## Contributor
* [Jungtaek Kim](https://jungtaek.github.io) (POSTECH)

## Citation
```
@misc{KimJ2017bayeso,
    author="Kim, Jungtaek and Choi, Seungjin",
    title="{BayesO}: A {Bayesian} optimization framework in {Python}",
    howpublished="\url{https://bayeso.org}",
    year="2017"
}
```

## License
[MIT License](https://github.com/jungtaekkim/bayeso/blob/main/LICENSE)
