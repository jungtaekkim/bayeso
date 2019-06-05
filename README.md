# bayeso
[![Build Status](https://travis-ci.org/jungtaekkim/bayeso.svg?branch=master)](https://travis-ci.org/jungtaekkim/bayeso)
[![Coverage Status](https://coveralls.io/repos/github/jungtaekkim/bayeso/badge.svg?branch=master)](https://coveralls.io/github/jungtaekkim/bayeso?branch=master)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/bayeso/badge/?version=latest)](http://bayeso.readthedocs.io/en/latest/?badge=latest)

Simple, but essential Bayesian optimization package.

* [Online documentation](http://bayeso.readthedocs.io)

## Installation
We recommend it should be installed in `virtualenv`.
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

If you would like to uninstall bayeso, command it.

```shell
$ pip uninstall bayeso
```

## Supported Python Version
We test our package in the following versions.

* Python 2.7
* Python 3.5
* Python 3.6
* Python 3.7

## Author
* [Jungtaek Kim](http://mlg.postech.ac.kr/~jtkim/) (POSTECH)
* [Seungjin Choi](http://mlg.postech.ac.kr/~seungjin/) (POSTECH)

## Citation
```
@misc{KimJ2017bayeso,
    author={Kim, Jungtaek and Choi, Seungjin},
    title={{bayeso}: A {Bayesian} optimization framework in {Python}},
    howpublished={\url{https://github.com/jungtaekkim/bayeso}},
    year={2017}
}
```

## Contact
* Jungtaek Kim: [jtkim@postech.ac.kr](mailto:jtkim@postech.ac.kr)

## License
[MIT License](LICENSE)
