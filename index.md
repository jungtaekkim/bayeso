---
layout: default
title: Home
---

![Bayesian Optimization]({{ site.baseurl }}public/ei.gif)

Simple, but essential Bayesian optimization package.

* [BayesO: GitHub Repository](https://github.com/jungtaekkim/bayeso)
* [BayesO Benchmarks: GitHub Repository](https://github.com/jungtaekkim/bayeso-benchmarks)
* [Batch BayesO: GitHub Repository](https://github.com/jungtaekkim/batch-bayeso)

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
