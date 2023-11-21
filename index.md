---
layout: default
title: Home
---

![Bayesian Optimization]({{ site.baseurl }}public/ei.gif)

Simple, but essential Bayesian optimization package.

* [BayesO: GitHub Repository]({{ site.github.repo-bayeso }})
* [BayesO Benchmarks: GitHub Repository]({{ site.github.repo-bench }})
* [Batch BayesO: GitHub Repository]({{ site.github.repo-batch }})

## Installation

Detailed installation guides can be found in the respective repositories.

* [BayesO]({{ site.github.repo-bayeso }})

To install a released version in the PyPI repository, command it.

```shell
$ pip install bayeso
```

* [BayesO Benchmarks]({{ site.github.repo-bench }})

Similar to BayesO, command it to install a released version.

```shell
$ pip install bayeso-benchmarks
```

* [Batch BayesO]({{ site.github.repo-batch }})

Now, it is not released through the PyPI repository.
Command it in the root directory of Batch BayesO.

```shell
$ pip install .
```

## Citation

```
@article{KimJ2023joss,
    author="Kim, Jungtaek and Choi, Seungjin",
    title="{BayesO}: A {Bayesian} optimization framework in {Python}",
    journal="Journal of Open Source Software",
    volume="8",
    number="90",
    pages="5320",
    year="2023"
}
```

## License

All software is licensed under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
You can look at the LICENSE files in the respective repositories.

* [BayesO]({{ site.github.repo-bayeso }}/blob/main/LICENSE)
* [BayesO Benchmarks]({{ site.github.repo-bench }}/blob/main/LICENSE)
* [Batch BayesO]({{ site.github.repo-batch }}/blob/main/LICENSE)
