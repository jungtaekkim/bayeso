---
title: 'BayesO: A Bayesian optimization framework in Python'
tags:
  - Python
  - Bayesian optimization
  - global optimization
  - black-box optimization
  - optimization
authors:
  - name: Jungtaek Kim
    orcid: 0000-0002-1905-1399
    affiliation: 1
  - name: Seungjin Choi
    orcid: 0000-0002-7873-4616
    affiliation: 2
affiliations:
 - name: University of Pittsburgh, USA
   index: 1
 - name: Intellicode, South Korea
   index: 2
date: 6 December 2022
bibliography: paper.bib

---

# Summary

Bayesian optimization is a sample-efficient method for solving the
optimization of a black-box function. In particular, it successfully shows
its effectiveness in diverse applications such as hyperparameter
optimization, automated machine learning, material design, sequential
assembly, and chemical reaction optimization. In this paper we present an
easy-to-use Bayesian optimization framework, referred to as *BayesO*, which
is written in Python and licensed under the MIT license. To briefly
introduce our software, we describe the functionality of BayesO and
components for software development.

# Statement of need

Bayesian
optimization [@BrochuE2010arxiv; @ShahriariB2016procieee; @GarnettR2022book]
is a sample-efficient method for solving the optimization of a black-box
function $f$:
\begin{equation}\label{eqn:global_opt}
    \mathbf{x}^\star = \underset{\mathbf{x} \in \mathcal{X}}{\mathrm{arg\,min}} f(\mathbf{x}) \quad \textrm{or} \quad \mathbf{x}^\star = \underset{\mathbf{x} \in \mathcal{X}}{\mathrm{arg\,max}} f(\mathbf{x}),
\end{equation}
where $\mathcal{X} \subset \mathbb{R}^d$ is a $d$-dimensional space. In
general, finding a solution $\mathbf{x}^\star$ of \autoref{eqn:global_opt},
i.e., a global optimizer on $\mathcal{X}$, is time-consuming since we
cannot employ any knowledge in solving this problem. Compared to other
possible approaches, e.g., random search and evolutionary algorithm,
Bayesian optimization successfully shows its effectiveness by utilizing a
probabilistic regression model and an acquisition function. In particular,
the sample-efficient approach of our interest enables us to apply it in
various real-world applications such as hyperparameter
optimization [@SnoekJ2012neurips], automated machine
learning [@FeurerM2015neurips; @FeurerM2022jmlr], neural architecture
search [@KandasamyK2018neurips], material design [@FrazierP2016ismdd],
free-electron laser configuration search [@DurisJ2020phrl], organic
molecule synthesis [@KorovinaK2020aistats], sequential
assembly [@KimJ2020ml4eng], and chemical reaction
optimization [@ShieldsBJ2021nature].

In this paper, we present an easy-to-use Bayesian optimization framework,
referred to as *BayesO* (pronounced "bayes-o"), to effortlessly utilize
Bayesian optimization in the problems of interest to practitioners. Our
BayesO is written in one of the most popular programming languages, Python,
and licensed under the MIT license. Moreover, it provides various features
including different types of input variables (e.g., vectors and
sets [@KimJ2021ml]) and different surrogate models (e.g., Gaussian process
regression [@RasmussenCE2006book] and Student-$t$ process
regression [@ShahA2014aistats]). Along with the description of such
functionality, we cover various components for software development in the
BayesO project. We hope that this BayesO project encourages researchers and
practitioners to readily utilize the powerful black-box optimization
technique in diverse academic and industrial fields.

![Visualization of Bayesian optimization procedure. Given an objective function, \autoref{eqn:simple} (colored by turquoise) and four initial points (denoted as light blue $\texttt{+}$ at iteration 1), a query point (denoted as pink $\texttt{x}$) is determined by constructing a surrogate model (colored by orange) and maximizing an acquisition function (colored by light green) every iteration.\label{fig:bo_steps}](figures/bo_step_global_local_ei.png)

# Bayesian Optimization

As discussed in the
work [@BrochuE2010arxiv; @ShahriariB2016procieee; @GarnettR2022book],
supposing that a target objective function is black-box, Bayesian
optimization is an approach to optimizing the objective in a
sample-efficient manner. It repeats three primary steps:

1. Building a probabilistic regression model, which is capable of estimating
the degrees of exploration and exploitation;
2. Optimizing an acquisition function, which is defined with the
probabilistic regression model;
3. Evaluating a query point, which is determined by optimizing the
acquisition function,

until a predefined stopping criterion, e.g., an iteration budget or a budget
of wall-clock time, is satisfied. Eventually, the best solution among the
queries evaluated is selected by considering the function evaluations. As
shown in \autoref{fig:bo_steps}, Bayesian optimization iteratively finds a
candidate of global optimizer, repeating the aforementioned steps. Note
that, for this example, an objective function is
\begin{equation}\label{eqn:simple}
    f(x) = 2 \sin(x) + 2 \cos(2x) + 0.05 x,
\end{equation}
where $x \in [-5, 5]$, and Gaussian process regression and expected
improvement are used as a surrogate model and an acquisition function,
respectively. To focus on the BayesO system, we omit the details of
surrogate models and acquisition functions here; see the seminal articles
and textbook on Bayesian
optimization [@BrochuE2010arxiv; @ShahriariB2016procieee; @GarnettR2022book]
for the details.

# Overview of BayesO

![Logo of BayesO.\label{fig:logo_bayeso}](figures/logo_bayeso_capitalized.png){ width=50% }

In this section we cover the overview of BayesO including probabilistic
regression models and acquisition functions. Note that this paper is written
by referring to BayesO v0.5.4. For higher versions of BayesO, see official
documentation.

BayesO supports the following probabilistic regression models:

- Gaussian process regression [@RasmussenCE2006book];
- Student-$t$ process regression [@ShahA2014aistats];
- Random forest regression [@BreimanL2001ml].

Although random forest regression is not a probabilistic model inherently,
we can compute its mean and variance functions as reported
by @HutterF2014ai.

We implement the following acquisition functions:

- pure exploration;
- pure exploitation;
- probability of improvement [@MockusJ1978tgo];
- expected improvement [@JonesDR1998jgo];
- augmented expected improvement [@HuangD2006jgo];
- Gaussian process upper confidence bound [@SrinivasN2010icml].

In addition to the aforementioned acquisition functions, we also include
Thompson sampling [@ThompsonWR1933biometrika] in BayesO.

Furthermore, to support an easy-to-use interface, we implement wrappers of
Bayesian optimization for the following scenarios:

- a run with randomly-chosen initial inputs;
- a run with initial inputs provided;
- a run with initial inputs provided and their evaluations.

# Software Development for BayesO

To manage BayesO productively, we actively utilize external development
management packages.

- Code analysis: The entire codes in our software are monitored and
inspected to satisfy the code conventions predefined in our software. Unless
otherwise specified, we do our best to satisfy all the conventions.
- Type hints: As supported in Python 3, we provide type hints for any
arguments including arguments with default values.
- Unit tests: All the unit tests for our software are included. We have
achieved 100\% coverage. In addition, unit tests for measuring execution
time are also provided.
- Dependency: Our package depends on NumPy [@HarrisCR2020nature],
SciPy [@VirtanenP2020nm], qmcpy [@ChoiSCT2022mcqmc],
pycma [@HansenN2019software], and tqdm.
- Installation: We upload our software in a popular repository for Python
packages, PyPI, accordingly BayesO can be easily installed in any supported
environments.
- Documentation: We create official documentation with docstring. A code
convention, docstring, is supported in Python and it is accomplished by
specific templates of comments.

# Conclusion

In this work we have presented our own Bayesian optimization framework,
named BayesO. We hope that our project enables many researchers to suggest
a new algorithm by modifying BayesO and many practitioners to utilize
Bayesian optimization in their applications.

# Acknowledgements

The BayesO project has been started when JK and SC were with POSTECH, and it
has been mostly developed at POSTECH.

# References
