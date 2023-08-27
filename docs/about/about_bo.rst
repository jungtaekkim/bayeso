About Bayesian Optimization
===========================

.. image:: ../_static/steps/ucb.*
    :width: 480
    :align: center
    :alt: bo_with_gp_and_ucb

Bayesian optimization is a *global optimization* strategy for *black-box* and *expensive-to-evaluate* functions.
Generic Bayesian optimization follows these steps:

#. Build a *surrogate function* $\hat{f}$ with historical inputs $\mathbf{X}$ and their observations $\mathbf{y}$, which is defined with mean and variance functions.
$$
\hat{f}(\mathbf{x} \mid \mathbf{X}, \mathbf{y}) = \mathcal{N}(\mu(\mathbf{x} \mid \mathbf{X}, \mathbf{y}), \sigma^2(\mathbf{x} \mid \mathbf{X}, \mathbf{y}))
$$
#. Compute and maximize an *acquisition function*, defined by the outputs of surrogate function.
#. Observe the *maximizer* of acquisition function from a true objective function.
#. Accumulate the maximizer and its observation.

This project helps us to execute this Bayesian optimization procedure.
In particular, *Gaussian process regression* is used as a surrogate function,
and various acquisition functions such as *probability improvement*, *expected improvement*, and *Gaussian process upper confidence bound* are included in this project.
