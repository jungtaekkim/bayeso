About Bayesian Optimization
===========================

.. image:: ../_static/steps/ucb.*
    :width: 480
    :align: center
    :alt: bo_with_gp_and_ucb

Bayesian optimization is a *global optimization* strategy for *black-box* and *expensive-to-evaluate* functions.
Generic Bayesian optimization follows these steps:

1. Build a *surrogate function* :math:`\hat{f}` with historical inputs :math:`\mathbf{X}` and their observations :math:`\mathbf{y}`, which is defined with mean and variance functions.

.. math::

    \hat{f}(\mathbf{x} \mid \mathbf{X}, \mathbf{y}) = \mathcal{N}(\mu(\mathbf{x} \mid \mathbf{X}, \mathbf{y}), \sigma^2(\mathbf{x} \mid \mathbf{X}, \mathbf{y}))
2. Compute and maximize an *acquisition function* :math:`a`, defined by the outputs of surrogate function, i.e., :math:`\mu(\mathbf{x} \mid \mathbf{X}, \mathbf{y})` and :math:`\sigma^2(\mathbf{x} \mid \mathbf{X}, \mathbf{y})`.

.. math::

    \mathbf{x}^{*} = {\arg \max} \ a(\mathbf{x} \mid \mu(\mathbf{x} \mid \mathbf{X}, \mathbf{y}), \sigma^2(\mathbf{x} \mid \mathbf{X}, \mathbf{y}))
3. Observe the *maximizer* of acquisition function from a true objective function :math:`f` where a random observation noise :math:`\epsilon` exists.

.. math::

    y = f(\mathbf{x}) + \epsilon
4. Accumulate the maximizer and its observation.

This project helps us to execute this Bayesian optimization procedure.
In particular, *Gaussian process regression* is used as a surrogate function,
and various acquisition functions such as *probability improvement*, *expected improvement*, and *Gaussian process upper confidence bound* are included in this project.
