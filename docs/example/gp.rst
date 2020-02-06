Building Gaussian Process Regression
====================================

This example is for building Gaussian process regression models.
First of all, import the packages we need and **bayeso**.

.. code-block:: python

    import numpy as np
    import os

    from bayeso import gp
    from bayeso.utils import utils_covariance
    from bayeso.utils import utils_plotting

Declare some parameters to control this example.

.. code-block:: python

    is_tex = False
    num_test = 200
    str_cov = 'matern52'

Make a simple synthetic dataset, which produces with cosine functions.

.. code-block:: python

    X_train = np.array([
        [-3.0],
        [-2.0],
        [-1.0],
        [2.0],
        [1.2],
        [1.1],
    ])
    Y_train = np.cos(X_train) + 10.0
    X_test = np.linspace(-3, 3, num_test)
    X_test = X_test.reshape((num_test, 1))
    Y_test_truth = np.cos(X_test) + 10.0

Build a Gaussian process regression model with fixed hyperparameters.
Then, plot the result.

.. code-block:: python

    hyps = utils_covariance.get_hyps(str_cov, 1)
    mu, sigma = gp.predict_test(X_train, Y_train, X_test, hyps, str_cov=str_cov)
    utils_plotting.plot_gp(
        X_train, Y_train, X_test, mu, sigma, Y_test_truth=Y_test_truth, is_tex=is_tex,
        str_x_axis='$x$', str_y_axis='$y$'
    )

.. image:: ../_static/examples/gp_fixed.*
    :width: 400
    :align: center
    :alt: gp_fixed

Build a Gaussian process regression model with the hyperparameters optimized by marginal likelihood maximization, and plot the result.

.. code-block:: python

    mu, sigma = gp.predict_optimized(X_train, Y_train, X_test, str_cov=str_cov)
    utils_plotting.plot_gp(
        X_train, Y_train, X_test, mu, sigma, Y_test_truth=Y_test_truth, is_tex=is_tex,
        str_x_axis='$x$', str_y_axis='$y$'
    )

.. image:: ../_static/examples/gp_optimized.*
    :width: 400
    :align: center
    :alt: gp_optimized

Declare some functions that would be employed as prior functions.

.. code-block:: python

    def cosine(X):
        return np.cos(X)

    def linear_down(X):
        list_up = []
        for elem_X in X:
            list_up.append([-0.5 * np.sum(elem_X)])
        return np.array(list_up)

    def linear_up(X):
        list_up = []
        for elem_X in X:
            list_up.append([0.5 * np.sum(elem_X)])
        return np.array(list_up)

Make an another synthetic dataset using a cosine function.

.. code-block:: python

    X_train = np.array([
        [-3.0],
        [-2.0],
        [-1.0],
    ])
    Y_train = np.cos(X_train) + 2.0
    X_test = np.linspace(-3, 6, num_test)
    X_test = X_test.reshape((num_test, 1))
    Y_test_truth = np.cos(X_test) + 2.0

Build Gaussian process regression models with the prior functions we declare above and the hyperparameters optimized by marginal likelihood maximization, and plot the result.

.. code-block:: python

    prior_mu = cosine
    mu, sigma = gp.predict_optimized(X_train, Y_train, X_test, str_cov=str_cov, prior_mu=prior_mu)
    utils_plotting.plot_gp(
        X_train, Y_train, X_test, mu, sigma, Y_test_truth=Y_test_truth, is_tex=is_tex,
        str_x_axis='$x$', str_y_axis='$y$'
    )

    prior_mu = linear_down
    mu, sigma = gp.predict_optimized(X_train, Y_train, X_test, str_cov=str_cov, prior_mu=prior_mu)
    utils_plotting.plot_gp(
        X_train, Y_train, X_test, mu, sigma, Y_test_truth, is_tex=is_tex,
        str_x_axis='$x$', str_y_axis='$y$'
    )

    prior_mu = linear_up
    mu, sigma = gp.predict_optimized(X_train, Y_train, X_test, str_cov=str_cov, prior_mu=prior_mu)
    utils_plotting.plot_gp(
        X_train, Y_train, X_test, mu, sigma, Y_test_truth=Y_test_truth, is_tex=is_tex,
        str_x_axis='$x$', str_y_axis='$y$'
    )

.. image:: ../_static/examples/gp_optimized_prior_cosine.*
    :width: 400
    :align: center
    :alt: gp_optimized_prior_cosine

.. image:: ../_static/examples/gp_optimized_prior_linear_down.*
    :width: 400
    :align: center
    :alt: gp_optimized_prior_linear_down

.. image:: ../_static/examples/gp_optimized_prior_linear_up.*
    :width: 400
    :align: center
    :alt: gp_optimized_prior_linear_up

Full code:

.. code-block:: python

    import numpy as np
    import os

    from bayeso import gp
    from bayeso.utils import utils_covariance
    from bayeso.utils import utils_plotting

    is_tex = False
    num_test = 200
    str_cov = 'matern52'

    X_train = np.array([
        [-3.0],
        [-2.0],
        [-1.0],
        [2.0],
        [1.2],
        [1.1],
    ])
    Y_train = np.cos(X_train) + 10.0
    X_test = np.linspace(-3, 3, num_test)
    X_test = X_test.reshape((num_test, 1))
    Y_test_truth = np.cos(X_test) + 10.0

    hyps = utils_covariance.get_hyps(str_cov, 1)
    mu, sigma = gp.predict_test(X_train, Y_train, X_test, hyps, str_cov=str_cov)
    utils_plotting.plot_gp(
        X_train, Y_train, X_test, mu, sigma, Y_test_truth=Y_test_truth, is_tex=is_tex,
        str_x_axis='$x$', str_y_axis='$y$'
    )

    mu, sigma = gp.predict_optimized(X_train, Y_train, X_test, str_cov=str_cov)
    utils_plotting.plot_gp(
        X_train, Y_train, X_test, mu, sigma, Y_test_truth=Y_test_truth, is_tex=is_tex,
        str_x_axis='$x$', str_y_axis='$y$'
    )

    def cosine(X):
        return np.cos(X)

    def linear_down(X):
        list_up = []
        for elem_X in X:
            list_up.append([-0.5 * np.sum(elem_X)])
        return np.array(list_up)

    def linear_up(X):
        list_up = []
        for elem_X in X:
            list_up.append([0.5 * np.sum(elem_X)])
        return np.array(list_up)

    X_train = np.array([
        [-3.0],
        [-2.0],
        [-1.0],
    ])
    Y_train = np.cos(X_train) + 2.0
    X_test = np.linspace(-3, 6, num_test)
    X_test = X_test.reshape((num_test, 1))
    Y_test_truth = np.cos(X_test) + 2.0

    prior_mu = cosine
    mu, sigma = gp.predict_optimized(X_train, Y_train, X_test, str_cov=str_cov, prior_mu=prior_mu)
    utils_plotting.plot_gp(
        X_train, Y_train, X_test, mu, sigma, Y_test_truth=Y_test_truth, is_tex=is_tex,
        str_x_axis='$x$', str_y_axis='$y$'
    )

    prior_mu = linear_down
    mu, sigma = gp.predict_optimized(X_train, Y_train, X_test, str_cov=str_cov, prior_mu=prior_mu)
    utils_plotting.plot_gp(
        X_train, Y_train, X_test, mu, sigma, Y_test_truth, is_tex=is_tex,
        str_x_axis='$x$', str_y_axis='$y$'
    )

    prior_mu = linear_up
    mu, sigma = gp.predict_optimized(X_train, Y_train, X_test, str_cov=str_cov, prior_mu=prior_mu)
    utils_plotting.plot_gp(
        X_train, Y_train, X_test, mu, sigma, Y_test_truth=Y_test_truth, is_tex=is_tex,
        str_x_axis='$x$', str_y_axis='$y$'
    )

