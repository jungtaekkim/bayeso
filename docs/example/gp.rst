Building Gaussian Process Regression
====================================

This example is for building Gaussian process regression models.
In this example, we cover three scenarios:
Gaussian process with fixed hyperparameters,
Gaussian process with learned hyperparameters,
and Gaussian process with particular priors.

First of all, import the package we need and **bayeso**.

.. code-block:: python

    import numpy as np

    from bayeso import covariance
    from bayeso.gp import gp
    from bayeso.utils import utils_covariance
    from bayeso.utils import utils_plotting

Declare some parameters to control this example.
`use_tex` is a flag for using a LaTeX style,
`num_test` is the number of test data points,
and `str_cov` is a kernel choice.

.. code-block:: python

    use_tex = False
    num_test = 200
    str_cov = 'matern52'

Make a simple synthetic dataset, which is produced with a cosine function.
The underlying true function is \cos(x) + 10.

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
    Y_test = np.cos(X_test) + 10.0

Sample functions from a prior distribution, which is zero mean.
As shown in the figure below, `num_samples` smooth functions are sampled.

.. code-block:: python

    mu = np.zeros(num_test)
    hyps = utils_covariance.get_hyps(str_cov, 1)
    Sigma = covariance.cov_main(str_cov, X_test, X_test, hyps, True)

    Ys = gp.sample_functions(mu, Sigma, num_samples=5)
    utils_plotting.plot_gp_via_sample(X_test, Ys, use_tex=use_tex,
                                      str_x_axis='$x$', str_y_axis='$y$')

.. image:: ../_static/examples/gp_sampled_prior.*
    :width: 320
    :align: center
    :alt: gp_sampled_prior

Build a Gaussian process regression model with fixed hyperparameters.
Fixed hyperparameters are brought through `get_hyps`.
`mu`, `sigma`, and `Sigma` are mean estimates, standard deviation estimates, and covariance estimates, respectively.
In addition, `num_samples` functions are sampled using `mu` and `Sigma`.
Then, plot the result.

.. code-block:: python

    hyps = utils_covariance.get_hyps(str_cov, 1)
    mu, sigma, Sigma = gp.predict_with_hyps(X_train, Y_train, X_test, hyps, str_cov=str_cov)
    utils_plotting.plot_gp_via_distribution(
        X_train, Y_train, X_test, mu, sigma,
        Y_test=Y_test, use_tex=use_tex,
        str_x_axis='$x$', str_y_axis='$y$'
    )

    Ys = gp.sample_functions(mu.flatten(), Sigma, num_samples=5)
    utils_plotting.plot_gp_via_sample(X_test, Ys, use_tex=use_tex,
                                      str_x_axis='$x$', str_y_axis='$y$')

.. image:: ../_static/examples/gp_fixed.*
    :width: 320
    :align: center
    :alt: gp_fixed

.. image:: ../_static/examples/gp_sampled_fixed.*
    :width: 320
    :align: center
    :alt: gp_sampled_fixed

Build a Gaussian process regression model with the hyperparameters optimized by marginal likelihood maximization, and plot the result.
Similar to the aforementioned case, `num_samples` functions are sampled from the distributions with `mu` and `Sigma`.

.. code-block:: python

    mu, sigma, Sigma = gp.predict_with_optimized_hyps(X_train, Y_train, X_test, str_cov=str_cov)
    utils_plotting.plot_gp_via_distribution(
        X_train, Y_train, X_test, mu, sigma,
        Y_test=Y_test, use_tex=use_tex,
        str_x_axis='$x$', str_y_axis='$y$'
    )

    Ys = gp.sample_functions(mu.flatten(), Sigma, num_samples=5)
    utils_plotting.plot_gp_via_sample(X_test, Ys, use_tex=use_tex,
                                      str_x_axis='$x$', str_y_axis='$y$')

.. image:: ../_static/examples/gp_optimized.*
    :width: 320
    :align: center
    :alt: gp_optimized

.. image:: ../_static/examples/gp_sampled_optimized.*
    :width: 320
    :align: center
    :alt: gp_sampled_optimized

Declare some functions that would be employed as prior functions.
`cosine` is a prior function with a cosine function.
`linear_down` is a prior function with a decreasing function.
`linear_up` is a prior function with an increasing function.

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
The true function is \cos(x) + 2.

.. code-block:: python

    X_train = np.array([
        [-3.0],
        [-2.0],
        [-1.0],
    ])
    Y_train = np.cos(X_train) + 2.0
    X_test = np.linspace(-3, 6, num_test)
    X_test = X_test.reshape((num_test, 1))
    Y_test = np.cos(X_test) + 2.0

Build Gaussian process regression models with the prior functions we declare above and the hyperparameters optimized by marginal likelihood maximization, and plot the result.
Also, `num_samples` functions are sampled from the distributions defined with `mu` and `Sigma`.

.. code-block:: python

    prior_mu = cosine
    mu, sigma, Sigma = gp.predict_with_optimized_hyps(X_train, Y_train, X_test,
                                                      str_cov=str_cov, prior_mu=prior_mu)
    utils_plotting.plot_gp_via_distribution(
        X_train, Y_train, X_test, mu, sigma,
        Y_test=Y_test, use_tex=use_tex,
        str_x_axis='$x$', str_y_axis='$y$'
    )

    Ys = gp.sample_functions(mu.flatten(), Sigma, num_samples=5)
    utils_plotting.plot_gp_via_sample(X_test, Ys, use_tex=use_tex,
                                      str_x_axis='$x$', str_y_axis='$y$')

    prior_mu = linear_down
    mu, sigma, Sigma = gp.predict_with_optimized_hyps(X_train, Y_train, X_test,
                                                      str_cov=str_cov, prior_mu=prior_mu)
    utils_plotting.plot_gp_via_distribution(
        X_train, Y_train, X_test, mu, sigma,
        Y_test=Y_test, use_tex=use_tex,
        str_x_axis='$x$', str_y_axis='$y$'
    )

    Ys = gp.sample_functions(mu.flatten(), Sigma, num_samples=5)
    utils_plotting.plot_gp_via_sample(X_test, Ys, use_tex=use_tex,
                                      str_x_axis='$x$', str_y_axis='$y$')

    prior_mu = linear_up
    mu, sigma, Sigma = gp.predict_with_optimized_hyps(X_train, Y_train, X_test,
                                                      str_cov=str_cov, prior_mu=prior_mu)
    utils_plotting.plot_gp_via_distribution(
        X_train, Y_train, X_test, mu, sigma,
        Y_test=Y_test, use_tex=use_tex,
        str_x_axis='$x$', str_y_axis='$y$'
    )

    Ys = gp.sample_functions(mu.flatten(), Sigma, num_samples=5)
    utils_plotting.plot_gp_via_sample(X_test, Ys, use_tex=use_tex,
                                      str_x_axis='$x$', str_y_axis='$y$')

.. image:: ../_static/examples/gp_optimized_prior_cosine.*
    :width: 320
    :align: center
    :alt: gp_optimized_prior_cosine

.. image:: ../_static/examples/gp_sampled_optimized_prior_cosine.*
    :width: 320
    :align: center
    :alt: gp_sampled_optimized_prior_cosine

.. image:: ../_static/examples/gp_optimized_prior_linear_down.*
    :width: 320
    :align: center
    :alt: gp_optimized_prior_linear_down

.. image:: ../_static/examples/gp_sampled_optimized_prior_linear_down.*
    :width: 320
    :align: center
    :alt: gp_sampled_optimized_prior_linear_down

.. image:: ../_static/examples/gp_optimized_prior_linear_up.*
    :width: 320
    :align: center
    :alt: gp_optimized_prior_linear_up

.. image:: ../_static/examples/gp_sampled_optimized_prior_linear_up.*
    :width: 320
    :align: center
    :alt: gp_sampled_optimized_prior_linear_up

Full code:

.. code-block:: python

    import numpy as np

    from bayeso import covariance
    from bayeso.gp import gp
    from bayeso.utils import utils_covariance
    from bayeso.utils import utils_plotting

    use_tex = False
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
    Y_test = np.cos(X_test) + 10.0

    mu = np.zeros(num_test)
    hyps = utils_covariance.get_hyps(str_cov, 1)
    Sigma = covariance.cov_main(str_cov, X_test, X_test, hyps, True)

    Ys = gp.sample_functions(mu, Sigma, num_samples=5)
    utils_plotting.plot_gp_via_sample(X_test, Ys, use_tex=use_tex,
                                      str_x_axis='$x$', str_y_axis='$y$')

    hyps = utils_covariance.get_hyps(str_cov, 1)
    mu, sigma, Sigma = gp.predict_with_hyps(X_train, Y_train, X_test, hyps, str_cov=str_cov)
    utils_plotting.plot_gp_via_distribution(
        X_train, Y_train, X_test, mu, sigma,
        Y_test=Y_test, use_tex=use_tex,
        str_x_axis='$x$', str_y_axis='$y$'
    )

    Ys = gp.sample_functions(mu.flatten(), Sigma, num_samples=5)
    utils_plotting.plot_gp_via_sample(X_test, Ys, use_tex=use_tex,
                                      str_x_axis='$x$', str_y_axis='$y$')

    mu, sigma, Sigma = gp.predict_with_optimized_hyps(X_train, Y_train, X_test, str_cov=str_cov)
    utils_plotting.plot_gp_via_distribution(
        X_train, Y_train, X_test, mu, sigma,
        Y_test=Y_test, use_tex=use_tex,
        str_x_axis='$x$', str_y_axis='$y$'
    )

    Ys = gp.sample_functions(mu.flatten(), Sigma, num_samples=5)
    utils_plotting.plot_gp_via_sample(X_test, Ys, use_tex=use_tex,
                                      str_x_axis='$x$', str_y_axis='$y$')

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
    Y_test = np.cos(X_test) + 2.0

    prior_mu = cosine
    mu, sigma, Sigma = gp.predict_with_optimized_hyps(X_train, Y_train, X_test,
                                                      str_cov=str_cov, prior_mu=prior_mu)
    utils_plotting.plot_gp_via_distribution(
        X_train, Y_train, X_test, mu, sigma,
        Y_test=Y_test, use_tex=use_tex,
        str_x_axis='$x$', str_y_axis='$y$'
    )

    Ys = gp.sample_functions(mu.flatten(), Sigma, num_samples=5)
    utils_plotting.plot_gp_via_sample(X_test, Ys, use_tex=use_tex,
                                      str_x_axis='$x$', str_y_axis='$y$')

    prior_mu = linear_down
    mu, sigma, Sigma = gp.predict_with_optimized_hyps(X_train, Y_train, X_test,
                                                      str_cov=str_cov, prior_mu=prior_mu)
    utils_plotting.plot_gp_via_distribution(
        X_train, Y_train, X_test, mu, sigma,
        Y_test=Y_test, use_tex=use_tex,
        str_x_axis='$x$', str_y_axis='$y$'
    )

    Ys = gp.sample_functions(mu.flatten(), Sigma, num_samples=5)
    utils_plotting.plot_gp_via_sample(X_test, Ys, use_tex=use_tex,
                                      str_x_axis='$x$', str_y_axis='$y$')

    prior_mu = linear_up
    mu, sigma, Sigma = gp.predict_with_optimized_hyps(X_train, Y_train, X_test,
                                                      str_cov=str_cov, prior_mu=prior_mu)
    utils_plotting.plot_gp_via_distribution(
        X_train, Y_train, X_test, mu, sigma,
        Y_test=Y_test, use_tex=use_tex,
        str_x_axis='$x$', str_y_axis='$y$'
    )

    Ys = gp.sample_functions(mu.flatten(), Sigma, num_samples=5)
    utils_plotting.plot_gp_via_sample(X_test, Ys, use_tex=use_tex,
                                      str_x_axis='$x$', str_y_axis='$y$')

