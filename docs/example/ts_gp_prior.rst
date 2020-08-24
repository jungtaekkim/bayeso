Optimizing Sampled Function via Thompson Sampling
=================================================

This example is to optimize a function sampled from a Gaussian process prior via Thompson sampling.
First of all, import the packages we need and **bayeso**.

.. code-block:: python

    import numpy as np

    from bayeso import covariance
    from bayeso.gp import gp
    from bayeso.utils import utils_covariance
    from bayeso.utils import utils_plotting

Declare some parameters to control this example, including zero-mean prior, and compute a covariance matrix.

.. code-block:: python

    num_points = 1000
    str_cov = 'se'
    int_init = 1
    int_iter = 50
    int_ts = 10

    list_Y_min = []

    X = np.expand_dims(np.linspace(-5, 5, num_points), axis=1)
    mu = np.zeros(num_points)
    hyps = utils_covariance.get_hyps(str_cov, 1)
    Sigma = covariance.cov_main(str_cov, X, X, hyps, True)

Optimize a function sampled from a Gaussian process prior.
At each iteration, we sample a query point that outputs the mininum value of the function sampled from a Gaussian process posterior.

.. code-block:: python

    for ind_ts in range(0, int_ts):
        print('TS:', ind_ts + 1, 'round')
        Y = gp.sample_functions(mu, Sigma, num_samples=1)[0]

        ind_init = np.argmin(Y)
        bx_min = X[ind_init]
        y_min = Y[ind_init]
    
        ind_random = np.random.choice(num_points)

        X_ = np.expand_dims(X[ind_random], axis=0)
        Y_ = np.expand_dims(np.expand_dims(Y[ind_random], axis=0), axis=1)

        for ind_iter in range(0, int_iter):
            print(ind_iter + 1, 'iteration')

            mu_, sigma_, Sigma_ = gp.predict_optimized(X_, Y_, X, str_cov=str_cov)
            ind_ = np.argmin(gp.sample_functions(np.squeeze(mu_, axis=1), Sigma_, num_samples=1)[0])

            X_ = np.concatenate([X_, [X[ind_]]], axis=0)
            Y_ = np.concatenate([Y_, [[Y[ind_]]]], axis=0)
        
        list_Y_min.append(Y_ - y_min)

    Ys = np.array(list_Y_min)
    Ys = np.squeeze(Ys, axis=2)
    print(Ys.shape)

Plot the result obtained from the code block above.

.. code-block:: python

    utils_plotting.plot_minimum(np.array([Ys]), ['TS'], 1, True,
                                is_tex=True, range_shade=1.0,
                                str_x_axis=r'\textrm{Iteration}',
                                str_y_axis=r'\textrm{Minimum regret}')

.. image:: ../_static/examples/ts_gp_prior.*
    :width: 320
    :align: center
    :alt: ts_gp_prior

Full code:

.. code-block:: python

    import numpy as np

    from bayeso import covariance
    from bayeso.gp import gp
    from bayeso.utils import utils_covariance
    from bayeso.utils import utils_plotting

    num_points = 1000
    str_cov = 'se'
    int_init = 1
    int_iter = 50
    int_ts = 10

    list_Y_min = []

    X = np.expand_dims(np.linspace(-5, 5, num_points), axis=1)
    mu = np.zeros(num_points)
    hyps = utils_covariance.get_hyps(str_cov, 1)
    Sigma = covariance.cov_main(str_cov, X, X, hyps, True)

    for ind_ts in range(0, int_ts):
        print('TS:', ind_ts + 1, 'round')
        Y = gp.sample_functions(mu, Sigma, num_samples=1)[0]

        ind_init = np.argmin(Y)
        bx_min = X[ind_init]
        y_min = Y[ind_init]
    
        ind_random = np.random.choice(num_points)

        X_ = np.expand_dims(X[ind_random], axis=0)
        Y_ = np.expand_dims(np.expand_dims(Y[ind_random], axis=0), axis=1)

        for ind_iter in range(0, int_iter):
            print(ind_iter + 1, 'iteration')

            mu_, sigma_, Sigma_ = gp.predict_optimized(X_, Y_, X, str_cov=str_cov)
            ind_ = np.argmin(gp.sample_functions(np.squeeze(mu_, axis=1), Sigma_, num_samples=1)[0])

            X_ = np.concatenate([X_, [X[ind_]]], axis=0)
            Y_ = np.concatenate([Y_, [[Y[ind_]]]], axis=0)
        
        list_Y_min.append(Y_ - y_min)

    Ys = np.array(list_Y_min)
    Ys = np.squeeze(Ys, axis=2)
    print(Ys.shape)

    utils_plotting.plot_minimum(np.array([Ys]), ['TS'], 1, True,
                                is_tex=True, range_shade=1.0,
                                str_x_axis=r'\textrm{Iteration}',
                                str_y_axis=r'\textrm{Minimum regret}')

