Optimizing Branin Function
==========================

This example is for optimizing Branin function.
It needs to install **bayeso-benchmarks**, which is included in **requirements-optional.txt**.
First, import some packages we need.

.. code-block:: python

    import numpy as np

    from bayeso import bo
    from bayeso_benchmarks.two_dim_branin import Branin
    from bayeso.wrappers import wrappers_bo
    from bayeso.utils import utils_plotting

Then, declare Branin function we will optimize and a search space for the function.

.. code-block:: python

    obj_fun = Branin()
    bounds = obj_fun.get_bounds()

    def fun_target(X):
        return obj_fun.output(X)

We optimize the objective function with 10 Bayesian optimization rounds and 50 iterations per round with 3 initial random evaluations.

.. code-block:: python

    str_fun = 'branin'

    num_bo = 10
    num_iter = 50
    num_init = 3

With BO class in `bayeso.bo`, optimize the objective function.

.. code-block:: python

    model_bo = bo.BO(bounds, debug=False)
    list_Y = []
    list_time = []

    for ind_bo in range(0, num_bo):
        print('BO Round', ind_bo + 1)
        X_final, Y_final, time_final, _, _ = wrappers_bo.run_single_round(
            model_bo, fun_target, num_init, num_iter,
            str_initial_method_bo='uniform', str_sampling_method_ao='uniform', num_samples_ao=100,
            seed=42 * ind_bo
        )
        list_Y.append(Y_final)
        list_time.append(time_final)

    arr_Y = np.array(list_Y)
    arr_time = np.array(list_time)

    arr_Y = np.expand_dims(np.squeeze(arr_Y), axis=0)
    arr_time = np.expand_dims(arr_time, axis=0)

Plot the results in terms of the number of iterations and time.

.. code-block:: python

    utils_plotting.plot_minimum_vs_iter(arr_Y, [str_fun], num_init, True,
        use_tex=True,
        str_x_axis=r'\textrm{Iteration}',
        str_y_axis=r'\textrm{Mininum function value}')
    utils_plotting.plot_minimum_vs_time(arr_time, arr_Y, [str_fun], num_init, True,
        use_tex=True,
        str_x_axis=r'\textrm{Time (sec.)}',
        str_y_axis=r'\textrm{Mininum function value}')

.. image:: ../_static/examples/bo_func_branin.*
    :width: 320
    :align: center
    :alt: bo_func_branin

.. image:: ../_static/examples/bo_time_branin.*
    :width: 320
    :align: center
    :alt: bo_time_branin

Full code:

.. code-block:: python

    import numpy as np

    from bayeso import bo
    from bayeso_benchmarks.two_dim_branin import Branin
    from bayeso.wrappers import wrappers_bo
    from bayeso.utils import utils_plotting

    obj_fun = Branin()
    bounds = obj_fun.get_bounds()

    def fun_target(X):
        return obj_fun.output(X)

    str_fun = 'branin'

    num_bo = 10
    num_iter = 50
    num_init = 3

    model_bo = bo.BO(bounds, debug=False)
    list_Y = []
    list_time = []

    for ind_bo in range(0, num_bo):
        print('BO Round', ind_bo + 1)
        X_final, Y_final, time_final, _, _ = wrappers_bo.run_single_round(
            model_bo, fun_target, num_init, num_iter,
            str_initial_method_bo='uniform', str_sampling_method_ao='uniform', num_samples_ao=100,
            seed=42 * ind_bo
        )
        list_Y.append(Y_final)
        list_time.append(time_final)

    arr_Y = np.array(list_Y)
    arr_time = np.array(list_time)

    arr_Y = np.expand_dims(np.squeeze(arr_Y), axis=0)
    arr_time = np.expand_dims(arr_time, axis=0)

    utils_plotting.plot_minimum_vs_iter(arr_Y, [str_fun], num_init, True,
        use_tex=True,
        str_x_axis=r'\textrm{Iteration}',
        str_y_axis=r'\textrm{Mininum function value}')
    utils_plotting.plot_minimum_vs_time(arr_time, arr_Y, [str_fun], num_init, True,
        use_tex=True,
        str_x_axis=r'\textrm{Time (sec.)}',
        str_y_axis=r'\textrm{Mininum function value}')

