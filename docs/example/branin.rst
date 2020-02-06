Optimizing Branin Function
==========================

This example is for simple Gaussian process regression.

.. code-block:: python

    import numpy as np
    import os

    from bayeso import bo
    from benchmarks.two_dim_branin import Branin
    from bayeso.utils import utils_bo
    from bayeso.utils import utils_plotting

.. code-block:: python

    obj_fun = Branin()
    bounds = obj_fun.get_bounds()

    def fun_target(X):
        return obj_fun.output(X)

.. code-block:: python

    str_fun = 'branin'

    int_bo = 10
    int_iter = 50
    int_init = 3

.. code-block:: python

    model_bo = bo.BO(bounds, debug=False)
    list_Y = []
    list_time = []

    for ind_bo in range(0, int_bo):
        print('BO Iteration', ind_bo + 1)
        X_final, Y_final, time_final, _, _ = utils_bo.optimize_many_with_random_init(
            model_bo, fun_target, int_init, int_iter,
            str_initial_method_bo='uniform', str_initial_method_ao='uniform', int_samples_ao=100,
            int_seed=42 * ind_bo
        )
        list_Y.append(Y_final)
        list_time.append(time_final)

    arr_Y = np.array(list_Y)
    arr_time = np.array(list_time)

    arr_Y = np.expand_dims(np.squeeze(arr_Y), axis=0)
    arr_time = np.expand_dims(arr_time, axis=0)

.. code-block:: python

    utils_plotting.plot_minimum(arr_Y, [str_fun], int_init, True,
        is_tex=True,
        str_x_axis=r'\textrm{Iteration}',
        str_y_axis=r'\textrm{Mininum function value}')
    utils_plotting.plot_minimum_time(arr_time, arr_Y, [str_fun], int_init, True,
        is_tex=True,
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
    import os

    from bayeso import bo
    from benchmarks.two_dim_branin import Branin
    from bayeso.utils import utils_bo
    from bayeso.utils import utils_plotting

    obj_fun = Branin()
    bounds = obj_fun.get_bounds()

    def fun_target(X):
        return obj_fun.output(X)

    str_fun = 'branin'

    int_bo = 10
    int_iter = 50
    int_init = 3

    model_bo = bo.BO(bounds, debug=False)
    list_Y = []
    list_time = []

    for ind_bo in range(0, int_bo):
        print('BO Iteration', ind_bo + 1)
        X_final, Y_final, time_final, _, _ = utils_bo.optimize_many_with_random_init(
            model_bo, fun_target, int_init, int_iter,
            str_initial_method_bo='uniform', str_initial_method_ao='uniform', int_samples_ao=100,
            int_seed=42 * ind_bo
        )
        list_Y.append(Y_final)
        list_time.append(time_final)

    arr_Y = np.array(list_Y)
    arr_time = np.array(list_time)

    arr_Y = np.expand_dims(np.squeeze(arr_Y), axis=0)
    arr_time = np.expand_dims(arr_time, axis=0)

    utils_plotting.plot_minimum(arr_Y, [str_fun], int_init, True,
        is_tex=True,
        str_x_axis=r'\textrm{Iteration}',
        str_y_axis=r'\textrm{Mininum function value}')
    utils_plotting.plot_minimum_time(arr_time, arr_Y, [str_fun], int_init, True,
        is_tex=True,
        str_x_axis=r'\textrm{Time (sec.)}',
        str_y_axis=r'\textrm{Mininum function value}')

