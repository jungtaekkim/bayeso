Constructing xgboost Classifier with Hyperparameter Optimization 
================================================================

This example is for optimizing hyperparameters for **xgboost** classifier.
In this example, we optimize `max_depth` and `n_estimators` for `xgboost.XGBClassifier`.
It needs to install **xgboost**, which is included in **requirements-examples.txt**.
First, import some packages we need.

.. code-block:: python

    import numpy as np
    import xgboost as xgb
    import sklearn.datasets
    import sklearn.metrics
    import sklearn.model_selection

    from bayeso import bo
    from bayeso.wrappers import wrappers_bo
    from bayeso.utils import utils_plotting

Get handwritten digits dataset, which contains digit images of 0 to 9,
and split the dataset to training and test datasets.

.. code-block:: python

    digits = sklearn.datasets.load_digits()
    data_digits = digits.images
    data_digits = np.reshape(data_digits,
        (data_digits.shape[0], data_digits.shape[1] * data_digits.shape[2]))
    labels_digits = digits.target

    data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(
        data_digits, labels_digits, test_size=0.3, stratify=labels_digits)

Declare an objective function we would like to optimize.
This function trains `xgboost.XGBClassifier` with the training dataset and given hyerparameter vector `bx` and returns (1 - accuracy), which computed by the test dataset.

.. code-block:: python

    def fun_target(bx):
        model_xgb = xgb.XGBClassifier(
            max_depth=int(bx[0]),
            n_estimators=int(bx[1])
        )
        model_xgb.fit(data_train, labels_train)
        preds_test = model_xgb.predict(data_test)
        return 1.0 - sklearn.metrics.accuracy_score(labels_test, preds_test)

We optimize the objective function with our `bayeso.bo.BO` for 50 iterations.
5 initial points would be given and 10 rounds would be run.

.. code-block:: python

    str_fun = 'xgboost'

    # (max_depth, n_estimators)
    bounds = np.array([[1, 10], [100, 500]])
    num_bo = 10
    num_iter = 50
    num_init = 5

Optimze the objective function, after declaring the `bayeso.bo.BO` object.

.. code-block:: python

    model_bo = bo.BO(bounds, debug=False)

    list_Y = []
    list_time = []

    for ind_bo in range(0, num_bo):
        print('BO Round:', ind_bo + 1)
        X_final, Y_final, time_final, _, _ = wrappers_bo.run_single_round(
            model_bo, fun_target, num_init, num_iter,
            str_initial_method_bo='uniform', str_sampling_method_ao='uniform',
            num_samples_ao=100, seed=42 * ind_bo)
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
        str_y_axis=r'$1 - $\textrm{Accuracy}')
    utils_plotting.plot_minimum_vs_time(arr_time, arr_Y, [str_fun], num_init, True,
        use_tex=True,
        str_x_axis=r'\textrm{Time (sec.)}',
        str_y_axis=r'$1 - $\textrm{Accuracy}')

.. image:: ../_static/examples/hpo_func_xgboost.*
    :width: 320
    :align: center
    :alt: hpo_func_xgboost

.. image:: ../_static/examples/hpo_time_xgboost.*
    :width: 320
    :align: center
    :alt: hpo_time_xgboost

Full code:

.. code-block:: python

    import numpy as np
    import xgboost as xgb
    import sklearn.datasets
    import sklearn.metrics
    import sklearn.model_selection

    from bayeso import bo
    from bayeso.wrappers import wrappers_bo
    from bayeso.utils import utils_plotting

    digits = sklearn.datasets.load_digits()
    data_digits = digits.images
    data_digits = np.reshape(data_digits,
        (data_digits.shape[0], data_digits.shape[1] * data_digits.shape[2]))
    labels_digits = digits.target

    data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(
        data_digits, labels_digits, test_size=0.3, stratify=labels_digits)

    def fun_target(bx):
        model_xgb = xgb.XGBClassifier(
            max_depth=int(bx[0]),
            n_estimators=int(bx[1])
        )
        model_xgb.fit(data_train, labels_train)
        preds_test = model_xgb.predict(data_test)
        return 1.0 - sklearn.metrics.accuracy_score(labels_test, preds_test)

    str_fun = 'xgboost'

    # (max_depth, n_estimators)
    bounds = np.array([[1, 10], [100, 500]])
    num_bo = 10
    num_iter = 50
    num_init = 5

    model_bo = bo.BO(bounds, debug=False)

    list_Y = []
    list_time = []

    for ind_bo in range(0, num_bo):
        print('BO Round:', ind_bo + 1)
        X_final, Y_final, time_final, _, _ = wrappers_bo.run_single_round(
            model_bo, fun_target, num_init, num_iter,
            str_initial_method_bo='uniform', str_sampling_method_ao='uniform',
            num_samples_ao=100, seed=42 * ind_bo)
        list_Y.append(Y_final)
        list_time.append(time_final)

    arr_Y = np.array(list_Y)
    arr_time = np.array(list_time)

    arr_Y = np.expand_dims(np.squeeze(arr_Y), axis=0)
    arr_time = np.expand_dims(arr_time, axis=0)

    utils_plotting.plot_minimum_vs_iter(arr_Y, [str_fun], num_init, True,
        use_tex=True,
        str_x_axis=r'\textrm{Iteration}',
        str_y_axis=r'$1 - $\textrm{Accuracy}')
    utils_plotting.plot_minimum_vs_time(arr_time, arr_Y, [str_fun], num_init, True,
        use_tex=True,
        str_x_axis=r'\textrm{Time (sec.)}',
        str_y_axis=r'$1 - $\textrm{Accuracy}')

