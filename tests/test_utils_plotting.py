# test_utils_plotting
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: July 10, 2018

import pytest
import numpy as np

from bayeso.utils import utils_plotting


TEST_EPSILON = 1e-5

def test_plot_gp():
    dim_X = 1
    dim_Y = 1
    num_train = 5
    num_test = 10

    X_train = np.zeros((num_train, dim_X))
    Y_train = np.ones((num_train, dim_Y))
    X_test = np.zeros((num_test, dim_X))
    mu = np.zeros((num_test, dim_Y))
    sigma = np.zeros((num_test, dim_Y))
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, Y_train, X_test, mu, 1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, Y_train, X_test, mu, np.arange(0, num_test))
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, Y_train, X_test, 1, sigma)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, Y_train, X_test, np.arange(0, num_test), sigma)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, Y_train, 1, mu, sigma)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, Y_train, np.arange(0, num_test), mu, sigma)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, 1, X_test, mu, sigma)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, np.arange(0, num_train), X_test, mu, sigma)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(1, Y_train, X_test, mu, sigma)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(np.arange(0, num_test), Y_train, X_test, mu, sigma)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(np.zeros((num_train, 2)), Y_train, X_test, mu, sigma)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, Y_train, np.zeros((num_train, 2)), mu, sigma)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, np.ones((num_train, 2)), X_test, mu, sigma)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, np.ones((10, 1)), X_test, mu, sigma)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, Y_train, X_test, np.zeros((num_test, 2)), sigma)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, Y_train, X_test, mu, np.zeros((num_test, 2)))
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, Y_train, X_test, mu, np.zeros((11, 1)))
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, Y_train, X_test, np.zeros((11, 1)), sigma)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, Y_train, X_test, mu, sigma, Y_test_truth=np.arange(0, num_test))
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, Y_train, X_test, mu, sigma, Y_test_truth=np.zeros((num_test, 2)))
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, Y_train, X_test, mu, sigma, Y_test_truth=np.zeros((20, 1)))
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, Y_train, X_test, mu, sigma, Y_test_truth=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, Y_train, X_test, mu, sigma, path_save=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, Y_train, X_test, mu, sigma, str_postfix=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, Y_train, X_test, mu, sigma, str_x_axis=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, Y_train, X_test, mu, sigma, str_y_axis=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, Y_train, X_test, mu, sigma, is_tex=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, Y_train, X_test, mu, sigma, is_zero_axis=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, Y_train, X_test, mu, sigma, time_pause='abc')
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, Y_train, X_test, mu, sigma, range_shade='abc')
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_gp(X_train, Y_train, X_test, mu, sigma, colors='abc')

def test_plot_minimum():
    num_model = 2
    num_bo = 3
    num_iter = 10
    arr_minima = np.ones((num_model, num_bo, num_iter))
    list_str_label = ['abc', 'def']
    int_init = 3
    is_std = True

    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum(arr_minima, list_str_label, int_init, 1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum(arr_minima, list_str_label, 'abc', is_std)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum(arr_minima, 1, int_init, is_std)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum(1, list_str_label, int_init, is_std)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum(np.ones((10, 2)), list_str_label, int_init, is_std)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum(np.ones(2), list_str_label, int_init, is_std)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum(arr_minima, ['abc', 'def', 'ghi'], int_init, is_std)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum(arr_minima, list_str_label, 12, is_std)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum(arr_minima, list_str_label, int_init, is_std, is_marker=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum(arr_minima, list_str_label, int_init, is_std, is_legend=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum(arr_minima, list_str_label, int_init, is_std, is_tex=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum(arr_minima, list_str_label, int_init, is_std, path_save=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum(arr_minima, list_str_label, int_init, is_std, str_postfix=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum(arr_minima, list_str_label, int_init, is_std, str_x_axis=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum(arr_minima, list_str_label, int_init, is_std, str_y_axis=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum(arr_minima, list_str_label, int_init, is_std, time_pause='abc')
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum(arr_minima, list_str_label, int_init, is_std, range_shade='abc')
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum(arr_minima, list_str_label, int_init, is_std, markers='abc')
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum(arr_minima, list_str_label, int_init, is_std, colors='abc')

def test_plot_minimum_time():
    num_model = 2
    num_bo = 3
    num_iter = 10
    arr_times = np.ones((num_model, num_bo, num_iter))
    arr_minima = np.ones((num_model, num_bo, num_iter))
    list_str_label = ['abc', 'def']
    int_init = 3
    is_std = True

    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum_time(arr_times, arr_minima, list_str_label, int_init, 1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum_time(arr_times, arr_minima, list_str_label, 'abc', is_std)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum_time(arr_times, arr_minima, 1, int_init, is_std)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum_time(arr_times, 1, list_str_label, int_init, is_std)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum_time(1, arr_minima, list_str_label, int_init, is_std)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum_time(np.ones((4, num_bo, num_iter)), arr_minima, list_str_label, int_init, is_std)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum_time(np.ones((num_model, 4, num_iter)), arr_minima, list_str_label, int_init, is_std)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum_time(np.ones((num_model, num_bo, 25)), arr_minima, list_str_label, int_init, is_std)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum_time(np.ones((num_bo, num_iter)), arr_minima, list_str_label, int_init, is_std)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum_time(np.ones(num_iter), arr_minima, list_str_label, int_init, is_std)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum_time(arr_times, np.ones((10, 2)), list_str_label, int_init, is_std)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum_time(arr_times, np.ones(2), list_str_label, int_init, is_std)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum_time(arr_times, arr_minima, ['abc', 'def', 'ghi'], int_init, is_std)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum_time(arr_times, arr_minima, list_str_label, 12, is_std)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum_time(arr_times, arr_minima, list_str_label, int_init, is_std, is_marker=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum_time(arr_times, arr_minima, list_str_label, int_init, is_std, is_legend=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum_time(arr_times, arr_minima, list_str_label, int_init, is_std, is_tex=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum_time(arr_times, arr_minima, list_str_label, int_init, is_std, path_save=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum_time(arr_times, arr_minima, list_str_label, int_init, is_std, str_postfix=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum_time(arr_times, arr_minima, list_str_label, int_init, is_std, str_x_axis=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum_time(arr_times, arr_minima, list_str_label, int_init, is_std, str_y_axis=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum_time(arr_times, arr_minima, list_str_label, int_init, is_std, time_pause='abc')
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum_time(arr_times, arr_minima, list_str_label, int_init, is_std, range_shade='abc')
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum_time(arr_times, arr_minima, list_str_label, int_init, is_std, markers='abc')
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_minimum_time(arr_times, arr_minima, list_str_label, int_init, is_std, colors='abc')

def test_plot_bo_step():
    num_dim_X = 1
    num_dim_Y = 1
    num_train = 5
    num_test = 10
    X_train = np.ones((num_train, num_dim_X))
    Y_train = np.ones((num_train, num_dim_Y))
    X_test = np.ones((num_test, num_dim_X))
    Y_test = np.ones((num_test, num_dim_Y))
    mean_test = np.ones((num_test, num_dim_Y))
    std_test = np.ones((num_test, num_dim_Y))

    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, 1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, Y_train, X_test, Y_test, 1, std_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, Y_train, X_test, 1, mean_test, std_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, Y_train, 1, Y_test, mean_test, std_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, 1, X_test, Y_test, mean_test, std_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(1, Y_train, X_test, Y_test, mean_test, std_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(np.arange(0, 10), Y_train, X_test, Y_test, mean_test, std_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, np.arange(0, 10), X_test, Y_test, mean_test, std_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, Y_train, np.arange(0, 10), Y_test, mean_test, std_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, Y_train, X_test, np.arange(0, 10), mean_test, std_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, Y_train, X_test, Y_test, np.arange(0, 10), std_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, np.arange(0, 10))
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(np.ones((num_train, 2)), Y_train, X_test, Y_test, mean_test, std_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, Y_train, np.ones((num_test, 2)), Y_test, mean_test, std_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, np.ones((num_train, 2)), X_test, Y_test, mean_test, std_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(np.ones((30, num_dim_X)), Y_train, X_test, Y_test, mean_test, std_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, Y_train, X_test, Y_test, np.ones((num_test, 2)), std_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, np.ones((num_test, 2)))
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, Y_train, np.ones((30, num_dim_X)), Y_test, mean_test, std_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, np.ones((30, num_dim_Y)))
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, std_test, int_init=20)

    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, std_test, path_save=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, std_test, str_postfix=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, std_test, str_x_axis=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, std_test, str_y_axis=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, std_test, int_init='abc')
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, std_test, is_tex=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, std_test, is_zero_axis=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, std_test, time_pause='abc')
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, std_test, range_shade='abc')

def test_plot_bo_step_acq():
    num_dim_X = 1
    num_dim_Y = 1
    num_train = 5
    num_test = 10
    X_train = np.ones((num_train, num_dim_X))
    Y_train = np.ones((num_train, num_dim_Y))
    X_test = np.ones((num_test, num_dim_X))
    Y_test = np.ones((num_test, num_dim_Y))
    mean_test = np.ones((num_test, num_dim_Y))
    std_test = np.ones((num_test, num_dim_Y))
    acq_test = np.ones((num_test, num_dim_Y))

    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(np.arange(0, 10), Y_train, X_test, Y_test, mean_test, std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, np.arange(0, 10), X_test, Y_test, mean_test, std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, np.arange(0, 10), Y_test, mean_test, std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, X_test, np.arange(0, 10), mean_test, std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, X_test, Y_test, np.arange(0, 10), std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, X_test, Y_test, mean_test, np.arange(0, 10), acq_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, np.arange(0, 10))
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(np.ones((num_train, 2)), Y_train, X_test, Y_test, mean_test, std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, np.ones((num_test, 2)), Y_test, mean_test, std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, np.ones((num_train, 2)), X_test, Y_test, mean_test, std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, X_test, np.ones((num_test, 2)), mean_test, std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, np.ones((num_test, 2)), Y_test, mean_test, std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(np.ones((30, num_dim_X)), Y_train, X_test, Y_test, mean_test, std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, np.ones((30, num_dim_Y)), X_test, Y_test, mean_test, std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, X_test, Y_test, np.ones((num_test, 2)), std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, X_test, Y_test, mean_test, np.ones((num_test, 2)), acq_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, np.ones((num_test, 2)))
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, np.ones((30, num_dim_X)), Y_test, mean_test, std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, np.ones((30, num_dim_Y)))
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, X_test, Y_test, mean_test, np.ones((30, num_dim_Y)), acq_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, X_test, Y_test, np.ones((30, num_dim_Y)), std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test, int_init=30)

    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test, path_save=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test, str_postfix=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test, str_x_axis=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test, str_y_axis=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test, str_acq_axis=1)
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test, int_init='abc')
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test, is_tex='abc')
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test, is_zero_axis='abc')
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test, is_acq_axis_small='abc')
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test, time_pause='abc')
    with pytest.raises(AssertionError) as error:
        utils_plotting.plot_bo_step_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test, range_shade='abc')

