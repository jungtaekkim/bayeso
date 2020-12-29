#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: September 24, 2020
#
"""test_utils_plotting"""

import typing
import pytest
import numpy as np

from bayeso.utils import utils_plotting as package_target


TEST_EPSILON = 1e-5

def test_plot_gp_via_sample_typing():
    annos = package_target.plot_gp_via_sample.__annotations__

    assert annos['X'] == np.ndarray
    assert annos['Ys'] == np.ndarray
    assert annos['path_save'] == typing.Union[str, type(None)]
    assert annos['str_postfix'] == typing.Union[str, type(None)]
    assert annos['str_x_axis'] == str
    assert annos['str_y_axis'] == str
    assert annos['use_tex'] == bool
    assert annos['draw_zero_axis'] == bool
    assert annos['pause_figure'] == bool
    assert annos['time_pause'] == typing.Union[int, float]
    assert annos['colors'] == list
    assert annos['return'] == type(None)

def test_plot_gp_via_sample():
    dim_X = 1
    num_Ys = 10
    num_train = 50

    X = np.zeros((num_train, dim_X))
    Ys = np.ones((num_train, num_Ys))

    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_sample('abc', Ys)
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_sample(X, 'abc')
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_sample(np.zeros(num_train), Ys)
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_sample(X, np.ones(num_train))
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_sample(X, np.ones((10, num_Ys)))

    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_sample(X, Ys, path_save=123)
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_sample(X, Ys, str_postfix=123)
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_sample(X, Ys, str_x_axis=123)
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_sample(X, Ys, str_y_axis=123)
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_sample(X, Ys, use_tex='abc')

    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_sample(X, Ys, draw_zero_axis='abc')
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_sample(X, Ys, pause_figure='abc')
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_sample(X, Ys, time_pause='abc')
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_sample(X, Ys, colors='abc')

def test_plot_gp_via_distribution_typing():
    annos = package_target.plot_gp_via_distribution.__annotations__

    assert annos['X_train'] == np.ndarray
    assert annos['Y_train'] == np.ndarray
    assert annos['X_test'] == np.ndarray
    assert annos['mean_test'] == np.ndarray
    assert annos['std_test'] == np.ndarray
    assert annos['Y_test'] == typing.Union[np.ndarray, type(None)]
    assert annos['path_save'] == typing.Union[str, type(None)]
    assert annos['str_postfix'] == typing.Union[str, type(None)]
    assert annos['str_x_axis'] == str
    assert annos['str_y_axis'] == str
    assert annos['use_tex'] == bool
    assert annos['draw_zero_axis'] == bool
    assert annos['pause_figure'] == bool
    assert annos['time_pause'] == typing.Union[int, float]
    assert annos['range_shade'] == float
    assert annos['colors'] == list
    assert annos['return'] == type(None)

def test_plot_gp_via_distribution():
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
        package_target.plot_gp_via_distribution(X_train, Y_train, X_test, mu, 1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, Y_train, X_test, mu, np.arange(0, num_test))
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, Y_train, X_test, 1, sigma)
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, Y_train, X_test, np.arange(0, num_test), sigma)
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, Y_train, 1, mu, sigma)

    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, Y_train, np.arange(0, num_test), mu, sigma)
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, 1, X_test, mu, sigma)
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, np.arange(0, num_train), X_test, mu, sigma)
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(1, Y_train, X_test, mu, sigma)
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(np.arange(0, num_test), Y_train, X_test, mu, sigma)

    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(np.zeros((num_train, 2)), Y_train, X_test, mu, sigma)
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, Y_train, np.zeros((num_train, 2)), mu, sigma)
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, np.ones((num_train, 2)), X_test, mu, sigma)
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, np.ones((10, 1)), X_test, mu, sigma)
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, Y_train, X_test, np.zeros((num_test, 2)), sigma)

    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, Y_train, X_test, mu, np.zeros((num_test, 2)))
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, Y_train, X_test, mu, np.zeros((11, 1)))
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, Y_train, X_test, np.zeros((11, 1)), sigma)
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, Y_train, X_test, mu, sigma, Y_test=np.arange(0, num_test))
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, Y_train, X_test, mu, sigma, Y_test=np.zeros((num_test, 2)))

    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, Y_train, X_test, mu, sigma, Y_test=np.zeros((20, 1)))
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, Y_train, X_test, mu, sigma, Y_test=1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, Y_train, X_test, mu, sigma, path_save=1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, Y_train, X_test, mu, sigma, str_postfix=1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, Y_train, X_test, mu, sigma, str_x_axis=1)

    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, Y_train, X_test, mu, sigma, str_y_axis=1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, Y_train, X_test, mu, sigma, use_tex=1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, Y_train, X_test, mu, sigma, draw_zero_axis=1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, Y_train, X_test, mu, sigma, pause_figure='abc')
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, Y_train, X_test, mu, sigma, time_pause='abc')

    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, Y_train, X_test, mu, sigma, range_shade='abc')
    with pytest.raises(AssertionError) as error:
        package_target.plot_gp_via_distribution(X_train, Y_train, X_test, mu, sigma, colors='abc')

def test_plot_minimum_vs_iter_typing():
    annos = package_target.plot_minimum_vs_iter.__annotations__

    assert annos['minima'] == np.ndarray
    assert annos['list_str_label'] == list
    assert annos['num_init'] == int
    assert annos['draw_std'] == bool
    assert annos['include_marker'] == bool
    assert annos['include_legend'] == bool
    assert annos['use_tex'] == bool
    assert annos['path_save'] == typing.Union[str, type(None)]
    assert annos['str_postfix'] == typing.Union[str, type(None)]
    assert annos['str_x_axis'] == str
    assert annos['str_y_axis'] == str
    assert annos['pause_figure'] == bool
    assert annos['time_pause'] == typing.Union[int, float]
    assert annos['range_shade'] == float
    assert annos['markers'] == list
    assert annos['colors'] == list
    assert annos['return'] == type(None)

def test_plot_minimum_vs_iter():
    num_model = 2
    num_bo = 3
    num_iter = 10
    arr_minima = np.ones((num_model, num_bo, num_iter))
    list_str_label = ['abc', 'def']
    num_init = 3
    draw_std = True

    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_iter(arr_minima, list_str_label, num_init, 1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_iter(arr_minima, list_str_label, 'abc', draw_std)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_iter(arr_minima, 1, num_init, draw_std)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_iter(1, list_str_label, num_init, draw_std)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_iter(np.ones((10, 2)), list_str_label, num_init, draw_std)

    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_iter(np.ones(2), list_str_label, num_init, draw_std)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_iter(arr_minima, ['abc', 'def', 'ghi'], num_init, draw_std)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_iter(arr_minima, list_str_label, 12, draw_std)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_iter(arr_minima, list_str_label, num_init, draw_std, include_marker=1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_iter(arr_minima, list_str_label, num_init, draw_std, include_legend=1)

    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_iter(arr_minima, list_str_label, num_init, draw_std, use_tex=1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_iter(arr_minima, list_str_label, num_init, draw_std, path_save=1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_iter(arr_minima, list_str_label, num_init, draw_std, str_postfix=1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_iter(arr_minima, list_str_label, num_init, draw_std, str_x_axis=1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_iter(arr_minima, list_str_label, num_init, draw_std, str_y_axis=1)

    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_iter(arr_minima, list_str_label, num_init, draw_std, pause_figure='abc')
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_iter(arr_minima, list_str_label, num_init, draw_std, time_pause='abc')
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_iter(arr_minima, list_str_label, num_init, draw_std, range_shade='abc')
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_iter(arr_minima, list_str_label, num_init, draw_std, markers='abc')
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_iter(arr_minima, list_str_label, num_init, draw_std, colors='abc')

def test_plot_minimum_vs_time_typing():
    annos = package_target.plot_minimum_vs_time.__annotations__

    assert annos['times'] == np.ndarray
    assert annos['minima'] == np.ndarray
    assert annos['list_str_label'] == list
    assert annos['num_init'] == int
    assert annos['draw_std'] == bool
    assert annos['include_marker'] == bool
    assert annos['include_legend'] == bool
    assert annos['use_tex'] == bool
    assert annos['path_save'] == typing.Union[str, type(None)]
    assert annos['str_postfix'] == typing.Union[str, type(None)]
    assert annos['str_x_axis'] == str
    assert annos['str_y_axis'] == str
    assert annos['pause_figure'] == bool
    assert annos['time_pause'] == typing.Union[int, float]
    assert annos['range_shade'] == float
    assert annos['markers'] == list
    assert annos['colors'] == list
    assert annos['return'] == type(None)

def test_plot_minimum_vs_time():
    num_model = 2
    num_bo = 3
    num_iter = 10
    arr_times = np.ones((num_model, num_bo, num_iter))
    arr_minima = np.ones((num_model, num_bo, num_iter))
    list_str_label = ['abc', 'def']
    num_init = 3
    draw_std = True

    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(arr_times, arr_minima, list_str_label, num_init, 1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(arr_times, arr_minima, list_str_label, 'abc', draw_std)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(arr_times, arr_minima, 1, num_init, draw_std)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(arr_times, 1, list_str_label, num_init, draw_std)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(1, arr_minima, list_str_label, num_init, draw_std)

    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(np.ones((4, num_bo, num_iter)), arr_minima, list_str_label, num_init, draw_std)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(np.ones((num_model, 4, num_iter)), arr_minima, list_str_label, num_init, draw_std)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(np.ones((num_model, num_bo, 25)), arr_minima, list_str_label, num_init, draw_std)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(np.ones((num_bo, num_iter)), arr_minima, list_str_label, num_init, draw_std)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(np.ones(num_iter), arr_minima, list_str_label, num_init, draw_std)

    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(arr_times, np.ones((10, 2)), list_str_label, num_init, draw_std)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(arr_times, np.ones(2), list_str_label, num_init, draw_std)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(arr_times, arr_minima, ['abc', 'def', 'ghi'], num_init, draw_std)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(arr_times, arr_minima, list_str_label, 12, draw_std)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(arr_times, arr_minima, list_str_label, num_init, draw_std, include_marker=1)

    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(arr_times, arr_minima, list_str_label, num_init, draw_std, include_legend=1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(arr_times, arr_minima, list_str_label, num_init, draw_std, use_tex=1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(arr_times, arr_minima, list_str_label, num_init, draw_std, path_save=1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(arr_times, arr_minima, list_str_label, num_init, draw_std, str_postfix=1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(arr_times, arr_minima, list_str_label, num_init, draw_std, str_x_axis=1)

    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(arr_times, arr_minima, list_str_label, num_init, draw_std, str_y_axis=1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(arr_times, arr_minima, list_str_label, num_init, draw_std, pause_figure='abc')
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(arr_times, arr_minima, list_str_label, num_init, draw_std, time_pause='abc')
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(arr_times, arr_minima, list_str_label, num_init, draw_std, range_shade='abc')
    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(arr_times, arr_minima, list_str_label, num_init, draw_std, markers='abc')

    with pytest.raises(AssertionError) as error:
        package_target.plot_minimum_vs_time(arr_times, arr_minima, list_str_label, num_init, draw_std, colors='abc')

def test_plot_bo_step_typing():
    annos = package_target.plot_bo_step.__annotations__

    assert annos['X_train'] == np.ndarray
    assert annos['Y_train'] == np.ndarray
    assert annos['X_test'] == np.ndarray
    assert annos['Y_test'] == np.ndarray
    assert annos['mean_test'] == np.ndarray
    assert annos['std_test'] == np.ndarray
    assert annos['path_save'] == typing.Union[str, type(None)]
    assert annos['str_postfix'] == typing.Union[str, type(None)]
    assert annos['str_x_axis'] == str
    assert annos['str_y_axis'] == str
    assert annos['num_init'] == typing.Union[int, type(None)]
    assert annos['use_tex'] == bool
    assert annos['draw_zero_axis'] == bool
    assert annos['pause_figure'] == bool
    assert annos['time_pause'] == typing.Union[int, float]
    assert annos['range_shade'] == float
    assert annos['return'] == type(None)

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
        package_target.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, 1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, Y_train, X_test, Y_test, 1, std_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, Y_train, X_test, 1, mean_test, std_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, Y_train, 1, Y_test, mean_test, std_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, 1, X_test, Y_test, mean_test, std_test)

    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(1, Y_train, X_test, Y_test, mean_test, std_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(np.arange(0, 10), Y_train, X_test, Y_test, mean_test, std_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, np.arange(0, 10), X_test, Y_test, mean_test, std_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, Y_train, np.arange(0, 10), Y_test, mean_test, std_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, Y_train, X_test, np.arange(0, 10), mean_test, std_test)

    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, Y_train, X_test, Y_test, np.arange(0, 10), std_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, np.arange(0, 10))
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(np.ones((num_train, 2)), Y_train, X_test, Y_test, mean_test, std_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, Y_train, np.ones((num_test, 2)), Y_test, mean_test, std_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, np.ones((num_train, 2)), X_test, Y_test, mean_test, std_test)

    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(np.ones((30, num_dim_X)), Y_train, X_test, Y_test, mean_test, std_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, Y_train, X_test, Y_test, np.ones((num_test, 2)), std_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, np.ones((num_test, 2)))
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, Y_train, np.ones((30, num_dim_X)), Y_test, mean_test, std_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, np.ones((30, num_dim_Y)))

    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, std_test, num_init=20)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, std_test, path_save=1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, std_test, str_postfix=1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, std_test, str_x_axis=1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, std_test, str_y_axis=1)

    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, std_test, num_init='abc')
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, std_test, use_tex=1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, std_test, draw_zero_axis=1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, std_test, pause_figure='abc')
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, std_test, time_pause='abc')

    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, std_test, range_shade='abc')

def test_plot_bo_step_with_acq_typing():
    annos = package_target.plot_bo_step_with_acq.__annotations__

    assert annos['X_train'] == np.ndarray
    assert annos['Y_train'] == np.ndarray
    assert annos['X_test'] == np.ndarray
    assert annos['Y_test'] == np.ndarray
    assert annos['mean_test'] == np.ndarray
    assert annos['std_test'] == np.ndarray
    assert annos['acq_test'] == np.ndarray
    assert annos['path_save'] == typing.Union[str, type(None)]
    assert annos['str_postfix'] == typing.Union[str, type(None)]
    assert annos['str_x_axis'] == str
    assert annos['str_y_axis'] == str
    assert annos['str_acq_axis'] == str
    assert annos['num_init'] == typing.Union[int, type(None)]
    assert annos['use_tex'] == bool
    assert annos['draw_zero_axis'] == bool
    assert annos['pause_figure'] == bool
    assert annos['time_pause'] == typing.Union[int, float]
    assert annos['range_shade'] == float
    assert annos['return'] == type(None)

def test_plot_bo_step_with_acq():
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
        package_target.plot_bo_step_with_acq(np.arange(0, 10), Y_train, X_test, Y_test, mean_test, std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, np.arange(0, 10), X_test, Y_test, mean_test, std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, np.arange(0, 10), Y_test, mean_test, std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, X_test, np.arange(0, 10), mean_test, std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, X_test, Y_test, np.arange(0, 10), std_test, acq_test)

    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, X_test, Y_test, mean_test, np.arange(0, 10), acq_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, np.arange(0, 10))
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(np.ones((num_train, 2)), Y_train, X_test, Y_test, mean_test, std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, np.ones((num_test, 2)), Y_test, mean_test, std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, np.ones((num_train, 2)), X_test, Y_test, mean_test, std_test, acq_test)

    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, X_test, np.ones((num_test, 2)), mean_test, std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, np.ones((num_test, 2)), Y_test, mean_test, std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(np.ones((30, num_dim_X)), Y_train, X_test, Y_test, mean_test, std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, np.ones((30, num_dim_Y)), X_test, Y_test, mean_test, std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, X_test, Y_test, np.ones((num_test, 2)), std_test, acq_test)

    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, X_test, Y_test, mean_test, np.ones((num_test, 2)), acq_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, np.ones((num_test, 2)))
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, np.ones((30, num_dim_X)), Y_test, mean_test, std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, np.ones((30, num_dim_Y)))
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, X_test, Y_test, mean_test, np.ones((30, num_dim_Y)), acq_test)

    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, X_test, Y_test, np.ones((30, num_dim_Y)), std_test, acq_test)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test, num_init=30)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test, path_save=1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test, str_postfix=1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test, str_x_axis=1)

    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test, str_y_axis=1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test, str_acq_axis=1)
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test, num_init='abc')
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test, use_tex='abc')
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test, draw_zero_axis='abc')

    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test, pause_figure='abc')
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test, time_pause='abc')
    with pytest.raises(AssertionError) as error:
        package_target.plot_bo_step_with_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test, range_shade='abc')
