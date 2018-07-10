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

