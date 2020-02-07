# utils_plotting
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 07, 2020

import os
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    plt = None
try:
    import pylab
except:
    pylab = None

from bayeso import constants
from bayeso.utils import utils_common


def _set_ax_config(ax, str_x_axis, str_y_axis,
    size_labels=32,
    size_ticks=22,
    xlim_min=None,
    xlim_max=None,
    is_box=True,
    is_zero_axis=False,
    is_grid=True,
): # pragma: no cover
    """
    It sets an axis configuration.

    :param ax: inputs for acquisition function. Shape: (n, d).
    :type ax: matplotlib.axes._subplots.AxesSubplot
    :param str_x_axis: the name of x axis.
    :type str_x_axis: str.
    :param str_y_axis: the name of y axis.
    :type str_y_axis: str.
    :param size_labels: label size.
    :type size_labels: int., optional
    :param size_ticks: tick size.
    :type size_ticks: int., optional
    :param xlim_min: None, or minimum for x limit.
    :type xlim_min: NoneType or float, optional
    :param xlim_max: None, or maximum for x limit.
    :type xlim_max: NoneType or float, optional
    :param is_box: flag for drawing a box.
    :type is_box: bool., optional
    :param is_zero_axis: flag for drawing a zero axis.
    :type is_zero_axis: bool., optional
    :param is_grid: flag for drawing grids.
    :type is_grid: bool., optional

    :returns: None.
    :rtype: NoneType

    """

    if str_x_axis is not None:
        ax.set_xlabel(str_x_axis, fontsize=size_labels)
    ax.set_ylabel(str_y_axis, fontsize=size_labels)
    ax.tick_params(labelsize=size_ticks)

    if not is_box:
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
    if xlim_min is not None and xlim_max is not None:
        ax.set_xlim([xlim_min, xlim_max])
    if is_zero_axis:
        ax.spines['bottom'].set_position('zero')
    if is_grid:
        ax.grid()
    return

def _save_figure(path_save, str_postfix, str_prefix=''): # pragma: no cover
    """
    It saves a figure.

    :param path_save: path for saving a figure.
    :type path_save: str.
    :param str_postfix: the name of postfix.
    :type str_postfix: str.
    :param str_prefix: the name of prefix.
    :type str_prefix: str., optional

    :returns: None.
    :rtype: NoneType

    """

    if path_save is not None and str_postfix is not None:
        str_figure = str_prefix + str_postfix
        plt.savefig(os.path.join(path_save, str_figure + '.pdf'),
            format='pdf',
            transparent=True,
            bbox_inches='tight'
        )
    return

def _show_figure(is_pause, time_pause): # pragma: no cover
    """
    It shows a figure.

    :param is_pause: flag for pausing before closing a figure.
    :type is_pause: bool.
    :param time_pause: pausing time.
    :type time_pause: float

    :returns: None.
    :rtype: NoneType

    """

    plt.ion()
    if is_pause:
        plt.pause(time_pause)
    plt.close('all')
    return

def plot_gp(X_train, Y_train, X_test, mu, sigma,
    Y_test_truth=None,
    path_save=None,
    str_postfix=None,
    str_x_axis='x',
    str_y_axis='y',
    is_tex=False,
    is_zero_axis=False,
    is_pause=True,
    time_pause=constants.TIME_PAUSE,
    range_shade=constants.RANGE_SHADE,
    colors=constants.COLORS,
): # pragma: no cover
    """
    It is for plotting Gaussian process regression.

    :param X_train: training inputs. Shape: (n, 1).
    :type X_train: numpy.ndarray
    :param Y_train: training outputs. Shape: (n, 1).
    :type Y_train: numpy.ndarray
    :param X_test: test inputs. Shape: (m, 1).
    :type X_test: numpy.ndarray
    :param mu: posterior predictive mean function values over `X_test`. Shape: (m, 1).
    :type mu: numpy.ndarray
    :param sigma: posterior predictive standard deviation function values over `X_test`. Shape: (m, 1).
    :type sigma: numpy.ndarray
    :param Y_test_truth: None, or true test outputs. Shape: (m, 1).
    :type Y_test_truth: NoneType or numpy.ndarray, optional
    :param path_save: None, or path for saving a figure.
    :type path_save: NoneType or str., optional
    :param str_postfix: None, or the name of postfix.
    :type str_postfix: NoneType or str., optional
    :param str_x_axis: the name of x axis.
    :type str_x_axis: str., optional
    :param str_y_axis: the name of y axis.
    :type str_y_axis: str., optional
    :param is_tex: flag for using latex.
    :type is_tex: bool., optional
    :param is_zero_axis: flag for drawing a zero axis.
    :type is_zero_axis: bool., optional
    :param is_pause: flag for pausing before closing a figure.
    :type is_pause: bool., optional
    :param time_pause: pausing time.
    :type time_pause: float, optional
    :param range_shade: shade range for standard deviation.
    :type range_shade: float, optional
    :param colors: list of colors.
    :type colors: list, optional

    :returns: None.
    :rtype: NoneType

    :raises: AssertionError

    """

    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(mu, np.ndarray)
    assert isinstance(sigma, np.ndarray)
    assert isinstance(Y_test_truth, np.ndarray) or Y_test_truth is None
    assert isinstance(path_save, str) or path_save is None
    assert isinstance(str_postfix, str) or str_postfix is None
    assert isinstance(str_x_axis, str)
    assert isinstance(str_y_axis, str)
    assert isinstance(is_tex, bool)
    assert isinstance(is_zero_axis, bool)
    assert isinstance(is_pause, bool)
    assert isinstance(time_pause, float)
    assert isinstance(range_shade, float)
    assert isinstance(colors, list)
    assert len(X_train.shape) == 2
    assert len(X_test.shape) == 2
    assert len(Y_train.shape) == 2
    assert len(mu.shape) == 2
    assert len(sigma.shape) == 2
    assert X_train.shape[1] == X_test.shape[1] == 1
    assert Y_train.shape[1] == 1
    assert X_train.shape[0] == Y_train.shape[0]
    assert mu.shape[1] == 1
    assert sigma.shape[1] == 1
    assert X_test.shape[0] == mu.shape[0] == sigma.shape[0]
    if Y_test_truth is not None:
        assert len(Y_test_truth.shape) == 2
        assert Y_test_truth.shape[1] == 1
        assert X_test.shape[0] == Y_test_truth.shape[0]

    if plt is None or pylab is None:
        return
    if is_tex:
        plt.rc('text', usetex=True)
    else:
        plt.rc('pdf', fonttype=42)

    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    if Y_test_truth is not None:
        ax.plot(X_test.flatten(), Y_test_truth.flatten(),
            c=colors[1],
            linewidth=4,
            marker='None')
    ax.plot(X_test.flatten(), mu.flatten(), 
        c=colors[2], 
        linewidth=4, 
        marker='None')
    ax.fill_between(X_test.flatten(), 
        mu.flatten() - range_shade * sigma.flatten(), 
        mu.flatten() + range_shade * sigma.flatten(), 
        color=colors[2], 
        alpha=0.3)
    ax.plot(X_train.flatten(), Y_train.flatten(), 
        'x', 
        c=colors[0], 
        markersize=10, 
        mew=4)

    _set_ax_config(ax, str_x_axis, str_y_axis, xlim_min=np.min(X_test), xlim_max=np.max(X_test), is_zero_axis=is_zero_axis)

    if path_save is not None and str_postfix is not None:
        _save_figure(path_save, str_postfix, str_prefix='gp_')
    _show_figure(is_pause, time_pause)
    return

def plot_minimum(arr_minima, list_str_label, int_init, is_std,
    is_marker=True,
    is_legend=False,
    is_tex=False,
    path_save=None,
    str_postfix=None,
    str_x_axis='Iteration',
    str_y_axis='Minimum function value',
    is_pause=True,
    time_pause=constants.TIME_PAUSE,
    range_shade=constants.RANGE_SHADE,
    markers=constants.MARKERS,
    colors=constants.COLORS,
): # pragma: no cover
    """
    It is for plotting optimization results of Bayesian optimization, in terms of iterations.

    :param arr_minima: function values over acquired examples. Shape: (b, r, n) where b is the number of experiments, r is the number of rounds, and n is the number of iterations per round.
    :type arr_minima: numpy.ndarray
    :param list_str_label: list of label strings. Shape: (b, ).
    :type list_str_label: list
    :param int_init: the number of initial examples < n.
    :type int_init: int.
    :param is_std: flag for drawing standard deviations.
    :type is_std: bool.
    :param is_marker: flag for drawing markers.
    :type is_marker: bool., optional
    :param is_legend: flag for drawing a legend.
    :type is_legend: bool., optional
    :param is_tex: flag for using latex.
    :type is_tex: bool., optional
    :param path_save: None, or path for saving a figure.
    :type path_save: NoneType or str., optional
    :param str_postfix: None, or the name of postfix.
    :type str_postfix: NoneType or str., optional
    :param str_x_axis: the name of x axis.
    :type str_x_axis: str., optional
    :param str_y_axis: the name of y axis.
    :type str_y_axis: str., optional
    :param is_pause: flag for pausing before closing a figure.
    :type is_pause: bool., optional
    :param time_pause: pausing time.
    :type time_pause: float, optional
    :param range_shade: shade range for standard deviation.
    :type range_shade: float, optional
    :param markers: list of markers.
    :type markers: list, optional
    :param colors: list of colors.
    :type colors: list, optional

    :returns: None.
    :rtype: NoneType

    :raises: AssertionError

    """

    assert isinstance(arr_minima, np.ndarray)
    assert isinstance(list_str_label, list)
    assert isinstance(int_init, int)
    assert isinstance(is_std, bool)
    assert isinstance(is_marker, bool)
    assert isinstance(is_legend, bool)
    assert isinstance(is_tex, bool)
    assert isinstance(path_save, str) or path_save is None
    assert isinstance(str_postfix, str) or str_postfix is None
    assert isinstance(str_x_axis, str)
    assert isinstance(str_y_axis, str)
    assert isinstance(time_pause, float)
    assert isinstance(range_shade, float)
    assert isinstance(markers, list)
    assert isinstance(colors, list)
    assert len(arr_minima.shape) == 3
    assert arr_minima.shape[0] == len(list_str_label)
    assert arr_minima.shape[2] >= int_init

    if plt is None or pylab is None:
        return
    if is_tex:
        plt.rc('text', usetex=True)
    else:
        plt.rc('pdf', fonttype=42)

    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    for ind_minimum, arr_minimum in enumerate(arr_minima):
        ind_color = ind_minimum % len(colors)
        ind_marker = ind_minimum % len(markers)
        _, mean_min, std_min = utils_common.get_minimum(arr_minimum, int_init)
        x_data = range(0, mean_min.shape[0])
        y_data = mean_min
        std_data = std_min
        if is_marker:
            ax.plot(x_data, y_data,
                label=list_str_label[ind_minimum],
                c=colors[ind_color],
                linewidth=4,
                marker=markers[ind_marker],
                markersize=10,
                mew=3,
            )
        else:
            ax.plot(x_data, y_data, 
                label=list_str_label[ind_minimum], 
                c=colors[ind_color], 
                linewidth=4,
                marker='None')
           
        if is_std:
            ax.fill_between(x_data, 
                y_data - range_shade * std_data, 
                y_data + range_shade * std_data, 
                color=colors[ind_color], 
                alpha=0.3)
    lines, labels = ax.get_legend_handles_labels()

    _set_ax_config(ax, str_x_axis, str_y_axis, xlim_min=0, xlim_max=mean_min.shape[0]-1)

    if is_legend:
        plt.legend(loc='upper right', fancybox=False, edgecolor='black', fontsize=24)

    if path_save is not None and str_postfix is not None:
        str_figure = 'minimum_mean_std_' + str_postfix if is_std else 'minimum_mean_only_' + str_postfix
        _save_figure(path_save, str_figure)

        fig_legend = pylab.figure(figsize=(3, 2))
        fig_legend.legend(lines, list_str_label, 'center', fancybox=False, edgecolor='black', fontsize=32)
        fig_legend.savefig(os.path.join(path_save, 'legend_{}.pdf'.format(str_postfix)), format='pdf', transparent=True, bbox_inches='tight')

    _show_figure(is_pause, time_pause)
    return

def plot_minimum_time(arr_times, arr_minima, list_str_label, int_init, is_std,
    is_marker=True,
    is_legend=False,
    is_tex=False,
    path_save=None,
    str_postfix=None,
    str_x_axis='Time (sec.)',
    str_y_axis='Minimum function value',
    is_pause=True,
    time_pause=constants.TIME_PAUSE,
    range_shade=constants.RANGE_SHADE,
    markers=constants.MARKERS,
    colors=constants.COLORS,
): # pragma: no cover
    """
    It is for plotting optimization results of Bayesian optimization, in terms of execution time.

    :param arr_times: execution times. Shape: (b, r, n), or (b, r, `int_init` + n) where b is the number of experiments, r is the number of rounds, and n is the number of iterations per round.
    :type arr_times: numpy.ndarray
    :param arr_minima: function values over acquired examples. Shape: (b, r, `int_init` + n) where b is the number of experiments, r is the number of rounds, and n is the number of iterations per round.
    :type arr_minima: numpy.ndarray
    :param list_str_label: list of label strings. Shape: (b, ).
    :type list_str_label: list
    :param int_init: the number of initial examples.
    :type int_init: int.
    :param is_std: flag for drawing standard deviations.
    :type is_std: bool.
    :param is_marker: flag for drawing markers.
    :type is_marker: bool., optional
    :param is_legend: flag for drawing a legend.
    :type is_legend: bool., optional
    :param is_tex: flag for using latex.
    :type is_tex: bool., optional
    :param path_save: None, or path for saving a figure.
    :type path_save: NoneType or str., optional
    :param str_postfix: None, or the name of postfix.
    :type str_postfix: NoneType or str., optional
    :param str_x_axis: the name of x axis.
    :type str_x_axis: str., optional
    :param str_y_axis: the name of y axis.
    :type str_y_axis: str., optional
    :param is_pause: flag for pausing before closing a figure.
    :type is_pause: bool., optional
    :param time_pause: pausing time.
    :type time_pause: float, optional
    :param range_shade: shade range for standard deviation.
    :type range_shade: float, optional
    :param markers: list of markers.
    :type markers: list, optional
    :param colors: list of colors.
    :type colors: list, optional

    :returns: None.
    :rtype: NoneType

    :raises: AssertionError

    """

    assert isinstance(arr_times, np.ndarray)
    assert isinstance(arr_minima, np.ndarray)
    assert isinstance(list_str_label, list)
    assert isinstance(int_init, int)
    assert isinstance(is_std, bool)
    assert isinstance(is_marker, bool)
    assert isinstance(is_legend, bool)
    assert isinstance(is_tex, bool)
    assert isinstance(path_save, str) or path_save is None
    assert isinstance(str_postfix, str) or str_postfix is None
    assert isinstance(str_x_axis, str)
    assert isinstance(str_y_axis, str)
    assert isinstance(time_pause, float)
    assert isinstance(range_shade, float)
    assert isinstance(markers, list)
    assert isinstance(colors, list)
    assert len(arr_times.shape) == 3
    assert len(arr_minima.shape) == 3
    assert arr_times.shape[0] == arr_minima.shape[0] == len(list_str_label)
    assert arr_times.shape[1] == arr_minima.shape[1]
    assert arr_minima.shape[2] >= int_init
    assert arr_times.shape[2] == arr_minima.shape[2] or arr_times.shape[2] + int_init == arr_minima.shape[2]

    if plt is None or pylab is None:
        return
    if is_tex:
        plt.rc('text', usetex=True)
    else:
        plt.rc('pdf', fonttype=42)

    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    list_x_data = []
    for ind_minimum, arr_minimum in enumerate(arr_minima):
        ind_color = ind_minimum % len(colors)
        ind_marker = ind_minimum % len(markers)
        _, mean_min, std_min = utils_common.get_minimum(arr_minimum, int_init)
        x_data = utils_common.get_time(arr_times[ind_minimum], int_init, arr_times.shape[2] == arr_minima.shape[2])
        y_data = mean_min
        std_data = std_min
        if is_marker:
            ax.plot(x_data, y_data,
                label=list_str_label[ind_minimum],
                c=colors[ind_color],
                linewidth=4,
                marker=markers[ind_marker],
                markersize=10,
                mew=3,
            )
        else:
            ax.plot(x_data, y_data, 
                label=list_str_label[ind_minimum], 
                c=colors[ind_color], 
                linewidth=4,
                marker='None')
           
        if is_std:
            ax.fill_between(x_data, 
                y_data - range_shade * std_data, 
                y_data + range_shade * std_data, 
                color=colors[ind_color], 
                alpha=0.3)
        list_x_data.append(x_data)
    lines, labels = ax.get_legend_handles_labels()

    _set_ax_config(ax, str_x_axis, str_y_axis, xlim_min=np.min(list_x_data), xlim_max=np.max(list_x_data))

    if is_legend:
        plt.legend(loc='upper right', fancybox=False, edgecolor='black', fontsize=24)

    if path_save is not None and str_postfix is not None:
        str_figure = 'minimum_time_mean_std_' + str_postfix if is_std else 'minimum_time_mean_only_' + str_postfix
        _save_figure(path_save, str_figure)

        fig_legend = pylab.figure(figsize=(3, 2))
        fig_legend.legend(lines, list_str_label, 'center', fancybox=False, edgecolor='black', fontsize=32)
        fig_legend.savefig(os.path.join(path_save, 'legend_{}.pdf'.format(str_postfix)), format='pdf', transparent=True, bbox_inches='tight')

    _show_figure(is_pause, time_pause)
    return

def plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, std_test,
    path_save=None,
    str_postfix=None,
    str_x_axis='x',
    str_y_axis='y',
    int_init=None,
    is_tex=False,
    is_zero_axis=False,
    is_pause=True,
    time_pause=constants.TIME_PAUSE,
    range_shade=constants.RANGE_SHADE,
): # pragma: no cover
    """
    It is for plotting Bayesian optimization results step by step.

    :param X_train: training inputs. Shape: (n, 1).
    :type X_train: numpy.ndarray
    :param Y_train: training outputs. Shape: (n, 1).
    :type Y_train: numpy.ndarray
    :param X_test: test inputs. Shape: (m, 1).
    :type X_test: numpy.ndarray
    :param Y_test: None, or true test outputs. Shape: (m, 1).
    :type Y_test: NoneType or numpy.ndarray, optional
    :param mean_test: posterior predictive mean function values over `X_test`. Shape: (m, 1).
    :type mean_test: numpy.ndarray
    :param std_test: posterior predictive standard deviation function values over `X_test`. Shape: (m, 1).
    :type std_test: numpy.ndarray
    :param path_save: None, or path for saving a figure.
    :type path_save: NoneType or str., optional
    :param str_postfix: None, or the name of postfix.
    :type str_postfix: NoneType or str., optional
    :param str_x_axis: the name of x axis.
    :type str_x_axis: str., optional
    :param str_y_axis: the name of y axis.
    :type str_y_axis: str., optional
    :param int_init: None, or the number of initial examples.
    :type int_init: NoneType or int., optional
    :param is_tex: flag for using latex.
    :type is_tex: bool., optional
    :param is_zero_axis: flag for drawing a zero axis.
    :type is_zero_axis: bool., optional
    :param is_pause: flag for pausing before closing a figure.
    :type is_pause: bool., optional
    :param time_pause: pausing time.
    :type time_pause: float, optional
    :param range_shade: shade range for standard deviation.
    :type range_shade: float, optional

    :returns: None.
    :rtype: NoneType

    :raises: AssertionError

    """

    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(Y_test, np.ndarray)
    assert isinstance(mean_test, np.ndarray)
    assert isinstance(std_test, np.ndarray)
    assert isinstance(path_save, str) or path_save is None
    assert isinstance(str_postfix, str) or str_postfix is None
    assert isinstance(str_x_axis, str)
    assert isinstance(str_y_axis, str)
    assert isinstance(int_init, int) or int_init is None
    assert isinstance(is_tex, bool)
    assert isinstance(is_zero_axis, bool)
    assert isinstance(time_pause, float)
    assert isinstance(range_shade, float)
    assert len(X_train.shape) == 2
    assert len(X_test.shape) == 2
    assert len(Y_train.shape) == 2
    assert len(Y_test.shape) == 2
    assert len(mean_test.shape) == 2
    assert len(std_test.shape) == 2
    assert X_train.shape[1] == X_test.shape[1] == 1
    assert Y_train.shape[1] == Y_test.shape[1] == 1
    assert X_train.shape[0] == Y_train.shape[0]
    assert mean_test.shape[1] == 1
    assert std_test.shape[1] == 1
    assert X_test.shape[0] == Y_test.shape[0] == mean_test.shape[0] == std_test.shape[0]
    if int_init is not None:
        assert X_train.shape[0] >= int_init

    if plt is None or pylab is None:
        return
    if is_tex:
        plt.rc('text', usetex=True)
    else:
        plt.rc('pdf', fonttype=42)

    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    ax.plot(X_test, Y_test, 'g', linewidth=4)
    ax.plot(X_test, mean_test, 'b', linewidth=4)
    ax.fill_between(X_test.flatten(), 
        mean_test.flatten() - range_shade * std_test.flatten(), 
        mean_test.flatten() + range_shade * std_test.flatten(), 
        color='b', 
        alpha=0.3,
    )
    if int_init is not None:
        if X_train.shape[0] > int_init:
            ax.plot(X_train[:int_init, :], Y_train[:int_init, :], 'x', c='saddlebrown', ms=14, markeredgewidth=6)
            ax.plot(X_train[int_init:X_train.shape[0]-1, :], Y_train[int_init:X_train.shape[0]-1, :], 'rx', ms=14, markeredgewidth=6)
            ax.plot(X_train[X_train.shape[0]-1, :], Y_train[X_train.shape[0]-1, :], c='orange', marker='+', ms=18, markeredgewidth=6)
        else:
            ax.plot(X_train, Y_train, 'x', c='saddlebrown', ms=14, markeredgewidth=6)
    else:
        ax.plot(X_train[:X_train.shape[0]-1, :], Y_train[:X_train.shape[0]-1, :], 'rx', ms=14, markeredgewidth=6)
        ax.plot(X_train[X_train.shape[0]-1, :], Y_train[X_train.shape[0]-1, :], c='orange', marker='+', ms=18, markeredgewidth=6)

    _set_ax_config(ax, str_x_axis, str_y_axis, xlim_min=np.min(X_test), xlim_max=np.max(X_test), is_zero_axis=is_zero_axis)

    if path_save is not None and str_postfix is not None:
        _save_figure(path_save, str_postfix, str_prefix='bo_step_')
    _show_figure(is_pause, time_pause)
    return

def plot_bo_step_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test,
    path_save=None,
    str_postfix=None,
    str_x_axis='x',
    str_y_axis='y',
    str_acq_axis='acq.',
    int_init=None,
    is_tex=False,
    is_zero_axis=False,
    is_pause=True,
    time_pause=constants.TIME_PAUSE,
    range_shade=constants.RANGE_SHADE,
): # pragma: no cover
    """
    It is for plotting Bayesian optimization results step by step.

    :param X_train: training inputs. Shape: (n, 1).
    :type X_train: numpy.ndarray
    :param Y_train: training outputs. Shape: (n, 1).
    :type Y_train: numpy.ndarray
    :param X_test: test inputs. Shape: (m, 1).
    :type X_test: numpy.ndarray
    :param Y_test: None, or true test outputs. Shape: (m, 1).
    :type Y_test: NoneType or numpy.ndarray, optional
    :param mean_test: posterior predictive mean function values over `X_test`. Shape: (m, 1).
    :type mean_test: numpy.ndarray
    :param std_test: posterior predictive standard deviation function values over `X_test`. Shape: (m, 1).
    :type std_test: numpy.ndarray
    :param acq_test: acquisition funcion values over `X_test`. Shape: (m, 1).
    :type acq_test: numpy.ndarray
    :param path_save: None, or path for saving a figure.
    :type path_save: NoneType or str., optional
    :param str_postfix: None, or the name of postfix.
    :type str_postfix: NoneType or str., optional
    :param str_x_axis: the name of x axis.
    :type str_x_axis: str., optional
    :param str_y_axis: the name of y axis.
    :type str_y_axis: str., optional
    :param str_acq_axis: the name of acquisition function axis.
    :type str_acq_axis: str., optional
    :param int_init: None, or the number of initial examples.
    :type int_init: NoneType or int., optional
    :param is_tex: flag for using latex.
    :type is_tex: bool., optional
    :param is_zero_axis: flag for drawing a zero axis.
    :type is_zero_axis: bool., optional
    :param is_pause: flag for pausing before closing a figure.
    :type is_pause: bool., optional
    :param time_pause: pausing time.
    :type time_pause: float, optional
    :param range_shade: shade range for standard deviation.
    :type range_shade: float, optional

    :returns: None.
    :rtype: NoneType

    :raises: AssertionError

    """

    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(Y_test, np.ndarray)
    assert isinstance(mean_test, np.ndarray)
    assert isinstance(std_test, np.ndarray)
    assert isinstance(path_save, str) or path_save is None
    assert isinstance(str_postfix, str) or str_postfix is None
    assert isinstance(str_x_axis, str)
    assert isinstance(str_y_axis, str)
    assert isinstance(str_acq_axis, str)
    assert isinstance(int_init, int) or int_init is None
    assert isinstance(is_tex, bool)
    assert isinstance(is_zero_axis, bool)
    assert isinstance(time_pause, float)
    assert isinstance(range_shade, float)
    assert len(X_train.shape) == 2
    assert len(X_test.shape) == 2
    assert len(Y_train.shape) == 2
    assert len(Y_test.shape) == 2
    assert len(mean_test.shape) == 2
    assert len(std_test.shape) == 2
    assert len(acq_test.shape) == 2
    assert X_train.shape[1] == X_test.shape[1] == 1
    assert Y_train.shape[1] == Y_test.shape[1] == 1
    assert X_train.shape[0] == Y_train.shape[0]
    assert mean_test.shape[1] == 1
    assert std_test.shape[1] == 1
    assert acq_test.shape[1] == 1
    assert X_test.shape[0] == Y_test.shape[0] == mean_test.shape[0] == std_test.shape[0] == acq_test.shape[0]
    if int_init is not None:
        assert X_train.shape[0] >= int_init

    if plt is None or pylab is None:
        return
    if is_tex:
        plt.rc('text', usetex=True)
    else:
        plt.rc('pdf', fonttype=42)

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[3, 1]})

    ax1.plot(X_test, Y_test, 'g', linewidth=4)
    ax1.plot(X_test, mean_test, 'b', linewidth=4)
    ax1.fill_between(X_test.flatten(), 
        mean_test.flatten() - range_shade * std_test.flatten(), 
        mean_test.flatten() + range_shade * std_test.flatten(), 
        color='b', 
        alpha=0.3)
    if int_init is not None:
        if X_train.shape[0] > int_init:
            ax1.plot(X_train[:int_init, :], Y_train[:int_init, :], 'x', c='saddlebrown', ms=14, markeredgewidth=6)
            ax1.plot(X_train[int_init:X_train.shape[0]-1, :], Y_train[int_init:X_train.shape[0]-1, :], 'rx', ms=14, markeredgewidth=6)
            ax1.plot(X_train[X_train.shape[0]-1, :], Y_train[X_train.shape[0]-1, :], c='orange', marker='+', ms=18, markeredgewidth=6)
        else:
            ax1.plot(X_train, Y_train, 'x', c='saddlebrown', ms=14, markeredgewidth=6)
    else:
        ax1.plot(X_train[:X_train.shape[0]-1, :], Y_train[:X_train.shape[0]-1, :], 'rx', ms=14, markeredgewidth=6)
        ax1.plot(X_train[X_train.shape[0]-1, :], Y_train[X_train.shape[0]-1, :], c='orange', marker='+', ms=18, markeredgewidth=6)

    _set_ax_config(ax1, None, str_y_axis, xlim_min=np.min(X_test), xlim_max=np.max(X_test), is_zero_axis=is_zero_axis)

    ax2.plot(X_test, acq_test, 'b', linewidth=4)
    _set_ax_config(ax2, str_x_axis, str_y_axis, xlim_min=np.min(X_test), xlim_max=np.max(X_test), is_zero_axis=is_zero_axis)

    if path_save is not None and str_postfix is not None:
        _save_figure(path_save, str_postfix, str_prefix='bo_step_acq_')
    _show_figure(is_pause, time_pause)
    return
