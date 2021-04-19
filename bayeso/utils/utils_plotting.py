#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: September 24, 2020
#
"""It is utilities for plotting figures."""

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

from bayeso.utils import utils_common
from bayeso.utils import utils_logger
from bayeso import constants

logger = utils_logger.get_logger('utils_plotting')


@utils_common.validate_types
def _set_font_config(use_tex: bool) -> constants.TYPE_NONE: # pragma: no cover
    """
    It sets a font configuration.

    :param use_tex: flag for using latex.
    :type use_tex: bool.

    :returns: None.
    :rtype: NoneType

    """

    if use_tex:
        plt.rc('text', usetex=True)
    else:
        plt.rc('pdf', fonttype=42)

@utils_common.validate_types
def _set_ax_config(ax: 'matplotlib.axes._subplots.AxesSubplot', str_x_axis: str, str_y_axis: str,
    size_labels: int=32,
    size_ticks: int=22,
    xlim_min: constants.TYPING_UNION_FLOAT_NONE=None,
    xlim_max: constants.TYPING_UNION_FLOAT_NONE=None,
    draw_box: bool=True,
    draw_zero_axis: bool=False,
    draw_grid: bool=True,
) -> constants.TYPE_NONE: # pragma: no cover
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
    :param draw_box: flag for drawing a box.
    :type draw_box: bool., optional
    :param draw_zero_axis: flag for drawing a zero axis.
    :type draw_zero_axis: bool., optional
    :param draw_grid: flag for drawing grids.
    :type draw_grid: bool., optional

    :returns: None.
    :rtype: NoneType

    """

    if str_x_axis is not None:
        ax.set_xlabel(str_x_axis, fontsize=size_labels)
    ax.set_ylabel(str_y_axis, fontsize=size_labels)
    ax.tick_params(labelsize=size_ticks)

    if not draw_box:
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
    if xlim_min is not None and xlim_max is not None:
        ax.set_xlim([xlim_min, xlim_max])
    if draw_zero_axis:
        ax.spines['bottom'].set_position('zero')
    if draw_grid:
        ax.grid()

@utils_common.validate_types
def _save_figure(path_save: str, str_postfix: str,
    str_prefix: str=''
) -> constants.TYPE_NONE: # pragma: no cover
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
            bbox_inches='tight')

@utils_common.validate_types
def _show_figure(pause_figure: bool, time_pause: constants.TYPING_UNION_INT_FLOAT
) -> constants.TYPE_NONE: # pragma: no cover
    """
    It shows a figure.

    :param pause_figure: flag for pausing before closing a figure.
    :type pause_figure: bool.
    :param time_pause: pausing time.
    :type time_pause: int. or float

    :returns: None.
    :rtype: NoneType

    """

    if pause_figure:
        if time_pause < np.inf:
            plt.ion()
            plt.pause(time_pause)
            plt.close('all')
        else:
            plt.show()

@utils_common.validate_types
def plot_gp_via_sample(X: np.ndarray, Ys: np.ndarray,
    path_save: constants.TYPING_UNION_STR_NONE=None,
    str_postfix: constants.TYPING_UNION_STR_NONE=None,
    str_x_axis: str='x',
    str_y_axis: str='y',
    use_tex: bool=False,
    draw_zero_axis: bool=False,
    pause_figure: bool=True,
    time_pause: constants.TYPING_UNION_INT_FLOAT=constants.TIME_PAUSE,
    colors: np.ndarray=constants.COLORS,
) -> constants.TYPE_NONE: # pragma: no cover
    """
    It is for plotting sampled functions from multivariate distributions.

    :param X: training inputs. Shape: (n, 1).
    :type X: numpy.ndarray
    :param Ys: training outputs. Shape: (m, n).
    :type Ys: numpy.ndarray
    :param path_save: None, or path for saving a figure.
    :type path_save: NoneType or str., optional
    :param str_postfix: None, or the name of postfix.
    :type str_postfix: NoneType or str., optional
    :param str_x_axis: the name of x axis.
    :type str_x_axis: str., optional
    :param str_y_axis: the name of y axis.
    :type str_y_axis: str., optional
    :param use_tex: flag for using latex.
    :type use_tex: bool., optional
    :param draw_zero_axis: flag for drawing a zero axis.
    :type draw_zero_axis: bool., optional
    :param pause_figure: flag for pausing before closing a figure.
    :type pause_figure: bool., optional
    :param time_pause: pausing time.
    :type time_pause: int. or float, optional
    :param colors: array of colors.
    :type colors: np.ndarray, optional

    :returns: None.
    :rtype: NoneType

    :raises: AssertionError

    """

    assert isinstance(X, np.ndarray)
    assert isinstance(Ys, np.ndarray)
    assert isinstance(path_save, (str, type(None)))
    assert isinstance(str_postfix, (str, type(None)))
    assert isinstance(str_x_axis, str)
    assert isinstance(str_y_axis, str)
    assert isinstance(use_tex, bool)
    assert isinstance(draw_zero_axis, bool)
    assert isinstance(pause_figure, bool)
    assert isinstance(time_pause, (int, float))
    assert isinstance(colors, np.ndarray)
    assert len(X.shape) == 2
    assert len(Ys.shape) == 2
    assert X.shape[1] == 1
    assert X.shape[0] == Ys.shape[1]

    if plt is None or pylab is None:
        logger.info('matplotlib or pylab is not installed.')
        return
    _set_font_config(use_tex)

    _ = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    for Y in Ys:
        ax.plot(X.flatten(), Y,
            c=colors[0],
            lw=4,
            alpha=0.3,
        )

    _set_ax_config(ax, str_x_axis, str_y_axis, xlim_min=np.min(X), xlim_max=np.max(X),
        draw_zero_axis=draw_zero_axis)

    if path_save is not None and str_postfix is not None:
        _save_figure(path_save, str_postfix, str_prefix='gp_sampled_')
    _show_figure(pause_figure, time_pause)

@utils_common.validate_types
def plot_gp_via_distribution(X_train: np.ndarray, Y_train: np.ndarray,
    X_test: np.ndarray, mean_test: np.ndarray, std_test: np.ndarray,
    Y_test: constants.TYPING_UNION_ARRAY_NONE=None,
    path_save: constants.TYPING_UNION_STR_NONE=None,
    str_postfix: constants.TYPING_UNION_STR_NONE=None,
    str_x_axis: str='x',
    str_y_axis: str='y',
    use_tex: bool=False,
    draw_zero_axis: bool=False,
    pause_figure: bool=True,
    time_pause: constants.TYPING_UNION_INT_FLOAT=constants.TIME_PAUSE,
    range_shade: float=constants.RANGE_SHADE,
    colors: np.ndarray=constants.COLORS,
) -> constants.TYPE_NONE: # pragma: no cover
    """
    It is for plotting Gaussian process regression.

    :param X_train: training inputs. Shape: (n, 1).
    :type X_train: numpy.ndarray
    :param Y_train: training outputs. Shape: (n, 1).
    :type Y_train: numpy.ndarray
    :param X_test: test inputs. Shape: (m, 1).
    :type X_test: numpy.ndarray
    :param mean_test: posterior predictive mean function values over `X_test`.
        Shape: (m, 1).
    :type mean_test: numpy.ndarray
    :param std_test: posterior predictive standard deviation function values
        over `X_test`. Shape: (m, 1).
    :type std_test: numpy.ndarray
    :param Y_test: None, or true test outputs. Shape: (m, 1).
    :type Y_test: NoneType or numpy.ndarray, optional
    :param path_save: None, or path for saving a figure.
    :type path_save: NoneType or str., optional
    :param str_postfix: None, or the name of postfix.
    :type str_postfix: NoneType or str., optional
    :param str_x_axis: the name of x axis.
    :type str_x_axis: str., optional
    :param str_y_axis: the name of y axis.
    :type str_y_axis: str., optional
    :param use_tex: flag for using latex.
    :type use_tex: bool., optional
    :param draw_zero_axis: flag for drawing a zero axis.
    :type draw_zero_axis: bool., optional
    :param pause_figure: flag for pausing before closing a figure.
    :type pause_figure: bool., optional
    :param time_pause: pausing time.
    :type time_pause: int. or float, optional
    :param range_shade: shade range for standard deviation.
    :type range_shade: float, optional
    :param colors: array of colors.
    :type colors: np.ndarray, optional

    :returns: None.
    :rtype: NoneType

    :raises: AssertionError

    """

    assert isinstance(X_train, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(mean_test, np.ndarray)
    assert isinstance(std_test, np.ndarray)
    assert isinstance(Y_test, (np.ndarray, type(None)))
    assert isinstance(path_save, (str, type(None)))
    assert isinstance(str_postfix, (str, type(None)))
    assert isinstance(str_x_axis, str)
    assert isinstance(str_y_axis, str)
    assert isinstance(use_tex, bool)
    assert isinstance(draw_zero_axis, bool)
    assert isinstance(pause_figure, bool)
    assert isinstance(time_pause, (int, float))
    assert isinstance(range_shade, float)
    assert isinstance(colors, np.ndarray)
    assert len(X_train.shape) == 2
    assert len(X_test.shape) == 2
    assert len(Y_train.shape) == 2
    assert len(mean_test.shape) == 2
    assert len(std_test.shape) == 2
    assert X_train.shape[1] == X_test.shape[1] == 1
    assert Y_train.shape[1] == 1
    assert X_train.shape[0] == Y_train.shape[0]
    assert mean_test.shape[1] == 1
    assert std_test.shape[1] == 1
    assert X_test.shape[0] == mean_test.shape[0] == std_test.shape[0]
    if Y_test is not None:
        assert len(Y_test.shape) == 2
        assert Y_test.shape[1] == 1
        assert X_test.shape[0] == Y_test.shape[0]

    if plt is None or pylab is None:
        logger.info('matplotlib or pylab is not installed.')
        return
    _set_font_config(use_tex)

    _ = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    if Y_test is not None:
        ax.plot(X_test.flatten(), Y_test.flatten(),
            c=colors[1],
            linewidth=4,
            marker='None')
    ax.plot(X_test.flatten(), mean_test.flatten(),
        c=colors[2],
        linewidth=4,
        marker='None')
    ax.fill_between(X_test.flatten(),
        mean_test.flatten() - range_shade * std_test.flatten(),
        mean_test.flatten() + range_shade * std_test.flatten(),
        color=colors[2],
        alpha=0.3)
    ax.plot(X_train.flatten(), Y_train.flatten(),
        'x',
        c=colors[0],
        markersize=10,
        mew=4)

    _set_ax_config(ax, str_x_axis, str_y_axis, xlim_min=np.min(X_test),
        xlim_max=np.max(X_test), draw_zero_axis=draw_zero_axis)

    if path_save is not None and str_postfix is not None:
        _save_figure(path_save, str_postfix, str_prefix='gp_')
    _show_figure(pause_figure, time_pause)

@utils_common.validate_types
def plot_minimum_vs_iter(minima: np.ndarray, list_str_label: constants.TYPING_LIST[str],
    num_init: int, draw_std: bool,
    include_marker: bool=True,
    include_legend: bool=False,
    use_tex: bool=False,
    path_save: constants.TYPING_UNION_STR_NONE=None,
    str_postfix: constants.TYPING_UNION_STR_NONE=None,
    str_x_axis: str='Iteration',
    str_y_axis: str='Minimum function value',
    pause_figure: bool=True,
    time_pause: constants.TYPING_UNION_INT_FLOAT=constants.TIME_PAUSE,
    range_shade: float=constants.RANGE_SHADE,
    markers: np.ndarray=constants.MARKERS,
    colors: np.ndarray=constants.COLORS,
) -> constants.TYPE_NONE: # pragma: no cover
    """
    It is for plotting optimization results of Bayesian optimization, in
    terms of iterations.

    :param minima: function values over acquired examples. Shape: (b, r, n)
        where b is the number of experiments, r is the number of rounds,
        and n is the number of iterations per round.
    :type minima: numpy.ndarray
    :param list_str_label: list of label strings. Shape: (b, ).
    :type list_str_label: list
    :param num_init: the number of initial examples < n.
    :type num_init: int.
    :param draw_std: flag for drawing standard deviations.
    :type draw_std: bool.
    :param include_marker: flag for drawing markers.
    :type include_marker: bool., optional
    :param include_legend: flag for drawing a legend.
    :type include_legend: bool., optional
    :param use_tex: flag for using latex.
    :type use_tex: bool., optional
    :param path_save: None, or path for saving a figure.
    :type path_save: NoneType or str., optional
    :param str_postfix: None, or the name of postfix.
    :type str_postfix: NoneType or str., optional
    :param str_x_axis: the name of x axis.
    :type str_x_axis: str., optional
    :param str_y_axis: the name of y axis.
    :type str_y_axis: str., optional
    :param pause_figure: flag for pausing before closing a figure.
    :type pause_figure: bool., optional
    :param time_pause: pausing time.
    :type time_pause: int. or float, optional
    :param range_shade: shade range for standard deviation.
    :type range_shade: float, optional
    :param markers: array of markers.
    :type markers: np.ndarray, optional
    :param colors: array of colors.
    :type colors: np.ndarray, optional

    :returns: None.
    :rtype: NoneType

    :raises: AssertionError

    """

    assert isinstance(minima, np.ndarray)
    assert isinstance(list_str_label, list)
    assert isinstance(num_init, int)
    assert isinstance(draw_std, bool)
    assert isinstance(include_marker, bool)
    assert isinstance(include_legend, bool)
    assert isinstance(use_tex, bool)
    assert isinstance(path_save, (str, type(None)))
    assert isinstance(str_postfix, (str, type(None)))
    assert isinstance(str_x_axis, str)
    assert isinstance(str_y_axis, str)
    assert isinstance(pause_figure, bool)
    assert isinstance(time_pause, (int, float))
    assert isinstance(range_shade, float)
    assert isinstance(markers, np.ndarray)
    assert isinstance(colors, np.ndarray)
    assert len(minima.shape) == 3
    assert minima.shape[0] == len(list_str_label)
    assert minima.shape[2] >= num_init

    if plt is None or pylab is None:
        logger.info('matplotlib or pylab is not installed.')
        return
    _set_font_config(use_tex)

    _ = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    for ind_minimum, arr_minimum in enumerate(minima):
        ind_color = ind_minimum % len(colors)
        ind_marker = ind_minimum % len(markers)

        _, mean_min, std_min = utils_common.get_minimum(arr_minimum, num_init)
        x_data = range(0, mean_min.shape[0])
        y_data = mean_min
        std_data = std_min

        if include_marker:
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

        if draw_std:
            ax.fill_between(x_data,
                y_data - range_shade * std_data,
                y_data + range_shade * std_data,
                color=colors[ind_color],
                alpha=0.3)
    lines, _ = ax.get_legend_handles_labels()

    _set_ax_config(ax, str_x_axis, str_y_axis, xlim_min=0, xlim_max=mean_min.shape[0]-1)

    if include_legend:
        plt.legend(loc='upper right', fancybox=False, edgecolor='black', fontsize=24)

    if path_save is not None and str_postfix is not None:
        if draw_std:
            str_figure = 'minimum_mean_std_' + str_postfix
        else:
            str_figure = 'minimum_mean_only_' + str_postfix
        _save_figure(path_save, str_figure)

        fig_legend = pylab.figure(figsize=(3, 2))
        fig_legend.legend(lines, list_str_label, 'center', fancybox=False,
            edgecolor='black', fontsize=32)
        fig_legend.savefig(os.path.join(path_save, 'legend_{}.pdf'.format(
            str_postfix)), format='pdf', transparent=True,
            bbox_inches='tight')

    _show_figure(pause_figure, time_pause)

@utils_common.validate_types
def plot_minimum_vs_time(times: np.ndarray, minima: np.ndarray,
    list_str_label: constants.TYPING_LIST[str], num_init: int, draw_std: bool,
    include_marker: bool=True,
    include_legend: bool=False,
    use_tex: bool=False,
    path_save: constants.TYPING_UNION_STR_NONE=None,
    str_postfix: constants.TYPING_UNION_STR_NONE=None,
    str_x_axis: str='Time (sec.)',
    str_y_axis: str='Minimum function value',
    pause_figure: bool=True,
    time_pause: constants.TYPING_UNION_INT_FLOAT=constants.TIME_PAUSE,
    range_shade: float=constants.RANGE_SHADE,
    markers: np.ndarray=constants.MARKERS,
    colors: np.ndarray=constants.COLORS,
) -> constants.TYPE_NONE: # pragma: no cover
    """
    It is for plotting optimization results of Bayesian optimization, in terms of execution time.

    :param times: execution times. Shape: (b, r, n), or (b, r, `num_init` + n)
        where b is the number of experiments, r is the number of rounds,
        and n is the number of iterations per round.
    :type times: numpy.ndarray
    :param minima: function values over acquired examples. Shape: (b, r, `num_init` + n)
        where b is the number of experiments, r is the number of rounds,
        and n is the number of iterations per round.
    :type minima: numpy.ndarray
    :param list_str_label: list of label strings. Shape: (b, ).
    :type list_str_label: list
    :param num_init: the number of initial examples.
    :type num_init: int.
    :param draw_std: flag for drawing standard deviations.
    :type draw_std: bool.
    :param include_marker: flag for drawing markers.
    :type include_marker: bool., optional
    :param include_legend: flag for drawing a legend.
    :type include_legend: bool., optional
    :param use_tex: flag for using latex.
    :type use_tex: bool., optional
    :param path_save: None, or path for saving a figure.
    :type path_save: NoneType or str., optional
    :param str_postfix: None, or the name of postfix.
    :type str_postfix: NoneType or str., optional
    :param str_x_axis: the name of x axis.
    :type str_x_axis: str., optional
    :param str_y_axis: the name of y axis.
    :type str_y_axis: str., optional
    :param pause_figure: flag for pausing before closing a figure.
    :type pause_figure: bool., optional
    :param time_pause: pausing time.
    :type time_pause: int. or float, optional
    :param range_shade: shade range for standard deviation.
    :type range_shade: float, optional
    :param markers: array of markers.
    :type markers: np.ndarray, optional
    :param colors: array of colors.
    :type colors: np.ndarray, optional

    :returns: None.
    :rtype: NoneType

    :raises: AssertionError

    """

    assert isinstance(times, np.ndarray)
    assert isinstance(minima, np.ndarray)
    assert isinstance(list_str_label, list)
    assert isinstance(num_init, int)
    assert isinstance(draw_std, bool)
    assert isinstance(include_marker, bool)
    assert isinstance(include_legend, bool)
    assert isinstance(use_tex, bool)
    assert isinstance(path_save, (str, type(None)))
    assert isinstance(str_postfix, (str, type(None)))
    assert isinstance(str_x_axis, str)
    assert isinstance(str_y_axis, str)
    assert isinstance(pause_figure, bool)
    assert isinstance(time_pause, (int, float))
    assert isinstance(range_shade, float)
    assert isinstance(markers, np.ndarray)
    assert isinstance(colors, np.ndarray)
    assert len(times.shape) == 3
    assert len(minima.shape) == 3
    assert times.shape[0] == minima.shape[0] == len(list_str_label)
    assert times.shape[1] == minima.shape[1]
    assert minima.shape[2] >= num_init
    assert times.shape[2] == minima.shape[2] or times.shape[2] + num_init == minima.shape[2]

    if plt is None or pylab is None:
        logger.info('matplotlib or pylab is not installed.')
        return
    _set_font_config(use_tex)

    _ = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    list_x_data = []
    for ind_minimum, arr_minimum in enumerate(minima):
        ind_color = ind_minimum % len(colors)
        ind_marker = ind_minimum % len(markers)

        _, mean_min, std_min = utils_common.get_minimum(arr_minimum, num_init)
        x_data = utils_common.get_time(times[ind_minimum], num_init,
            times.shape[2] == minima.shape[2])
        y_data = mean_min
        std_data = std_min

        if include_marker:
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

        if draw_std:
            ax.fill_between(x_data,
                y_data - range_shade * std_data,
                y_data + range_shade * std_data,
                color=colors[ind_color],
                alpha=0.3)
        list_x_data.append(x_data)
    lines, _ = ax.get_legend_handles_labels()

    _set_ax_config(ax, str_x_axis, str_y_axis, xlim_min=np.min(list_x_data),
        xlim_max=np.max(list_x_data))

    if include_legend:
        plt.legend(loc='upper right', fancybox=False, edgecolor='black', fontsize=24)

    if path_save is not None and str_postfix is not None:
        if draw_std:
            str_figure = 'minimum_time_mean_std_' + str_postfix
        else:
            str_figure = 'minimum_time_mean_only_' + str_postfix
        _save_figure(path_save, str_figure)

        fig_legend = pylab.figure(figsize=(3, 2))
        fig_legend.legend(lines, list_str_label, 'center', fancybox=False,
            edgecolor='black', fontsize=32)
        fig_legend.savefig(os.path.join(path_save, 'legend_{}.pdf'.format(str_postfix)),
            format='pdf', transparent=True, bbox_inches='tight')

    _show_figure(pause_figure, time_pause)

@utils_common.validate_types
def plot_bo_step(X_train: np.ndarray, Y_train: np.ndarray,
    X_test: np.ndarray, Y_test: np.ndarray,
    mean_test: np.ndarray, std_test: np.ndarray,
    path_save: constants.TYPING_UNION_STR_NONE=None,
    str_postfix: constants.TYPING_UNION_STR_NONE=None,
    str_x_axis: str='x',
    str_y_axis: str='y',
    num_init: constants.TYPING_UNION_INT_NONE=None,
    use_tex: bool=False,
    draw_zero_axis: bool=False,
    pause_figure: bool=True,
    time_pause: constants.TYPING_UNION_INT_FLOAT=constants.TIME_PAUSE,
    range_shade: float=constants.RANGE_SHADE,
) -> constants.TYPE_NONE: # pragma: no cover
    """
    It is for plotting Bayesian optimization results step by step.

    :param X_train: training inputs. Shape: (n, 1).
    :type X_train: numpy.ndarray
    :param Y_train: training outputs. Shape: (n, 1).
    :type Y_train: numpy.ndarray
    :param X_test: test inputs. Shape: (m, 1).
    :type X_test: numpy.ndarray
    :param Y_test: true test outputs. Shape: (m, 1).
    :type Y_test: numpy.ndarray
    :param mean_test: posterior predictive mean function values over `X_test`.
        Shape: (m, 1).
    :type mean_test: numpy.ndarray
    :param std_test: posterior predictive standard deviation function values
        over `X_test`. Shape: (m, 1).
    :type std_test: numpy.ndarray
    :param path_save: None, or path for saving a figure.
    :type path_save: NoneType or str., optional
    :param str_postfix: None, or the name of postfix.
    :type str_postfix: NoneType or str., optional
    :param str_x_axis: the name of x axis.
    :type str_x_axis: str., optional
    :param str_y_axis: the name of y axis.
    :type str_y_axis: str., optional
    :param num_init: None, or the number of initial examples.
    :type num_init: NoneType or int., optional
    :param use_tex: flag for using latex.
    :type use_tex: bool., optional
    :param draw_zero_axis: flag for drawing a zero axis.
    :type draw_zero_axis: bool., optional
    :param pause_figure: flag for pausing before closing a figure.
    :type pause_figure: bool., optional
    :param time_pause: pausing time.
    :type time_pause: int. or float, optional
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
    assert isinstance(path_save, (str, type(None)))
    assert isinstance(str_postfix, (str, type(None)))
    assert isinstance(str_x_axis, str)
    assert isinstance(str_y_axis, str)
    assert isinstance(num_init, (int, type(None)))
    assert isinstance(use_tex, bool)
    assert isinstance(draw_zero_axis, bool)
    assert isinstance(pause_figure, bool)
    assert isinstance(time_pause, (int, float))
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
    if num_init is not None:
        assert X_train.shape[0] >= num_init

    if plt is None or pylab is None:
        logger.info('matplotlib or pylab is not installed.')
        return
    _set_font_config(use_tex)

    _ = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    ax.plot(X_test, Y_test, 'g', linewidth=4)
    ax.plot(X_test, mean_test, 'b', linewidth=4)
    ax.fill_between(X_test.flatten(),
        mean_test.flatten() - range_shade * std_test.flatten(),
        mean_test.flatten() + range_shade * std_test.flatten(),
        color='b',
        alpha=0.3)

    if num_init is not None:
        if X_train.shape[0] > num_init:
            ax.plot(X_train[:num_init, :], Y_train[:num_init, :], 'x',
                c='saddlebrown', ms=14, markeredgewidth=6)
            ax.plot(X_train[num_init:X_train.shape[0]-1, :],
                Y_train[num_init:X_train.shape[0]-1, :], 'rx',
                ms=14, markeredgewidth=6)
            ax.plot(X_train[X_train.shape[0]-1, :],
                Y_train[X_train.shape[0]-1, :], c='orange', marker='+',
                ms=18, markeredgewidth=6)
        else:
            ax.plot(X_train, Y_train, 'x', c='saddlebrown', ms=14,
                markeredgewidth=6)
    else:
        ax.plot(X_train[:X_train.shape[0]-1, :],
            Y_train[:X_train.shape[0]-1, :], 'rx', ms=14, markeredgewidth=6)
        ax.plot(X_train[X_train.shape[0]-1, :],
            Y_train[X_train.shape[0]-1, :], c='orange', marker='+', ms=18,
            markeredgewidth=6)

    _set_ax_config(ax, str_x_axis, str_y_axis, xlim_min=np.min(X_test),
        xlim_max=np.max(X_test), draw_zero_axis=draw_zero_axis)

    if path_save is not None and str_postfix is not None:
        _save_figure(path_save, str_postfix, str_prefix='bo_step_')
    _show_figure(pause_figure, time_pause)

@utils_common.validate_types
def plot_bo_step_with_acq(X_train: np.ndarray, Y_train: np.ndarray,
    X_test: np.ndarray, Y_test: np.ndarray, mean_test: np.ndarray,
    std_test: np.ndarray, acq_test: np.ndarray,
    path_save: constants.TYPING_UNION_STR_NONE=None,
    str_postfix: constants.TYPING_UNION_STR_NONE=None,
    str_x_axis: str='x',
    str_y_axis: str='y',
    str_acq_axis: str='acq.',
    num_init: constants.TYPING_UNION_INT_NONE=None,
    use_tex: bool=False,
    draw_zero_axis: bool=False,
    pause_figure: bool=True,
    time_pause: constants.TYPING_UNION_INT_FLOAT=constants.TIME_PAUSE,
    range_shade: float=constants.RANGE_SHADE,
) -> constants.TYPE_NONE: # pragma: no cover
    """
    It is for plotting Bayesian optimization results step by step.

    :param X_train: training inputs. Shape: (n, 1).
    :type X_train: numpy.ndarray
    :param Y_train: training outputs. Shape: (n, 1).
    :type Y_train: numpy.ndarray
    :param X_test: test inputs. Shape: (m, 1).
    :type X_test: numpy.ndarray
    :param Y_test: true test outputs. Shape: (m, 1).
    :type Y_test: numpy.ndarray
    :param mean_test: posterior predictive mean function values over `X_test`.
        Shape: (m, 1).
    :type mean_test: numpy.ndarray
    :param std_test: posterior predictive standard deviation function values
        over `X_test`. Shape: (m, 1).
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
    :param num_init: None, or the number of initial examples.
    :type num_init: NoneType or int., optional
    :param use_tex: flag for using latex.
    :type use_tex: bool., optional
    :param draw_zero_axis: flag for drawing a zero axis.
    :type draw_zero_axis: bool., optional
    :param pause_figure: flag for pausing before closing a figure.
    :type pause_figure: bool., optional
    :param time_pause: pausing time.
    :type time_pause: int. or float, optional
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
    assert isinstance(path_save, (str, type(None)))
    assert isinstance(str_postfix, (str, type(None)))
    assert isinstance(str_x_axis, str)
    assert isinstance(str_y_axis, str)
    assert isinstance(str_acq_axis, str)
    assert isinstance(num_init, (int, type(None)))
    assert isinstance(use_tex, bool)
    assert isinstance(draw_zero_axis, bool)
    assert isinstance(pause_figure, bool)
    assert isinstance(time_pause, (int, float))
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
    assert X_test.shape[0] == Y_test.shape[0] == mean_test.shape[0] \
        == std_test.shape[0] == acq_test.shape[0]
    if num_init is not None:
        assert X_train.shape[0] >= num_init

    if plt is None or pylab is None:
        logger.info('matplotlib or pylab is not installed.')
        return
    _set_font_config(use_tex)

    _, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[3, 1]})

    ax1.plot(X_test, Y_test, 'g', linewidth=4)
    ax1.plot(X_test, mean_test, 'b', linewidth=4)
    ax1.fill_between(X_test.flatten(),
        mean_test.flatten() - range_shade * std_test.flatten(),
        mean_test.flatten() + range_shade * std_test.flatten(),
        color='b',
        alpha=0.3)
    if num_init is not None:
        if X_train.shape[0] > num_init:
            ax1.plot(X_train[:num_init, :], Y_train[:num_init, :], 'x',
                c='saddlebrown', ms=14, markeredgewidth=6)
            ax1.plot(X_train[num_init:X_train.shape[0]-1, :],
                Y_train[num_init:X_train.shape[0]-1, :], 'rx',
                ms=14, markeredgewidth=6)
            ax1.plot(X_train[X_train.shape[0]-1, :],
                Y_train[X_train.shape[0]-1, :],
                c='orange', marker='+', ms=18, markeredgewidth=6)
        else:
            ax1.plot(X_train, Y_train, 'x', c='saddlebrown', ms=14, markeredgewidth=6)
    else:
        ax1.plot(X_train[:X_train.shape[0]-1, :],
            Y_train[:X_train.shape[0]-1, :], 'rx', ms=14, markeredgewidth=6)
        ax1.plot(X_train[X_train.shape[0]-1, :],
            Y_train[X_train.shape[0]-1, :], c='orange', marker='+',
            ms=18, markeredgewidth=6)

    _set_ax_config(ax1, None, str_y_axis, xlim_min=np.min(X_test),
        xlim_max=np.max(X_test), draw_zero_axis=draw_zero_axis)

    ax2.plot(X_test, acq_test, 'b', linewidth=4)
    _set_ax_config(ax2, str_x_axis, str_acq_axis, xlim_min=np.min(X_test),
        xlim_max=np.max(X_test), draw_zero_axis=draw_zero_axis)

    if path_save is not None and str_postfix is not None:
        _save_figure(path_save, str_postfix, str_prefix='bo_step_acq_')
    _show_figure(pause_figure, time_pause)
