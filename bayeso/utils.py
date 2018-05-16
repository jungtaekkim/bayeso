import os
import numpy as np
import matplotlib.pyplot as plt
import pylab

TIME_PAUSE = 2.0
RANGE_SHADE = 1.96

COLORS = [
    'red',
    'blue',
    'green',
    'orange',
    'purple',
    'olive',
    'darkred',
    'deepskyblue',
    'limegreen',
    'lightsalmon',
    'navy',
    'aquamarine',
    'rosybrown',
    'darkslategray',
    'darkkhaki',
]

MARKERS = [
    '.',
    'x',
    '*',
    '+',
    '^',
    'v',
    '<',
    '>',
    'd',
    ',',
    '8',
    'h',
    '1',
    '2',
    '3',
]

def get_minimum(list_read, num_init):
    list_all = []
    for list_file in list_read:
        cur_min = np.inf
        list_min = []
        for ind_cur in range(0, num_init):
#            print list_file
            if cur_min > float(list_file[ind_cur]):
                cur_min = float(list_file[ind_cur])
        list_min.append(float(cur_min))
        for ind_cur in range(num_init, len(list_file)):
            if cur_min > float(list_file[ind_cur]):
                cur_min = float(list_file[ind_cur])
            list_min.append(float(cur_min))
        list_all.append(list_min)
    list_all = np.array(list_all)
    mean_min = np.mean(list_all, axis=0)
    std_min = np.std(list_all, axis=0)
#    print mean_min, std_min
    return mean_min, std_min

def plot_gp(X_train, Y_train, X_test, mu, sigma, path_save=None, str_postfix=None, is_tex=False):
    if is_tex:
        plt.rc('text', usetex=True)
    else:
        plt.rc('pdf', fonttype=42)
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.plot(X_train.flatten(), Y_train.flatten(), 
        'x', 
        c=COLORS[0], 
        markersize=10, 
        mew=4)
    ax.plot(X_test.flatten(), mu.flatten(), 
        c=COLORS[1], 
        linewidth=4, 
        marker='None')
           
    ax.fill_between(X_test.flatten(), 
        mu.flatten() - RANGE_SHADE * sigma.flatten(), 
        mu.flatten() + RANGE_SHADE * sigma.flatten(), 
        color=COLORS[1], 
        alpha=0.3)
    ax.set_xlabel('$x$', fontsize=32)
    ax.set_ylabel('$y$', fontsize=32)
    ax.set_xlim([np.min(X_test), np.max(X_test)])
    ax.tick_params(labelsize=22)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
#    ax.spines['bottom'].set_position('zero')
    if path_save is not None and str_figure is not None:
        str_figure = 'gp_' + str_postfix
        plt.savefig(os.path.join(path_save, str_figure + '.pdf'), format='pdf', transparent=True, bbox_inches='tight', frameon=False)

    plt.ion()
    plt.pause(TIME_PAUSE)
    plt.close('all')

def plot_minimum(list_all, list_str_label, num_init, is_std, is_marker=True, is_legend=False, is_tex=False, path_save=None, str_postfix=None):
    if is_tex:
        plt.rc('text', usetex=True)
    else:
        plt.rc('pdf', fonttype=42)
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    for ind_read, list_read in enumerate(list_all):
        ind_color = ind_read % len(COLORS)
        ind_marker = ind_read % len(MARKERS)
        mean_min, std_min = get_minimum(list_read, num_init)
        x_data = range(0, mean_min.shape[0])
        y_data = mean_min
        std_data = std_min
        if is_marker:
            ax.plot(x_data, y_data, 
                label=list_str_label[ind_read], 
                c=COLORS[ind_color], 
                linewidth=4, 
                marker=MARKERS[ind_marker], 
                markersize=10, 
                mew=3)
        else:
            ax.plot(x_data, y_data, 
                label=list_str_label[ind_read], 
                c=COLORS[ind_color], 
                linewidth=4,
                marker='None')
           
        if is_std:
            ax.fill_between(x_data, 
                y_data - RANGE_SHADE * std_data, 
                y_data + RANGE_SHADE * std_data, 
                color=COLORS[ind_color], 
                alpha=0.3)
    lines, labels = ax.get_legend_handles_labels()
    ax.set_xlabel('Iteration', fontsize=27)
    ax.set_ylabel('Minimum function value', fontsize=27)
    ax.set_xlim([0, mean_min.shape[0]-1])
    ax.tick_params(labelsize=22)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    if is_legend:
        plt.legend(loc='upper right', fancybox=False, edgecolor='black', fontsize=24)
    if path_save is not None and str_postfix is not None:
        if is_std:
            str_figure = 'minimum_mean_std_' + str_postfix
        else:
            str_figure = 'minimum_mean_only_' + str_postfix
        plt.savefig(os.path.join(path_save, str_figure + '.pdf'), format='pdf', transparent=True, bbox_inches='tight', frameon=False)

        if not is_legend:
            fig_legend = pylab.figure(figsize=(3, 2))
            fig_legend.legend(lines, list_str_label, 'center', fancybox=False, edgecolor='black', fontsize=32)
            fig_legend.savefig(os.path.join(path_save, 'legend_' + str_postfix + '.pdf'), format='pdf', transparent=True, bbox_inches='tight', frameon=False)
    plt.ion()
    plt.pause(TIME_PAUSE)
    plt.close('all')

def plot_bo_step(X_train, Y_train, X_test, Y_test, mean_test, std_test, path_save=None, str_postfix=None, num_init=None, is_tex=False):
    if is_tex:
        plt.rc('text', usetex=True)
    else:
        plt.rc('pdf', fonttype=42)
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
#    ax.spines['bottom'].set_position('zero')
    ax.plot(X_test, Y_test, 'g', linewidth=4)
    ax.plot(X_test, mean_test, 'b', linewidth=4)
    ax.fill_between(X_test.flatten(), 
        mean_test.flatten() - RANGE_SHADE * std_test.flatten(), 
        mean_test.flatten() + RANGE_SHADE * std_test.flatten(), 
        color='blue', 
        alpha=0.3)
    if num_init is not None:
        if X_train.shape[0] > num_init:
            ax.plot(X_train[:num_init, :], Y_train[:num_init, :], 'x', c='saddlebrown', ms=14, markeredgewidth=6)
            ax.plot(X_train[num_init:X_train.shape[0]-1, :], Y_train[num_init:X_train.shape[0]-1, :], 'rx', ms=14, markeredgewidth=6)
            ax.plot(X_train[X_train.shape[0]-1, :], Y_train[X_train.shape[0]-1, :], c='orange', marker='+', ms=18, markeredgewidth=6)
        else:
            ax.plot(X_train, Y_train, 'x', c='saddlebrown', ms=14, markeredgewidth=6)
    else:
        ax.plot(X_train[:X_train.shape[0]-1, :], Y_train[:X_train.shape[0]-1, :], 'rx', ms=14, markeredgewidth=6)
        ax.plot(X_train[X_train.shape[0]-1, :], Y_train[X_train.shape[0]-1, :], c='orange', marker='+', ms=18, markeredgewidth=6)

    ax.set_xlabel('$x$', fontsize=32)
    ax.set_ylabel('$y$', fontsize=32)
    ax.set_xlim([np.min(X_test), np.max(X_test)])
    ax.tick_params(labelsize=22)
#    plt.legend(loc='upper right', fancybox=False, edgecolor='black', fontsize=24)
    if path_save is not None and str_postfix is not None:
        plt.savefig(os.path.join(path_save, 'bo_step_' + str_postfix + '.pdf'), format='pdf', transparent=True, bbox_inches='tight', frameon=False)
    plt.ion()
    plt.pause(TIME_PAUSE)
    plt.close('all')

def plot_bo_step_acq(X_train, Y_train, X_test, Y_test, mean_test, std_test, acq_test, path_save=None, str_postfix=None, num_init=None, is_tex=False):
    if is_tex:
        plt.rc('text', usetex=True)
    else:
        plt.rc('pdf', fonttype=42)
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[3, 1]})
    ax1.spines['top'].set_color('none')
    ax1.spines['right'].set_color('none')
#    ax1.spines['bottom'].set_position('zero')
    ax1.plot(X_test, Y_test, 'g', linewidth=4)
    ax1.plot(X_test, mean_test, 'b', linewidth=4)
    ax1.fill_between(X_test.flatten(), 
        mean_test.flatten() - RANGE_SHADE * std_test.flatten(), 
        mean_test.flatten() + RANGE_SHADE * std_test.flatten(), 
        color='blue', 
        alpha=0.3)
    if num_init is not None:
        if X_train.shape[0] > num_init:
            ax1.plot(X_train[:num_init, :], Y_train[:num_init, :], 'x', c='saddlebrown', ms=14, markeredgewidth=6)
            ax1.plot(X_train[num_init:X_train.shape[0]-1, :], Y_train[num_init:X_train.shape[0]-1, :], 'rx', ms=14, markeredgewidth=6)
            ax1.plot(X_train[X_train.shape[0]-1, :], Y_train[X_train.shape[0]-1, :], c='orange', marker='+', ms=18, markeredgewidth=6)
        else:
            ax1.plot(X_train, Y_train, 'x', c='saddlebrown', ms=14, markeredgewidth=6)
    ax1.set_ylabel('$y$', fontsize=32)
    ax1.set_xlim([np.min(X_test), np.max(X_test)])
    ax1.tick_params(labelsize=22)

    ax2.plot(X_test, acq_test, 'b', linewidth=4)
    ax2.set_xlabel('$x$', fontsize=32)
    ax2.set_ylabel('Acq.', fontsize=32)
    ax2.spines['top'].set_color('none')
    ax2.spines['right'].set_color('none')
    ax2.set_xlim([np.min(X_test), np.max(X_test)])
    ax2.tick_params(labelsize=22)
    ax2.tick_params('y', labelsize=14)
#    plt.legend(loc='upper right', fancybox=False, edgecolor='black', fontsize=24)
    if path_save is not None and str_postfix is not None:
        plt.savefig(os.path.join(path_save, 'bo_step_acq_' + str_postfix + '.pdf'), format='pdf', transparent=True, bbox_inches='tight', frameon=False)
    plt.ion()
    plt.pause(TIME_PAUSE)
    plt.close('all')


if __name__ == '__main__':
    pass

