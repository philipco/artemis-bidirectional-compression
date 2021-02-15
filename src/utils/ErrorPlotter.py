"""
Created by Philippenko, 6th March 2020.

This python file provide facilities to plot the results of a (multiple) gradient descent run.
"""
import math

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from numpy import quantile

from src.utils.Utilities import drop_nan_values, keep_until_found_nan

markers = ["o", "v", "s", "p", "X", "d", "P", "*", "<"]
markersize = 1
curve_size=4
fontsize=30
fontsize_legend=14
# figsize=(15,7)
figsize=(8,7)
fourfigsize=(13, 8)
sixfigsize=(13, 11)

Y_LEGENDS = {"loss": r"$\log_{10}(F(w^k) - F(w^*))$",
             "ef": r"$\log_{10}(\| \| EF_k \| \|)$",
             "rand_dist": r"$\log_{10}(\mathbb{E} \| \| w_k - w_k^i \| \|^2)$",
             "rand_var": r"$\log_{10}( \| \| \mathbb{V}~[w_k^i] \| \| )$"}

nb_bars = 1  # = 3 when running 400 iterations, to plot 1 on nb_bars error bars.


def plot_error_dist(all_losses, legend, nb_devices, nb_dim, batch_size=None, all_error=None,
                    x_points=None, x_legend=None, one_on_two_points=True, xlabels=None,
                    ylegends="loss", ylim=False, omega_c = None, picture_name=None):

    assert ylegends in Y_LEGENDS.keys(), "Possible values for ylegend are : " + str([key for key in Y_LEGENDS.keys()])

    legend = [l if l != "ArtemisEF" else "Dore" for l in legend]

    N_it = len(all_losses[0])

    # If there is less than 50 points, we plot each point !
    if N_it < 50:
        one_on_two_points = False
    xlog = (x_points is not None)

    fig, ax = plt.subplots(figsize=figsize)
    # if xlog:
    #     axins = zoomed_inset_axes(ax, zoom=2.5, loc=3)
    # else:
    #     axins = zoomed_inset_axes(ax, zoom=3, loc=2)
    it = 0

    nb_curves = min(len(all_losses), len(markers))

    for i in range(nb_curves):
        abscisse = [i for i in range(N_it)]
        error_distance = all_losses[i]
        lw = curve_size-1 if len(error_distance) > 40 else curve_size
        ms = markersize-1 if len(error_distance) > 40 else markersize

        if x_points is not None:
            abscisse = x_points[i]

        if all_error is not None:
            if one_on_two_points:
                # if we plot error bar we don't take all elements
                objectives_dist = [error_distance[0]] + list(error_distance[i + 1:N_it - 1:nb_bars * (len(all_losses)-1)]) + [
                    error_distance[-1]]
                abscisse = [abscisse[0]] + abscisse[i + 1:N_it - 1:nb_bars * (len(all_losses)-1)] + [abscisse[-1]]

                error_to_plot = [all_error[i][0]] + list(all_error[i][i + 1:N_it - 1:nb_bars * (len(all_losses)-1)]) + [
                    all_error[i][-1]]
            else:
                objectives_dist, error_to_plot = error_distance, all_error[i]
            legend_i = legend[i]
            if omega_c:
                legend_i = legend_i + " {0}".format(str(omega_c[i]))[:4]
            ax.errorbar(abscisse, objectives_dist, yerr=error_to_plot, label=legend_i, lw=lw, marker=markers[it],
                         markersize=ms)
            # setup_zoom(ax, axins, abscisse, objectives_dist, xlog, legend, i, it, ms, lw)

        else:
            objectives_dist = error_distance
            ax.plot(abscisse, objectives_dist, label=legend[i], lw=lw, marker=markers[it], markersize=ms)
            # setup_zoom(ax, axins, abscisse, objectives_dist, xlog, legend, i, it, ms, lw)
        it += 1

    if batch_size is None:
        title_precision = "\n(N=" + str(nb_devices) +", d=" + str(nb_dim) + ")"
    else:
        title_precision = "\n(N=" + str(nb_devices) + ", d=" + str(nb_dim) + ", b=" + str(batch_size) + ")"

    x_legend = x_legend if x_legend is not None else "Number of passes on data"
    setup_plot(x_legend + title_precision, ylegends=ylegends, xlog=xlog, xlabels=xlabels,
               ylim=ylim, picture_name=picture_name, ax=ax, fig=fig)


def plot_multiple_run_each_curve_different_objectives(x_points, all_losses, nb_dim, legend, obj_min, objective_keys,
                                                      xlabels, x_legend, subplot=None):
    if subplot is None:
        plt.figure(figsize=(8, 6))
    else:
        plt.subplot(subplot)
    it = 0

    for i in range(len(all_losses)):
        objectives = all_losses[i]
        objectives_dist = [objectives.get_loss(obj_min[objective_keys[i]])[j][-1] for j in range(len(objectives.names))]
        objectives_error = [objectives.getter_std(obj_min[objective_keys[i]])[j][-1] for j in range(len(objectives.names))]
        lw = curve_size-1 if len(objectives_dist) > 40 else curve_size
        ms = markersize-1 if len(objectives_dist) > 40 else markersize
        plt.errorbar(x_points, objectives_dist, yerr=objectives_error, label=legend[i], lw=lw, marker=markers[it], markersize=ms)
        it += 1

    plt.xticks([i for i in range(1, len(xlabels) + 1)], xlabels, rotation=25)

    title_precision = "\n(d={0})".format(nb_dim)

    setup_plot(x_legend + title_precision, xticks_fontsize=15, xlog=False)



def setup_zoom(ax, axins, abscisse, objectives_dist, xlog, legend, i, it, ms, lw):

    axins.plot(abscisse, objectives_dist, label=legend[i], lw=lw, marker=markers[it], markersize=ms)

    objectives_dist = drop_nan_values(objectives_dist)
    abscisse = abscisse[:len(objectives_dist)]

    max_x = abscisse[-1]

    axins.set_xlim(390, 500)  # Limit the region for zoom
    axins.set_ylim(-1.63, -1.4)
    plt.xticks(visible=False)  # Not present ticks
    plt.yticks(visible=False)
    ## draw a bbox of the region of the inset axes in the parent axes and
    ## connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.4")
    plt.draw()


def setup_plot(xlegends, ylegends="loss", fontsize=fontsize, xticks_fontsize=fontsize, ylog: bool = False, xlog: bool = False,
               xlabels=None, ylim=False, picture_name=None, ax = None, fig=None):
  
    if ylog:
        ax.yscale("log")
    if ylim:
        ax.set_ylim(top=2)
    if xlog:
        ax.set_xscale("log")
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.grid()
    if xlabels:
        plt.xticks([i for i in range(0, len(xlabels))], xlabels, rotation=40, fontsize=xticks_fontsize-3)
    else:
        # To limit the number of ticks on xaxis.
        if not xlog:
            plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        plt.xticks(fontsize=xticks_fontsize)
    ax.set_xlabel(xlegends, fontsize=fontsize)
    ax.set_ylabel(Y_LEGENDS[ylegends], fontsize=fontsize)
    ax.legend(loc='upper right', fontsize=fontsize_legend)
    fig.tight_layout()
    if True:
        plt.savefig('{0}.eps'.format(picture_name), format='eps')
    else:
        plt.show()


def logistic_plot(X, Y):
    """Plot the logistic distribution of a dataset of dimension 2."""
    plt.scatter(*X[Y == 1].T, color='b', s=10, label=r'$y_i=1$')
    plt.scatter(*X[Y == -1].T, color='r', s=10, label=r'$y_i=-1$')
    plt.legend(loc='upper left')
    plt.xlabel(r"$x_i^1$", fontsize=16)
    plt.ylabel(r"$x_i^2$", fontsize=16)
    plt.title("Logistic regression simulation", fontsize=18)
