"""
Created by Philippenko, 6th March 2020.

This python file provide facilities to plot the results of a (multiple) gradient descent run.
"""

import matplotlib.pyplot as plt
import numpy as np

markers = ["o", "v", "s", "p", "X", "d", "P", "*"]
markersize = 2
curve_size=4
fontsize=30
fontsize_legend=21
figsize=(8,7)
fourfigsize=(13, 8)
sixfigsize=(13, 11)

nb_bars = 1  # = 3 when running 400 iterations, to plot 1 on nb_bars error bars.


def plot_error_dist(all_losses, legend, nb_devices, nb_dim, batch_size=None, all_error=None,
                    x_points=None, x_legend=None, one_on_two_points=True, xlabels=None, ylim=False):
    N_it = len(all_losses[0])
    plt.figure(figsize=figsize)
    it = 0

    for i in range(min(len(all_losses), len(markers))):
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
            plt.errorbar(abscisse, objectives_dist, yerr=error_to_plot, label=legend[i], lw=lw, marker=markers[it], markersize=ms)

        else:
            plt.plot(abscisse, objectives_dist, label=legend[i], lw=lw, marker=markers[it], markersize=ms)
        it += 1

    if batch_size is None:
        title_precision = "\n(N=" + str(nb_devices) +", d=" + str(nb_dim) + ")"
    else:
        title_precision = "\n(N=" + str(nb_devices) + ", d=" + str(nb_dim) + ", b=" + str(batch_size) + ")"

    x_legend = x_legend if x_legend is not None else "Number of passes on data"
    setup_plot(x_legend + title_precision, r"$\log_{10}(F(w^k) - F(w^*))$", xlog=(x_points is not None), xlabels=xlabels,
               ylim=ylim)


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
        objectives_error = [objectives.get_std(obj_min[objective_keys[i]])[j][-1] for j in range(len(objectives.names))]
        lw = curve_size-1 if len(objectives_dist) > 40 else curve_size
        ms = markersize-1 if len(objectives_dist) > 40 else markersize
        plt.errorbar(x_points, objectives_dist, yerr=objectives_error, label=legend[i], lw=lw, marker=markers[it], markersize=ms)
        it += 1

    plt.xticks([i for i in range(1, len(xlabels) + 1)], xlabels, rotation=25)

    title_precision = "\n(d={0})".format(nb_dim)

    setup_plot(x_legend + title_precision, r"$\log_{10}(F(w^k) - F(w^*))$", xticks_fontsize=15,
               xlog=False)


def setup_plot(xlegends, ylegends, fontsize=fontsize, xticks_fontsize=fontsize, ylog: bool = False, xlog: bool = False,
               xlabels=None, ylim=False):
    if ylog:
        plt.yscale("log")
    if ylim:
        plt.ylim(top=1)
    if xlog:
        plt.xscale("log")
    plt.yticks(fontsize=fontsize)
    plt.grid()
    if xlabels:
        plt.xticks([i for i in range(0, len(xlabels))], xlabels, rotation=40, fontsize=xticks_fontsize-3)
    else:
        plt.xticks(fontsize=xticks_fontsize)
        # plt.xticks(np.arange(0, 401, step=100), fontsize=xticks_fontsize)
    plt.xlabel(xlegends, fontsize=fontsize)
    plt.ylabel(ylegends, fontsize=fontsize)
    plt.legend(loc='best', fontsize=fontsize_legend)
    plt.tight_layout()
    plt.show()


def logistic_plot(X, Y):
    """Plot the logistic distribution of a dataset of dimension 2."""
    plt.scatter(*X[Y == 1].T, color='b', s=10, label=r'$y_i=1$')
    plt.scatter(*X[Y == -1].T, color='r', s=10, label=r'$y_i=-1$')
    plt.legend(loc='upper left')
    plt.xlabel(r"$x_i^1$", fontsize=16)
    plt.ylabel(r"$x_i^2$", fontsize=16)
    plt.title("Logistic regression simulation", fontsize=18)
