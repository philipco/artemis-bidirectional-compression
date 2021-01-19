"""
Created by Philippenko, 6th March 2020.

This python file provide facilities to plot the results of a (multiple) gradient descent run.
"""
import matplotlib
import matplotlib.pyplot as plt

markers = ["o", "v", "s", "p", "X", "d", "P", "*", "<"]
markersize = 1
curve_size=3
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

    plt.figure(figsize=figsize)
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
            plt.errorbar(abscisse, objectives_dist, yerr=error_to_plot, label=legend_i, lw=lw, marker=markers[it],
                         markersize=ms)

        else:
            objectives_dist = error_distance
            plt.plot(abscisse, objectives_dist, label=legend[i], lw=lw, marker=markers[it], markersize=ms)
        it += 1

    if batch_size is None:
        title_precision = "\n(N=" + str(nb_devices) +", d=" + str(nb_dim) + ")"
    else:
        title_precision = "\n(N=" + str(nb_devices) + ", d=" + str(nb_dim) + ", b=" + str(batch_size) + ")"

    x_legend = x_legend if x_legend is not None else "Number of passes on data"
    setup_plot(x_legend + title_precision, ylegends=ylegends, xlog=(x_points is not None), xlabels=xlabels,
               ylim=ylim, picture_name=picture_name)



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


def setup_plot(xlegends, ylegends="loss", fontsize=fontsize, xticks_fontsize=fontsize, ylog: bool = False, xlog: bool = False,
               xlabels=None, ylim=False, picture_name=None, ax = None):
  
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
        # To limit the number of ticks on xaxis.
        if not xlog:
            plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        plt.xticks(fontsize=xticks_fontsize)
    plt.xlabel(xlegends, fontsize=fontsize)
    plt.ylabel(Y_LEGENDS[ylegends], fontsize=fontsize)
    plt.legend(loc='best', fontsize=fontsize_legend)
    plt.tight_layout()
    if picture_name:
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
