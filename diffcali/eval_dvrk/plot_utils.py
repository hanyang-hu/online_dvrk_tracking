import numpy as np

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


# data: n * 2 (* layer) (x, y, number of curves)
def curve2D(
    data,
    colors,
    x_label,
    y_label,
    x_lim=None,
    y_lim=None,
    title=None,
    legends=None,
    legend_loc="best",
    marker=None,
    grid=False,
    figsize=None,
    fontsize=None,
    fig_name=None,
):

    # plt.clf()

    curve_num = 1 if len(data.shape) == 2 else data.shape[2]
    assert len(colors) == curve_num
    if not legends is None:
        assert len(legends) == curve_num
    else:
        legends = [None for _ in range(curve_num)]

    fig, ax = plt.subplots(figsize=figsize)

    if curve_num == 1:
        data = data[:, :, np.newaxis]
    for c in range(curve_num):
        ax.plot(
            data[:, 0, c],
            data[:, 1, c],
            color=colors[c],
            marker=marker,
            label=legends[c],
        )

    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    if not x_lim is None:
        plt.xlim(x_lim)
    if not y_lim is None:
        plt.ylim(y_lim)

    """
    for tick in ax.xaxis.get_major_ticks(): 
        tick.label.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks(): 
        tick.label.set_fontsize(fontsize)
    """
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)

    if not title is None:
        plt.title(title)
    if not legends[0] is None:
        plt.legend(loc=legend_loc)

    if grid:
        plt.grid()

    if fig_name is None:
        plt.show()
    else:
        plt.savefig(fig_name)


# data_x: n (x)
# data_mean/max/min: n (* layer) (y, number of curves)
def curveRange2D(
    data_x,
    data_mean,
    data_max,
    data_min,
    colors,
    x_label,
    y_label,
    x_lim=None,
    y_lim=None,
    title=None,
    legends=None,
    legend_loc="best",
    marker=None,
    grid=False,
    fig_name=None,
):

    plt.clf()
    curve_num = 1 if len(data_mean.shape) == 1 else data_mean.shape[1]
    assert len(colors) == curve_num
    if not legends is None:
        assert len(legends) == curve_num
    else:
        legends = [None for _ in range(curve_num)]

    if curve_num == 1:
        data_mean = data_mean[:, np.newaxis]
        data_max = data_max[:, np.newaxis]
        data_min = data_min[:, np.newaxis]
    for c in range(curve_num):
        plt.plot(
            data_x, data_mean[:, c], color=colors[c], marker=marker, label=legends[c]
        )
        plt.fill_between(
            data_x, data_max[:, c], data_min[:, c], facecolor=colors[c], alpha=0.5
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if not x_lim is None:
        plt.xlim(x_lim)
    if not y_lim is None:
        plt.ylim(y_lim)

    if not title is None:
        plt.title(title)
    if not legends[0] is None:
        plt.legend(loc=legend_loc)

    if grid:
        plt.grid()

    if fig_name is None:
        plt.show()
    else:
        plt.savefig(fig_name)


def boxPlot(
    data, x_labels, y_label=None, y_lim=None, title=None, grid=True, fig_name=None
):
    fig, ax = plt.subplots()
    ax.boxplot(data)
    ax.set_xticklabels(x_labels)

    if not y_label is None:
        ax.set_ylabel(y_label)
    if not y_lim is None:
        ax.set_ylim(y_lim)
    if not title is None:
        plt.title(title)

    if grid:
        plt.grid()

    if fig_name is None:
        plt.show()
    else:
        plt.savefig(fig_name)


def violinPlot(
    data,
    x_labels,
    y_label=None,
    y_lim=None,
    figsize=None,
    title=None,
    grid=False,
    fontsize=None,
    fig_name=None,
):
    fig, ax = plt.subplots(figsize=figsize)
    parts = ax.violinplot(data, showextrema=False)

    for pc in parts["bodies"]:
        pc.set_facecolor("blue")
        pc.set_edgecolor("black")
        pc.set_alpha(0.5)

    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)

        return lower_adjacent_value, upper_adjacent_value

    quartile1, medians, quartile3 = (
        np.zeros(len(data)),
        np.zeros(len(data)),
        np.zeros(len(data)),
    )
    for idx in range(len(data)):
        quartile1[idx], medians[idx], quartile3[idx] = np.percentile(
            data[idx], [25, 50, 75]
        )
    whiskers = np.array(
        [
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data, quartile1, quartile3)
        ]
    )
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker="o", color="white", s=30, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color="k", linestyle="-", lw=5)
    ax.vlines(inds, whiskers_min, whiskers_max, color="k", linestyle="-", lw=1)

    ax.set_xticks(np.arange(1, len(x_labels) + 1))
    ax.set_xticklabels(x_labels, fontsize=fontsize)

    if not y_label is None:
        ax.set_ylabel(y_label, fontsize=fontsize)
    if not y_lim is None:
        ax.set_ylim(y_lim)
    if not title is None:
        plt.title(title)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    if grid:
        plt.grid()

    if fig_name is None:
        plt.show()
    else:
        plt.savefig(fig_name, bbox_inches="tight")


# data: [n * 3 (x,y,z), n * 3, ...]
def scatter3DPlot(
    data,
    color=["b"],
    marker=["."],
    marker_size=[5],
    x_lim=None,
    y_lim=None,
    z_lim=None,
    x_label=None,
    y_label=None,
    z_label=None,
    show=True,
    fig_name=None,
):
    assert len(data) == len(color) and len(data) == len(marker)

    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(len(data)):
        ax.scatter(
            data[i][:, 0],
            data[i][:, 1],
            data[i][:, 2],
            color=color[i],
            marker=marker[i],
            s=marker_size[i],
        )

    if not x_lim is None:
        ax.set_xlim(x_lim)
    if not y_lim is None:
        ax.set_ylim(y_lim)
    if not z_lim is None:
        ax.set_zlim(z_lim)

    if not x_label is None:
        ax.set_xlabel(x_label)
    if not y_label is None:
        ax.set_ylabel(y_label)
    if not z_label is None:
        ax.set_zlabel(z_label)

    if not show:
        return ax

    if fig_name is None:
        plt.show()
    else:
        plt.savefig(fig_name)
