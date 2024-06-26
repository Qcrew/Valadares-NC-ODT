import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


main_path = None


def plot_fig2c(main_path):
    print("Plotting figure 2c...")
    DIR = f"{main_path}/fig2/data"
    FILE_NAME = "20231101_183019_somerset_pi_pulse_scope.h5"
    PISCOPE_1D_CUT_NAME = "pi_scope_1d_cut.npz"

    QUBIT_COLOR = "#4A9F79"

    FULL_PATH = f"{DIR}/{FILE_NAME}"
    CUT_PATH = f"{DIR}/{PISCOPE_1D_CUT_NAME}"
    LO = 6.2e9  # Hz
    ON_TIME = 18 * 4
    infile = h5py.File(FULL_PATH, "r")

    data = np.array(infile["data"])
    detuning = 1 * np.array(infile["x"]) / 1e6
    time = 4 * np.array(infile["y"])

    line_fit = np.load(CUT_PATH)
    line_fit_data = line_fit["data"]

    font_size = 5
    line_width = 1
    cm = 1 / 2.54  # centimeters in inches

    HEIGHT = 7
    WIDTH = 5
    PLOT_CBAR = False

    fig = plt.figure(figsize=(HEIGHT * cm, WIDTH * cm))
    if PLOT_CBAR:
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[29, 1])
        top_plot = gs[0, 0]
        bottom_plot = gs[1, 0]
    else:
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])
        top_plot = gs[0]
        bottom_plot = gs[1]

    line_fit_data = -1 * (line_fit_data - line_fit_data[0])
    stable_comp = line_fit_data[200:1000]
    avg_stable = np.average(stable_comp)
    norm_line = line_fit_data / avg_stable
    ax0 = plt.subplot(top_plot)
    ax0.plot((norm_line), color=QUBIT_COLOR, linestyle="-")
    ax0.set_yticks([0, 1])
    ax0.set_yticklabels([])
    ax0.grid()
    ax0.set_xlim(0, len(line_fit_data))
    ax0.tick_params(axis="y", direction="in")

    ax1 = plt.subplot(bottom_plot, sharex=ax0)

    # add inset
    axins = inset_axes(
        ax1,
        width="100%",
        height="100%",
        bbox_to_anchor=(0.2, 0.05, 0.3, 0.4),
        bbox_transform=ax1.transAxes,
        loc=3,
    )
    axins.plot(line_fit_data[45:100], color=QUBIT_COLOR)
    axins.tick_params(labelleft=False, labelbottom=False)
    axins.set_xticks([])
    axins.set_yticks([])
    sample = line_fit_data[45:100]
    offset = 1.5 * data.min()
    data = data - offset

    map = ax1.pcolormesh(
        time,
        detuning,
        data,
        norm=mcolors.LogNorm(vmin=data.min(), vmax=data.max()),
        cmap="Blues_r",
        shading="auto",
        # linewidth=0,
        rasterized=True,
    )
    ax1.tick_params(axis="x", direction="in")
    ax1.tick_params(axis="y", direction="in")
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_xticks([0, 500, 1000, 1500])
    ax1.set_yticks([-210, -170, -130, -90])
    plt.setp(ax0.get_xticklabels(), visible=False)
    # remove last tick label for the second subplot
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)

    if PLOT_CBAR:
        cbar_ticks_scale = np.array([0.3, 0.4, 0.6, 0.9])
        cbar_ticks = cbar_ticks_scale * data.max()
        cbar_actual = (cbar_ticks) + offset
        cbar_labels = [f"{i*10000:.3f}" for i in cbar_actual]

        cbar = plt.colorbar(map, orientation="vertical", cax=plt.subplot(gs[-1, 1:]))
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticks([], minor=True)
        cbar.set_ticklabels(cbar_labels)
        # cbar.set_ticklabels(cbar_ticks + offset)
        # cbar.set_ticklabels(["2", "5", "10", "15", "20"])
        for t in cbar.ax.get_xticklabels():
            t.set_fontsize(font_size)
        cbar.outline.set_linewidth(line_width)

    # plt.suptitle("Actual qubit response")
    # plt.ylabel("Change in qubit frequency (MHz)")
    # plt.xlabel("Time (ns)")
    plt.subplots_adjust(hspace=0.0)
    infile.close()


if __name__ == "__main__":
    plot_fig2c(main_path=main_path)
