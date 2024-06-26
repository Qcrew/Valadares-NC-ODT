import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors


main_path = None


def plot_supp_fig2(main_path):
    file = np.load(
        f"{main_path}/supp_fig2/data/224810_somerset_qubit_spec_current_sweep_cut.npz"
    )
    z_signal = file["z_signal"]
    x_currents = file["x_currents"]
    y_frequencies = file["y_frequencies"]
    plt.rcParams["savefig.dpi"] = 300

    WIDTH = 8
    HEIGTH = 8

    font_size = 5
    line_width = 1
    cm = 1 / 2.54

    fig, ax = plt.subplots(figsize=(WIDTH * cm, HEIGTH * cm))
    plt.setp(ax.spines.values(), linewidth=line_width)

    color = [(0.0, 0.02, 0.35), (0.0, 0.6, 0.3), (1.0, 1.0, 0.0)]
    # Dark Teal (more visible), Green, Yellow
    # Teal (less blue), Green, Yellow

    custom_cmap = colors.LinearSegmentedColormap.from_list("custom_cmap", color, N=256)

    z = z_signal.T
    z = z - (2 * z.min())
    map = ax.pcolormesh(
        x_currents,
        y_frequencies,
        z,
        norm=colors.LogNorm(vmin=z.min(), vmax=z.max()),
        cmap=custom_cmap,
        shading="auto",
        # linewidth=0,
        rasterized=True,
    )

    # cbar = plt.colorbar(map, orientation="vertical", pad=0.2)
    # cbar.set_ticks([2e-5, 5e-5, 10e-5, 15e-5, 20e-5])
    # cbar.set_ticks([], minor=True)
    # cbar.set_ticklabels(["2", "5", "10", "15", "20"])
    # for t in cbar.ax.get_xticklabels():
    #     t.set_fontsize(font_size)
    # cbar.outline.set_linewidth(line_width)

    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")

    cav_frequency = 5.793e9 - 49e6
    plot_1_freq = 5.766e9 - 43.50e6
    plot_2_freq = 5.766e9 - 56.66e6
    plot_3_freq = 5.766e9 - 78.4e6

    # ax.plot(
    #     [min(x_phis), max(x_phis)], [plot_1_freq, plot_1_freq], linestyle="--", c="white"
    # )
    # ax.plot(
    #     [min(x_phis), max(x_phis)], [plot_2_freq, plot_2_freq], linestyle="--", c="white"
    # )

    grid_range = [cav_frequency - 20e6 * i for i in range(-4, 3, 2)]
    ax.set_yticks(
        ticks=grid_range, labels=[f"{(x - cav_frequency)/1e6:.0f}" for x in grid_range]
    )
    # print(grid_range)
    # ax.set_yticks(ticks=[-40, 0, 40, 80])
    # ax.set_xticks(ticks=[0.540, 0.545, 0.550, 0.555])
    ax.set_xticks(ticks=[-1e-3, -0.75e-3, -0.5e-3, -0.25e-3, 0])
    # ax.set_ylabel("Detuning from cavity (MHz)")
    # ax.set_xlabel(r"Threaded reduced flux (rad/$2/pi$)")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.grid(False)
    fig.savefig(
        f"{main_path}/supp_fig2/supp_fig2.pdf",
    )
    return z


if __name__ == "__main__":
    plot_supp_fig2(main_path=main_path)
