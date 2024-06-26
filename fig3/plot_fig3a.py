import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize


main_path = None


def plot_fig3a(main_path):
    print("Plotting figure 3a...")
    cutoff = 2000
    fname = "104524_somerset_vacuum_rabi.h5"
    input_file = f"{main_path}/fig3/data/{fname}"
    font_size = 5
    line_width = 0.5

    # Define a custom colormap with enhanced visibility
    colors = [(0.0, 0.02, 0.35), (0.0, 0.6, 0.3), (1.0, 1.0, 0.0)]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    flipped_cmap = custom_cmap.reversed()

    infile = h5py.File(input_file, "r")
    x = np.array(infile["data"]["x"])
    y = np.array(infile["data"]["internal sweep"])
    z = np.array(infile["data"]["state"])
    raw_I = np.array(infile["data"]["I"])
    threshold = infile["operations"]["readout_pulse"].attrs["threshold"]

    cut_I = raw_I[:cutoff]
    states = [[(i < threshold) for i in row] for row in cut_I]
    reshaped_states = np.reshape(states, (cutoff, 73, 50))
    avg_states = np.mean(reshaped_states, axis=0)
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(6 * cm, 5 * cm))

    map = ax.pcolormesh(
        x,
        y,
        avg_states,
        cmap=flipped_cmap,  # "summer_r",
        norm=Normalize(vmin=0, vmax=1),
        shading="auto",
        linewidth=0,
        rasterized=True,
    )
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    plt.setp(ax.spines.values(), linewidth=line_width)
    plt.xticks(fontsize=font_size, ticks=[0.40, 0.47, 0.55])
    plt.yticks(fontsize=font_size, ticks=[50, 150, 250, 350])

    cbar = plt.colorbar(map)
    cbar.ax.set_yticks([0, 0.5, 1.0])
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(font_size)
    cbar.outline.set_linewidth(line_width)

    fig.savefig(f"{main_path}/fig3/fig3a.pdf")
    infile.close()


if __name__ == "__main__":
    plot_fig3a(main_path=main_path)
