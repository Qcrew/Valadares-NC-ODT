import numpy as np
import matplotlib.pyplot as plt

main_path = None


def plot_fig2b(main_path):
    print("Plotting figure 2b...")
    FILE_NAME = "constcos10ns_1k5on_E2pF2pG2pH2.npz"

    data_file_path = f"{main_path}/fig2/data/{FILE_NAME}"
    FLUX_COLOR = "#F68721"

    file = np.load(data_file_path)
    data = file["I_quad"]
    data = data[:1616]

    # pad to match with pi-scope date
    padding = [0] * 60
    data = list(padding) + list(data)
    # print(len(data))
    # print(data[65:74])
    cm = 1 / 2.54

    fig, ax = plt.subplots(figsize=(7 * cm, 3 * cm))

    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # ax.set_title("Predistorted flux pulse")
    ax.set_yticks(ticks=[1.0, 0, -1.0])
    ax.set_xticks([0, 500, 1000, 1500])
    # ax.axhline(y=0, color="k", linewidth=0.5)
    plt.grid()
    # ax.axvline(x=0, color="k")

    plt.ylim([-1.1, 1.1])
    plt.xlim((0, 1676))
    ax.plot(data, color=FLUX_COLOR)
    plt.savefig(f"{main_path}/fig2/fig_2b.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    plot_fig2b(main_path=main_path)
