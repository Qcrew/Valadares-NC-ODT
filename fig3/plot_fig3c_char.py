from re import X
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from scipy.optimize import curve_fit

main_path = None


def reshape_and_avg(data_array, buffer):
    return np.average(np.reshape(data_array, buffer), axis=0)


def correct_ofs_and_amp(data_array, amp, ofs):
    return (data_array - ofs) / amp


def gaussian(x, a, b, sigma, c):
    return a * np.exp(-((x - b) ** 2) / (2 * sigma**2)) + c


def gaussian_fit(x, y):
    mean_arg = np.argmax(y)
    mean = x[mean_arg]
    # fit_range = int(0.2 * len(x))
    x_sample = x  # [mean_arg - fit_range : mean_arg + fit_range]
    y_sample = y  # [mean_arg - fit_range : mean_arg + fit_range]
    popt, pcov = curve_fit(
        gaussian,
        x_sample,
        y_sample,
        bounds=(
            (-np.inf, min(x_sample), -np.inf, -np.inf),
            (np.inf, max(x_sample), np.inf, np.inf),
        ),
        p0=[max(y) - min(y), 0, 0.5, max(y)],
    )
    return popt


def plot_fig3c_char(main_path):
    print("Plotting figure 3c char plots ...")
    cal_filename = (
        f"{main_path}/fig3/data/172001_somerset_characteristic_function_1D.h5"
    )
    data_filename = (
        f"{main_path}/fig3/data/174656_somerset_characteristic_function_2D.h5"
    )

    contrasts_filename = (
        f"{main_path}/fig3/data/173008_somerset_characteristic_function_1D.h5"
    )

    threshold = -0.0002898168253436553

    # Get offset and amplitude from 1D vacuum data
    file = h5py.File(cal_filename, "r")
    data = file["data"]
    state_correction = np.array(data["state"])
    # ((np.array(data["I"])) < threshold).astype(int)
    # state_correction = np.average(state_correction, axis=0)
    x = np.arange(-1.9, 1.91 + 0.1 / 2, 0.1)
    file.close()
    popt = gaussian_fit(x, state_correction)
    y_fit = gaussian(x, *popt)
    amp = popt[0]
    sig = popt[2]
    ofs = popt[3]

    # Get high-average contrasts for each state
    file = h5py.File(contrasts_filename, "r")
    data = file["data"]
    states_contrast = np.array(data["state"])
    contrast_0 = np.max((states_contrast[:, 0] - ofs) / amp)
    contrast_1 = np.max((states_contrast[:, 1] - ofs) / amp)
    contrast_2 = np.max((states_contrast[:, 2] - ofs) / amp)

    file = h5py.File(data_filename, "r")
    data = file["data"]
    state = (np.array(data["I"]) < threshold).astype(int)
    state = state.flatten()  # revert the buffering of qcore
    file.close()

    # separate internal loop into even (0.0) and odd (0.5) and state
    vacuum_real = state[::6]
    vacuum_imag = state[1::6]
    fock1_real = state[2::6]
    fock1_imag = state[3::6]
    fock2_real = state[4::6]
    fock2_imag = state[5::6]

    # Get x and y sweeps for buffering and plotting
    x = np.arange(-1.9, 1.91 + 0.1 / 2, 0.1)
    y = np.arange(-1.9, 1.91 + 0.1 / 2, 0.1)

    reps = len(vacuum_real) // (len(y) * len(x))
    buffer = (reps, len(y), len(x))

    vacuum_real_avg = np.average(np.reshape(vacuum_real, buffer), axis=0)
    vacuum_imag_avg = np.average(np.reshape(vacuum_imag, buffer), axis=0)
    fock1_real_avg = np.average(np.reshape(fock1_real, buffer), axis=0)
    fock1_imag_avg = np.average(np.reshape(fock1_imag, buffer), axis=0)
    fock2_real_avg = np.average(np.reshape(fock2_real, buffer), axis=0)
    fock2_imag_avg = np.average(np.reshape(fock2_imag, buffer), axis=0)

    ofs = np.average(vacuum_imag_avg)  # second estimate of offset
    vacuum_real_cor = correct_ofs_and_amp(
        reshape_and_avg(vacuum_real, buffer), amp, ofs
    )
    vacuum_imag_cor = correct_ofs_and_amp(
        reshape_and_avg(vacuum_imag, buffer), amp, ofs
    )
    ofs_f1 = np.average(fock1_imag_avg)
    # ofs_f1 = ofs
    fock1_real_cor = correct_ofs_and_amp(
        reshape_and_avg(fock1_real, buffer), amp, ofs_f1
    )
    fock1_imag_cor = correct_ofs_and_amp(
        reshape_and_avg(fock1_imag, buffer), amp, ofs_f1
    )
    ofs_f2 = np.average(fock2_imag_avg)
    # ofs_f2 = ofs
    fock2_real_cor = correct_ofs_and_amp(
        reshape_and_avg(fock2_real, buffer), amp, ofs_f2
    )
    fock2_imag_cor = correct_ofs_and_amp(
        reshape_and_avg(fock2_imag, buffer), amp, ofs_f2
    )

    # print(ofs, amp)
    x = x / sig
    y = y / sig

    WIDTH = 2.5806259314456037
    HEIGTH = 2.5974

    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(WIDTH * cm, HEIGTH * cm))
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks(ticks=[2, 0, -2])
    ax.set_yticks(ticks=[2, 0, -2])

    im = ax.pcolormesh(
        x,
        y,
        fock1_real_cor,
        norm=colors.Normalize(vmin=-1, vmax=1),
        cmap="bwr",
        linewidth=0,
        rasterized=True,
    )
    ax.set_aspect("equal", "box")
    fig.savefig(f"{main_path}/fig3/fig3c_fock1_char.pdf", dpi=300)

    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(WIDTH * cm, HEIGTH * cm))
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks(ticks=[2, 0, -2])
    ax.set_yticks(ticks=[2, 0, -2])
    im = ax.pcolormesh(
        x,
        y,
        fock2_real_cor,
        norm=colors.Normalize(vmin=-1, vmax=1),
        cmap="bwr",
        linewidth=0,
        rasterized=True,
    )
    ax.set_aspect("equal", "box")
    # uncomment this part to plot colour bar
    # cbar = plt.colorbar(im, pad=0.2)
    # cbar.ax.set_xticks([])
    # cbar.ax.set_yticks([])
    fig.savefig(f"{main_path}/fig3/fig3c_fock2_char.pdf", dpi=300)


if __name__ == "__main__":
    plot_fig3c_char(main_path=main_path)
