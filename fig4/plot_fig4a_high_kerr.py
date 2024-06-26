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


def plot_fig4a_high_kerr(main_path):
    print("Plotting fig4b high kerr...")
    cal_filename = (
        f"{main_path}/fig4/data/215713_somerset_characteristic_function_1D.h5"
    )
    data_filename = (
        f"{main_path}/fig4/data/230309_somerset_characteristic_function_2D.h5"
    )

    threshold = -0.00017624835481500393

    # Get offset and amplitude from 1D vacuum data
    file = h5py.File(cal_filename, "r")
    data = file["data"]
    state_correction = np.array(data["state"])[:, 0]
    # ((np.array(data["I"])) < threshold).astype(int)
    # state_correction = np.average(state_correction, axis=0)
    x = np.arange(-1.9, 1.91 + 0.1 / 2, 0.1)
    file.close()
    popt = gaussian_fit(x, state_correction)
    amp = popt[0]
    sig = popt[2]
    ofs = popt[3]

    # Get data and apply threshold
    file = h5py.File(data_filename, "r")
    data = file["data"]
    state = (np.array(data["I"]) < threshold).astype(int)
    state = state.flatten()  # revert the buffering of qcore
    file.close()

    # separate internal loop into even (0.0) and odd (0.5) and state
    cohstate1_imag = state[0::5]
    cohstate2_imag = state[1::5]
    cohstate3_imag = state[2::5]
    cohstate4_imag = state[3::5]
    cohstate5_imag = state[4::5]

    # Get x and y sweeps for buffering and plotting
    x = np.arange(-1.9, 1.91 + 0.1 / 2, 0.1) / sig
    y = np.arange(-1.9, 1.91 + 0.1 / 2, 0.1) / sig

    reps = len(cohstate5_imag) // (len(y) * len(x))
    buffer = (reps, len(y), len(x))

    cohstate1_imag = correct_ofs_and_amp(
        reshape_and_avg(cohstate1_imag, buffer), amp, ofs
    )
    cohstate2_imag = correct_ofs_and_amp(
        reshape_and_avg(cohstate2_imag, buffer), amp, ofs
    )
    cohstate3_imag = correct_ofs_and_amp(
        reshape_and_avg(cohstate3_imag, buffer), amp, ofs
    )
    cohstate4_imag = correct_ofs_and_amp(
        reshape_and_avg(cohstate4_imag, buffer), amp, ofs
    )
    cohstate5_imag = correct_ofs_and_amp(
        reshape_and_avg(cohstate5_imag, buffer), amp, ofs
    )

    WIDTH = 2.5806259314456037
    HEIGTH = 2.5974

    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(WIDTH * cm, HEIGTH * cm))
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.set_xticks(ticks=[2, 0, -2])
    ax.set_yticks(ticks=[2, 0, -2])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    im = ax.pcolormesh(
        x,
        y,
        cohstate5_imag,
        norm=colors.Normalize(vmin=-1, vmax=1),
        cmap="bwr",
        linewidth=0,
        rasterized=True,
    )
    fig.savefig(f"{main_path}/fig4/fig4a_high_kerr.pdf")
    return cohstate5_imag


if __name__ == "__main__":
    plot_fig4a_high_kerr(main_path=main_path)
