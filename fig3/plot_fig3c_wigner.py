from re import X
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from scipy.optimize import curve_fit

main_path = None


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
            (0, min(x_sample), -np.inf, -np.inf),
            (np.inf, max(x_sample), np.inf, np.inf),
        ),
        p0=[max(y) - min(y), 0, 0.5, min(y)],
    )
    return popt


def plot_fig3c_wigner(main_path):
    print("Plotting figure 3c wigner plots ...")
    wigner_calibration_list = (
        f"{main_path}/fig3/data/160440_somerset_wigner_1d.h5",
        f"{main_path}/fig3/data/165511_somerset_wigner_1d.h5",
        f"{main_path}/fig3/data/175556_somerset_wigner_1d.h5",
        f"{main_path}/fig3/data/190334_somerset_wigner_1d.h5",
        f"{main_path}/fig3/data/202234_somerset_wigner_1d.h5",
        f"{main_path}/fig3/data/222038_somerset_wigner_1d.h5",
        f"{main_path}/fig3/data/232647_somerset_wigner_1d.h5",
        f"{main_path}/fig3/data/003032_somerset_wigner_1d.h5",
    )

    filepath_list = [
        f"{main_path}/fig3/data/161939_somerset_wigner_2d.h5",
        f"{main_path}/fig3/data/171210_somerset_wigner_2d.h5",
        f"{main_path}/fig3/data/181404_somerset_wigner_2d.h5",
        f"{main_path}/fig3/data/193655_somerset_wigner_2d.h5",
        f"{main_path}/fig3/data/203858_somerset_wigner_2d.h5",
        f"{main_path}/fig3/data/223536_somerset_wigner_2d.h5",
        f"{main_path}/fig3/data/235515_somerset_wigner_2d.h5",
        f"{main_path}/fig3/data/004541_somerset_wigner_2d.h5",
    ]

    threshold = -0.00014472961900504353

    vacuum_corrected = []
    fock1_corrected = []
    fock2_corrected = []

    for i in range(len(filepath_list)):
        cal_filename = wigner_calibration_list[i]
        data_filename = filepath_list[i]

        ### Get amplitude and offset
        file = h5py.File(cal_filename, "r")
        data = file["data"]
        state_correction_A = np.array(data["state"][:, 0])
        state_correction_B = np.array(data["state"][:, 1])
        state_correction = state_correction_A - state_correction_B
        x = np.array(data["x"][:, 0])
        file.close()
        fig, ax = plt.subplots()
        im = ax.scatter(x, state_correction, label="data")
        popt = gaussian_fit(x, state_correction)
        y_fit = gaussian(x, *popt)
        amp = popt[0]
        sig = popt[2]
        ofs = popt[3]
        ax.plot(x, y_fit, label=f"amp = {amp:.3f}\nsig = {sig:.3f}\nofs = {ofs:.3f}")

        plt.title(f"1D Wigner for amplitude and offset correction #{i}")
        plt.legend()

        file = h5py.File(data_filename, "r")
        data = file["data"]
        state = (np.array(data["I"]) < threshold).astype(int)
        state = state.flatten()  # revert the buffering of qcore

        # separate internal loop into even (0.0) and odd (0.5) and state
        even_vacuum = state[::6]
        odd_vacuum = state[1::6]
        even_fock1 = state[2::6]
        odd_fock1 = state[3::6]
        even_fock2 = state[4::6]
        odd_fock2 = state[5::6]

        # Get x and y sweeps for buffering and plotting
        y = np.arange(-1.9, 1.91 + 0.1 / 2, 0.1)
        x = np.arange(-1.9, 1.91 + 0.1 / 2, 0.1)
        # reps for buffering and averaging
        reps = len(even_vacuum) // (len(y) * len(x))
        buffer = (reps, len(y), len(x))

        even_vacuum_avg = np.average(np.reshape(even_vacuum, buffer), axis=0)
        odd_vacuum_avg = np.average(np.reshape(odd_vacuum, buffer), axis=0)
        even_fock1_avg = np.average(np.reshape(even_fock1, buffer), axis=0)
        odd_fock1_avg = np.average(np.reshape(odd_fock1, buffer), axis=0)
        even_fock2_avg = np.average(np.reshape(even_fock2, buffer), axis=0)
        odd_fock2_avg = np.average(np.reshape(odd_fock2, buffer), axis=0)

        vacuum_corrected.append(((even_vacuum_avg - odd_vacuum_avg) - ofs) / amp)
        fock1_corrected.append(((even_fock1_avg - odd_fock1_avg) - ofs) / amp)
        fock2_corrected.append(((even_fock2_avg - odd_fock2_avg) - ofs) / amp)

    vacuum_final = np.average(vacuum_corrected, axis=0)
    fock1_final = np.average(fock1_corrected, axis=0)
    fock2_final = np.average(fock2_corrected, axis=0)

    WIDTH = 2.5806259314456037
    HEIGTH = 2.5974

    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(WIDTH * cm, HEIGTH * cm))
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    im = ax.pcolormesh(
        x,
        y,
        fock1_final,
        norm=colors.Normalize(vmin=-1, vmax=1),
        cmap="bwr",
        linewidth=0,
        rasterized=True,
    )
    fig.savefig(f"{main_path}/fig3/fig3c_wigner_fock1.pdf")

    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(WIDTH * cm, HEIGTH * cm))
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    im = ax.pcolormesh(
        x,
        y,
        fock2_final,
        norm=colors.Normalize(vmin=-1, vmax=1),
        cmap="bwr",
        linewidth=0,
        rasterized=True,
    )
    # cbar = plt.colorbar(map, pad=0.2)
    fig.savefig(f"{main_path}/fig3/fig3c_wigner_fock2.pdf")


if __name__ == "__main__":
    plot_fig3c_wigner(main_path=main_path)
