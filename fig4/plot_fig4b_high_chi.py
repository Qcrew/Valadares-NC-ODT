from re import X
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from scipy.optimize import curve_fit
import os

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


from numpy import linalg as LA


def characteristic_coherent(xy, A, B, C, alpha, theta):
    x = xy[:, 0]
    y = xy[:, 1]
    envelope = np.exp(-(A * x**2 + 2 * B * x * y + C * y**2))
    # envelope = np.exp(-(x ** 2) / 2 / A ** 2) * np.exp(-(y ** 2) / 2 / B ** 2)
    oscillation = np.exp(
        1j * y * 2 * alpha * np.cos(theta) - 1j * x * 2 * alpha * np.sin(theta)
    )
    return np.real(envelope * oscillation)


def characteristic_coherent_fit(x, y, z, bounds, guess):
    popt, pcov = curve_fit(
        characteristic_coherent,
        (x, y),
        z,
        bounds=bounds,
        p0=guess,
    )
    return popt


from scipy.interpolate import interp2d


def cut_indexes(y_inter, angle):
    k = np.tan(angle)
    max_y_index = np.max(y_inter)  # define maximal y_index allowed
    max_x_index = max_y_index
    xvec_fine = np.linspace(0, max_x_index, len(y_inter))
    for i, x in enumerate(xvec_fine):
        if np.abs(x * k) > max_y_index:
            max_x_index = xvec_fine[i - 1]
            break
    x_indicies = np.linspace(-max_x_index, max_x_index, len(y_inter))
    y_indicies = x_indicies * k
    return x_indicies, y_indicies


def plot_fig4b_high_chi(main_path):
    print("Plotting fig4b low chi...")
    # high chi with 10 pi2
    folder_path_1 = f"{main_path}/fig4/data/20231130_high_chi/"
    all_1d_files_1 = [
        x for x in os.listdir(folder_path_1) if "char_func_1D_preselection" in x
    ]
    starting_time_1d_1 = "222226"
    # stopping_time_1d_1 = "055532"
    char_1d_files_1 = [
        x for x in all_1d_files_1 if int(x.split("_")[0]) >= int(starting_time_1d_1)
    ]

    folder_path_2 = f"{main_path}/fig4/data/20231201_high_chi/"
    all_1d_files_2 = [
        x for x in os.listdir(folder_path_2) if "char_func_1D_preselection" in x
    ]
    starting_time_1d_2 = "000000"
    stopping_time_1d_3 = "053811"
    char_1d_files_2 = [
        x
        for x in all_1d_files_2
        if int(stopping_time_1d_3) >= int(x.split("_")[0]) >= int(starting_time_1d_2)
    ]

    char1d_filenames_1 = [os.path.join(folder_path_1, file) for file in char_1d_files_1]
    char1d_filenames_2 = [os.path.join(folder_path_2, file) for file in char_1d_files_2]

    combined_char_1d_files = char1d_filenames_1 + char1d_filenames_2

    ####################################################################################
    ####################################################################################
    # 2D Files
    folder_path_1 = f"{main_path}/fig4/data/20231130_high_chi/"
    all_1d_files_1 = [x for x in os.listdir(folder_path_1) if "char_func_2D" in x]
    starting_time_1d_1 = "223750"
    # stopping_time_1d_1 = "055532"
    char_1d_files_1 = [
        x for x in all_1d_files_1 if int(x.split("_")[0]) >= int(starting_time_1d_1)
    ]

    folder_path_2 = f"{main_path}/fig4/data/20231201_high_chi/"
    all_1d_files_2 = [x for x in os.listdir(folder_path_2) if "char_func_2D" in x]
    starting_time_1d_2 = "000000"
    stopping_time_1d_3 = "055335"
    char_1d_files_2 = [
        x
        for x in all_1d_files_2
        if int(stopping_time_1d_3) >= int(x.split("_")[0]) >= int(starting_time_1d_2)
    ]

    char2d_filenames_1 = [os.path.join(folder_path_1, file) for file in char_1d_files_1]
    char2d_filenames_2 = [os.path.join(folder_path_2, file) for file in char_1d_files_2]

    combined_char_2d_files = char2d_filenames_1 + char2d_filenames_2

    cal_filenames = combined_char_1d_files
    char2d_filenames = combined_char_2d_files

    # threshold = -9.98635242498581e-05
    threshold = -0.00010658777118336782
    total_reps = 0
    total_reps_cal = 0
    all_select_cal_data_real = []
    all_select_cal_data_imag = []
    no_select_cal_data_real = []
    no_select_cal_data_imag = []
    final_data = []

    sel = -1

    for i in range(len(cal_filenames[:sel])):
        cal_filename = cal_filenames[i]
        char2d_filename = char2d_filenames[i]

        # Get offset and amplitude from 1D vacuum data
        file = h5py.File(cal_filename, "r")
        data = file["data"]
        state_cal = np.array((np.array(data["I"])) < threshold).astype(int)
        file.close()

        state_cal = state_cal.flatten()

        m1_cal = state_cal[::2]
        m2_cal = state_cal[1::2]

        # Get x and y sweeps for buffering and plotting
        x_cal = np.arange(-1.9, 1.91 + 0.1 / 2, 0.1)
        y_cal = ["Real", "Imag"]

        reps_cal = len(m2_cal) // (len(y_cal) * len(x_cal))
        #     print("reps_cal", reps_cal)
        total_reps_cal += reps_cal
        #     print("total_reps_cal", total_reps_cal)
        buffer_cal = (reps_cal, len(x_cal), len(y_cal))

        select_cal_data = np.where(m1_cal == 0, m2_cal, np.nan)
        select_data_rate = np.average(np.where(m1_cal == 0, 1, 0))  # first_selection

        select_cal_data = np.reshape(select_cal_data, buffer_cal)
        select_cal_data = np.nanmean(select_cal_data, axis=0)

        m2_cal = np.reshape(m2_cal, buffer_cal)
        m2_cal = np.nanmean(m2_cal, axis=0)

        no_select_cal_data_real.append(m2_cal[:, 0])
        no_select_cal_data_imag.append(m2_cal[:, 1])

        select_cal_data_real = select_cal_data[:, 0]
        select_cal_data_imag = select_cal_data[:, 1]

        all_select_cal_data_real.append(select_cal_data_real)
        all_select_cal_data_imag.append(select_cal_data_imag)

    # Convert the list of arrays into a numpy array
    no_select_cal_data_real = np.array(no_select_cal_data_real)
    no_select_cal_data_imag = np.array(no_select_cal_data_imag)
    all_select_cal_data_real = np.array(all_select_cal_data_real)
    all_select_cal_data_imag = np.array(all_select_cal_data_imag)

    # Calculate the average along the first axis (axis=0)
    average_no_select_cal_data_real = np.mean(no_select_cal_data_real, axis=0)
    average_no_select_cal_data_imag = np.mean(no_select_cal_data_imag, axis=0)
    average_select_cal_data_real = np.mean(all_select_cal_data_real, axis=0)
    average_select_cal_data_imag = np.mean(all_select_cal_data_imag, axis=0)

    popt = gaussian_fit(x_cal, average_select_cal_data_real)

    amp = popt[0]
    sig = popt[2]
    ofs = popt[3]

    for i in range(len(cal_filenames[:sel])):
        char2d_filename = char2d_filenames[i]
        # 2D and Get offset and amplitude from 1D vacuum data
        file = h5py.File(char2d_filename, "r")
        data = file["data"]
        state = np.array((np.array(data["I"])) < threshold).astype(int)
        file.close()
        state = state.flatten()
        m1 = state[::2]
        m2 = state[1::2]

        # Get x and y sweeps for buffering and plotting
        x = np.arange(-1.9, 1.91 + 0.1 / 2, 0.1) / sig
        y = np.arange(-1.9, 1.91 + 0.1 / 2, 0.1) / sig

        reps = len(m2) // (len(y) * len(x))
        total_reps += reps
        buffer = (reps, len(x), len(y))

        select_data = np.where(m1 == 0, m2, np.nan)  # first_selection
        select_data_rate = np.average(np.where(m1 == 0, 1, 0))  # first_selection

        select_data = np.reshape(select_data, buffer)
        #     ofs = 0.5
        # print("ofs", ofs)
        select_data = correct_ofs_and_amp(select_data, amp, ofs)
        final_data.append(select_data)
        m2 = np.reshape(m2, buffer)
        select_data = np.nanmean(select_data, axis=0)
        m2 = np.nanmean(m2, axis=0)

    # Use last sigma for scaling
    x = np.arange(-1.9, 1.91 + 0.1 / 2, 0.1) / sig
    y = np.arange(-1.9, 1.91 + 0.1 / 2, 0.1) / sig
    compiled_final_data = np.concatenate(final_data, axis=0)
    nan_percentage = np.count_nonzero(np.isnan(compiled_final_data)) / len(
        compiled_final_data.flatten()
    )
    compiled_final_data = np.nanmean(compiled_final_data, axis=0)
    cm = 1 / 2.54
    WIDTH = 2.5806259314456037
    HEIGTH = 2.5974
    fig, ax = plt.subplots(figsize=(WIDTH * cm, HEIGTH * cm))
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    ax.set_xticks(ticks=[2, 0, -2])
    ax.set_yticks(ticks=[2, 0, -2])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    im = ax.pcolormesh(
        x, y, compiled_final_data, norm=colors.Normalize(vmin=-1, vmax=1), cmap="bwr"
    )
    ax.set_aspect("equal", "box")
    ax.set_xticks([-2.0, 0.0, 2.0])
    ax.set_yticks([-2.0, 0.0, 2.0])
    fig.tight_layout()
    fig.savefig(f"{main_path}/fig4/fig4b_high_chi.pdf")
    return compiled_final_data


if __name__ == "__main__":
    plot_fig4b_high_chi(main_path=main_path)
