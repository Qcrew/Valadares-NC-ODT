from re import X
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from scipy.optimize import curve_fit
import os
from numpy import linalg as LA
from scipy.interpolate import interp2d

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


# Define the Gaussian function with slope and offset
def gaussian_with_slope_offset(x, A, mu, sigma, B, C):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + B * x + C


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


# Function to perform a 2D rotation
def rotate_point(x, y, angle):
    # Convert angle to radians
    theta = np.radians(angle)

    # Create the rotation matrix
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    # Apply the rotation matrix to the original point
    rotated_point = np.dot(rotation_matrix, np.array([x, y]))

    return rotated_point[0], rotated_point[1]


def plot_fig4b_low_chi(main_path):
    print("Plotting fig4b high chi...")
    # low chi with 10 pi2
    folder_path_1 = f"{main_path}/fig4/data/20231203_low_chi/"
    all_1d_files_1 = [
        x for x in os.listdir(folder_path_1) if "char_func_1D_preselection" in x
    ]
    starting_time_1d_1 = "205835"
    # stopping_time_1d_1 = "055532"
    char_1d_files_1 = [
        x for x in all_1d_files_1 if int(x.split("_")[0]) >= int(starting_time_1d_1)
    ]

    folder_path_2 = f"{main_path}/fig4/data/20231204_low_chi/"
    all_1d_files_2 = [
        x for x in os.listdir(folder_path_2) if "char_func_1D_preselection" in x
    ]
    starting_time_1d_2 = "000000"
    stopping_time_1d_2 = "103600"
    char_1d_files_2 = [
        x
        for x in all_1d_files_2
        if int(stopping_time_1d_2) >= int(x.split("_")[0]) >= int(starting_time_1d_2)
    ]

    char1d_filenames_1 = [os.path.join(folder_path_1, file) for file in char_1d_files_1]
    char1d_filenames_2 = [os.path.join(folder_path_2, file) for file in char_1d_files_2]

    combined_char_1d_files = char1d_filenames_1 + char1d_filenames_2

    ####################################################################################
    ####################################################################################
    # 2D Files
    folder_path_1 = f"{main_path}/fig4/data/20231203_low_chi/"
    all_2d_files_1 = [
        x for x in os.listdir(folder_path_1) if "2D_chi_preselect_lk" in x
    ]
    starting_time_2d_1 = "205835"
    # stopping_time_2d_1 = "055532"
    char_2d_files_1 = [
        x for x in all_2d_files_1 if int(x.split("_")[0]) >= int(starting_time_2d_1)
    ]

    folder_path_2 = f"{main_path}/fig4/data/20231204_low_chi/"
    all_2d_files_2 = [x for x in os.listdir(folder_path_2) if "char_func_2D" in x]
    starting_time_2d_2 = "000000"
    stopping_time_2d_2 = "103600"
    char_2d_files_2 = [
        x
        for x in all_2d_files_2
        if int(stopping_time_2d_2) >= int(x.split("_")[0]) >= int(starting_time_2d_2)
    ]

    char2d_filenames_1 = [os.path.join(folder_path_1, file) for file in char_2d_files_1]
    char2d_filenames_2 = [os.path.join(folder_path_2, file) for file in char_2d_files_2]

    combined_char_2d_files = char2d_filenames_1 + char2d_filenames_2

    cal_filenames = combined_char_1d_files
    char2d_filenames = combined_char_2d_files

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
        #     print("ofs", ofs)
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

    ## fitting
    # Get x and y sweeps for buffering and plotting
    state_corrected = compiled_final_data
    x = x
    y = y

    # Get fit
    x_mesh, y_mesh = np.meshgrid(x, y)
    xdata = np.c_[x_mesh.flatten(), y_mesh.flatten()]
    ydata = state_corrected.flatten()
    # bounds = [
    #     (0.0, 0.0, 0.0, 0.0, int(np.pi/180 * (0))),
    #     (3, 0.5, 3, 5, int(np.pi/180 * (60))),
    # ]  # (_, A, B, C, alpha, theta)
    # guess = (0.5, 0, 0.5, 3, int(np.pi/180*(20)) )

    bounds = [
        (0.0, 0.0, 0.0, 0.0, int(np.pi / 180 * (-120))),
        (3, 0.5, 3, 6, int(np.pi / 180 * (20))),
    ]  # (_, A, B, C, alpha, theta)
    guess = (0.5, 0, 0.5, 3, int(np.pi / 180 * (-80)))

    popt, pcov = curve_fit(
        characteristic_coherent,
        xdata,
        ydata,
        bounds=bounds,
        p0=guess,
    )
    state_fit = characteristic_coherent(xdata, *popt).reshape(len(y), len(x))
    # print("state_fit", state_fit.shape)

    # print(popt)
    # print("A", popt[0], "B", popt[1], "C", popt[2], "alpha", popt[3], "theta", popt[4])

    # sm = ax.imshow(state_fit, cmap=plt.get_cmap("coolwarm"), origin="lower")
    A, B, C = popt[:3]
    cov_matrix = np.array([[A, B], [B, C]])
    lamb1, lamb2 = LA.eig(cov_matrix)[0]
    sig1, sig2 = 1 / (2 * lamb1) ** 0.5, 1 / (2 * lamb2) ** 0.5

    label_txt = (
        f"sig1: {sig1:.2f}, sig2: {sig2:.2f}\n|a|: {popt[3]:.3f}, theta: {popt[4]:.3f}"
    )

    theta_angle = popt[4] * 180 / np.pi

    # Example usage
    original_x = 1
    original_y = 0
    rotation_angle = 90 + theta_angle  # Degrees

    # Perform the rotation
    rotated_x, rotated_y = rotate_point(original_x, original_y, rotation_angle)

    # Assuming lamb1 and lamb2 are the coordinates
    x1, y1 = 0, 0
    x2, y2 = lamb2.real, lamb1.real  # Assuming lamb1 and lamb2 are complex numbers

    # Calculate the slope and intercept for lines with y = mx + c
    slope1 = (y2 - y1) / (x2 - x1)
    intercept1 = 0  # Set the intercept as needed

    # Your existing code for interpolation and line cut
    angle = np.pi / 2 * (-(90 + theta_angle)) / 90
    func = interp2d(x, y, compiled_final_data, kind="linear")
    x_inter = x
    y_inter = y
    x_index, y_index = cut_indexes(y_inter, np.pi / 2 - angle)
    cut = [float(func(x_index[j], y_index[j])) for j in range(len(x_index))]

    # Example usage
    original_x = 1
    original_y = 0
    rotation_angle = 90 + theta_angle  # Degrees

    # Perform the rotation
    rotated_x, rotated_y = rotate_point(original_x, original_y, rotation_angle)

    # Example usage
    original_x = 0
    original_y = 1
    rotation_angle = 90 + theta_angle  # Degrees

    # Perform the rotation
    rotated_x, rotated_y = rotate_point(original_x, original_y, rotation_angle)

    # Generate example data
    x_data = x_inter
    y_true = cut

    # # Add some noise to the data
    # # np.random.seed(42)
    # y_noisy = y_true #+ 0.1 * np.random.normal(size=len(x_data))

    # Fit the model to the noisy data
    initial_guess = [1, 0, 1, 0, 1]  # Initial parameter guess
    params, covariance = curve_fit(
        gaussian_with_slope_offset, x_data, y_true, p0=initial_guess
    )

    # Print the fitted parameters
    # print("Fitted parameters:")
    # print("Amplitude (A):", params[0])
    # print("Mean (mu):", params[1])
    # print("Standard Deviation (sigma):", params[2])
    # print("Slope (B):", params[3])
    # print("Offset (C):", params[4])

    # print("corresponding x_inter", x_inter)
    # print("to be subtracted", params[3] * x_inter + params[4])

    new_compiled_final_data = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        for j in range(len(y)):
            original_x = x[i]
            original_y = y[j]
            rotation_angle = theta_angle  # Degrees
            # Perform the rotation
            rotated_x, rotated_y = rotate_point(original_x, original_y, rotation_angle)
            new_compiled_final_data[i][j] = compiled_final_data[i][j] - (
                params[3] * rotated_y + params[4]
            )

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
    ax.pcolormesh(
        x_inter, y_inter, np.rot90(new_compiled_final_data), vmin=-1, vmax=1, cmap="bwr"
    )
    ax.set_aspect("equal", "box")
    fig.savefig(f"{main_path}/fig4/fig4b_low_chi.pdf")

    return np.rot90(new_compiled_final_data)


if __name__ == "__main__":
    plot_fig4b_low_chi(main_path=main_path)
