from re import X
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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
    return np.imag(envelope * oscillation)  # change this if you want real or imag


def characteristic_coherent_fit(x, y, z, bounds, guess):
    popt, pcov = curve_fit(
        characteristic_coherent,
        (x, y),
        z,
        bounds=bounds,
        p0=guess,
    )
    return popt


def cut_indexes(y_inter, angle, y_offset=0):
    # Calculate the slope of the line
    k = np.tan(angle)

    # Define the maximal y index allowed
    max_y_index = np.max(y_inter)

    # Initialize max_x_index with max_y_index
    max_x_index = max_y_index

    # Create a fine x vector
    xvec_fine = np.linspace(0, max_x_index, len(y_inter))

    # Iterate through the x vector to find the maximum x index based on the slope
    for i, x in enumerate(xvec_fine):
        if np.abs(x * k) > (max_y_index + y_offset):
            max_x_index = xvec_fine[i - 1]
            break

    # Create x and y indices based on the calculated max_x_index, slope, and offset
    x_indices = np.linspace(-max_x_index, max_x_index, len(y_inter))
    y_indices = x_indices * k + y_offset

    return x_indices, y_indices


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


def plot_fig4a_low_kerr(main_path):
    print("Plotting fig4a low kerr...")
    cal_filename = (
        f"{main_path}/fig4/data/132926_somerset_characteristic_function_1D.h5"
    )
    data_filename = (
        f"{main_path}/fig4/data/142514_somerset_characteristic_function_2D.h5"
    )

    threshold = -0.00028143484898768604

    # Get offset and amplitude from 1D vacuum data
    file = h5py.File(cal_filename, "r")
    data = file["data"]
    state_correction = np.array(data["state"])[:, 0]

    x = np.arange(-1.9, 1.91 + 0.1 / 2, 0.1)
    file.close()

    popt = gaussian_fit(x, state_correction)
    y_fit = gaussian(x, *popt)
    amp = popt[0]
    sig = popt[2]
    ofs = popt[3]  # + 0.02

    # Get data and apply threshold
    file = h5py.File(data_filename, "r")
    data = file["data"]
    state = (np.array(data["I"]) < threshold).astype(int)
    state = state.flatten()  # revert the buffering of qcore
    file.close()

    # Get x and y sweeps for buffering and plotting
    x = np.arange(-1.9, 1.91 + 0.1 / 2, 0.1) / sig
    y = np.arange(-1.9, 1.91 + 0.1 / 2, 0.1) / sig

    reps = len(state) // (len(y) * len(x))
    buffer = (reps, len(y), len(x))

    cohstate5_imag = correct_ofs_and_amp(reshape_and_avg(state, buffer), amp, ofs)

    compiled_final_data = cohstate5_imag
    state_corrected = cohstate5_imag

    # Get fit
    x_mesh, y_mesh = np.meshgrid(x, y)
    xdata = np.c_[x_mesh.flatten(), y_mesh.flatten()]
    ydata = state_corrected.flatten()

    bounds = [
        (0.0, 0.0, 0.0, 0.0, int(np.pi / 180 * (-120))),
        (3, 0.5, 3, 6, int(np.pi / 180 * (20))),
    ]  # (_, A, B, C, alpha, theta)
    guess = (0.5, 0, 0.5, 3, int(np.pi / 180 * (0)))

    popt, pcov = curve_fit(
        characteristic_coherent,
        xdata,
        ydata,
        bounds=bounds,
        p0=guess,
    )
    state_fit = characteristic_coherent(xdata, *popt).reshape(len(y), len(x))

    # sm = ax.imshow(state_fit, cmap=plt.get_cmap("coolwarm"), origin="lower")
    A, B, C = popt[:3]
    cov_matrix = np.array([[A, B], [B, C]])
    lamb1, lamb2 = LA.eig(cov_matrix)[0]
    sig1, sig2 = 1 / (2 * lamb1) ** 0.5, 1 / (2 * lamb2) ** 0.5
    theta_angle = popt[4] * 180 / np.pi  # popt[4]*180/np.pi

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

    x_index, y_index = cut_indexes(
        y_inter, np.pi / 2 - angle, y_offset=0.35
    )  # (y_inter, np.pi / 2 - angle)
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
    x_values = np.linspace(-2, 2, 5)

    # Generate example data
    x_data = x_inter
    y_true = cut

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
            #         print("original_x", original_x)
            original_y = y[j]
            rotation_angle = theta_angle  # 90+theta_angle # Degrees
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
        x_inter, y_inter, new_compiled_final_data, vmin=-1, vmax=1, cmap="bwr"
    )
    ax.set_aspect("equal", "box")
    fig.savefig(f"{main_path}/fig4/fig4a_low_kerr.pdf")
    return new_compiled_final_data


if __name__ == "__main__":
    plot_fig4a_low_kerr(main_path=main_path)
