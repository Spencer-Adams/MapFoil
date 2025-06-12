import numpy as np # type: ignore
import os # type: ignore
import math # type: ignore
import sympy as sp # type: ignore
from sympy import symbols, cos, sin, I, re, im, sqrt # type: ignore
import time # type: ignore
import threading # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.image as mpimg # type: ignore
import matplotlib.animation as animation  # type: ignore
from matplotlib.animation import PillowWriter  # type: ignore
from matplotlib.offsetbox import OffsetImage, AnnotationBbox # type: ignore
from io import BytesIO
from tqdm import tqdm # type: ignore
# Helper functions below

def parse_dictionary_or_return_default(dictionary, keys, default):
    """Safely get nested dictionary values. It makes it so that if a json key is not found, it returns a default value instead of throwing an error."""
    for key in keys:
        dictionary = dictionary.get(key, {})
        if not isinstance(dictionary, dict) and key != keys[-1]:
            return default
    if dictionary == {} and default is not None:
        print("The dictionary ", keys, " is empty. Returning default value of ", default, ".")
        return default
    elif dictionary != {}:
        return dictionary 
    else:
        raise ValueError(f"Key {keys} not found in dictionary and no default value provided.")
    
def plot_xy_array(xy_array: np.array):
    plt.plot(xy_array[:, 0], xy_array[:, 1], color='black', linewidth=0.5)

def vector_magnitude(vector: np.array): # Used to help A6 in the project. The streamlines need to be integrated forward using the unit velocity vector
    """
    Calculate the magnitude of a vector.

    Parameters:
    vector (list): A list representing the vector.

    Returns:
    float: The magnitude of the vector.
    """
    magnitude = math.sqrt(sum([element**2 for element in vector]))
    return magnitude

def unit_vector(vector: np.array): # Used to help A6 in the project. The streamlines need to be integrated forward using the unit velocity vector
    """Calculate the unit vector of a given vector."""
    # make sure that vector is a list of two or three float elements
    if len(vector) < 2 or not all(isinstance(coord, float) for coord in vector):
        raise ValueError("The vector must be a list of two or three float elements.")
    # calculate the magnitude of the vector
    magnitude = vector_magnitude(vector)
    # calculate the unit vector
    unit_vector = [element/magnitude for element in vector]
    return unit_vector

def rk4(start: np.array, direction: int, step_size: float,  move_off_direction_func: callable, function: callable):
    """This function performs a Runge-Kutta 4th order integration

    Args:
        - start (list): A list of two float values representing x and y coordinates.
        - direction (int): An integer value representing the direction of integration.
        - step_size (float): The step size for the integration.
        - function (callable): A function that calculates the derivative at a given point.

    Returns:
        list: A numpy array of float values representing the new coordinates after integration.
    """
    # make sure that start is a list of two or three float elements
    if len(start) < 2 or not all(isinstance(coord, float) for coord in start):
        raise ValueError("The start must be a list of two float values representing x and y coordinates.")
    # make sure that direction is an integer
    if not isinstance(direction, int):
        raise TypeError("The direction must be an integer. Please provide an integer value.")
    # set the step size
    # print(direction)
    point = start
    h = direction*step_size
    # if all of the function values in the np.array are really close to zero, step the point off in the move off direction. 
    while np.all(np.abs(function(point)) < 1e-12):
        for i in range(len(point)):
            print("The function values are too small. Stepping the point off in the move off direction.")
            print("point: ", point)
            print("velocity: ", function(point))
            move_off_direction = move_off_direction_func(point[0])[i]
            point[i] = point[i] + move_off_direction[i]*1e-6
            break
    # set the initial values of k1, k2, k3, and k4
    k1 = np.array(function(point))
    k2 = np.array(function(point + 0.5 * h * k1))
    k3 = np.array(function(point + 0.5 * h * k2))
    k4 = np.array(function(point + h * k3))
    point_new = point + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    point_new = np.array(point_new)
    return point_new

def sort_points_counterclockwise(points: np.ndarray) -> np.ndarray:
    """
    Sort an array of 2D points in counterclockwise order around their centroid.

    Parameters:
    - points (np.ndarray): shape (n, 2), each row is an (x, y) point.

    Returns:
    - sorted_points (np.ndarray): shape (n, 2), points in CCW order.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Input must be an (n, 2) array of xy points.")
    # Step 1: Compute centroid
    centroid = np.mean(points, axis=0)
    # Step 2: Compute angle to each point
    angles = np.arctan2(points[:,1] - centroid[1], points[:,0] - centroid[0])
    # Step 3: Sort points by angle
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    return sorted_points


def list_to_range(three_element_list: list):
    """First element is the start, second element is the end, and the third element is the step size."""
    if len(three_element_list) != 3:
        raise ValueError("The list must contain three elements. The first element is the start, the second element is the end, and the third element is the step size.")
    
    start = three_element_list[0]
    end = three_element_list[1]
    step_size = three_element_list[2]
    
    if start != end:
        # Calculate the number of points to include in the range
        num_points = int(round((end - start) / step_size)) + 1
        values = np.linspace(start, end, num_points)
    else:
        values = np.array([start])
    
    return values

def xy_to_r_theta(x: float, y: float):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def r_theta_to_xy(r: float, theta: float):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def central_difference(f_plus: float, f_minus: float, step_size: float):
    derivative = (f_plus - f_minus)/(2*step_size)
    return derivative

def polar_vector(theta, cartesian_vector):
    """This function converts the cartesian velocity to polar velocity (can go from z, zeta, or chi plane to polar velocity)"""
    r = cartesian_vector[0]*np.cos(theta) + cartesian_vector[1]*np.sin(theta)
    # print("\nradial velocity", velocity_r)
    theta = cartesian_vector[1]*np.cos(theta) - cartesian_vector[0]*np.sin(theta)
    # print("theta velocity", velocity_theta)
    polar_vector = np.array([r, theta])
    return polar_vector

# Function to remove the middle element
def remove_middle_element(arr):
    mid_idx = len(arr) // 2
    return np.delete(arr, mid_idx, axis=0)

def numerical_derivative(func, x, r_values=np.array([0.0]), theta_values=np.array([0.0]), h=1e-6, D=0.0):
    """ Compute first derivative using central difference. """
    f_plus = func(x + h, r_values, theta_values, D)
    f_minus = func(x - h, r_values, theta_values, D)
    # print(f"f_plus: {f_plus}, f_minus: {f_minus}")  # Debug print
    return (f_plus - f_minus) / (2 * h)

def numerical_second_derivative(func, x, r_values=np.array([0.0]), theta_values=np.array([0.0]), h=1e-6, D=0.0):
    """ Compute second derivative using central difference. """
    f_plus = func(x + h, r_values, theta_values, D)
    f = func(x, r_values, theta_values, D)
    f_minus = func(x - h, r_values, theta_values, D)
    # print(f"f_plus: {f_plus}, f: {f}, f_minus: {f_minus}")  # Debug print
    return (f_plus - 2 * f + f_minus) / (h**2)

def newtons_method(func, x0, r_values=np.array([0.0]), theta_values=np.array([0.0]), tol=1e-10, max_iter=1000, D=0.0):
    """ Newton's method to find extrema of a function. """
    x = x0
    
    for i in range(max_iter):
        f_prime = numerical_derivative(func, x, r_values, theta_values)
        f_double_prime = numerical_second_derivative(func, x, r_values, theta_values)
        
        if abs(f_double_prime) < 1e-12:
            raise ValueError("Second derivative is too small — possible inflection point or flat region.")
        
        x_new = x - f_prime / f_double_prime
        epsilon = abs(x_new - x)
        # stop if epsilon is less than tolerance or if the previous x is the same as the new x out to 14 decimal places
        if epsilon < tol:# or np.isclose(x, x_new, atol=1e-16):
            if epsilon < tol:
                print(f"Converged in {i+1} iterations.")
                # print epsilon at convergence
                print(f"epsilon = {epsilon:.16f}")
            elif np.isclose(x, x_new, atol=1e-16):
                print(f"Converged in {i+1} iterations because the previous iteration was the same as this one to 16 digits.")
                print(f"epsilon = {epsilon:.16f}")
            return x_new, func(x_new, r_values, theta_values, D)  # Return the extremum and its value
        
        # print iteration, and epsilon compared to tol every 10 steps
        if i % 5 == 0:
            print(f"Iteration {i+1}: x = {x_new:.16f}, epsilon = {epsilon:.16f}")
            # print(f"Iteration {i+1}: x = {x_new:.6f}")
        
        x = x_new

    raise ValueError("Newton's method did not converge.")

def polyfit(func, r_values, gamma_vals, order_of_polynomial, is_plot, xlabel, ylabel, plot_title, D):
    time_1 = time.time()
    # print("length of gamma_vals", len(gamma_vals))
    
    # Evaluate function at all gamma values
    appellian_vals = np.array([[gamma, func(gamma, r_values, D)] for gamma in gamma_vals])

    # Fit a polynomial of specified order
    coeffs = np.polyfit(appellian_vals[:, 0], appellian_vals[:, 1], order_of_polynomial)

    # Warn if leading coefficient is negative
    if coeffs[0] < 0:
        print("Warning: The highest order coefficient is negative. This may indicate that the polynomial is not a good fit for the data.")
        print("This warning is at D =", D)

    # Derivative and roots (extrema)
    derivative_coeffs = np.polyder(coeffs)
    extrema = np.roots(derivative_coeffs)
    # print(f"Extrema found: {extrema}")
    # Separate real and complex extrema (with tolerance)
    real_extrema = [x.real for x in extrema if np.isclose(x.imag, 0)]
    complex_extrema = [x for x in extrema if not np.isclose(x.imag, 0)]

    # Determine which extrema to evaluate
    if len(real_extrema) == 1 and len(complex_extrema) == 2:
        # print("Case: One real root and a complex conjugate pair — selecting the real root as the minimum.")
        selected_extrema = [real_extrema[0]]
    elif len(real_extrema) == 3 and len(set(np.round(real_extrema, 12))) == 1:
        # print("Case: Triple real root — all roots identical. Choosing one.")
        selected_extrema = [real_extrema[0]]
    elif len(real_extrema) >= 1:
        # print("Case: All real distinct roots — evaluating all to find the minimum.")
        selected_extrema = real_extrema
    else:
        raise ValueError("Unexpected root configuration in polynomial derivative.")
    # print(f"Selected extrema: {selected_extrema}")
    # Evaluate the true function and polynomial at extrema
    # print("length of selected extrema", len(selected_extrema))
    func_vals = np.array([func(x, r_values,  D) for x in selected_extrema])
    poly_vals = np.array([np.polyval(coeffs, x) for x in selected_extrema])

    # for i, (f_val, p_val, x_val) in enumerate(zip(func_vals, poly_vals, selected_extrema)):
    #     if abs(f_val - p_val) > 1e-12:
    #         print("\nWarning: The function and polynomial values differ at extremum.")
    #         print(f"This warning is at D = {D}")
    #         print(f"extremum = {x_val}")
    #         print(f"func_val = {f_val}, poly_val = {p_val}")
    #         print(f"absolute difference = {abs(f_val - p_val)}\n")

    # Choose minimum among selected extrema
    min_index = np.argmin(func_vals)
    extremum_min = selected_extrema[min_index]
    value_at_min = func_vals[min_index]

    # Optional plot
    if is_plot:
        plt.plot(appellian_vals[:, 0], appellian_vals[:, 1], marker='o', linestyle='-', color='black', markersize=3)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(plot_title)

        # Add extrema to plot
        y_range = plt.ylim()[1] - plt.ylim()[0]
        y_offset = 0.05 * y_range
        x_text = 0.5 * (plt.xlim()[0] + plt.xlim()[1])
        y_text = plt.ylim()[1] - y_offset

        for i, x_val in enumerate(selected_extrema):
            plt.text(
                x_text,
                y_text - i * y_offset,
                f"({x_val:.8f}, {func_vals[i]:.2f})",
                fontsize=8,
                ha='center',
                va='top'
            )
        plt.show()

    time_2 = time.time()
    # print("Time taken to find extremum using polyfit:", time_2 - time_1)

    return extremum_min, value_at_min

def combine_two_plots(plot1_filename, plot2_filename, output_filename):
    """
    Combine two plots into one figure with two subplots stacked vertically.
    
    Parameters:
    - plot1_filename: str, path to the first plot image file.
    - plot2_filename: str, path to the second plot image file.
    - output_filename: str, path to save the combined figure.
    """
    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 12))
    
    # Load and display the first plot
    img1 = mpimg.imread(plot1_filename)
    axes[0].imshow(img1)
    axes[0].axis('off')  # Turn off axes for the image
    # axes[0].set_title("Plot 1", fontsize=16)

    # Load and display the second plot
    img2 = mpimg.imread(plot2_filename)
    axes[1].imshow(img2)
    axes[1].axis('off')  # Turn off axes for the image
    # axes[1].set_title("Plot 2", fontsize=16)

    # Adjust layout and save the combined figure
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close(fig)

def combine_plots(plot_names_array, output_filename):
    num_plots = len(plot_names_array)
    num_rows = math.ceil(num_plots / 2)

    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=2,
        figsize=(6.5, 7), # size of the figure in inches 
        gridspec_kw={'wspace': 0.05, 'hspace': -0.45} # wspace adjusts the space between columns, hspace adjusts the space between rows
    )

    axes = axes.flatten()  # flatten to make indexing simpler

    for i, plot_name in enumerate(plot_names_array):
        img = mpimg.imread(plot_name)
        axes[i].imshow(img)
        axes[i].axis('off')

    # Turn off any unused axes if number of plots is odd
    for j in range(len(plot_names_array), len(axes)):
        axes[j].axis('off')

    # plt.tight_layout(pad=0.5, w_pad=0.1, h_pad=0.1)
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)

def create_animation_from_all_figs_in_folder(folder, output_filename):
    """
    Create an animation from all .png figures in a specified folder.

    Parameters:
    - folder: str, path to the folder containing the .png figures.
    - output_filename: str, path to save the animation.
    """

    # Create a figure and axis for the animation
    fig, ax = plt.subplots(figsize=(10, 8))

    # List to store sorted image filenames
    image_filenames = []

    # Find all .png files in the folder that end with a number
    for filename in os.listdir(folder):
        if filename.endswith(".png") and filename[-5].isdigit():
            image_filenames.append(os.path.join(folder, filename))

    # Sort the filenames based on the numeric suffix
    image_filenames.sort(key=lambda x: int(x.split("__")[-1].split(".")[0]))

    # Load .png images into memory for animation
    png_images = [plt.imread(img_file) for img_file in image_filenames]

    # Function to update the frame in the animation
    def update_frame(frame_idx):
        ax.clear()
        ax.imshow(png_images[frame_idx])
        ax.axis("off")  # Turn off axes for clean animation

    # Set interval to 125ms for 1/8 second per frame
    interval = 125
    fps = 8  # 1000 / 125 ms = 8 fps

    # Create an animation from the images
    ani = animation.FuncAnimation(fig, update_frame, frames=len(png_images), interval=interval)

    # Save the animation as a GIF
    ani.save(output_filename, writer="pillow", fps=fps)

    print("Animation created successfully!")

    # Clean up the images
    for filename in image_filenames:
        os.remove(filename)
    
# def create_animation_from_all_figs_in_folder_mp4(folder, output_filename):
#     """
#     Create an MP4 animation from all .png figures in a specified folder.

#     Parameters:
#     - folder: str, path to the folder containing the .png figures.
#     - output_filename: str, path to save the animation (should end in .mp4).
#     """
#     fig, ax = plt.subplots(figsize=(10, 8))
#     image_filenames = []

#     # Find all .png files that end with a number
#     for filename in os.listdir(folder):
#         if filename.endswith(".png") and filename[-5].isdigit():
#             image_filenames.append(os.path.join(folder, filename))

#     # Sort the filenames by numeric suffix
#     image_filenames.sort(key=lambda x: int(x.split("__")[-1].split(".")[0]))

#     png_images = [plt.imread(img_file) for img_file in image_filenames]

#     def update_frame(frame_idx):
#         ax.clear()
#         ax.imshow(png_images[frame_idx])
#         ax.axis("off")

#     interval = 125  # milliseconds (1/8 second per frame)
#     fps = 1000 / interval

#     ani = animation.FuncAnimation(
#         fig, update_frame, frames=len(png_images), interval=interval
#     )

#     # Save the animation as MP4 using ffmpeg
#     Writer = animation.writers['ffmpeg']
#     writer = Writer(fps=fps, metadata=dict(artist='AutoGen'), bitrate=1800)

#     ani.save(output_filename, writer=writer)
#     print("MP4 animation created successfully!")

#     # Optional: delete input images
#     # for filename in image_filenames:
#     #     os.remove(filename)

if __name__ == "__main__":
    combine_two_plots("figures/zeta_-0.25_D_sweep_alpha_5.png", "figures/zeta_-0.25_D_0_alpha_5.png", "combined_plot.svg")
    # Create an animation
    # fig, ax = plt.subplots(figsize=(10, 8))
    # print("Creating animation...")
    # ani = animation.FuncAnimation(fig, update_frame, frames=len(image_filenames), interval=100) # fig is the figure, update_frame is the function to update the frame, frames is the number of frames, interval is the time between frames in milliseconds

    # # Save the animation as a video or GIF
    # ani.save("Distributions_Animation.gif", writer="pillow", fps=5)  # Save as a GIF
    # print("Animation created successfully!")

    # # clean up the images 
    # for filename in image_filenames:
    #     os.remove(filename)

