import numpy as np # type: ignore
import math 
import sympy # type: ignore
from sympy import symbols, cos, sin, I, re, im, sqrt # type: ignore
import time
import threading
# Helper functions below
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

def rk4(start: np.array, direction: int, step_size: float,  move_off_direction_func: callable, function: callable): # Helps with A6 on the project
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
            print("point", point)
            print("velocity", function(point))
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



def list_to_range(three_element_list: list):
    """first element is the start, second element is the end, and the third element is the step size."""
    if len(three_element_list) != 3:
        raise ValueError("The list must contain three elements. The first element is the start, the second element is the end, and the third element is the step size.")
    start = three_element_list[0]
    end = three_element_list[1]
    step_size = three_element_list[2]
    if start != end:
        values = np.arange(start, end + step_size, step_size)
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

def numerical_derivative(func, x, r_values=np.array([0.0]), theta_values=np.array([0.0]), is_analytic_accel=False, h=1e-6):
    """ Compute first derivative using central difference. """
    f_plus = func(x + h, r_values, theta_values, is_analytic_accel)
    f_minus = func(x - h, r_values, theta_values, is_analytic_accel)
    # print(f"f_plus: {f_plus}, f_minus: {f_minus}")  # Debug print
    return (f_plus - f_minus) / (2 * h)

def numerical_second_derivative(func, x, r_values=np.array([0.0]), theta_values=np.array([0.0]), is_analytic_accel=False, h=1e-6):
    """ Compute second derivative using central difference. """
    f_plus = func(x + h, r_values, theta_values, is_analytic_accel)
    f = func(x, r_values, theta_values, is_analytic_accel)
    f_minus = func(x - h, r_values, theta_values, is_analytic_accel)
    # print(f"f_plus: {f_plus}, f: {f}, f_minus: {f_minus}")  # Debug print
    return (f_plus - 2 * f + f_minus) / (h**2)

def newtons_method(func, x0, r_values=np.array([0.0]), theta_values=np.array([0.0]), is_analytic_accel=False, tol=1e-10, max_iter=1000):
    """ Newton's method to find extrema of a function. """
    x = x0
    
    for i in range(max_iter):
        f_prime = numerical_derivative(func, x, r_values, theta_values, is_analytic_accel)
        f_double_prime = numerical_second_derivative(func, x, r_values, theta_values, is_analytic_accel)
        
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
            return x_new
        
        # print iteration, and epsilon compared to tol every 10 steps
        if i % 5 == 0:
            print(f"Iteration {i+1}: x = {x_new:.16f}, epsilon = {epsilon:.16f}")
            # print(f"Iteration {i+1}: x = {x_new:.6f}")
        
        x = x_new
    
    raise ValueError("Newton's method did not converge.")

def example_func(x, r_values, theta_values, is_analytic_accel):  # Fix argument order
    return np.sin(x) + 0.1 * x**2

if __name__ == "__main__":
    # Example usage
    # initial_guess = 2.0
    # xi_eta = 1.0
    # r_values = 1
    # theta_values = 1
    # is_analytic_accel = False
    # print("\n")
    # extremum = newtons_method(example_func, initial_guess, r_values, theta_values, is_analytic_accel)
    # print(f"Extremum at Gamma = {extremum:.6f}")
    # print("\n")

    # r = -0.25406965
    # q = 0.14441865
    # s = np.sqrt(r**2 + q**2)
    # s /= 2
    # print("s", s)

    # Define variables
    r, theta, alpha, Gamma, V_inf, R, epsilon, r_0, rho = sympy.symbols('r theta alpha Gamma V_inf R epsilon r_0 rho', real=True)

    # Define components of the function
    rei_theta = r * sympy.exp(sympy.I * theta)
    r0ei_theta0 = r_0 * sympy.exp(sympy.I * theta)

    print("Defining the mathematical expression...")

    term1 = sympy.exp(-sympy.I * alpha) + sympy.I * (Gamma / (2 * sympy.pi * V_inf)) * (1 / rei_theta) - (R**2 * sympy.exp(sympy.I * alpha)) / (rei_theta**2)
    term2 = (-sympy.I * (Gamma / (2 * sympy.pi * V_inf)) * (1 / (rei_theta**2))) + (2 * R**2 * sympy.exp(sympy.I * alpha) / (rei_theta**3))
    term3 = 2 * (R - epsilon)**2 / (rei_theta + r0ei_theta0)**3
    term4 = 1 - ((R - epsilon)**2 / (rei_theta + r0ei_theta0)**2)

    numerator = -2 * term1 * (term2 - (term1 * term3 * (1 / term4)))
    denominator = term4**2

    integrand = numerator / denominator * r

    print("Expression successfully defined.")
    print("Starting integration over θ from 0 to 2π...")

    # Function to perform the integration
    def perform_integration():
        global integrated_theta
        integrated_theta = sympy.integrate(integrand, (theta, 0, 2 * sympy.pi))
        print("\nFirst integration step complete.")

    # Function to print time ticks every 60 seconds
    def time_ticker():
        start_time = time.time()
        while integration_thread.is_alive():
            time.sleep(60)  # Wait for 60 seconds
            elapsed_time = int(time.time() - start_time)
            print(f"\n[Progress Update] Integration running for {elapsed_time // 60} minutes...")

    # Start integration in a separate thread
    integration_thread = threading.Thread(target=perform_integration)
    integration_thread.start()

    # Start time ticker
    time_ticker()
    integration_thread.join()  # Wait for integration to finish

    # Multiply by constant factors
    S = (V_inf**4 / 8) * rho * integrated_theta

    print("Final expression for S obtained.")
    print("S =", S)