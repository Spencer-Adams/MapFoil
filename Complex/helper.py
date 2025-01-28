import numpy as np # type: ignore
import math 
import sympy # type: ignore
from sympy import symbols, cos, sin, I, re, im, sqrt # type: ignore

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

def symbolically_factorize(expression: str): 
    """This function symbolically factorizes a given expression.

    Args:
        - expression (str): A string representing the expression to be factorized.

    Returns:
        str: A string representing the factorized expression.
    """
    # make sure that expression is a string
    if not isinstance(expression, str):
        raise TypeError("The expression must be a string. Please provide a string value.")
    # factorize the expression
    # factorization is when you take an expression and write it as a product of other expressions
    factorized_expression = sympy.factor(expression)
    return factorized_expression

def symbolically_expand(expression: str):
    """This function symbolically expands a given expression.

    Args:
        - expression (str): A string representing the expression to be expanded.

    Returns:
        str: A string representing the expanded expression.
    """
    # make sure that expression is a string
    if not isinstance(expression, str):
        raise TypeError("The expression must be a string. Please provide a string value.")
    # expand the expression
    # expansion is when you take an expression and write it as a sum of other expressions
    expanded_expression = sympy.expand(expression)
    return expanded_expression

def symbolically_simplify(expression: str):
    """This function symbolically simplifies a given expression.

    Args:
        - expression (str): A string representing the expression to be simplified.

    Returns:
        str: A string representing the simplified expression.
    """
    # make sure that expression is a string
    if not isinstance(expression, str):
        raise TypeError("The expression must be a string. Please provide a string value.")
    # simplify the expression
    # simplification is when you take an expression and write it in a simpler form
    simplified_expression = sympy.simplify(expression)
    return simplified_expression

def symbolically_differentiate(expression: str, variable: str):
    """This function symbolically differentiates a given expression.

    Args:
        - expression (str): A string representing the expression to be differentiated.
        - variable (str): A string representing the variable to differentiate with respect to.

    Returns:
        str: A string representing the differentiated expression.
    """
    # make sure that expression is a string
    if not isinstance(expression, str):
        raise TypeError("The expression must be a string. Please provide a string value.")
    # make sure that variable is a string
    if not isinstance(variable, str):
        raise TypeError("The variable must be a string. Please provide a string value.")
    # differentiate the expression
    # differentiation is when you find the rate at which a quantity changes with respect to another quantity
    differentiated_expression = sympy.diff(expression, variable)
    return differentiated_expression

def symbolically_integrate(expression: str, variable: str):
    """This function symbolically integrates a given expression.

    Args:
        - expression (str): A string representing the expression to be integrated.
        - variable (str): A string representing the variable to integrate with respect to.

    Returns:
        str: A string representing the integrated expression.
    """
    # make sure that expression is a string
    if not isinstance(expression, str):
        raise TypeError("The expression must be a string. Please provide a string value.")
    # make sure that variable is a string
    if not isinstance(variable, str):
        raise TypeError("The variable must be a string. Please provide a string value.")
    # integrate the expression
    # integration is when you find the area under a curve
    integrated_expression = sympy.integrate(expression, variable)
    return integrated_expression

def symbolically_solve(expression: str, variable: str):
    """This function symbolically solves a given expression.

    Args:
        - expression (str): A string representing the expression to be solved.
        - variable (str): A string representing the variable to solve for.

    Returns:
        str: A string representing the solution to the expression.
    """
    # make sure that expression is a string
    if not isinstance(expression, str):
        raise TypeError("The expression must be a string. Please provide a string value.")
    # make sure that variable is a string
    if not isinstance(variable, str):
        raise TypeError("The variable must be a string. Please provide a string value.")
    # solve the expression
    # solving is when you find the value of a variable that makes the equation true
    solved_expression = sympy.solve(expression, variable)
    return solved_expression

def symbolically_vectorize(expression: str):
    """This function symbolically vectorizes a given expression.

    Args:
        - expression (str): A string representing the expression to be vectorized.

    Returns:
        str: A string representing the vectorized expression.
    """
    # make sure that expression is a string
    if not isinstance(expression, str):
        raise TypeError("The expression must be a string. Please provide a string value.")
    # vectorize the expression
    # vectorization is when you convert a scalar expression into a vector expression
    vectorized_expression = sympy.Matrix([expression])
    return vectorized_expression

def symbolically_complex_vectorize(expression: str):
    """This function symbolically vectorizes a given expression.

    Args:
        - expression (str): A string representing the expression to be vectorized.

    Returns:
        str: A string representing the vectorized expression.
    """
    # make sure that expression is a string
    if not isinstance(expression, str):
        raise TypeError("The expression must be a string. Please provide a string value.")
    # vectorize the expression
    # vectorization is when you convert a scalar expression into a vector expression
    vectorized_expression = sympy.Matrix([expression])
    return vectorized_expression

def string_to_latex(expression: str):
    """This function converts a given expression to LaTeX format.

    Args:
        - expression (str): A string representing the expression to be converted to LaTeX format.

    Returns:
        str: A string representing the LaTeX formatted expression.
    """
    # make sure that expression is a string
    if not isinstance(expression, str):
        raise TypeError("The expression must be a string. Please provide a string value.")
    # convert the expression to LaTeX format
    latex_expression = sympy.latex(expression)
    return latex_expression

if __name__ == "__main__":
    # Define symbols
    V_inf, alpha, Gamma, zeta, zeta0, eta, etao, xi, xio, R, eps, Pie = symbols(
        'V_inf alpha Gamma zeta zeta0 eta etao xi xio R eps Pie', real=True
    )
    i = I  # Complex unit
    # Define the complex expression (replace with your full expression)
    w_omega = V_inf*((cos(alpha)-i*sin(alpha))+i*(Gamma/(2*Pie*V_inf)) * (
                1 / ((xi + i * eta) - (xio + i * etao)))-((R**2*(cos(alpha)+i*sin(alpha)))/((xi+i*eta)-(xio+i*etao))**2))/(1-((R-eps)**2/(xi+i*eta)**2))
    # Split into real and imaginary parts
    real_part = re(w_omega)
    imag_part = im(w_omega)

    # make the real and imaginary parts into strings
    real_part = str(real_part)
    imag_part = str(imag_part)

    # Print the results
    print("\nReal Part:", real_part)
    print("Type of Real Part:", type(real_part))
    print("\nImaginary Part:", imag_part)
    print("Type of Imaginary Part:", type(imag_part))
    print("\n")

    # simplify the real part
    # simplified_real_part = symbolically_simplify(real_part)
    # print("Simplified Real Part:", simplified_real_part)
    latex_real_part = string_to_latex("(eta**2 + xi**2)**2*(eta*xi*(R - eps)**2*(Gamma*(xi - xio)*((eta - etao)**2 + (xi - xio)**2) + 2*Pie*R**2*V_inf*(2*(eta - etao)*(xi - xio)*cos(alpha) + ((-eta + etao)**2 - (xi - xio)**2)*sin(alpha)) - 2*Pie*V_inf*((eta - etao)**2 + (xi - xio)**2)**2*sin(alpha)) + ((R - eps)**2*(eta**2 - xi**2) + (eta**2 + xi**2)**2)*(Gamma*(eta - etao)*((eta - etao)**2 + (xi - xio)**2) + 2*Pie*R**2*V_inf*(-2*(eta - etao)*(xi - xio)*sin(alpha) + ((-eta + etao)**2 - (xi - xio)**2)*cos(alpha)) + 2*Pie*V_inf*((eta - etao)**2 + (xi - xio)**2)**2*cos(alpha))/2)/(Pie*(4*eta**2*xi**2*(R - eps)**4 + ((R - eps)**2*(eta**2 - xi**2) + (eta**2 + xi**2)**2)**2)*((eta - etao)**2 + (xi - xio)**2)**2)")
    # retrieve the latex version of the simplified real part
    print("\nLatex Real Part:", latex_real_part)

    # simplify the imaginary part
    # simplified_imag_part = symbolically_simplify(imag_part)
    # print("\nSimplified Imaginary Part:", simplified_imag_part)
    latex_imag_part = string_to_latex("(eta**2 + xi**2)**2*(-eta*xi*(R - eps)**2*(Gamma*(eta - etao)*((eta - etao)**2 + (xi - xio)**2) + 2*Pie*R**2*V_inf*(-2*(eta - etao)*(xi - xio)*sin(alpha) + ((-eta + etao)**2 - (xi - xio)**2)*cos(alpha)) + 2*Pie*V_inf*((eta - etao)**2 + (xi - xio)**2)**2*cos(alpha)) + ((R - eps)**2*(eta**2 - xi**2) + (eta**2 + xi**2)**2)*(Gamma*(xi - xio)*((eta - etao)**2 + (xi - xio)**2) + 2*Pie*R**2*V_inf*(2*(eta - etao)*(xi - xio)*cos(alpha) + ((-eta + etao)**2 - (xi - xio)**2)*sin(alpha)) - 2*Pie*V_inf*((eta - etao)**2 + (xi - xio)**2)**2*sin(alpha))/2)/(Pie*(4*eta**2*xi**2*(R - eps)**4 + ((R - eps)**2*(eta**2 - xi**2) + (eta**2 + xi**2)**2)**2)*((eta - etao)**2 + (xi - xio)**2)**2)")
    # retrieve the latex version of the simplified imaginary part
    print("\nLatex Imaginary Part:", latex_imag_part)


    

