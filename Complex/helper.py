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

def list_to_range(three_element_list: list):
    step_size = three_element_list[2]
    values = np.arange(three_element_list[0], three_element_list[1] + step_size, step_size)
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

if __name__ == "__main__":
    # Define symbols
    V_inf, alpha, Gamma, zeta, zeta0, eta, etao, xi, xio, R, eps, Pie, r, r0, theta, theta0 = symbols(
        'V_inf alpha Gamma zeta zeta0 eta etao xi xio R eps Pie, r, r0, theta, theta0', real=True
    )
    i = I  # Complex unit

    # G1 and G2 Expressions 
    G1 = (r*sin(theta) - r0*sin(theta0))/(r**2 + r0**2 - 2*r*r0*cos(theta - theta0))
    G2 = (r*cos(theta) - r0*cos(theta0))/(r**2 + r0**2 - 2*r*r0*cos(theta - theta0))
    # make G1 and G2 into strings
    # G1 = str(G1)
    # G2 = str(G2)
    # simplify G1 and G2
    # G1_simplified = symbolically_simplify(G1)
    # G2_simplified = symbolically_simplify(G2)
    # print("\nG1:", G1) # this is already simplified
    # print("\nG2:", G2) # this is already simplified


    # G3 and G4 Expressions
    Numerator = R**2*((cos(alpha)+i*sin(alpha))*((r**2*cos(2*theta)-2*r*r0*cos(theta+theta0)+r0**2*cos(2*theta0))-i*(r**2*sin(2*theta)-2*r*r0*sin(theta+theta0)+r0**2*sin(2*theta0))))
    Denom = (r**2*cos(2*theta)-2*r*r0*cos(theta+theta0)+r0**2*cos(2*theta0))**2 + (r**2*sin(2*theta)-2*r*r0*sin(theta+theta0)+r0**2*sin(2*theta0))**2
    Expression = (Numerator)/Denom

    real_expression = re(Expression)
    imag_expression = im(Expression)
    # make expressions into strings 
    # G3 = str(real_expression)
    # G4 = str(imag_expression)
    # print the real and imaginary parts
    # print("\nReal Part:", real_expression)
    # print("\nImaginary Part:", imag_expression)

    # simplify the real part
    # real_simplified_complex_polar_velocity = symbolically_simplify(G3)
    G3 =  R**2*(2*r**2*sin(theta)*sin(alpha - theta) + r**2*cos(alpha) - 2*r*r0*cos(-alpha + theta + theta0) + 2*r0**2*sin(theta0)*sin(alpha - theta0) + r0**2*cos(alpha))/((r**2*sin(2*theta) - 2*r*r0*sin(theta + theta0) + r0**2*sin(2*theta0))**2 + (r**2*cos(2*theta) - 2*r*r0*cos(theta + theta0) + r0**2*cos(2*theta0))**2)
    # simplify the imaginary part
    # imag_simplified_complex_polar_velocity = symbolically_simplify(G4)
    G4 =  R**2*(r**2*sin(alpha) - 2*r**2*sin(theta)*cos(alpha - theta) + 2*r*r0*sin(-alpha + theta + theta0) + r0**2*sin(alpha) - 2*r0**2*sin(theta0)*cos(alpha - theta0))/((r**2*sin(2*theta) - 2*r*r0*sin(theta + theta0) + r0**2*sin(2*theta0))**2 + (r**2*cos(2*theta) - 2*r*r0*cos(theta + theta0) + r0**2*cos(2*theta0))**2)
    # print("\nG3:", G3)
    # print("\nG4:", G4)

    G56 = 1 - ((R-eps)**2*(r**2*cos(2*theta)-i*r**2*sin(2*theta)))/((r**2*cos(2*theta))**2+(r**2*sin(2*theta))**2)
    G5 = re(G56)
    G6 = im(G56)
    # make expressions into strings
    G5 = str(G5)
    G6 = str(G6)
    # simplify the real part
    G5 = 1 - ((R - eps)**2*cos(2*theta))/r**2
    # simplify the imaginary part
    G6 = ((R - eps)**2*sin(2*theta))/r**2
    # print("G56:", G56)
    # print("\nG5:", G5)
    # print("\nG6:", G6)

    # print("\n")

    # Squared_terms = (1 - ((R - eps)**2*cos(2*theta))/r**2)**2 + (((R - eps)**2*sin(2*theta))/r**2)**2
    # Squared_terms = str(Squared_terms)
    # # simplify the squared terms
    # Squared_terms = symbolically_simplify(Squared_terms)
    # print("\nSquared Terms:", Squared_terms)

    # full_real_expression = (V_inf*(G5*(cos(alpha)+G1-G3)+G6*(G2-sin(alpha)-G4)))/(G5**2+G6**2)
    full_real_expression = V_inf*((1 - ((R-eps)**2*(cos(2*theta)))/(r**2))*(cos(alpha)+(Gamma*(r*sin(theta)-r0*sin(theta0))/(2*Pie*V_inf*(r**2 + r0**2 - 2*r*r0*cos(theta-theta0))))-(R**2*(2*r**2*sin(theta)*sin(alpha - theta) + r**2*cos(alpha) - 2*r*r0*cos(-alpha + theta + theta0) + 2*r0**2*sin(theta0)*sin(alpha - theta0) + r0**2*cos(alpha))/((r**2*sin(2*theta) - 2*r*r0*sin(theta + theta0) + r0**2*sin(2*theta0))**2 + (r**2*cos(2*theta) - 2*r*r0*cos(theta + theta0) + r0**2*cos(2*theta0))**2)))+(((R-eps)**2*(sin(2*theta)))/(r**2))*((Gamma*(r*cos(theta)-r0*cos(theta0))/(2*Pie*V_inf*(r**2 + r0**2 - 2*r*r0*cos(theta-theta0))))-sin(alpha)-((R**2*(r**2*sin(alpha)-2*r**2*sin(theta)*cos(alpha-theta)+2*r*r0*sin(-alpha+theta+theta0)+r0**2*sin(alpha)-2*r0**2*sin(theta0)*cos(alpha-theta0)))/((r**2*cos(2*theta) - 2*r*r0*cos(theta+theta0) + r0**2*cos(2*theta0))**2 + (r**2*sin(2*theta) - 2*r*r0*sin(theta+theta0) + r0**2*sin(2*theta0))**2))))/((1 - ((R-eps)**2*(cos(2*theta)))/(r**2))**2 + (((R-eps)**2*(sin(2*theta)))/(r**2))**2)
    full_real_expression = str(full_real_expression)
    # simplify the real part
    print("Finding a simplified real expression...")
    # real_simplified_complex_cartesian_velocity = symbolically_simplify(full_real_expression)
    # print("\nReal Part:\n", real_simplified_complex_cartesian_velocity)
    V_real =  r**2*(-(R - eps)**2*(-Gamma*(r*cos(theta) - r0*cos(theta0))*((r**2*sin(2*theta) - 2*r*r0*sin(theta + theta0) + r0**2*sin(2*theta0))**2 + (r**2*cos(2*theta) - 2*r*r0*cos(theta + theta0) + r0**2*cos(2*theta0))**2) + 2*Pie*R**2*V_inf*(r**2 - 2*r*r0*cos(theta - theta0) + r0**2)*(r**2*sin(alpha) - 2*r**2*sin(theta)*cos(alpha - theta) + 2*r*r0*sin(-alpha + theta + theta0) + r0**2*sin(alpha) - 2*r0**2*sin(theta0)*cos(alpha - theta0)) + 2*Pie*V_inf*((r**2*sin(2*theta) - 2*r*r0*sin(theta + theta0) + r0**2*sin(2*theta0))**2 + (r**2*cos(2*theta) - 2*r*r0*cos(theta + theta0) + r0**2*cos(2*theta0))**2)*(r**2 - 2*r*r0*cos(theta - theta0) + r0**2)*sin(alpha))*sin(2*theta) + (r**2 - (R - eps)**2*cos(2*theta))*(Gamma*(r*sin(theta) - r0*sin(theta0))*((r**2*sin(2*theta) - 2*r*r0*sin(theta + theta0) + r0**2*sin(2*theta0))**2 + (r**2*cos(2*theta) - 2*r*r0*cos(theta + theta0) + r0**2*cos(2*theta0))**2) - 2*Pie*R**2*V_inf*(r**2 - 2*r*r0*cos(theta - theta0) + r0**2)*(2*r**2*sin(theta)*sin(alpha - theta) + r**2*cos(alpha) - 2*r*r0*cos(-alpha + theta + theta0) + 2*r0**2*sin(theta0)*sin(alpha - theta0) + r0**2*cos(alpha)) + 2*Pie*V_inf*((r**2*sin(2*theta) - 2*r*r0*sin(theta + theta0) + r0**2*sin(2*theta0))**2 + (r**2*cos(2*theta) - 2*r*r0*cos(theta + theta0) + r0**2*cos(2*theta0))**2)*(r**2 - 2*r*r0*cos(theta - theta0) + r0**2)*cos(alpha)))/(2*Pie*((R - eps)**4*sin(2*theta)**2 + (r**2 - (R - eps)**2*cos(2*theta))**2)*((r**2*sin(2*theta) - 2*r*r0*sin(theta + theta0) + r0**2*sin(2*theta0))**2 + (r**2*cos(2*theta) - 2*r*r0*cos(theta + theta0) + r0**2*cos(2*theta0))**2)*(r**2 - 2*r*r0*cos(theta - theta0) + r0**2))
    print("\nReal Part:\n", V_real)

    # full_imag_expression = (V_inf*(G5*(G2-sin(alpha)-G4)-G6*(cos(alpha)+G1-G3)))/(G5**2+G6**2)
    full_imag_expression = -V_inf*((1 - ((R-eps)**2*(cos(2*theta)))/(r**2))*((Gamma*(r*cos(theta)-r0*cos(theta0))/(2*Pie*V_inf*(r**2 + r0**2 - 2*r*r0*cos(theta-theta0))))-sin(alpha)-((R**2*(r**2*sin(alpha)-2*r**2*sin(theta)*cos(alpha-theta)+2*r*r0*sin(-alpha+theta+theta0)+r0**2*sin(alpha)-2*r0**2*sin(theta0)*cos(alpha-theta0)))/((r**2*cos(2*theta) - 2*r*r0*cos(theta+theta0) + r0**2*cos(2*theta0))**2 + (r**2*sin(2*theta) - 2*r*r0*sin(theta+theta0) + r0**2*sin(2*theta0))**2)))-(((R-eps)**2*(sin(2*theta)))/(r**2))*(cos(alpha)+(Gamma*(r*sin(theta)-r0*sin(theta0))/(2*Pie*V_inf*(r**2 + r0**2 - 2*r*r0*cos(theta-theta0))))-(R**2*(2*r**2*sin(theta)*sin(alpha - theta) + r**2*cos(alpha) - 2*r*r0*cos(-alpha + theta + theta0) + 2*r0**2*sin(theta0)*sin(alpha - theta0) + r0**2*cos(alpha))/((r**2*sin(2*theta) - 2*r*r0*sin(theta + theta0) + r0**2*sin(2*theta0))**2 + (r**2*cos(2*theta) - 2*r*r0*cos(theta + theta0) + r0**2*cos(2*theta0))**2))))/((1 - ((R-eps)**2*(cos(2*theta)))/(r**2))**2 + (((R-eps)**2*(sin(2*theta)))/(r**2))**2)
    full_imag_expression = str(full_imag_expression)
    # simplify the imaginary part
    print("\nFinding a simplified imaginary expression...")
    imag_simplified_complex_cartesian_velocity = symbolically_simplify(full_imag_expression)
    print("\nImaginary Part:\n", imag_simplified_complex_cartesian_velocity)

    # Define the complex expression (replace with your full expression)
    # w_omega = V_inf*((cos(alpha)-i*sin(alpha))+i*(Gamma/(2*Pie*V_inf)) * (
    #             1 / ((xi + i * eta) - (xio + i * etao)))-((R**2*(cos(alpha)+i*sin(alpha)))/((xi+i*eta)-(xio+i*etao))**2))/(1-((R-eps)**2/(xi+i*eta)**2))
    # # Split into real and imaginary parts
    # real_part = re(w_omega)
    # imag_part = im(w_omega)

    # # make the real and imaginary parts into strings
    # real_part = str(real_part)
    # imag_part = str(imag_part)

    # # Print the results
    # print("\nReal Part:", real_part)
    # print("Type of Real Part:", type(real_part))
    # print("\nImaginary Part:", imag_part)
    # print("Type of Imaginary Part:", type(imag_part))
    # print("\n")


    # simplify the real part
    # real_simplified_complex_cartesian_velocity = symbolically_simplify(real_part)
    # real_simplified_complex_cartesian_velocity = "(eta**2 + xi**2)**2*(eta*xi*(R - eps)**2*(Gamma*(xi - xio)*((eta - etao)**2 + (xi - xio)**2) + 2*Pie*R**2*V_inf*(2*(eta - etao)*(xi - xio)*cos(alpha) + ((-eta + etao)**2 - (xi - xio)**2)*sin(alpha)) - 2*Pie*V_inf*((eta - etao)**2 + (xi - xio)**2)**2*sin(alpha)) + ((R - eps)**2*(eta**2 - xi**2) + (eta**2 + xi**2)**2)*(Gamma*(eta - etao)*((eta - etao)**2 + (xi - xio)**2) + 2*Pie*R**2*V_inf*(-2*(eta - etao)*(xi - xio)*sin(alpha) + ((-eta + etao)**2 - (xi - xio)**2)*cos(alpha)) + 2*Pie*V_inf*((eta - etao)**2 + (xi - xio)**2)**2*cos(alpha))/2)/(Pie*(4*eta**2*xi**2*(R - eps)**4 + ((R - eps)**2*(eta**2 - xi**2) + (eta**2 + xi**2)**2)**2)*((eta - etao)**2 + (xi - xio)**2)**2)" 

    # simplify the imaginary part
    # imag_simplified_complex_cartesian_velocity = symbolically_simplify(imag_part)
    # imag_simplified_complex_cartesian_velocity = "(eta**2 + xi**2)**2*(-eta*xi*(R - eps)**2*(Gamma*(eta - etao)*((eta - etao)**2 + (xi - xio)**2) + 2*Pie*R**2*V_inf*(-2*(eta - etao)*(xi - xio)*sin(alpha) + ((-eta + etao)**2 - (xi - xio)**2)*cos(alpha)) + 2*Pie*V_inf*((eta - etao)**2 + (xi - xio)**2)**2*cos(alpha)) + ((R - eps)**2*(eta**2 - xi**2) + (eta**2 + xi**2)**2)*(Gamma*(xi - xio)*((eta - etao)**2 + (xi - xio)**2) + 2*Pie*R**2*V_inf*(2*(eta - etao)*(xi - xio)*cos(alpha) + ((-eta + etao)**2 - (xi - xio)**2)*sin(alpha)) - 2*Pie*V_inf*((eta - etao)**2 + (xi - xio)**2)**2*sin(alpha))/2)/(Pie*(4*eta**2*xi**2*(R - eps)**4 + ((R - eps)**2*(eta**2 - xi**2) + (eta**2 + xi**2)**2)**2)*((eta - etao)**2 + (xi - xio)**2)**2)" 
