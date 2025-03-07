import json
import numpy as np # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import scipy.integrate as sp_int # type: ignore
from scipy.integrate import dblquad # type: ignore
# import bisection from scipy.optimize
import helper as hlp
from complex_potential_flow_class import potential_flow_object
# import tqdm for progress bar
from tqdm import tqdm


class cylinder(potential_flow_object):
    """This is a class that creates a cylinder object and performs calculations specific to a cylinder"""
    def __init__(self, json_file):
        self.cyl_json_file = json_file
        self.parse_cylinder_json()
        super().__init__(json_file) # this is doing self.() of all the variables we want to in the super class

    def parse_cylinder_json(self):
        """This function reads the json file and stores the cylinder specific data in a dictionary"""
        with open(self.cyl_json_file, 'r') as json_handle:
            input = json.load(json_handle)
            print("\n")
            self.is_D = input["geometry"]["is_trailing_edge_sharpness_D"]
            self.D = input["geometry"]["trailing_edge_sharpness_D"]
            self.is_appellian = input["appellian"]["compute_appellian"]
            self.is_analytic_accel = input["appellian"]["is_analytic_accel"]
            self.is_chi = input["appellian"]["is_chi"]
            self.is_z = input["appellian"]["is_z"]
            self.is_pressure_gradient = input["appellian"]["is_pressure_gradient"]
            self.step_size = input["appellian"]["step_size"]
            self.polynomial_order = input["appellian"]["polynomial_order"]
            self.is_plot_appellian = input["appellian"]["is_plot_appellian"]
            self.is_plot_D_sweep = input["plot"]["is_plot_D_sweep"]
            self.save_fig = input["plot"]["save_fig"]
            self.show_fig = input["plot"]["show_fig"]
            self.plot_text = input["plot"]["plot_text"]
            self.do_plot_streamlines = input["plot"]["do_plot_streamlines"]
            self.cylinder_radius = input["geometry"]["cylinder_radius"] # this is the radius of the cylinder in meters. We are normalizing everything by this value. This is used in the geometry function to calculate the geometry of the cylinder.
            self.original_cylinder_radius = self.cylinder_radius # this is the original cylinder radius in meters. We are normalizing everything by this value. This is used in the geometry function to calculate the geometry of the cylinder.
            self.cylinder_radius/=self.original_cylinder_radius # normalize the cylinder radius to 1. This makes the math easier. We are normalizing everything by the cylinder radius. This is used in the geometry function to calculate the geometry of the cylinder.
            self.is_airfoil = False
            self.is_text_file = False
            self.is_element = False
            self.is_cylinder = True
            self.type = input["geometry"]["type"]
            angle_of_attack = input["operating"]["angle_of_attack[deg]"]
            self.angle_of_attack = np.radians(angle_of_attack)
            self.freestream_velocity = input["operating"]["freestream_velocity"]
            self.design_CL = input["geometry"]["design_CL"]
            self.design_thickness = input["geometry"]["design_thickness"]
            self.output_points = input["geometry"]["output_points"]
            if self.output_points % 2 == 0:
                self.even_num_points = True
            else:
                self.even_num_points = False
            self.n_geom_points = self.output_points
            if self.type == "cylinder":
                self.circulation = input["operating"]["vortex_strength"]
                zeta_center = input["geometry"]["zeta_0"] # this is a list of two elements, real and imaginary parts of zeta_0. We need to convert it to a complex number
                self.zeta_center = (zeta_center[0] + 1j*zeta_center[1])/self.cylinder_radius
                if self.is_D:
                    self.epsilon = self.calc_epsilon_from_D(self.D)
                    print("epsilon from D = " + str(self.D) + " and zeta0 = " + str(self.zeta_center) + " is " + str(self.epsilon))
                    self.kutta_circulation = self.calc_circulation_J_airfoil()
                    print("Kutta circulation", self.kutta_circulation)
                else:
                    self.epsilon = input["geometry"]["epsilon"]
                    D = self.calc_D_from_epsilon()
                    print("D from epsilon = " + str(self.epsilon) + " and zeta0 = " + str(self.zeta_center) + " is " + str(D))
            else:
                self.zeta_center = self.calc_J_airfoil_zeta_center() # uses design CL and thickness to calculate zeta center
                print("J airfoil zeta center", self.zeta_center)
                self.epsilon = self.calc_J_airfoil_epsilon() # uses the cylinder radius, and zeta center to calculate epsilon
                print("J airfoil epsilon", self.epsilon)
                self.circulation = self.calc_circulation_J_airfoil() # uses the Kutta condition to calculate circulation
                print("J airfoil circulation", self.circulation)
                # calculate gamma so that it satisfies the Kutta condition for the airfoil

    def calc_epsilon_from_D(self, D: float):
        """This function calculates epsilon from D"""
        epsilon_o = self.calc_J_airfoil_epsilon()
        epsilon = D*(1-epsilon_o)+epsilon_o
        return epsilon
    
    def calc_D_from_epsilon(self):
        """This function calculates D from epsilon"""
        epsilon_o = self.calc_J_airfoil_epsilon()
        D = (self.epsilon-epsilon_o)/(1-epsilon_o)
        return D
    
    def geometry_zeta(self, x_coord: float): # A1 on project
        """
        Calculate the geometry of an airfoil at a given x-coordinate.

        Parameters:
        - x_coord (float): The x-coordinate at which to calculate the airfoil geometry.

        Returns:
        - upper (list): The upper surface coordinates of the airfoil at the given x-coordinate.
        - lower (list): The lower surface coordinates of the airfoil at the given x-coordinate.
        - camber (list): The camber line coordinates of the airfoil at the given x-coordinate.
        """
        x_shifted = (x_coord - np.real(self.zeta_center))/self.cylinder_radius
        radius = self.cylinder_radius
        theta = np.arccos(x_shifted/radius)
        upper = (x_coord + 1j*radius*np.sin(theta) + 1j*self.zeta_center.imag)/self.cylinder_radius
        lower = (x_coord - 1j*radius*np.sin(theta) + 1j*self.zeta_center.imag)/self.cylinder_radius
        camber = (x_coord + 1j*self.zeta_center.imag)/self.cylinder_radius
        # shift the coordinates to the center of the cylinder based on zeta_center
        return [upper.real, upper.imag], [lower.real, lower.imag], [camber.real, camber.imag]

    def get_full_geometry_zeta(self):
        """This function calls the geometry function across every point on the cylinder"""
        # create an empty array for the upper and lower surfaces and camber 
        n = 1000
        upper = np.zeros((n,2))
        lower = np.zeros((n,2))
        camber = np.zeros((n,2))
        x_coords = np.linspace(self.x_leading_edge, self.x_trailing_edge, n) # this is creating an array of x coordinates
        for i in range(n):
            upper[i], lower[i], camber[i] = self.geometry_zeta(np.real(x_coords[i]))
            # print("upper", upper[i], "lower", lower[i])
        self.upper_zeta_coords = upper
        self.lower_zeta_coords = lower
        self.camber_zeta_coords = camber
    
    def plot_geometry_zeta(self):
        """"""
        # make the line type dashed
        linetype = '--'
        color = 'black'
        plt.plot(self.upper_zeta_coords[:,0], self.upper_zeta_coords[:,1], label = "Cyl", linestyle=linetype, color=color)
        plt.plot(self.lower_zeta_coords[:,0], self.lower_zeta_coords[:,1], linestyle=linetype, color=color)
        # plt.plot(self.camber_zeta_coords[:,0], self.camber_zeta_coords[:,1], label="Camber Line", linestyle=linetype, color=color)
        # plot the zeta center with a hallow circle marker
        size = 5
        plt.scatter(self.zeta_center.real, self.zeta_center.imag, color=color, marker='o', s = size, facecolors='none', label="$\\zeta_0$")

    def geometry(self, x_from_zeta: float): # A1 on project
        """This function calculates the geometry of the cylinder at a given x-coordinate"""
        zeta_upper = self.geometry_zeta(x_from_zeta)[0][0] + 1j*self.geometry_zeta(x_from_zeta)[0][1]
        zeta_lower = self.geometry_zeta(x_from_zeta)[1][0] + 1j*self.geometry_zeta(x_from_zeta)[1][1]
        zeta_camber = self.geometry_zeta(x_from_zeta)[2][0] + 1j*self.geometry_zeta(x_from_zeta)[2][1]

        z_upper = self.zeta_to_z(zeta_upper, self.epsilon)
        z_lower = self.zeta_to_z(zeta_lower, self.epsilon)
        z_camber = self.zeta_to_z(zeta_camber, self.epsilon)
        return [z_upper.real, z_upper.imag], [z_lower.real, z_lower.imag], [z_camber.real, z_camber.imag]  

    def get_full_geometry(self):
        """This function calculates the geometry of the cylinder at every point"""
        if self.type == "airfoil" and self.output_points <= 3:
            # raise value error if the number of points is less than or equal to  3
            print("\n")
            raise ValueError("The number of points must be greater than 3")
        elif self.type == "airfoil" and self.output_points > 3:
            num_points = self.output_points//2 
        else:
            num_points = self.output_points//2 
        # num_points = 50000
        upper = np.zeros((num_points, 2))  # Ensure upper has the correct size
        lower = np.zeros((num_points, 2))
        camber = np.zeros((num_points, 2))
        
        # Make sure that x_coords has points at the leading and trailing edges
        x_coords = np.linspace(self.x_leading_edge, self.x_trailing_edge, num_points)
        
        for i in range(num_points):
            upper[i], lower[i], camber[i] = self.geometry(x_coords[i])
        
        self.upper_coords = upper
        self.lower_coords = lower
        self.camber_coords = camber
        # export upper coords and lower coords to a text file
        if self.is_plot_D_sweep:
            # concatenate the upper and lower coordinates
            combined_upper_lower = np.concatenate((upper, lower), axis=0)
            np.savetxt("combined_upper_lower_coords_at_D_is_"+str(self.D)+"_and_zeta0_is_" + str(self.zeta_center) + ".txt", combined_upper_lower)
            print("combined upper and lower coords saved to combined_upper_lower_coords_at_D_is_"+str(self.D)+"_and_zeta0_is_" + str(self.zeta_center) + ".txt")
        return self.upper_coords, self.lower_coords, self.camber_coords
    
    def loop_geometry_for_D_sweep(self):
        """loops the get_full_geometry function for a range of D values"""
    
    def get_and_plot_foci(self):
        """This function calculates and plots the foci of the ellipse using epsilon"""
        self.zeta_trailing_edge_focus = 1*(self.cylinder_radius- self.epsilon)/self.cylinder_radius
        self.zeta_leading_edge_focus = -1*(self.cylinder_radius - self.epsilon)/self.cylinder_radius
        self.z_trailing_edge_focus = self.zeta_to_z(self.zeta_trailing_edge_focus, self.epsilon)#2*(self.cylinder_radius - self.epsilon)
        self.z_leading_edge_focus = self.zeta_to_z(self.zeta_leading_edge_focus, self.epsilon)#-2*(self.cylinder_radius - self.epsilon)
        # print("Real component of z_leading_edge singularity", self.z_leading_edge_focus)
        # print("Real component of z_trailing_edge singularity", self.z_trailing_edge_focus)
        # list_of_all_possible_markers = ['o', 's', 'D', 'v', '^', '<', '>', 'p', 'P', '*', 'X', 'd', 'H', 'h', '+', '|', '_', '1', '2', '3', '4', '8']
        # D is a diamond, s is a square, o is a circle, v is a triangle pointing down, ^ is a triangle pointing up
        # reduce size of markers 
        size = 5
        size_tri = 10
        plt.scatter(self.zeta_trailing_edge_focus, 0.0, color='black', marker='^', s = size_tri, label="sing")
        plt.scatter(self.zeta_leading_edge_focus, 0.0, color='black', marker='^', s = size_tri)
        plt.scatter(self.z_trailing_edge_focus, 0.0, color='black', marker='s', s = size, label = "J-np.sing")
        plt.scatter(self.z_leading_edge_focus, 0.0, color='black', marker='s', s = size)
        return self.zeta_trailing_edge_focus, self.zeta_leading_edge_focus, self.z_trailing_edge_focus, self.z_leading_edge_focus

    def velocity(self, point_xi_eta_in_z_plane, Gamma):
        """This function calculates the velocity at a given point in the flow field in the z plane"""
        z = point_xi_eta_in_z_plane[0] + 1j*point_xi_eta_in_z_plane[1]
        zeta = self.z_to_zeta(z, self.epsilon)
        V_inf, epsilon, R, alpha, zeta0 = self.freestream_velocity, self.epsilon, self.cylinder_radius, self.angle_of_attack, self.zeta_center
        velocity = V_inf*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf*(zeta-zeta0)) - np.exp(1j*alpha)*R**2/(zeta-zeta0)**2) / (1 - (R-epsilon)**2/(zeta)**2) # eq 107
        velocity_complex = np.array([velocity.real, -velocity.imag])
        return velocity_complex
    
    def z_analytic_acceleration(self, point_xi_eta_in_z_plane, Gamma):
        """This function also calculates the acceleration at a given point in the flow field in the z plane according"""
        z = point_xi_eta_in_z_plane[0] + 1j*point_xi_eta_in_z_plane[1]
        zeta = self.z_to_zeta(z, self.epsilon)
        V_inf, epsilon, R, alpha, zeta0 = self.freestream_velocity, self.epsilon, self.cylinder_radius, self.angle_of_attack, self.zeta_center
        plus = (z + np.sqrt(z**2 - 4*(R-epsilon)**2))/2
        minus = (z - np.sqrt(z**2 - 4*(R-epsilon)**2))/2
        if zeta == plus:
            dzeta_dz_squared = (z+1)/(4*(2*epsilon-2*R+z)*(2*R-2*epsilon+z))
            d2zeta_dz2 = -(2*(R-epsilon)**2)/(z**2-4*(R-epsilon)**2)**(3/2)
        elif zeta == minus:
            dzeta_dz_squared = (1-z)/(4*(2*epsilon-2*R+z)*(2*R-2*epsilon+z))
            d2zeta_dz2 = (2*(R-epsilon)**2)/(z**2-4*(R-epsilon)**2)**(3/2)
        acceleration_1 = V_inf*(-1j*Gamma/(2*np.pi*V_inf*(zeta-zeta0)**2) + 2*np.exp(1j*alpha)*R**2/((zeta-zeta0)**3))*dzeta_dz_squared
        acceleration_2 = V_inf*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf*(zeta-zeta0)) - np.exp(1j*alpha)*R**2/(zeta-zeta0)**2)*d2zeta_dz2
        acceleration = acceleration_1 + acceleration_2
        # acceleration *= 2*V_inf*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf*(zeta-zeta0)) - np.exp(1j*alpha)*R**2/(zeta-zeta0)**2) / (1 - (R-epsilon)**2/(zeta)**2)
        acceleration_complex = np.array([acceleration.real, -acceleration.imag])
        theta = np.arctan2(point_xi_eta_in_z_plane[1], point_xi_eta_in_z_plane[0])
        polar_acceleration = self.polar_vector(theta, acceleration_complex)
        return polar_acceleration

    # def pressure_gradient(self, point_xi_eta_in_z_plane, Gamma):
    #     """This function calculates the pressure gradient at a given point in the flow field in the z plane"""
    #     z = point_xi_eta_in_z_plane[0] + 1j*point_xi_eta_in_z_plane[1]
    #     zeta = self.z_to_zeta(z, self.epsilon)
    #     point_xi_eta_in_zeta_plane = [zeta.real, zeta.imag]
    #     V_inf, epsilon, R, alpha, zeta0 = self.freestream_velocity, self.epsilon, self.cylinder_radius, self.angle_of_attack, self.zeta_center
    #     plus = (z + np.sqrt(z**2 - 4*(R-epsilon)**2))/2
    #     minus = (z - np.sqrt(z**2 - 4*(R-epsilon)**2))/2
    #     first = -2*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf)*1/(zeta-zeta0)-(R**2*np.exp(1j*alpha))/(zeta-zeta0)**2)
    #     second = -1j*Gamma/(2*np.pi*V_inf)*1/(zeta-zeta0)**2 + 2*R**2*np.exp(1j*alpha)/(zeta-zeta0)**3
    #     third = np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf)*1/(zeta-zeta0) - (R**2*np.exp(1j*alpha))/(zeta-zeta0)**2
    #     fourth = (1 - ((R - epsilon)**2)/(zeta**2))
    #     if zeta == plus:
    #         fifth = (1+z)/(4*(2*epsilon-2*R+z)*(2*R-2*epsilon+z))
    #         sixth = (-2*(R - epsilon)**2)/(z**2 - 4*(R - epsilon)**2)**(3/2)
    #     if zeta == minus:
    #         fifth = (1-z)/(4*(2*epsilon-2*R+z)*(2*R-2*epsilon+z))
    #         sixth = (2*(R - epsilon)**2)/((z**2 - 4*(R - epsilon)**2)**(3/2))
    #     pressure_gradient = first*(second*fifth + third*sixth) / fourth
    #     pressure_gradient_complex = np.array([pressure_gradient.real, -pressure_gradient.imag])
    #     theta = np.arctan2(point_xi_eta_in_z_plane[1], point_xi_eta_in_z_plane[0])
    #     pressure_gradient_polar = self.polar_vector(theta, pressure_gradient_complex)
    #     magnitude_pressure_gradient_complex = np.linalg.norm(pressure_gradient_complex)
    #     return magnitude_pressure_gradient_complex

    def polar_vector(self, theta, cartesian_vector):
        """This function converts the cartesian velocity to polar velocity (can go from z, zeta, or chi plane to polar velocity)"""
        r = cartesian_vector[0]*np.cos(theta) + cartesian_vector[1]*np.sin(theta)
        # print("\nradial velocity", velocity_r)
        theta = cartesian_vector[1]*np.cos(theta) - cartesian_vector[0]*np.sin(theta)
        # print("theta velocity", velocity_theta)
        polar_velocity = np.array([r, theta])
        return polar_velocity
    
    def numerical_convective_acceleration(self, point_xi_eta, Gamma, step=1e-6):
        """calculates the convective acceleration numerically in the z or chi plane"""
        if self.is_z:
            z = point_xi_eta[0] + 1j*point_xi_eta[1]
            zeta = self.z_to_zeta(z, self.epsilon)
            point_z = [z.real, z.imag]
            r, theta = hlp.xy_to_r_theta(zeta.real, zeta.imag)
            point_r_theta = [hlp.xy_to_r_theta(z.real, z.imag)[0], hlp.xy_to_r_theta(z.real, z.imag)[1]]
            # r, theta = [point_r_theta[0], point_r_theta[1]]
            cartesian_z_velocity = self.velocity(point_z, Gamma)
            polar_z_velocity = self.polar_vector(theta, cartesian_z_velocity)
            omega_r, omega_theta = polar_z_velocity[0], polar_z_velocity[1]
            omega_r_plus_dr, omega_r_minus_dr, omega_r_plus_dtheta, omega_r_minus_dtheta, omega_theta_plus_dr, omega_theta_minus_dr, omega_theta_plus_dtheta, omega_theta_minus_dtheta = self.function_plus_minus_step_variable(point_r_theta, Gamma, step, self.velocity)
        elif self.is_chi:
            r, theta = hlp.xy_to_r_theta(point_xi_eta[0], point_xi_eta[1])
            point_r_theta = [r, theta]
            cartesian_Chi_velocity = self.velocity_chi(point_xi_eta, Gamma)
            polar_Chi_velocity = self.polar_vector(theta, cartesian_Chi_velocity)
            omega_r, omega_theta = polar_Chi_velocity[0], polar_Chi_velocity[1]
            omega_r_plus_dr, omega_r_minus_dr, omega_r_plus_dtheta, omega_r_minus_dtheta, omega_theta_plus_dr, omega_theta_minus_dr, omega_theta_plus_dtheta, omega_theta_minus_dtheta = self.function_plus_minus_step_variable(point_r_theta, Gamma, step, self.velocity_chi)

        # calculate the partial derivatives of omega r with respect to r and theta
        partial_omega_r_wrt_r = hlp.central_difference(omega_r_plus_dr, omega_r_minus_dr, step)
        partial_omega_r_wrt_theta = hlp.central_difference(omega_r_plus_dtheta, omega_r_minus_dtheta, step)
        # calculate the partial derivatives of omega theta with respect to r and theta
        partial_omega_theta_wrt_r = hlp.central_difference(omega_theta_plus_dr, omega_theta_minus_dr, step)
        partial_omega_theta_wrt_theta = hlp.central_difference(omega_theta_plus_dtheta, omega_theta_minus_dtheta, step)
        # calculate the convective acceleration
        convective_acceleration_r = omega_r*partial_omega_r_wrt_r + (omega_theta/r)*partial_omega_r_wrt_theta - omega_theta**2/r
        convective_acceleration_theta = omega_r*partial_omega_theta_wrt_r + (omega_theta/r)*partial_omega_theta_wrt_theta + omega_r*omega_theta/r
        convective_acceleration = np.array([convective_acceleration_r, convective_acceleration_theta])
        return convective_acceleration
    
    def function_plus_minus_step_variable(self, point_r_theta, Gamma, stepsize, vel_func: callable):
        """takes in a function and returns the function plus and minus a stepsize"""
        r, theta = point_r_theta[0], point_r_theta[1]
        r_plus, r_minus = r + stepsize, r - stepsize
        theta_plus, theta_minus = theta + stepsize, theta - stepsize

        x_r_plus, x_r_minus, y_r_plus, y_r_minus = hlp.r_theta_to_xy(r_plus, theta)[0], hlp.r_theta_to_xy(r_minus, theta)[0], hlp.r_theta_to_xy(r_plus, theta)[1], hlp.r_theta_to_xy(r_minus, theta)[1]
        x_theta_plus, x_theta_minus, y_theta_plus, y_theta_minus = hlp.r_theta_to_xy(r, theta_plus)[0], hlp.r_theta_to_xy(r, theta_minus)[0], hlp.r_theta_to_xy(r, theta_plus)[1], hlp.r_theta_to_xy(r, theta_minus)[1]
        omega_xy_plus_r, omega_xy_minus_r = vel_func([x_r_plus, y_r_plus], Gamma), vel_func([x_r_minus, y_r_minus], Gamma)
        omega_xy_plus_theta, omega_xy_minus_theta = vel_func([x_theta_plus, y_theta_plus], Gamma), vel_func([x_theta_minus, y_theta_minus], Gamma)
        omega_r_plus_dr, omega_r_minus_dr = self.polar_vector(theta, omega_xy_plus_r)[0], self.polar_vector(theta, omega_xy_minus_r)[0]
        omega_r_plus_dtheta, omega_r_minus_dtheta = self.polar_vector(theta, omega_xy_plus_theta)[0], self.polar_vector(theta, omega_xy_minus_theta)[0]
        omega_theta_plus_dr, omega_theta_minus_dr = self.polar_vector(theta, omega_xy_plus_r)[1], self.polar_vector(theta, omega_xy_minus_r)[1]
        omega_theta_plus_dtheta, omega_theta_minus_dtheta = self.polar_vector(theta, omega_xy_plus_theta)[1], self.polar_vector(theta, omega_xy_minus_theta)[1]
        return omega_r_plus_dr, omega_r_minus_dr, omega_r_plus_dtheta, omega_r_minus_dtheta, omega_theta_plus_dr, omega_theta_minus_dr, omega_theta_plus_dtheta, omega_theta_minus_dtheta

    def appellian_acceleration_loop(self, Gamma: float, r_values: np.array, theta_values: np.array, dr: float, dtheta: float, is_Chi: bool, acceleration: callable):
        """"""
        # print("Gamma", Gamma)
        # print("r_values", r_values)
        # print("theta_values", theta_values)
        # print("dr", dr)
        # print("dtheta", dtheta)
        # print("is_Chi", is_Chi)
        # print("acceleration", acceleration)

        Gamma_values = np.array([Gamma])
        # index = 0
        Appellian_value = 0.0
        Appellian_array = np.zeros((len(Gamma_values), 2))
        xi_eta_values = np.zeros((len(r_values)*len(theta_values), 2))
        if is_Chi:
                for j in range(len(r_values)):
                    for k in range(len(theta_values)):
                        Chi_val = hlp.r_theta_to_xy(r_values[j], theta_values[k])[0] + 1j*hlp.r_theta_to_xy(r_values[j], theta_values[k])[1] # the r, theta point in the Chi plane is converted to a complex number
                        # print("Chi_val", Chi_val)
                        area_element = r_values[j]*dr*dtheta
                        # print("area element", area_element)
                        # print("\n")
                        convective_acceleration = acceleration([Chi_val.real, Chi_val.imag], Gamma)
                        Appellian_value += np.dot(convective_acceleration, convective_acceleration)*area_element
                        # xi_eta_values[index] = [Chi_val.real, Chi_val.imag]
                        # index += 1
                Appellian_array[0] = [Gamma, 0.5*Appellian_value]
        else:
                for j in range(len(r_values)):
                    for k in range(len(theta_values)):
                        Chi = hlp.r_theta_to_xy(r_values[j], theta_values[k])[0] + 1j*hlp.r_theta_to_xy(r_values[j], theta_values[k])[1] # the r, theta point in the Chi plane is converted to a complex number
                        zeta = self.Chi_to_zeta(Chi)
                        z = self.zeta_to_z(zeta, self.epsilon)
                        r_z = hlp.xy_to_r_theta(z.real, z.imag)[0]
                        area_element = r_z * dr * dtheta
                        convective_acceleration = acceleration([z.real, z.imag], Gamma)
                        Appellian_value += np.dot(convective_acceleration, convective_acceleration)*area_element
                        # index += 1
                Appellian_array[0] = [Gamma, 0.5*Appellian_value]
                # Appellian_value = 0.0
                # index = 0
        Appellian_array = 0.5*Appellian_value
        return Appellian_array
        
    def numerically_integrate_appellian(self, Gamma: float, r_values: np.array, theta_values: np.array, is_analytic_accel: bool):
        """This function calculates the Appellian function numerically requires evenly spaced ranges for Gamma, r, and theta in Chi"""
        # create a meshgrid of r and theta values, the first value in r_range is the lower bound the second value is the upper bound, the third value is the increment size
        # calculate the area element 
        if len(r_values) == 1:
            dr = 1.0
        else:        
            dr = r_values[1] - r_values[0]
        if len(theta_values) == 1:
            dtheta = 1.0
        else:
            dtheta = theta_values[1] - theta_values[0]
        dr_original, dtheta_original = dr, dtheta
        # now get dr and dtheta in z plane using the zeta to z function 
        d_zeta = hlp.r_theta_to_xy(dr, dtheta)[0] + 1j*hlp.r_theta_to_xy(dr, dtheta)[1]
        d_z = self.zeta_to_z(d_zeta, self.epsilon)
        d_xi_z, d_eta_z = d_z.real, d_z.imag
        # now these are the dr and dtheta values in the z plane
        dr, dtheta = hlp.xy_to_r_theta(d_xi_z, d_eta_z)
        n = len(r_values)*len(theta_values)
        if is_analytic_accel and self.is_chi:
            # print("ANALYTIC CONVECTIVE ACCELERATION IN CHI")
            Appellian_array = self.appellian_acceleration_loop(Gamma, r_values, theta_values, dr_original, dtheta_original, True, self.chi_convective_acceleration)
        elif is_analytic_accel and self.is_z:
            # print("ANALYTIC CONVECTIVE ACCELERATION IN Z")
            Appellian_array = self.appellian_acceleration_loop(Gamma, r_values, theta_values, dr, dtheta, False, self.z_convective_acceleration)
        elif is_analytic_accel and self.is_pressure_gradient: 
            # print("ANALYTIC PRESSURE GRADIENT")
            Appellian_array = self.appellian_acceleration_loop(Gamma, r_values, theta_values, dr, dtheta, False, self.pressure_gradient)
        elif not is_analytic_accel and self.is_z:
            # print("NUMERICAL CONVECTIVE ACCELERATION IN Z")
            Appellian_array = self.appellian_acceleration_loop(Gamma, r_values, theta_values, dr, dtheta, False, self.numerical_convective_acceleration)
        elif not is_analytic_accel and self.is_chi:
            # print("NUMERICAL CONVECTIVE ACCELERATION IN CHI")
            Appellian_array = self.appellian_acceleration_loop(Gamma, r_values, theta_values, dr_original, dtheta_original, True, self.numerical_convective_acceleration)
        else:
            raise ValueError("The function you are trying to use is not implemented")
        # print("\n")
        # print("\n")
        # print("Appellian array", Appellian_array)
        # print("\n")
        # print("\n")
        return Appellian_array
    
    
    def integrand(self, theta, r, is_Chi, acceleration, Gamma):
        if is_Chi:
            Chi_val = hlp.r_theta_to_xy(r, theta)[0] + 1j * hlp.r_theta_to_xy(r, theta)[1]
            convective_acceleration = acceleration([Chi_val.real, Chi_val.imag], Gamma)
        else:
            Chi = hlp.r_theta_to_xy(r, theta)[0] + 1j * hlp.r_theta_to_xy(r, theta)[1]
            zeta = self.Chi_to_zeta(Chi)
            z = self.zeta_to_z(zeta, self.epsilon)
            r_z = hlp.xy_to_r_theta(z.real, z.imag)[0]
            convective_acceleration = acceleration([z.real, z.imag], Gamma)
            r = r_z  # Modify r for area element
        area_element = r  # Since dA = r dr dtheta in polar coordinates
        return np.dot(convective_acceleration, convective_acceleration) * area_element

    
    def appellian_acceleration_integral(self, Gamma, r_min, r_max, theta_min, theta_max, is_Chi, acceleration):
        """
        Compute the integral using scipy.dblquad instead of loops.
        """
        # Perform double integration over r and theta
        f = lambda r, theta: self.integrand(theta, r, is_Chi, acceleration, Gamma)
        result, _ = dblquad(f, theta_min, theta_max, r_min, r_max)
        return 0.5 * result

    def numerically_dbquad_integrate_appellian(self, Gamma_range, r_range, theta_range, is_analytic_accel):
        """
        Integrate the Appellian function using dblquad.
        """
        Gamma_values = hlp.list_to_range(Gamma_range)
        r_values = hlp.list_to_range(r_range)
        theta_values = hlp.list_to_range(theta_range)

        r_min, r_max = r_values[0], r_values[-1]
        theta_min, theta_max = theta_values[0], theta_values[-1]

        Appellian_array = np.zeros((len(Gamma_values), 2))
        if is_analytic_accel and self.is_chi:
            print("ANALYTIC CONVECTIVE ACCELERATION IN CHI")
            acceleration_func = self.chi_convective_acceleration
        elif is_analytic_accel and self.is_z:
            print("ANALYTIC CONVECTIVE ACCELERATION IN Z")
            acceleration_func = self.z_convective_acceleration
        elif is_analytic_accel and self.is_pressure_gradient:
            print("ANALYTIC PRESSURE GRADIENT")
            acceleration_func = self.pressure_gradient
        elif not is_analytic_accel and self.is_z:
            print("NUMERICAL CONVECTIVE ACCELERATION IN Z")
            acceleration_func = self.numerical_convective_acceleration
        elif not is_analytic_accel and self.is_chi:
            print("NUMERICAL CONVECTIVE ACCELERATION IN CHI")
            acceleration_func = self.numerical_convective_acceleration
        else:
            raise ValueError("Invalid acceleration function.")
        for i in tqdm(range(len(Gamma_values)), desc="Calculating Appellian function"):
            Gamma = Gamma_values[i]
            Appellian_array[i] = [Gamma, self.appellian_acceleration_integral(Gamma, r_min, r_max, theta_min, theta_max, self.is_chi, acceleration_func)]
        return Appellian_array

    def r_theta_Chi_to_r_theta_zeta(self, r_Chi, theta_Chi):
        """This function takes in r and theta in the Chi plane and returns r and theta in the zeta plane"""
        xi_Chi, eta_Chi = hlp.r_theta_to_xy(r_Chi, theta_Chi)
        xi_zeta, eta_zeta = xi_Chi + self.zeta_center.real, eta_Chi + self.zeta_center.imag
        r_zeta, theta_zeta = hlp.xy_to_r_theta(xi_zeta, eta_zeta)
        return r_zeta, theta_zeta
    
    def r_theta_zeta_to_r_theta_z(self, r_zeta, theta_zeta):
        """This function takes in r and theta in the zeta plane and returns r and theta in the z plane"""
        xi_zeta, eta_zeta = hlp.r_theta_to_xy(r_zeta, theta_zeta)
        point_xi_eta_in_zeta = xi_zeta + 1j*eta_zeta
        z = self.zeta_to_z(point_xi_eta_in_zeta, self.epsilon)
        r_z, theta_z = hlp.xy_to_r_theta(z.real, z.imag)
        return r_z, theta_z

    def zeta_to_z(self, zeta: complex, epsilon: float):
        """This function takes in a zeta coordinate and returns the z coordinate"""
        if np.isclose(zeta.real, 0.0) and np.isclose(zeta.imag, 0.0):
            z = zeta
        else:
            z = zeta + (self.cylinder_radius - epsilon)**2/zeta # eq 96
        return z
    
    def zeta_to_Chi(self, zeta: complex):
        """This function takes in a zeta coordinate and returns the Chi coordinate"""
        Chi = zeta - self.zeta_center
        return Chi

    def Chi_to_zeta(self, Chi: complex):
        """This function takes in a Chi coordinate and returns the zeta coordinate"""
        zeta = Chi + self.zeta_center
        return zeta

    def z_to_zeta(self, z: complex, epsilon: float): # eq 104 
        """This function takes in a z coordinate and returns the zeta coordinate"""
        z_1 = z**2 - 4*(self.cylinder_radius - epsilon)**2
        if z_1.real > 0:
            zeta = (z + np.sqrt(z_1))/2
            zeta_2 = (z - np.sqrt(z_1))/2
        elif z_1.real < 0:
            zeta = (z - 1j*np.sqrt(-z_1))/2
            zeta_2 = (z + 1j*np.sqrt(-z_1))/2
        elif z_1.imag >= 0:
            zeta = (z + np.sqrt(z_1))/2
            zeta_2 = (z - np.sqrt(z_1))/2
        else:
            zeta = (z - 1j*np.sqrt(-z_1))/2
            zeta_2 = (z + 1j*np.sqrt(-z_1))/2
        if abs(zeta_2 - self.zeta_center) > abs(zeta - self.zeta_center):
            zeta = zeta_2
        return zeta
    
    def calc_J_airfoil_zeta_center(self):
        """This function calculates the zeta center of the airfoil"""
        first = (-4*self.design_thickness/(3*np.sqrt(3)))
        second = 1j*self.design_CL/(2*np.pi*(1+4*self.design_thickness/(3*np.sqrt(3))))
        zeta_center = self.cylinder_radius*(first + second) # eq 152 in complex variables
        return zeta_center
    
    def calc_J_airfoil_thickness(self, CLd):
        """This function calculates the thickness of the airfoil"""
        thickness = -3*np.sqrt(3)/4 + (3*np.sqrt(3)*CLd)/(8*np.pi*(self.zeta_center.imag/self.cylinder_radius))  # rearrangement of eq 152 in complex variables
        return thickness

    def calc_J_airfoil_epsilon(self):
        """This function calculates the epsilon of the airfoil"""
        epsilon = self.cylinder_radius - 1*np.sqrt(self.cylinder_radius**2-self.zeta_center.imag**2)-self.zeta_center.real # eq 113 in complex variables
        return epsilon
    
    def calc_circulation_J_airfoil(self):
        """This function calculates the circulation of the airfoil based on the Kutta condition"""
        gamma = 4*np.pi*self.freestream_velocity*(np.sin(self.angle_of_attack)*np.sqrt(self.cylinder_radius**2 - self.zeta_center.imag**2) + self.zeta_center.imag*np.cos(self.angle_of_attack)) # eq 122 in complex variables
        return gamma
    
    def calc_zeta_real_intercepts_J_airfoil(self):
        """This function calculates the real intercepts of the airfoil"""
        zeta_real_leading_edge = -1*np.sqrt(self.cylinder_radius**2 - self.zeta_center.imag**2) + self.zeta_center.real
        zeta_real_trailing_edge = np.sqrt(self.cylinder_radius**2 - self.zeta_center.imag**2) + self.zeta_center.real
        return zeta_real_leading_edge, zeta_real_trailing_edge
    
    def calc_zeta_real_intercepts(self):
        """This function calculates the real intercepts of the airfoil"""
        zeta_real_leading_edge = -1*np.sqrt(self.cylinder_radius**2 - self.zeta_center.imag**2) + self.zeta_center.real 
        zeta_real_trailing_edge = np.sqrt(self.cylinder_radius**2 - self.zeta_center.imag**2) + self.zeta_center.real 
        return zeta_real_leading_edge, zeta_real_trailing_edge
    
    def calc_J_airfoil_CL(self):
        """This function calculates the lift of the airfoil"""
        self.lift_j_airfoil = (2*np.pi*(np.sin(self.angle_of_attack)+(self.zeta_center.imag*np.cos(self.angle_of_attack))/np.sqrt(self.cylinder_radius**2 - self.zeta_center.imag**2)))/(1+self.zeta_center.real/(np.sqrt(self.cylinder_radius**2 - self.zeta_center.imag**2)-self.zeta_center.real)) # eq 124
        return self.lift_j_airfoil
    
    def calc_J_airfoil_Cmz(self, point_z_plane): # eq 141
        """This function calculates the moment coefficient of the airfoil""" 
        First = (np.sin(2*self.angle_of_attack)*np.pi/4 * ((self.cylinder_radius**2-self.zeta_center.imag**2-self.zeta_center.real**2)/(self.cylinder_radius**2-self.zeta_center.imag**2))**2 )
        Second = ((0.25*self.lift_j_airfoil))
        Third = ((point_z_plane.real-self.zeta_center.real)*np.cos(self.angle_of_attack) + (point_z_plane.imag-self.zeta_center.imag)*np.sin(self.angle_of_attack))/(self.cylinder_radius**2-self.zeta_center.imag**2)
        Fourth = (np.sqrt(self.cylinder_radius**2-self.zeta_center.imag**2)-self.zeta_center.real)
        Cmz = First + Second*Third*Fourth
        return Cmz
    
    def calc_J_airfoil_c4(self):
        """This function calculates the c4 location of the airfoil"""
        zeta_leading_edge, zeta_trailing_edge = self.calc_zeta_real_intercepts()
        z_leading_edge = self.zeta_to_z(zeta_leading_edge, self.epsilon)
        z_trailing_edge = self.zeta_to_z(zeta_trailing_edge, self.epsilon)
        c4 = (3*z_leading_edge+z_trailing_edge)/4
        return c4
    
    def shift_joukowski_airfoil(self):
        """This function finds shifts all the points back to where zeta_center = 0, finds how far to the left the leading edge is, and shifts all points by that much (so the leading edge is at 0)
        Then it finds the right most point, and finds the length from the right most point to the left most point, and scales all points by that much"""
        # shift all the points back to where zeta_center = 0 by shifting the x coordinates by the real part of zeta_center
        self.surface_points[:,0] = self.surface_points[:,0] - self.zeta_center.real
        # now shift the y coordinates by the imaginary part of zeta_center
        self.surface_points[:,1] = self.surface_points[:,1] - self.zeta_center.imag
        # find how far to the left the leading edge is
        leading_edge = np.min(self.surface_points[:,0])
        # shift all points by that much
        self.surface_points[:,0] = self.surface_points[:,0] - leading_edge
        # find the right most point
        trailing_edge = np.max(self.surface_points[:,0])
        self.surface_points = self.surface_points/trailing_edge
        return self.surface_points
    
    def make_surface_points_go_from_lower_trailing_to_upper_trailing(self):
        """This function removes duplicate points from self.surface_points while preserving order, and reorders 
        the surface points so that they go from the lower trailing edge to the upper trailing edge."""
        # Remove duplicates while preserving order
        unique_points = []
        seen = set()
        for point in self.surface_points:
            # Convert point to a tuple to make it hashable for the set
            point_tuple = tuple(point)
            if point_tuple not in seen:
                seen.add(point_tuple)
                unique_points.append(point)
        # Update surface_points to only unique points, maintaining order
        self.surface_points = np.array(unique_points)
        
    def output_J_airfoil(self):
        # flip the lower coordinates so that they go from the trailing edge to the leading edge
        self.lower_coords = np.flip(self.lower_coords, axis=0)
        # make total surface_points array that starts with lower, then does upper
        self.surface_points = np.concatenate((self.lower_coords, self.upper_coords), axis=0)
        self.make_surface_points_go_from_lower_trailing_to_upper_trailing()
        self.surface_points = self.shift_joukowski_airfoil()
        # if self.export_geometry is True, then export the geometry to a txt file
        print("Exporting geometry to txt file...")
        file_name = "Joukowski" + self.type + "_" + str(self.n_geom_points) + ".txt"
        # save text file with the geometry in double precision
        np.savetxt(file_name, self.surface_points, delimiter = ",", header = "x, y", comments = "")
        print("Geometry exported to", file_name)

    def velocity_chi(self, point_xy_in_Chi_plane, Gamma):
        """Start with a Chi value that is shifted from zeta_center"""
        xi, eta = point_xy_in_Chi_plane[0], point_xy_in_Chi_plane[1]
        r, theta = hlp.xy_to_r_theta(xi, eta)
        zeta_center = self.zeta_center
        xio, etao = zeta_center.real, zeta_center.imag
        r0, theta0 = hlp.xy_to_r_theta(xio, etao)
        V_inf, R, alpha, epsilon = self.freestream_velocity, self.cylinder_radius, self.angle_of_attack, self.epsilon
        G1, G2, G3, G4, G5, G6 = self.calc_Chi_G_values(r, theta, alpha, epsilon, R, r0, theta0)
        V_real = (Gamma/(2*np.pi))*((G1*G5+G2*G6)/(G5**2 + G6**2)) + V_inf*((G5*np.cos(alpha)-G6*np.sin(alpha)-G3*G5-G4*G6)/(G5**2 + G6**2))
        V_imag = (-1*Gamma/(2*np.pi))*((G2*G5-G1*G6)/(G5**2 + G6**2)) + V_inf*((G6*np.cos(alpha)+G5*np.sin(alpha)+G4*G5-G3*G6)/(G5**2 + G6**2))
        velocity_complex = np.array([V_real, V_imag])
        return velocity_complex

    def calc_Chi_G_values(self, r, theta, alpha, epsilon, R, r0, theta0):
        """takes in r, theta, alpha, epsilon, R, r0, theta0 and calculates the G values"""
        G1 = np.sin(theta)/r
        G2 = np.cos(theta)/r
        G3 = R**2*np.cos(alpha-2*theta)/r**2
        G4 = R**2*np.sin(alpha-2*theta)/r**2
        G5 = 1 - ((R-epsilon)**2*(r**2*np.cos(2*theta)+r0**2*np.cos(2*theta0)+2*r*r0*np.cos(theta+theta0)))/((r**2*np.cos(2*theta)+r0**2*np.cos(2*theta0)+2*r*r0*np.cos(theta+theta0))**2+(r**2*np.sin(2*theta)+r0**2*np.sin(2*theta0)+2*r*r0*np.sin(theta+theta0))**2)
        G6 = ((R-epsilon)**2*(r**2*np.sin(2*theta)+r0**2*np.sin(2*theta0)+2*r*r0*np.sin(theta+theta0)))/((r**2*np.cos(2*theta)+r0**2*np.cos(2*theta0)+2*r*r0*np.cos(theta+theta0))**2+(r**2*np.sin(2*theta)+r0**2*np.sin(2*theta0)+2*r*r0*np.sin(theta+theta0))**2)
        return G1, G2, G3, G4, G5, G6
    
    def calc_z_G_values(self, r, theta, alpha, epsilon, R, r_0, theta_0):
        """takes in r, theta, alpha, epsilon, R, r0, theta0 and calculates the G values"""
        G1 = (r*np.sin(theta) - r_0*np.sin(theta_0))/(r**2 + r_0**2 - 2*r*r_0*np.cos(theta - theta_0))
        G2 = (r*np.cos(theta) - r_0*np.cos(theta_0))/(r**2 + r_0**2 - 2*r*r_0*np.cos(theta - theta_0))
        G3 = (R**2*(np.cos(alpha)*((r*np.cos(theta)-r_0*np.cos(theta_0))**2-(r*np.sin(theta)-r_0*np.sin(theta_0))**2)+2*np.sin(alpha)*((r*np.cos(theta)-r_0*np.cos(theta_0))*(r*np.sin(theta)-r_0*np.sin(theta_0)))))/(((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2+(2*(r*np.cos(theta)-r_0*np.cos(theta_0))*(r*np.sin(theta)-r_0*np.sin(theta_0)))**2)
        G4 = (R**2*(np.sin(alpha)*((r*np.cos(theta)-r_0*np.cos(theta_0))**2-(r*np.sin(theta)-r_0*np.sin(theta_0))**2)-2*np.cos(alpha)*((r*np.cos(theta)-r_0*np.cos(theta_0))*(r*np.sin(theta)-r_0*np.sin(theta_0)))))/(((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2+(2*(r*np.cos(theta)-r_0*np.cos(theta_0))*(r*np.sin(theta)-r_0*np.sin(theta_0)))**2)
        G5 = 1 - (np.cos(2*theta)*(R - epsilon)**2)/r**2
        G6 = (np.sin(2*theta)*(R - epsilon)**2)/r**2
        return G1, G2, G3, G4, G5, G6
    
    def partial_coefficients_of_omega_r_wrt_G_Chi(self, V_inf, theta, alpha, G1, G2, G3, G4, G5, G6):
        """This function retrieves the partial derivatives of omega r with respect to G values"""
        # G1, G2, G3, G4, G5, G6 = self.calc_Chi_G_values(r, theta, alpha, epsilon, R, r0, theta0)
        # omega r partials with respect to G1, G2, G3, G4, G5, G6
        A_omega_rG = 1/(2*np.pi)*((G5*np.cos(theta))/(G5**2 + G6**2) + (G6*np.sin(theta))/(G5**2 + G6**2))
        B_omega_rG = 1/(2*np.pi)*((G6*np.cos(theta))/(G5**2 + G6**2) - (G5*np.sin(theta))/(G5**2 + G6**2))
        C_omega_rG = - V_inf*(G5*np.cos(theta)/(G5**2 + G6**2) + G6*np.sin(theta)/(G5**2 + G6**2))
        D_omega_rG = V_inf*(G5*np.sin(theta)/(G5**2 + G6**2) - G6*np.cos(theta)/(G5**2 + G6**2))
        E_omega_rG = (G1*np.cos(theta))/(2*np.pi*(G5**2 + G6**2)) - (G2*np.sin(theta))/(2*np.pi*(G5**2 + G6**2)) - (G5*np.cos(theta)*(G1*G5 + G2*G6))/(np.pi*(G5**2 + G6**2)**2) - (G5*np.sin(theta)*(G1*G6 - G2*G5))/(np.pi*(G5**2 + G6**2)**2)
        F_omega_rG = V_inf*((np.sin(theta)*(G4+np.sin(alpha)))/(G5**2+G6**2)-(np.cos(theta)*(G3-np.cos(alpha)))/(G5**2+G6**2)+(2*G5*np.cos(theta)*(G6*np.sin(alpha)-G5*np.cos(alpha)+G3*G5+G4*G6))/(G5**2+G6**2)**2-(2*G5*np.sin(theta)*(G6*np.cos(alpha)+G5*np.sin(alpha)-G3*G6+G4*G5))/(G5**2+G6**2)**2)
        G_omega_rG = (G2*np.cos(theta))/(2*np.pi*(G5**2 + G6**2)) + (G1*np.sin(theta)/(2*np.pi*(G5**2+G6**2))) - (G6*np.cos(theta)*(G1*G5+G2*G6))/(np.pi*(G5**2+G6**2)**2) - (G6*np.sin(theta)*(G1*G6-G2*G5))/(np.pi*(G5**2+G6**2)**2)
        H_omega_rG = V_inf*((2*G6*np.cos(theta)*(G6*np.sin(alpha)-G5*np.cos(alpha)+G3*G5+G4*G6))/(G5**2+G6**2)**2-(np.sin(theta)*(G3-np.cos(alpha)))/(G5**2+G6**2)-(np.cos(theta)*(G4+np.sin(alpha)))/(G5**2+G6**2)-(2*G6*np.sin(theta)*(G6*np.cos(alpha)+G5*np.sin(alpha)-G3*G6+G4*G5))/(G5**2+G6**2)**2)
        return A_omega_rG, B_omega_rG, C_omega_rG, D_omega_rG, E_omega_rG, F_omega_rG, G_omega_rG, H_omega_rG
    
    def partial_coefficients_of_omega_r_wrt_G_z(self, V_inf, theta, alpha, G1, G2, G3, G4, G5, G6):
        """This function retrieves the partial derivatives of omega r with respect to G values"""
        A_omega_rG = 1/(2*np.pi)*((G5*np.cos(theta))/(G5**2 + G6**2) + (G6*np.sin(theta))/(G5**2 + G6**2))
        B_omega_rG = 1/(2*np.pi)*((G6*np.cos(theta))/(G5**2 + G6**2) - (G5*np.sin(theta))/(G5**2 + G6**2))
        C_omega_rG = - V_inf*(G5*np.cos(theta)/(G5**2 + G6**2) + G6*np.sin(theta)/(G5**2 + G6**2))
        D_omega_rG = V_inf*(G5*np.sin(theta)/(G5**2 + G6**2) - G6*np.cos(theta)/(G5**2 + G6**2))
        E_omega_rG = (G1*np.cos(theta))/(2*np.pi*(G5**2 + G6**2)) - (G2*np.sin(theta))/(2*np.pi*(G5**2 + G6**2)) - (G5*np.cos(theta)*(G1*G5 + G2*G6))/(np.pi*(G5**2 + G6**2)**2) - (G5*np.sin(theta)*(G1*G6 - G2*G5))/(np.pi*(G5**2 + G6**2)**2)
        F_omega_rG = V_inf*((np.sin(theta)*(G4+np.sin(alpha)))/(G5**2+G6**2)-(np.cos(theta)*(G3-np.cos(alpha)))/(G5**2+G6**2)+(2*G5*np.cos(theta)*(G6*np.sin(alpha)-G5*np.cos(alpha)+G3*G5+G4*G6))/(G5**2+G6**2)**2-(2*G5*np.sin(theta)*(G6*np.cos(alpha)+G5*np.sin(alpha)-G3*G6+G4*G5))/(G5**2+G6**2)**2)
        G_omega_rG = (G2*np.cos(theta))/(2*np.pi*(G5**2 + G6**2)) + (G1*np.sin(theta)/(2*np.pi*(G5**2+G6**2))) - (G6*np.cos(theta)*(G1*G5+G2*G6))/(np.pi*(G5**2+G6**2)**2) - (G6*np.sin(theta)*(G1*G6-G2*G5))/(np.pi*(G5**2+G6**2)**2)
        H_omega_rG = V_inf*((2*G6*np.cos(theta)*(G6*np.sin(alpha)-G5*np.cos(alpha)+G3*G5+G4*G6))/(G5**2+G6**2)**2-(np.sin(theta)*(G3-np.cos(alpha)))/(G5**2+G6**2)-(np.cos(theta)*(G4+np.sin(alpha)))/(G5**2+G6**2)-(2*G6*np.sin(theta)*(G6*np.cos(alpha)+G5*np.sin(alpha)-G3*G6+G4*G5))/(G5**2+G6**2)**2)
        return A_omega_rG, B_omega_rG, C_omega_rG, D_omega_rG, E_omega_rG, F_omega_rG, G_omega_rG, H_omega_rG

    def partial_coefficients_of_omega_theta_wrt_G_Chi(self, V_inf, theta, alpha, G1, G2, G3, G4, G5, G6):
        """This function retrieves the partial derivatives of omega theta with respect to G values"""
        A_omega_thetaG = 1/(2*np.pi)*((G6*np.cos(theta))/(G5**2 + G6**2) - (G5*np.sin(theta))/(G5**2 + G6**2))
        B_omega_thetaG = 1/(2*np.pi)*(-(G5*np.cos(theta))/(G5**2 + G6**2) - (G6*np.sin(theta))/(G5**2 + G6**2))
        C_omega_thetaG = V_inf*(G5*np.sin(theta)/(G5**2 + G6**2) - G6*np.cos(theta)/(G5**2 + G6**2))
        D_omega_thetaG = V_inf*(G5*np.cos(theta)/(G5**2 + G6**2) + G6*np.sin(theta)/(G5**2 + G6**2))
        E_omega_thetaG = -((G2*np.cos(theta))/(2*np.pi*(G5**2 + G6**2)) + (G1*np.sin(theta)/(2*np.pi*(G5**2+G6**2))) + (G5*np.cos(theta)*(G1*G6-G2*G5))/(np.pi*(G5**2+G6**2)**2) - (G5*np.sin(theta)*(G1*G5+G2*G6))/(np.pi*(G5**2+G6**2)**2))
        F_omega_thetaG = V_inf*((np.cos(theta)*(G4+np.sin(alpha)))/(G5**2+G6**2)+(np.sin(theta)*(G3-np.cos(alpha)))/(G5**2+G6**2)-(2*G5*np.cos(theta)*(G6*np.cos(alpha)+G5*np.sin(alpha)-G3*G6+G4*G5))/(G5**2+G6**2)**2-(2*G5*np.sin(theta)*(G6*np.sin(alpha)-G5*np.cos(alpha)+G3*G5+G4*G6))/(G5**2+G6**2)**2)
        G_omega_thetaG = (G1*np.cos(theta))/(2*np.pi*(G5**2 + G6**2)) - (G2*np.sin(theta)/(2*np.pi*(G5**2+G6**2))) - (G6*np.cos(theta)*(G1*G6-G2*G5))/(np.pi*(G5**2+G6**2)**2) + (G6*np.sin(theta)*(G1*G5+G2*G6))/(np.pi*(G5**2+G6**2)**2)
        H_omega_thetaG = V_inf*((np.sin(theta)*(G4+np.sin(alpha)))/(G5**2+G6**2)-(np.cos(theta)*(G3-np.cos(alpha)))/(G5**2+G6**2)-(2*G6*np.cos(theta)*(G6*np.cos(alpha)+G5*np.sin(alpha)-G3*G6+G4*G5))/(G5**2+G6**2)**2-(2*G6*np.sin(theta)*(G6*np.sin(alpha)-G5*np.cos(alpha)+G3*G5+G4*G6))/(G5**2+G6**2)**2)
        return A_omega_thetaG, B_omega_thetaG, C_omega_thetaG, D_omega_thetaG, E_omega_thetaG, F_omega_thetaG, G_omega_thetaG, H_omega_thetaG
    
    def partial_coefficients_of_omega_theta_wrt_G_z(self, V_inf, theta, alpha, G1, G2, G3, G4, G5, G6):
        """This function retrieves the partial derivatives of omega theta with respect to G values"""
        A_omega_thetaG = 1/(2*np.pi)*((G6*np.cos(theta))/(G5**2 + G6**2) - (G5*np.sin(theta))/(G5**2 + G6**2))
        B_omega_thetaG = 1/(2*np.pi)*(-(G5*np.cos(theta))/(G5**2 + G6**2) - (G6*np.sin(theta))/(G5**2 + G6**2))
        C_omega_thetaG = V_inf*(G5*np.sin(theta)/(G5**2 + G6**2) - G6*np.cos(theta)/(G5**2 + G6**2))
        D_omega_thetaG = V_inf*(G5*np.cos(theta)/(G5**2 + G6**2) + G6*np.sin(theta)/(G5**2 + G6**2))
        E_omega_thetaG = -((G2*np.cos(theta))/(2*np.pi*(G5**2 + G6**2)) + (G1*np.sin(theta)/(2*np.pi*(G5**2+G6**2))) + (G5*np.cos(theta)*(G1*G6-G2*G5))/(np.pi*(G5**2+G6**2)**2) - (G5*np.sin(theta)*(G1*G5+G2*G6))/(np.pi*(G5**2+G6**2)**2))
        F_omega_thetaG = V_inf*((np.cos(theta)*(G4+np.sin(alpha)))/(G5**2+G6**2)+(np.sin(theta)*(G3-np.cos(alpha)))/(G5**2+G6**2)-(2*G5*np.cos(theta)*(G6*np.cos(alpha)+G5*np.sin(alpha)-G3*G6+G4*G5))/(G5**2+G6**2)**2-(2*G5*np.sin(theta)*(G6*np.sin(alpha)-G5*np.cos(alpha)+G3*G5+G4*G6))/(G5**2+G6**2)**2)
        G_omega_thetaG = (G1*np.cos(theta))/(2*np.pi*(G5**2 + G6**2)) - (G2*np.sin(theta)/(2*np.pi*(G5**2+G6**2))) - (G6*np.cos(theta)*(G1*G6-G2*G5))/(np.pi*(G5**2+G6**2)**2) + (G6*np.sin(theta)*(G1*G5+G2*G6))/(np.pi*(G5**2+G6**2)**2)
        H_omega_thetaG = V_inf*((np.sin(theta)*(G4+np.sin(alpha)))/(G5**2+G6**2)-(np.cos(theta)*(G3-np.cos(alpha)))/(G5**2+G6**2)-(2*G6*np.cos(theta)*(G6*np.cos(alpha)+G5*np.sin(alpha)-G3*G6+G4*G5))/(G5**2+G6**2)**2-(2*G6*np.sin(theta)*(G6*np.sin(alpha)-G5*np.cos(alpha)+G3*G5+G4*G6))/(G5**2+G6**2)**2)
        return A_omega_thetaG, B_omega_thetaG, C_omega_thetaG, D_omega_thetaG, E_omega_thetaG, F_omega_thetaG, G_omega_thetaG, H_omega_thetaG
    
    def partial_coefficients_of_G_wrt_r_Chi(self, r, theta, alpha, epsilon, R, r0, theta0):
        """This function retrieves the partial derivatives of G values with respect to r"""
        AGr = -np.sin(theta)/r**2
        BGr = -np.cos(theta)/r**2
        CGr = (R**2*(r**2*np.cos(2*theta)*np.cos(alpha) + r**2*np.sin(2*theta)*np.sin(alpha))*(2*r**3*(np.cos(4*theta) - 1) - 4*r**3*np.cos(2*theta)**2))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2)**2 - (R**2*(2*r*np.cos(2*theta)*np.cos(alpha) + 2*r*np.sin(2*theta)*np.sin(alpha)))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2)
        DGr = (R**2*(r**2*np.cos(2*theta)*np.sin(alpha) - r**2*np.sin(2*theta)*np.cos(alpha))*(2*r**3*(np.cos(4*theta) - 1) - 4*r**3*np.cos(2*theta)**2))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2)**2 - (R**2*(2*r*np.cos(2*theta)*np.sin(alpha) - 2*r*np.sin(2*theta)*np.cos(alpha)))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2)
        EGr = ((R - epsilon)**2*(2*(2*r*np.cos(2*theta) + 2*r0*np.cos(theta + theta0))*(np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2) + 2*(2*r*np.sin(2*theta) + 2*r0*np.sin(theta + theta0))*(np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))*(np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2)**2 - ((R - epsilon)**2*(2*r*np.cos(2*theta) + 2*r0*np.cos(theta + theta0)))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2)
        FGr= ((2*r*np.sin(2*theta) + 2*r0*np.sin(theta + theta0))*(R - epsilon)**2)/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2) - ((R - epsilon)**2*(2*(2*r*np.cos(2*theta) + 2*r0*np.cos(theta + theta0))*(np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2) + 2*(2*r*np.sin(2*theta) + 2*r0*np.sin(theta + theta0))*(np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))*(np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2)**2
        return AGr, BGr, CGr, DGr, EGr, FGr
    
    def partial_coefficients_of_G_wrt_r_z(self, r, theta, alpha, epsilon, R, r_0, theta_0):
        """"""
        AGr = (r_0**2*np.sin(theta - 2*theta_0) - r**2*np.sin(theta) + 2*r*r_0*np.sin(theta_0))/(r**2 + r_0**2 - 2*r*r_0*np.cos(theta - theta_0))**2
        BGr = -(r**2*np.cos(theta) + r_0**2*np.cos(theta - 2*theta_0) - 2*r*r_0*np.cos(theta_0))/(r**2 + r_0**2 - 2*r*r_0*np.cos(theta - theta_0))**2
        CGr = - (R**2*(2*r_0*np.cos(theta - alpha + theta_0) - 2*r*np.cos(alpha - 2*theta)))/(((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2 + 4*(r*np.cos(theta) - r_0*np.cos(theta_0))**2*(r*np.sin(theta) - r_0*np.sin(theta_0))**2) - (R**2*(np.cos(alpha)*((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2) + 2*np.sin(alpha)*(r*np.cos(theta) - r_0*np.cos(theta_0))*(r*np.sin(theta) - r_0*np.sin(theta_0)))*(8*r*r_0**2 - 4*r_0**3*np.cos(theta - theta_0) + 4*r**3 + 4*r*r_0**2*np.cos(2*theta - 2*theta_0) - 12*r**2*r_0*np.cos(theta - theta_0)))/(((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2 + 4*(r*np.cos(theta) - r_0*np.cos(theta_0))**2*(r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2
        DGr = (R**2*(2*r_0*np.sin(theta - alpha + theta_0) + 2*r*np.sin(alpha - 2*theta)))/(((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2 + 4*(r*np.cos(theta) - r_0*np.cos(theta_0))**2*(r*np.sin(theta) - r_0*np.sin(theta_0))**2) - (R**2*(np.sin(alpha)*((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2) - 2*np.cos(alpha)*(r*np.cos(theta) - r_0*np.cos(theta_0))*(r*np.sin(theta) - r_0*np.sin(theta_0)))*(8*r*r_0**2 - 4*r_0**3*np.cos(theta - theta_0) + 4*r**3 + 4*r*r_0**2*np.cos(2*theta - 2*theta_0) - 12*r**2*r_0*np.cos(theta - theta_0)))/(((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2 + 4*(r*np.cos(theta) - r_0*np.cos(theta_0))**2*(r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2
        EGr = (2*np.cos(2*theta)*(R - epsilon)**2)/r**3
        FGr = -(2*np.sin(2*theta)*(R - epsilon)**2)/r**3
        return AGr, BGr, CGr, DGr, EGr, FGr

    def partial_coefficients_of_G_wrt_theta_Chi(self, r, theta, alpha, epsilon, R, r0, theta0):
        AGtheta = np.cos(theta)/r
        BGtheta  = -np.sin(theta)/r
        CGtheta = -(R**2*(2*r**2*np.cos(2*theta)*np.sin(alpha) - 2*r**2*np.sin(2*theta)*np.cos(alpha)))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2) - (R**2*(2*r**4*np.sin(4*theta) - 4*r**4*np.cos(2*theta)*np.sin(2*theta))*(r**2*np.cos(2*theta)*np.cos(alpha) + r**2*np.sin(2*theta)*np.sin(alpha)))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2)**2
        DGtheta = (R**2*(2*r**2*np.cos(2*theta)*np.cos(alpha) + 2*r**2*np.sin(2*theta)*np.sin(alpha)))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2) - (R**2*(2*r**4*np.sin(4*theta) - 4*r**4*np.cos(2*theta)*np.sin(2*theta))*(r**2*np.cos(2*theta)*np.sin(alpha) - r**2*np.sin(2*theta)*np.cos(alpha)))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2)**2
        EGtheta = ((R - epsilon)**2*(2*np.sin(2*theta)*r**2 + 2*r0*np.sin(theta + theta0)*r))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2) - ((2*(2*np.sin(2*theta)*r**2 + 2*r0*np.sin(theta + theta0)*r)*(np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2) - 2*(2*np.cos(2*theta)*r**2 + 2*r0*np.cos(theta + theta0)*r)*(np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))*(R - epsilon)**2*(np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2)**2
        FGtheta = ((R - epsilon)**2*(2*np.cos(2*theta)*r**2 + 2*r0*np.cos(theta + theta0)*r))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2) + ((2*(2*np.sin(2*theta)*r**2 + 2*r0*np.sin(theta + theta0)*r)*(np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2) - 2*(2*np.cos(2*theta)*r**2 + 2*r0*np.cos(theta + theta0)*r)*(np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))*(R - epsilon)**2*(np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2)**2
        return AGtheta, BGtheta, CGtheta, DGtheta, EGtheta, FGtheta
    
    def partial_coefficients_of_G_wrt_theta_z(self, r, theta, alpha, epsilon, R, r_0, theta_0):
        """"""
        AGtheta = (r*(r**2*np.cos(theta) + r_0**2*np.cos(theta - 2*theta_0) - 2*r*r_0*np.cos(theta_0)))/(r**2 + r_0**2 - 2*r*r_0*np.cos(theta - theta_0))**2
        BGtheta = (r*(r_0**2*np.sin(theta - 2*theta_0) - r**2*np.sin(theta) + 2*r*r_0*np.sin(theta_0)))/(r**2 + r_0**2 - 2*r*r_0*np.cos(theta - theta_0))**2
        CGtheta = (2*R**2*r*(r_0*np.sin(theta - alpha + theta_0) + r*np.sin(alpha - 2*theta)))/(((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2 + 4*(r*np.cos(theta) - r_0*np.cos(theta_0))**2*(r*np.sin(theta) - r_0*np.sin(theta_0))**2) - (4*R**2*r*r_0*(np.cos(alpha)*((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2) + 2*np.sin(alpha)*(r*np.cos(theta) - r_0*np.cos(theta_0))*(r*np.sin(theta) - r_0*np.sin(theta_0)))*(r**2*np.sin(theta - theta_0) + r_0**2*np.sin(theta - theta_0) - r*r_0*np.sin(2*theta - 2*theta_0)))/(((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2 + 4*(r*np.cos(theta) - r_0*np.cos(theta_0))**2*(r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2
        DGtheta = (2*R**2*r*(r_0*np.cos(theta - alpha + theta_0) - r*np.cos(alpha - 2*theta)))/(((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2 + 4*(r*np.cos(theta) - r_0*np.cos(theta_0))**2*(r*np.sin(theta) - r_0*np.sin(theta_0))**2) - (4*R**2*r*r_0*(np.sin(alpha)*((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2) - 2*np.cos(alpha)*(r*np.cos(theta) - r_0*np.cos(theta_0))*(r*np.sin(theta) - r_0*np.sin(theta_0)))*(r**2*np.sin(theta - theta_0) + r_0**2*np.sin(theta - theta_0) - r*r_0*np.sin(2*theta - 2*theta_0)))/(((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2 + 4*(r*np.cos(theta) - r_0*np.cos(theta_0))**2*(r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2
        EGtheta = (2*np.sin(2*theta)*(R - epsilon)**2)/r**2
        FGtheta = (2*np.cos(2*theta)*(R - epsilon)**2)/r**2
        return AGtheta, BGtheta, CGtheta, DGtheta, EGtheta, FGtheta
    
    def calc_A1_through_A12(self,  V_inf, theta, alpha, G1, G2, G3, G4, G5, G6, A_omega_rG, B_omega_rG, C_omega_rG, D_omega_rG, E_omega_rG, F_omega_rG, G_omega_rG, H_omega_rG, A_omega_thetaG, B_omega_thetaG, C_omega_thetaG, D_omega_thetaG, E_omega_thetaG, F_omega_thetaG, G_omega_thetaG, H_omega_thetaG, AGr, BGr, CGr, DGr, EGr, FGr, AGtheta, BGtheta, CGtheta, DGtheta, EGtheta, FGtheta):
        """This function calculates the A1 through A12 values"""
        A1 = 1/(2*np.pi)*(np.cos(theta)*(G1*G5+G2*G6)/(G5**2 + G6**2) - np.sin(theta)*(G2*G5-G1*G6)/(G5**2 + G6**2))
        A2 = V_inf*(np.cos(theta)*(G5*np.cos(alpha)-G6*np.sin(alpha)-G3*G5-G4*G6)/(G5**2 + G6**2) + np.sin(theta)*(G6*np.cos(alpha)+G5*np.sin(alpha)+G4*G5-G3*G6)/(G5**2 + G6**2))
        A3 = 1/(2*np.pi)*(-np.cos(theta)*(G2*G5-G1*G6)/(G5**2 + G6**2) - np.sin(theta)*(G1*G5+G2*G6)/(G5**2 + G6**2))
        A4 = V_inf*(np.cos(theta)*(G6*np.cos(alpha)+G5*np.sin(alpha)+G4*G5-G3*G6)/(G5**2 + G6**2) - np.sin(theta)*(G5*np.cos(alpha)-G6*np.sin(alpha)-G3*G5-G4*G6)/(G5**2 + G6**2))
        A5 = A_omega_rG*AGr + B_omega_rG*BGr + E_omega_rG*EGr + G_omega_rG*FGr
        A6 = C_omega_rG*CGr + D_omega_rG*DGr + F_omega_rG*EGr + H_omega_rG*FGr
        A7 = A_omega_rG*AGtheta + B_omega_rG*BGtheta + E_omega_rG*EGtheta + G_omega_rG*FGtheta
        A8 = C_omega_rG*CGtheta + D_omega_rG*DGtheta + F_omega_rG*EGtheta + H_omega_rG*FGtheta
        A9 = A_omega_thetaG*AGtheta + B_omega_thetaG*BGtheta + E_omega_thetaG*EGtheta + G_omega_thetaG*FGtheta
        A10 = C_omega_thetaG*CGtheta + D_omega_thetaG*DGtheta + F_omega_thetaG*EGtheta + H_omega_thetaG*FGtheta
        A11 = A_omega_thetaG*AGr + B_omega_thetaG*BGr + E_omega_thetaG*EGr + G_omega_thetaG*FGr
        A12 = C_omega_thetaG*CGr + D_omega_thetaG*DGr + F_omega_thetaG*EGr + H_omega_thetaG*FGr
        return A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12

    def chi_partial_derivatives(self, Gamma, A5, A6, A7, A8, A9, A10, A11, A12):
        """This function retrieves the values of the partial derivatives of omega r and omega theta with respect to r and theta"""
        partial_omega_r_wrt_r = Gamma*A5 + A6
        partial_omega_r_wrt_theta = Gamma*A7 + A8
        partial_omega_theta_wrt_theta = Gamma*A9 + A10
        partial_omega_theta_wrt_r = Gamma*A11 + A12
        return partial_omega_r_wrt_r, partial_omega_r_wrt_theta, partial_omega_theta_wrt_theta, partial_omega_theta_wrt_r
    
    def calc_B1_through_B6(self, r, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12):
        """This function calculates the B1 through B6 values"""
        B1 = A1*A5 + (1/r)*(A3*(A7-A3))
        B2 = A1*A6 + A2*A5 + (1/r)*(A3*A8 + A4*(A7-2*A3))
        B3 = A2*A6 + (1/r)*(A4*(A8-A4))
        B4 = A1*A11 + (1/r)*(A3*(A9+A1))
        B5 = A1*A12 + A2*A11 + (1/r)*(A3*(A10+A2) + A4*(A9+A1))
        B6 = A2*A12 + (1/r)*(A4*(A10+A2))
        return B1, B2, B3, B4, B5, B6
    
    def calc_C1_through_C5(self, B1, B2, B3, B4, B5, B6):
        """This function calculates the C1 through C5 values"""
        C1 = B1**2 + B4**2
        C2 = 2*(B1*B2 + B4*B5)
        C3 = 2*(B1*B3 + B4*B6) + B2**2 + B5**2
        C4 = 2*(B2*B3 + B5*B6)
        C5 = B3**2 + B6**2
        return C1, C2, C3, C4, C5
    
    def chi_convective_acceleration(self, point_xy_in_Chi_plane, Gamma):
        """Start with a Chi value that is shifted from zeta_center"""
        xi, eta = point_xy_in_Chi_plane[0], point_xy_in_Chi_plane[1]
        # xi, eta = point_xy_in_zeta_plane[0], point_xy_in_zeta_plane[1]
        r, theta = hlp.xy_to_r_theta(xi, eta)
        xio, etao = self.zeta_center.real, self.zeta_center.imag
        r0, theta0 = hlp.xy_to_r_theta(xio, etao)
        V_inf, R, alpha, epsilon = self.freestream_velocity, self.cylinder_radius, self.angle_of_attack, self.epsilon
        # omega_chi_unsplit = V_inf*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf*(xi+1j*eta)) - np.exp(1j*alpha)*R**2/(xi+1j*eta)**2) / (1 - (R-epsilon)**2/(xi+xio+1j*(eta+etao))**2)
        G1, G2, G3, G4, G5, G6 = self.calc_Chi_G_values(r, theta, alpha, epsilon, R, r0, theta0)
        # velocity_r = np.cos(theta)*Gamma/(2*np.pi)*(G1*G5+G2*G6)/(G5**2 + G6**2)+np.cos(theta)*V_inf*(G5*np.cos(alpha)-G6*np.sin(alpha)-G3*G5-G4*G6)/(G5**2 + G6**2) - np.sin(theta)*Gamma/(2*np.pi)*(G2*G5-G1*G6)/(G5**2 + G6**2) + np.sin(theta)*V_inf*(G6*np.cos(alpha)+G5*np.sin(alpha)+G4*G5-G3*G6)/(G5**2 + G6**2)
        # velocity_theta = np.cos(theta)*V_inf*(G6*np.cos(alpha)+G5*np.sin(alpha)+G4*G5-G3*G6)/(G5**2 + G6**2) - np.cos(theta)*Gamma/(2*np.pi)*(G2*G5-G1*G6)/(G5**2 + G6**2) - np.sin(theta)*Gamma/(2*np.pi)*(G1*G5+G2*G6)/(G5**2 + G6**2) - np.sin(theta)*V_inf*(G5*np.cos(alpha)-G6*np.sin(alpha)-G3*G5-G4*G6)/(G5**2 + G6**2)
        A_omega_rG, B_omega_rG, C_omega_rG, D_omega_rG, E_omega_rG, F_omega_rG, G_omega_rG, H_omega_rG = self.partial_coefficients_of_omega_r_wrt_G_Chi(V_inf, theta, alpha, G1, G2, G3, G4, G5, G6)
        A_omega_thetaG, B_omega_thetaG, C_omega_thetaG, D_omega_thetaG, E_omega_thetaG, F_omega_thetaG, G_omega_thetaG, H_omega_thetaG = self.partial_coefficients_of_omega_theta_wrt_G_Chi(V_inf, theta, alpha, G1, G2, G3, G4, G5, G6)
        AGr, BGr, CGr, DGr, EGr, FGr = self.partial_coefficients_of_G_wrt_r_Chi(r, theta, alpha, epsilon, R, r0, theta0)
        AGtheta, BGtheta, CGtheta, DGtheta, EGtheta, FGtheta = self.partial_coefficients_of_G_wrt_theta_Chi(r, theta, alpha, epsilon, R, r0, theta0)
        # calculate the A1 through A12 values
        A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12 = self.calc_A1_through_A12(V_inf, theta, alpha, G1, G2, G3, G4, G5, G6, A_omega_rG, B_omega_rG, C_omega_rG, D_omega_rG, E_omega_rG, F_omega_rG, G_omega_rG, H_omega_rG, A_omega_thetaG, B_omega_thetaG, C_omega_thetaG, D_omega_thetaG, E_omega_thetaG, F_omega_thetaG, G_omega_thetaG, H_omega_thetaG, AGr, BGr, CGr, DGr, EGr, FGr, AGtheta, BGtheta, CGtheta, DGtheta, EGtheta, FGtheta)
        # omega_r, omega_theta = Gamma*A1 + A2, Gamma*A3 + A4
        # partial_omega_r_wrt_r, partial_omega_r_wrt_theta, partial_omega_theta_wrt_theta, partial_omega_theta_wrt_r = self.chi_partial_derivatives(Gamma, A5, A6, A7, A8, A9, A10, A11, A12)
        B1, B2, B3, B4, B5, B6 = self.calc_B1_through_B6(r, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12)
        convective_acceleration = np.array([Gamma**2*B1 + Gamma*B2 + B3, Gamma**2*B4 + Gamma*B5 + B6])
        C1, C2, C3, C4, C5 = self.calc_C1_through_C5(B1, B2, B3, B4, B5, B6)
        return convective_acceleration
    
    def z_convective_acceleration(self, point_xy_in_z_plane, Gamma):
        """Start with a z value that is shifted from z_center"""
        z = point_xy_in_z_plane[0] + 1j*point_xy_in_z_plane[1]
        zeta = self.z_to_zeta(z, self.epsilon)
        r, theta = hlp.xy_to_r_theta(zeta.real, zeta.imag)
        xio, etao = self.zeta_center.real, self.zeta_center.imag
        r0, theta0 = hlp.xy_to_r_theta(xio, etao)
        V_inf, R, alpha, epsilon = self.freestream_velocity, self.cylinder_radius, self.angle_of_attack, self.epsilon
        G1, G2, G3, G4, G5, G6 = self.calc_z_G_values(r, theta, alpha, epsilon, R, r0, theta0)
        A_omega_rG, B_omega_rG, C_omega_rG, D_omega_rG, E_omega_rG, F_omega_rG, G_omega_rG, H_omega_rG = self.partial_coefficients_of_omega_r_wrt_G_z(V_inf, theta, alpha, G1, G2, G3, G4, G5, G6)
        A_omega_thetaG, B_omega_thetaG, C_omega_thetaG, D_omega_thetaG, E_omega_thetaG, F_omega_thetaG, G_omega_thetaG, H_omega_thetaG = self.partial_coefficients_of_omega_theta_wrt_G_z(V_inf, theta, alpha, G1, G2, G3, G4, G5, G6)
        AGr, BGr, CGr, DGr, EGr, FGr = self.partial_coefficients_of_G_wrt_r_z(r, theta, alpha, epsilon, R, r0, theta0)
        AGtheta, BGtheta, CGtheta, DGtheta, EGtheta, FGtheta = self.partial_coefficients_of_G_wrt_theta_z(r, theta, alpha, epsilon, R, r0, theta0)
        # calculate the A1 through A12 values
        A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12 = self.calc_A1_through_A12(V_inf, theta, alpha, G1, G2, G3, G4, G5, G6, A_omega_rG, B_omega_rG, C_omega_rG, D_omega_rG, E_omega_rG, F_omega_rG, G_omega_rG, H_omega_rG, A_omega_thetaG, B_omega_thetaG, C_omega_thetaG, D_omega_thetaG, E_omega_thetaG, F_omega_thetaG, G_omega_thetaG, H_omega_thetaG, AGr, BGr, CGr, DGr, EGr, FGr, AGtheta, BGtheta, CGtheta, DGtheta, EGtheta, FGtheta)
        B1, B2, B3, B4, B5, B6 = self.calc_B1_through_B6(r, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12)
        convective_acceleration = np.array([Gamma**2*B1 + Gamma*B2 + B3, Gamma**2*B4 + Gamma*B5 + B6])
        C1, C2, C3, C4, C5 = self.calc_C1_through_C5(B1, B2, B3, B4, B5, B6)
        return convective_acceleration
    
    def run_appellian_roots(self):
        """This function runs the Appellian stuff"""
        if self.is_analytic_accel and self.is_chi:
            print("ANALYTIC CONVECTIVE ACCELERATION IN CHI")
        elif self.is_analytic_accel and self.is_z:
            print("ANALYTIC CONVECTIVE ACCELERATION IN Z")
        elif self.is_analytic_accel and self.is_pressure_gradient: 
            print("ANALYTIC PRESSURE GRADIENT")
        elif not self.is_analytic_accel and self.is_z:
            print("NUMERICAL CONVECTIVE ACCELERATION IN Z")
        elif not self.is_analytic_accel and self.is_chi:
            print("NUMERICAL CONVECTIVE ACCELERATION IN CHI")
        else:
            raise ValueError("The function you are trying to use is not implemented")
        r_range = [cyl.cylinder_radius, cyl.cylinder_radius, 0.05*cyl.cylinder_radius]
        r_vals = hlp.list_to_range(r_range)
        theta_range = [0, 2*np.pi, np.pi/128]
        theta_vals = hlp.list_to_range(theta_range)
        xi_eta_array = np.zeros((len(r_vals)*len(theta_vals), 2))
        if self.is_chi:
            for i in range(len(r_vals)):
                for j in range(len(theta_vals)):
                    xi_eta_array[i*len(theta_vals) + j] = hlp.r_theta_to_xy(r_vals[i], theta_vals[j]) # this populates the xi_eta_array with the xi and eta values corresponding to the r and theta values in the chi plane
        else:
            # convert r and theta values to z values by first converting them to xi and eta values in the chi plane, transforming them to zeta values, and then transforming them to z values
            for i in range(len(r_vals)):
                for j in range(len(theta_vals)):
                    xi_eta_chi = hlp.r_theta_to_xy(r_vals[i], theta_vals[j])[0] + 1j*hlp.r_theta_to_xy(r_vals[i], theta_vals[j])[1]
                    xi_eta_zeta = self.Chi_to_zeta(xi_eta_chi)
                    xi_eta_z = self.zeta_to_z(xi_eta_zeta, self.epsilon)
                    xi_eta_array[i*len(theta_vals) + j] = np.array([xi_eta_z.real, xi_eta_z.imag])

        Gamma_start = self.kutta_circulation
        appellian_root = hlp.newtons_method(self.numerically_dbquad_integrate_appellian, Gamma_start, r_values = r_vals, theta_values = theta_vals, is_analytic_accel=self.is_analytic_accel)
        # print appellian root out to 8 decimal places
        print("Appellian Root", appellian_root)
        print("Appellian Root: ", round(appellian_root, 9))
        return appellian_root, xi_eta_array
    
    def pressure_gradient(self, point_xi_eta_in_z, Gamma):
        """Takes in a point in the zeta plane and returns the pressure gradient"""
        z = point_xi_eta_in_z[0] + 1j*point_xi_eta_in_z[1]
        xi, eta = point_xi_eta_in_z[0], point_xi_eta_in_z[1]
        zeta = self.z_to_zeta(z, self.epsilon)
        point_xi_eta_in_zeta_plane = [zeta.real, zeta.imag]
        V_inf, epsilon, R, alpha, zeta0 = self.freestream_velocity, self.epsilon, self.cylinder_radius, self.angle_of_attack, self.zeta_center      
        first = (-2*(V_inf*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf*(zeta-zeta0)) - np.exp(1j*alpha)*R**2/(zeta-zeta0)**2) / (1 - (R-epsilon)**2/(zeta)**2)))/V_inf**2
        second_d2 = ((V_inf*(-1j*Gamma/(2*np.pi*V_inf)*1/(zeta-zeta0)**2 + (2*R**2*np.exp(1j*alpha))/(zeta-zeta0)**3))-(V_inf*(np.exp(-1j*alpha)+1j*Gamma/(2*np.pi*V_inf)*1/(zeta-zeta0) - (R**2*np.exp(1j*alpha))/(zeta-zeta0)**2))*(2*(R-epsilon)**2/zeta**3)/((1-(R-epsilon)**2/zeta**2)))/((1 - 2*(R-epsilon)**2/zeta**2+(R-epsilon)**4/zeta**4))
        Dcp = first*second_d2
        pressure_gradient_complex = np.array([Dcp.real, -Dcp.imag])
        # theta = np.arctan2(point_xi_eta_in_z[1], point_xi_eta_in_z[0])
        # pressure_gradient_polar = self.polar_vector(theta, pressure_gradient_complex)
        magnitude_pressure_gradient_complex = np.linalg.norm(pressure_gradient_complex)
        return magnitude_pressure_gradient_complex

if __name__ == "__main__":
    ## initialize the cylinder object
    plt.rcParams["font.family"] = "Serif"
    plt.rcParams["font.size"] = 17.0
    plt.rcParams["axes.labelsize"] = 17.0
    plt.rcParams['lines.linewidth'] = 1.0 # 1.0
    plt.rcParams["xtick.minor.visible"] = True 
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["xtick.direction"] = plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.bottom"] = plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.left"] = plt.rcParams["ytick.right"] = True
    plt.rcParams["xtick.major.width"] = plt.rcParams["ytick.major.width"] = 0.75
    plt.rcParams["xtick.minor.width"] = plt.rcParams["ytick.minor.width"] = 0.75
    plt.rcParams["xtick.major.size"] = plt.rcParams["ytick.major.size"] = 5.0
    plt.rcParams["xtick.minor.size"] = plt.rcParams["ytick.minor.size"] = 2.5
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams['figure.dpi'] = 300.0
    ## change legend parameters
    plt.rcParams["legend.fontsize"] = 17.0
    plt.rcParams["legend.frameon"] = True
    subdict = {
        "figsize" : (3.25,3.5),
        "constrained_layout" : True,
        "sharex" : True
    }
    cyl = cylinder("Joukowski_Cylinder.json")
    print("\n")
    if cyl.is_appellian and cyl.type == "cylinder":
        appellian_roots, xi_eta_vals = cyl.run_appellian_roots()
        # appellian_roots, xi_eta_vals = cyl.run_appellian_stuff()
    plt.figure()
    cyl.get_full_geometry_zeta()
    cyl.plot_geometry_zeta()
    cyl.get_full_geometry()
    cyl.plot_geometry()
    zeta_trailing_edge_focus, zeta_leading_edge_focus, z_leading_edge_focus, z_trailing_edge_focus = cyl.get_and_plot_foci()
    # zeta_to_z_test_point = 1 + 1j
    if cyl.type == "airfoil":
        lift = cyl.calc_J_airfoil_CL()
        print("CL Joukowski: ", lift)
        Cmo = cyl.calc_J_airfoil_Cmz((0+1j*0))
        print("Cm0 Joukowski: ", Cmo)
        c4 = cyl.calc_J_airfoil_c4()
        print("C4 location Joukowski: ", c4)
        # now find the moment coefficient at c4
        Cmc4 = cyl.calc_J_airfoil_Cmz(c4)
        print("Cmc4 Joukowski: ", Cmc4)
        # now create the full geometry of the airfoil and export it to a text file
        cyl.output_J_airfoil()
        print("\n")
    if cyl.do_plot_streamlines:
        cyl.plot()
    # include y and x axes 
    plt.axhline(0, color='gray')
    plt.axvline(0, color='gray')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("$\\xi$/$R$")
    plt.ylabel("$\\eta$/$R$")
    plt.gca().xaxis.labelpad = 0.001
    plt.gca().yaxis.labelpad = -10
    plt.xlim(cyl.plot_x_lower_lim, cyl.plot_x_upper_lim)
    plt.ylim(cyl.plot_x_lower_lim, cyl.plot_x_upper_lim)
    x_tick_length = int(cyl.plot_x_upper_lim - cyl.plot_x_lower_lim)
    y_tick_length = int(cyl.plot_x_upper_lim - cyl.plot_x_lower_lim)
    x_ticks = np.linspace(cyl.plot_x_lower_lim, cyl.plot_x_upper_lim, x_tick_length + 1)[1:] # ticks everywhere except for the first element
    y_ticks = np.linspace(cyl.plot_x_lower_lim, cyl.plot_x_upper_lim, y_tick_length + 1)[1:] # ticks everywhere except for the first element
    plt.xticks(x_ticks)
    plt.yticks(y_ticks) 
    plt.text(-0.07, -0.01, str(int(cyl.plot_x_lower_lim)), transform=plt.gca().transAxes, fontsize=17, verticalalignment='top')
    epsilon_name = "epsilon_" + str(cyl.epsilon)
    zeta_0_name = "zeta_0_" + str(cyl.zeta_center)
    if cyl.plot_text:
        plt.text(0.05, 0.92, "$\\epsilon = $" + str(cyl.epsilon) + " $R$", transform=plt.gca().transAxes, fontsize=17, verticalalignment='top')
        plt.text(0.60, 0.96, "$\\xi_0 = $" + str(cyl.zeta_center.real) + " $R$", transform=plt.gca().transAxes, fontsize=17, verticalalignment='top')
        plt.text(0.60, 0.88, "$\\eta_0 = $" + str(cyl.zeta_center.imag) + " $R$", transform=plt.gca().transAxes, fontsize=17, verticalalignment='top')
    if cyl.show_fig:
        if cyl.type == "cylinder" and cyl.is_appellian and cyl.is_plot_appellian:
            plt.scatter(xi_eta_vals[:, 0], xi_eta_vals[:, 1], color='red', s=5)
            if cyl.save_fig:
                plt.savefig("Joukowski_Cylinder_" + epsilon_name + "_" +  zeta_0_name + "_" + ".svg")
        plt.show()
    print("\n")

    # test_point_in_z = [-2.5, 2.5]
    # pressure = cyl.pressure_gradient(test_point_in_z, cyl.circulation)
    # print("Pressure Gradient at ", test_point_in_z, " is ", pressure)
    # other_pressure = cyl.other_pressure_gradient(test_point_in_z, cyl.circulation)
    # print("Pressure Gradient at ", test_point_in_z, " is ", other_pressure)


    # D_range = [0, 1, 1]
    # D_values = hlp.list_to_range(D_range)
    # # make an array which contains zeros which is the length of D_values. The first column will be the D values and the second column will be the normalized gamma values
    # normalized_gamma = np.zeros((len(D_values), 2))
    # # Normalized_Gamma_wrt_D = np.zeros
    # for i in tqdm(range(len(D_values)), desc="Calculating normalized Gamma wrt D"):
    #     print("\nD VALUE", D_values[i])
    #     cyl.epsilon = cyl.calc_epsilon_from_D(D_values[i])
    #     appellian_roots, _ = cyl.run_appellian_stuff()
    #     print("appellian roots: ", appellian_roots)
    #     normalized_gamma[i, 0] = D_values[i]
    #     normalized_gamma[i, 1] = appellian_roots/cyl.kutta_circulation
    #     print("\n")
    # plt.figure()
    # plt.plot(normalized_gamma[:,0], normalized_gamma[:,1])
    # plt.xlabel("$D$")
    # plt.ylabel("$\\Gamma$/$\\Gamma_k$")
    # # plt.xlim(0, 1.1)
    # # plt.ylim(0, 1.1)
    # plt.show()
    # if cyl.save_fig:
    #     plt.savefig("Normalized_Gamma_wrt_D_at_zeta0_is" +  str(cyl.zeta_center) + ".svg")


    


    


