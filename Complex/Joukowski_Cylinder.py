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
                    self.kutta_circulation = self.calc_circulation_J_airfoil()
                    print("Kutta circulation", self.kutta_circulation)
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
        size = 15
        # plt.scatter(self.zeta_center.real, self.zeta_center.imag, color=color, marker='o', s = size, facecolors='none', label="$\\zeta_0$")
        plt.scatter(self.zeta_center.real, self.zeta_center.imag, color=color, s = size, label="$\\zeta_0$")

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
        size = 15
        size_tri = 30
        # plt.scatter(self.zeta_trailing_edge_focus, 0.0, color='black', marker='^', s = size_tri, label="sing")
        # plt.scatter(self.zeta_leading_edge_focus, 0.0, color='black', marker='^', s = size_tri)
        # plt.scatter(self.z_trailing_edge_focus, 0.0, color='black', marker='s', s = size, label = "J-np.sing")
        # plt.scatter(self.z_leading_edge_focus, 0.0, color='black', marker='s', s = size)
        return self.zeta_trailing_edge_focus, self.zeta_leading_edge_focus, self.z_trailing_edge_focus, self.z_leading_edge_focus

    def velocity(self, point_xi_eta_in_z_plane, Gamma):
        """This function calculates the velocity at a given point in the flow field in the z plane"""
        z = point_xi_eta_in_z_plane[0] + 1j*point_xi_eta_in_z_plane[1]
        zeta = self.z_to_zeta(z, self.epsilon)
        V_inf, epsilon, R, alpha, zeta0 = self.freestream_velocity, self.epsilon, self.cylinder_radius, self.angle_of_attack, self.zeta_center
        velocity = V_inf*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf*(zeta-zeta0)) - np.exp(1j*alpha)*R**2/(zeta-zeta0)**2) / (1 - (R-epsilon)**2/(zeta)**2) # eq 107
        velocity_complex = np.array([velocity.real, -velocity.imag])
        return velocity_complex
    
    def velocity_chi(self, point_xy_in_Chi_plane, Gamma):
        """Start with a Chi value that is shifted from zeta_center"""
        xi, eta = point_xy_in_Chi_plane[0], point_xy_in_Chi_plane[1]
        zeta0 = self.zeta_center
        Chi = xi + 1j*eta
        zeta = Chi + zeta0
        r, theta = hlp.xy_to_r_theta(xi, eta)
        zeta_center = self.zeta_center
        # xio, etao = zeta_center.real, zeta_center.imag
        # r0, theta0 = hlp.xy_to_r_theta(xio, etao)
        V_inf, R, alpha, epsilon = self.freestream_velocity, self.cylinder_radius, self.angle_of_attack, self.epsilon
        # G1, G2, G3, G4, G5, G6 = self.calc_Chi_G_values(r, theta, alpha, epsilon, R, r0, theta0)
        # V_real = (Gamma/(2*np.pi))*((G1*G5+G2*G6)/(G5**2 + G6**2)) + V_inf*((G5*np.cos(alpha)-G6*np.sin(alpha)-G3*G5-G4*G6)/(G5**2 + G6**2))
        # V_imag = (-1*Gamma/(2*np.pi))*((G2*G5-G1*G6)/(G5**2 + G6**2)) + V_inf*((G6*np.cos(alpha)+G5*np.sin(alpha)+G4*G5-G3*G6)/(G5**2 + G6**2))
        # velocity_complex = np.array([V_real, V_imag])
        velocity = V_inf*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf*(Chi)) - np.exp(1j*alpha)*R**2/(Chi)**2) / (1 - (R-epsilon)**2/(Chi+zeta0)**2) # eq 107
        velocity_complex = np.array([velocity.real, -velocity.imag])
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
        polar_acceleration = hlp.polar_vector(theta, acceleration_complex)
        return polar_acceleration

    def other_pressure_gradient(self, point_xi_eta_in_z_plane, Gamma):
        """This function calculates the pressure gradient at a given point in the flow field in the z plane"""
        z = point_xi_eta_in_z_plane[0] + 1j*point_xi_eta_in_z_plane[1]
        zeta = self.z_to_zeta(z, self.epsilon)
        point_xi_eta_in_zeta_plane = [zeta.real, zeta.imag]
        V_inf, epsilon, R, alpha, zeta0 = self.freestream_velocity, self.epsilon, self.cylinder_radius, self.angle_of_attack, self.zeta_center
        plus = (z + np.sqrt(z**2 - 4*(R-epsilon)**2))/2
        minus = (z - np.sqrt(z**2 - 4*(R-epsilon)**2))/2
        first = -2*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf)*1/(zeta-zeta0)-(R**2*np.exp(1j*alpha))/(zeta-zeta0)**2)
        second = -1j*Gamma/(2*np.pi*V_inf)*1/(zeta-zeta0)**2 + 2*R**2*np.exp(1j*alpha)/(zeta-zeta0)**3
        third = np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf)*1/(zeta-zeta0) - (R**2*np.exp(1j*alpha))/(zeta-zeta0)**2
        fourth = (1 - ((R - epsilon)**2)/(zeta**2))
        if zeta == plus:
            fifth = (1+z)/(4*(2*epsilon-2*R+z)*(2*R-2*epsilon+z))
            sixth = (-2*(R - epsilon)**2)/(z**2 - 4*(R - epsilon)**2)**(3/2)
        if zeta == minus:
            fifth = (1-z)/(4*(2*epsilon-2*R+z)*(2*R-2*epsilon+z))
            sixth = (2*(R - epsilon)**2)/((z**2 - 4*(R - epsilon)**2)**(3/2))
        pressure_gradient = first*(second*fifth + third*sixth) / fourth
        pressure_gradient_complex = np.array([pressure_gradient.real, -pressure_gradient.imag])
        theta = np.arctan2(point_xi_eta_in_z_plane[1], point_xi_eta_in_z_plane[0])
        pressure_gradient_polar = hlp.polar_vector(theta, pressure_gradient_complex)
        magnitude_pressure_gradient_complex = np.linalg.norm(pressure_gradient_complex)
        return pressure_gradient_complex

    
    # def numerical_cartesian_convective_acceleration(self, point_xi_eta, Gamma, step=1e-10):
    #     """calculates the convective acceleration numerically in the z or chi plane"""
    #     if self.is_z:
    #         z = point_xi_eta[0] + 1j*point_xi_eta[1]
    #         zeta = self.z_to_zeta(z, self.epsilon)
    #         point_z = [z.real, z.imag]
    #         r, theta = hlp.xy_to_r_theta(zeta.real, zeta.imag)
    #         point_r_theta = [hlp.xy_to_r_theta(z.real, z.imag)[0], hlp.xy_to_r_theta(z.real, z.imag)[1]]
    #         # r, theta = [point_r_theta[0], point_r_theta[1]]
    #         cartesian_z_velocity = self.velocity(point_z, Gamma)
    #         polar_z_velocity = hlp.polar_vector(theta, cartesian_z_velocity)
    #         omega_r, omega_theta = polar_z_velocity[0], polar_z_velocity[1]
    #         omega_r_plus_dr, omega_r_minus_dr, omega_r_plus_dtheta, omega_r_minus_dtheta, omega_theta_plus_dr, omega_theta_minus_dr, omega_theta_plus_dtheta, omega_theta_minus_dtheta = self.function_plus_minus_step_variable(point_r_theta, Gamma, step, self.velocity)
    #     elif self.is_chi:
    #         r, theta = hlp.xy_to_r_theta(point_xi_eta[0], point_xi_eta[1])
    #         point_r_theta = [r, theta]
    #         cartesian_Chi_velocity = self.velocity_chi(point_xi_eta, Gamma)
    #         polar_Chi_velocity = hlp.polar_vector(theta, cartesian_Chi_velocity)
    #         omega_r, omega_theta = polar_Chi_velocity[0], polar_Chi_velocity[1]
    #         omega_r_plus_dr, omega_r_minus_dr, omega_r_plus_dtheta, omega_r_minus_dtheta, omega_theta_plus_dr, omega_theta_minus_dr, omega_theta_plus_dtheta, omega_theta_minus_dtheta = self.function_plus_minus_step_variable(point_r_theta, Gamma, step, self.velocity_chi)

    #     # calculate the partial derivatives of omega r with respect to r and theta
    #     partial_omega_r_wrt_r = hlp.central_difference(omega_r_plus_dr, omega_r_minus_dr, step)
    #     partial_omega_r_wrt_theta = hlp.central_difference(omega_r_plus_dtheta, omega_r_minus_dtheta, step)
    #     # calculate the partial derivatives of omega theta with respect to r and theta
    #     partial_omega_theta_wrt_r = hlp.central_difference(omega_theta_plus_dr, omega_theta_minus_dr, step)
    #     partial_omega_theta_wrt_theta = hlp.central_difference(omega_theta_plus_dtheta, omega_theta_minus_dtheta, step)
    #     # calculate the convective acceleration
    #     convective_acceleration_r = omega_r*partial_omega_r_wrt_r + (omega_theta/r)*partial_omega_r_wrt_theta - omega_theta**2/r
    #     convective_acceleration_theta = omega_r*partial_omega_theta_wrt_r + (omega_theta/r)*partial_omega_theta_wrt_theta + omega_r*omega_theta/r
    #     convective_acceleration = np.array([convective_acceleration_r, convective_acceleration_theta])
    #     # convective_acceleration = np.linalg.norm(convective_acceleration)
    #     return convective_acceleration
    
    # def function_plus_minus_step_variable(self, point_r_theta, Gamma, stepsize, vel_func: callable):
    #     """takes in a function and returns the function plus and minus a stepsize"""
    #     r, theta = point_r_theta[0], point_r_theta[1]
    #     r_plus, r_minus = r + stepsize, r - stepsize
    #     theta_plus, theta_minus = theta + stepsize, theta - stepsize

    #     x_r_plus, x_r_minus, y_r_plus, y_r_minus = hlp.r_theta_to_xy(r_plus, theta)[0], hlp.r_theta_to_xy(r_minus, theta)[0], hlp.r_theta_to_xy(r_plus, theta)[1], hlp.r_theta_to_xy(r_minus, theta)[1]
    #     x_theta_plus, x_theta_minus, y_theta_plus, y_theta_minus = hlp.r_theta_to_xy(r, theta_plus)[0], hlp.r_theta_to_xy(r, theta_minus)[0], hlp.r_theta_to_xy(r, theta_plus)[1], hlp.r_theta_to_xy(r, theta_minus)[1]
    #     omega_xy_plus_r, omega_xy_minus_r = vel_func([x_r_plus, y_r_plus], Gamma), vel_func([x_r_minus, y_r_minus], Gamma)
    #     omega_xy_plus_theta, omega_xy_minus_theta = vel_func([x_theta_plus, y_theta_plus], Gamma), vel_func([x_theta_minus, y_theta_minus], Gamma)
    #     omega_r_plus_dr, omega_r_minus_dr = hlp.polar_vector(theta, omega_xy_plus_r)[0], hlp.polar_vector(theta, omega_xy_minus_r)[0]
    #     omega_r_plus_dtheta, omega_r_minus_dtheta = hlp.polar_vector(theta, omega_xy_plus_theta)[0], hlp.polar_vector(theta, omega_xy_minus_theta)[0]
    #     omega_theta_plus_dr, omega_theta_minus_dr = hlp.polar_vector(theta, omega_xy_plus_r)[1], hlp.polar_vector(theta, omega_xy_minus_r)[1]
    #     omega_theta_plus_dtheta, omega_theta_minus_dtheta = hlp.polar_vector(theta, omega_xy_plus_theta)[1], hlp.polar_vector(theta, omega_xy_minus_theta)[1]
    #     return omega_r_plus_dr, omega_r_minus_dr, omega_r_plus_dtheta, omega_r_minus_dtheta, omega_theta_plus_dr, omega_theta_minus_dr, omega_theta_plus_dtheta, omega_theta_minus_dtheta

    def numerical_cartesian_convective_acceleration(self, point_xi_eta, Gamma, step=1e-8):
        """Calculates the convective acceleration numerically in the Cartesian (x, y) plane."""
        if self.is_z:
            point_xy = [point_xi_eta[0], point_xi_eta[1]]
            velocity_func = self.velocity
        else:
            point_xy = point_xi_eta
            velocity_func = self.velocity_chi
        # Compute velocity components (u, v) at the given point
        omega_xi, omega_eta = velocity_func(point_xy, Gamma)
        # Compute finite difference derivatives for du/dx, du/dy, dv/dx, dv/dy
        derivs = self.function_plus_minus_step_variable(point_xy, Gamma, step, velocity_func)
        # Compute partial derivatives using central difference
        partials = [hlp.central_difference(derivs[i], derivs[i+1], step) for i in range(0, 8, 2)]  
        # [du/dx, du/dy, dv/dx, dv/dy]
        # Ensure the list has exactly 4 elements before accessing
        if len(partials) != 4:
            raise ValueError(f"Expected 4 partial derivatives, but got {len(partials)}. Check function_plus_minus_step_variable.")

        # Compute convective acceleration
        convective_acceleration = np.array([
            omega_xi * partials[0] + omega_eta * partials[1],  # a_xi = u du/dx + v du/dy
            omega_xi * partials[2] + omega_eta * partials[3]   # a_eta = u dv/dx + v dv/dy
        ])
        return convective_acceleration

    def function_plus_minus_step_variable(self, point_xy, Gamma, step, vel_func):
        """Computes velocity perturbations for numerical differentiation in Cartesian coordinates."""
        xi, eta = point_xy
        # Perturbations in x and y directions
        perturbations = [(xi + step, eta), (xi - step, eta),  (xi, eta + step), (xi, eta - step)]
        # Compute velocity components at perturbed locations
        velocities = [vel_func(p, Gamma) for p in perturbations]  # [(u,v) at each perturbed point]
        # Extract components separately
        omega_xi_plus_dxi, omega_eta_plus_dxi = velocities[0]
        omega_xi_minus_dxi, omega_eta_minus_dxi = velocities[1]
        omega_xi_plus_d_eta, omega_eta_plus_d_eta = velocities[2]
        omega_xi_minus_d_eta, omega_eta_minus_d_eta = velocities[3]
        return (omega_xi_plus_dxi, omega_xi_minus_dxi, omega_xi_plus_d_eta, omega_xi_minus_d_eta,omega_eta_plus_dxi, omega_eta_minus_dxi, omega_eta_plus_d_eta, omega_eta_minus_d_eta)  # For du/dx, du/dy, dv/dx, dv/dy

    def appellian_acceleration_loop(self, Gamma: float, r_values: np.array, theta_values: np.array, dr: float, dtheta: float, is_Chi: bool, acceleration: callable):
        """"""
        Gamma_values = np.array([Gamma])
        # index = 0
        Appellian_value = 0.0
        Appellian_array = np.zeros((len(Gamma_values), 2))
        xi_eta_values = np.zeros((len(r_values)*len(theta_values), 2))
        if is_Chi:
            for j in range(len(r_values)):
                for k in range(len(theta_values)):
                    area_element = r_values[j]*dr*dtheta
                    accel = np.real(acceleration([r_values[j], theta_values[k]], Gamma)) 
                    Appellian_value += accel*area_element
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
        if is_analytic_accel and self.is_z:
            # print("ANALYTIC CONVECTIVE ACCELERATION IN Z")
            raise ValueError("The function you are trying to use is not implemented")
            Appellian_array = self.appellian_acceleration_loop(Gamma, r_values, theta_values, dr, dtheta, False, self.z_convective_acceleration)
        elif is_analytic_accel and self.is_pressure_gradient and self.is_chi: 
            # print("ANALYTIC PRESSURE GRADIENT")
            Appellian_array = self.appellian_acceleration_loop(Gamma, r_values, theta_values, dr, dtheta, True, self.conv_accel_using_comp_conj)
        elif not is_analytic_accel and self.is_z:
            # print("NUMERICAL CONVECTIVE ACCELERATION IN Z")
            Appellian_array = self.appellian_acceleration_loop(Gamma, r_values, theta_values, dr, dtheta, False, self.numerical_cartesian_convective_acceleration)
        elif not is_analytic_accel and self.is_chi:
            # print("NUMERICAL CONVECTIVE ACCELERATION IN CHI")
            Appellian_array = self.appellian_acceleration_loop(Gamma, r_values, theta_values, dr_original, dtheta_original, True, self.numerical_cartesian_convective_acceleration)
        else:
            raise ValueError("The function you are trying to use is not implemented")
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
    
    def z_to_zeta_incorrect(self, z: complex, epsilon: float):
        """This function returns the opposite root that z_to_zeta returns"""
        zeta_correct = self.z_to_zeta(z, epsilon)
        z_1 = z**2 - 4*(self.cylinder_radius - epsilon)**2
        zeta_option_1 = (z + np.sqrt(z_1))/2
        zeta_option_2 = (z - np.sqrt(z_1))/2
        if zeta_correct == zeta_option_1:
            zeta = zeta_option_2
        elif zeta_correct == zeta_option_2:
            zeta = zeta_option_1
        else:
            zeta = zeta_correct
            print("z", z)
            print("zeta_correct", zeta_correct)
            print("zeta_option_1", zeta_option_1)
            print("zeta_option_2", zeta_option_2)
            raise ValueError("The zeta coordinate does not match either of the options")
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

    

    
    def compare_correct_and_incorrect_z_selection(self):
        """This function plots the zeta surface points using the correct and incorrect z selection methods"""
        # first plot the correct zeta surface points
        upper,lower,_= self.get_full_geometry()
        upper_complex = upper[:,0] + 1j*upper[:,1] # the type of upper_complex is an array of complex numbers
        lower_complex = lower[:,0] + 1j*lower[:,1] # the type of lower_complex is an array of complex numbers

        upper_complex_zeta = np.zeros(len(upper_complex), dtype=complex)
        lower_complex_zeta = np.zeros(len(lower_complex), dtype=complex)
        upper_complex_zeta_incorrect = np.zeros(len(upper_complex), dtype=complex)
        lower_complex_zeta_incorrect = np.zeros(len(lower_complex), dtype=complex)
        for i in range(len(upper_complex)):
            upper_complex_zeta[i] = self.z_to_zeta(upper_complex[i], self.epsilon)
        for j in range(len(lower_complex)):
            lower_complex_zeta[j] = self.z_to_zeta(lower_complex[j], self.epsilon)
        for k in range(len(upper_complex)):
            upper_complex_zeta_incorrect[k] = self.z_to_zeta_incorrect(upper_complex[k], self.epsilon)
        for l in range(len(lower_complex)):
            lower_complex_zeta_incorrect[l] = self.z_to_zeta_incorrect(lower_complex[l], self.epsilon)

        # now create arrays that hold the real and imaginary parts as x and y components for the complex zeta coordinates for the correct and incorrect methods
        upper_zeta = np.zeros((len(upper_complex_zeta), 2))
        lower_zeta = np.zeros((len(lower_complex_zeta), 2))
        upper_zeta_incorrect = np.zeros((len(upper_complex_zeta_incorrect), 2))
        lower_zeta_incorrect = np.zeros((len(lower_complex_zeta_incorrect), 2))
        for i in range(len(upper_complex_zeta)):
            upper_zeta[i] = [upper_complex_zeta[i].real, upper_complex_zeta[i].imag]
        for j in range(len(lower_complex_zeta)):
            lower_zeta[j] = [lower_complex_zeta[j].real, lower_complex_zeta[j].imag]
        for k in range(len(upper_complex_zeta_incorrect)):
            upper_zeta_incorrect[k] = [upper_complex_zeta_incorrect[k].real, upper_complex_zeta_incorrect[k].imag]
        for l in range(len(lower_complex_zeta_incorrect)):
            lower_zeta_incorrect[l] = [lower_complex_zeta_incorrect[l].real, lower_complex_zeta_incorrect[l].imag]      
        # now plot z, zeta, and zeta incorrect on the same plot
        plt.figure()
        plt.plot(upper[:,0], upper[:,1], label="$z$", color="black")
        plt.plot(lower[:,0], lower[:,1], color="black")
        plt.plot(upper_zeta[:,0], upper_zeta[:,1],linestyle="--", label="$\\zeta$", color="black")
        plt.plot(lower_zeta[:,0], lower_zeta[:,1],linestyle="--", color="black")
        plt.plot(upper_zeta_incorrect[:,0], upper_zeta_incorrect[:,1], label="$\\zeta$ incorrect", linestyle="dotted", color="black")
        plt.plot(lower_zeta_incorrect[:,0], lower_zeta_incorrect[:,1], linestyle="dotted", color="black")
        # plt.axhline(0, color='gray')
        # plt.axvline(0, color='gray')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("$\\xi$/$R$")
        plt.ylabel("$\\eta$/$R$")
        plt.gca().xaxis.labelpad = 0.001
        plt.gca().yaxis.labelpad = -10
        plt.xlim(self.plot_x_lower_lim, self.plot_x_upper_lim)
        plt.ylim(self.plot_x_lower_lim, self.plot_x_upper_lim)
        x_tick_length = int(self.plot_x_upper_lim - self.plot_x_lower_lim)
        y_tick_length = int(self.plot_x_upper_lim - self.plot_x_lower_lim)
        x_ticks = np.linspace(self.plot_x_lower_lim, self.plot_x_upper_lim, x_tick_length + 1)[1:] # ticks everywhere except for the first element
        y_ticks = np.linspace(self.plot_x_lower_lim, self.plot_x_upper_lim, y_tick_length + 1)[1:] # ticks everywhere except for the first element
        plt.xticks(x_ticks)
        plt.yticks(y_ticks) 
        plt.text(-0.07, -0.01, str(int(self.plot_x_lower_lim)), transform=plt.gca().transAxes, fontsize=17, verticalalignment='top')
        if self.plot_text:
            plt.text(0.05, 0.92, "$\\epsilon = $" + str(self.epsilon) + " $R$", transform=plt.gca().transAxes, fontsize=17, verticalalignment='top')
            plt.text(0.60, 0.96, "$\\xi_0 = $" + str(self.zeta_center.real) + " $R$", transform=plt.gca().transAxes, fontsize=17, verticalalignment='top')
            plt.text(0.60, 0.88, "$\\eta_0 = $" + str(self.zeta_center.imag) + " $R$", transform=plt.gca().transAxes, fontsize=17, verticalalignment='top')
        # plt.legend(fontsize=12.0, loc="lower right")
        plt.show()

    def run_appellian_roots(self):
        """This function runs the Appellian stuff"""
        if self.is_analytic_accel and self.is_chi:
            print("ANALYTIC PRESSURE GRADIENT")
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
        r_end = self.cylinder_radius
        r_step = 0.1*self.cylinder_radius
        if r_end != self.cylinder_radius:
            print("r stepsize: ", r_step)
        r_range = [self.cylinder_radius, r_end, r_step]
        r_vals = hlp.list_to_range(r_range)
        theta_step = np.pi/1000
        print("Theta Stepsize: ", theta_step)
        theta_range = [0, 2*np.pi, theta_step]
        theta_vals = hlp.list_to_range(theta_range)
        xi_eta_array = np.zeros((len(r_vals)*len(theta_vals), 2))
        print("Number of points to be integrated: ", len(r_vals)*len(theta_vals))
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
        appellian_root = hlp.newtons_method(self.numerically_integrate_appellian, Gamma_start, r_values = r_vals, theta_values = theta_vals, is_analytic_accel=self.is_analytic_accel)
        # print appellian root out to 8 decimal places
        print("Appellian Root", appellian_root)
        # print("Appellian Root: ", round(appellian_root, 9))
        if self.D == 0.0:
            # print the percent difference between the appellian root and the kutta circulation
            percent_difference = 100*(appellian_root - self.kutta_circulation)/self.kutta_circulation
            print("Percent Difference Appellian to Kutta: ", percent_difference)
        return appellian_root, xi_eta_array
    

    
    def reset_plot_stuff(self):
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("$\\xi$/$R$")
        plt.ylabel("$\\eta$/$R$")
        # plt.gca().xaxis.labelpad = 0.001
        # plt.gca().yaxis.labelpad = -10
        plt.xlim(self.plot_x_lower_lim, self.plot_x_upper_lim)
        plt.ylim(self.plot_x_lower_lim, self.plot_x_upper_lim)
        x_tick_length = int(self.plot_x_upper_lim - self.plot_x_lower_lim)
        y_tick_length = int(self.plot_x_upper_lim - self.plot_x_lower_lim)
        # x_ticks = np.linspace(self.plot_x_lower_lim, self.plot_x_upper_lim, x_tick_length + 1)[1:] # ticks everywhere except for the first element
        # y_ticks = np.linspace(self.plot_x_lower_lim, self.plot_x_upper_lim, y_tick_length + 1)[1:] # ticks everywhere except for the first element
        # plt.xticks(x_ticks)
        # plt.yticks(y_ticks) 
        # plt.text(-0.07, -0.01, str(int(self.plot_x_lower_lim)), transform=plt.gca().transAxes, fontsize=17, verticalalignment='top')
        # remove all ticks
        plt.xticks([])
        plt.yticks([])

    def grid_plots(self):
        # Reset the plot settings
        cyl.reset_plot_stuff()

        # Parameters for the polar grid
        R = cyl.cylinder_radius  # Cylinder radius
        zeta_center = cyl.zeta_center  # Center of the cylinder in the zeta plane
        theta_increment = np.pi / 32  # Spacing between radial lines
        r_increment = 0.1 * R  # Increment for radial spacing
        max_radius = 6.0 * R  # Maximum radius for the grid

        # Generate theta and r values
        theta_vals = np.arange(0, 2 * np.pi, theta_increment)  # Angles for radial lines
        r_vals = np.arange(R, max_radius + r_increment, r_increment)  # Radii for circular lines

        # Arrays to store the lines
        radial_lines = []
        circular_lines = []

        # Create radial lines in the zeta plane
        for theta in theta_vals:
            line = np.array([[zeta_center.real + r * np.cos(theta), zeta_center.imag + r * np.sin(theta)] for r in np.linspace(r_vals[0], r_vals[-1], 1000)])  # Offset by zeta_center
            radial_lines.append(line)  # Save the line
            plt.plot(line[:, 0], line[:, 1], color='black')  # Plot the radial line

        # Create circular lines in the zeta plane
        for r in r_vals:
            line = np.array([[zeta_center.real + r * np.cos(theta), zeta_center.imag + r * np.sin(theta)] for theta in np.linspace(0, 2 * np.pi, 1000)])  # Offset by zeta_center
            circular_lines.append(line)  # Save the line
            plt.plot(line[:, 0], line[:, 1], color='black')  # Plot the circular line

        # Plot the geometry of the circular cylinder in the zeta plane
        plt.plot(cyl.upper_zeta_coords[:, 0], cyl.upper_zeta_coords[:, 1], color='black')
        plt.plot(cyl.lower_zeta_coords[:, 0], cyl.lower_zeta_coords[:, 1], color='black')

        # plt.show()

        # Transform the grid to the z-plane
        plt.figure()
        cyl.reset_plot_stuff()

        # Arrays to store the transformed lines
        transformed_radial_lines = []
        transformed_circular_lines = []

        # Transform radial lines to the z-plane
        for line in radial_lines:
            transformed_line = []
            for point in line:
                zeta = point[0] + 1j * point[1]  # Convert to complex zeta
                z = cyl.zeta_to_z(zeta, cyl.epsilon)  # Transform to z-plane
                transformed_line.append([z.real, z.imag])  # Store the transformed point
            transformed_line = np.array(transformed_line)  # Convert to numpy array
            transformed_radial_lines.append(transformed_line)  # Save the transformed line
            plt.plot(transformed_line[:, 0], transformed_line[:, 1], color='black')  # Plot the transformed radial line

        # Transform circular lines to the z-plane
        for line in circular_lines:
            transformed_line = []
            for point in line:
                zeta = point[0] + 1j * point[1]  # Convert to complex zeta
                z = cyl.zeta_to_z(zeta, cyl.epsilon)  # Transform to z-plane
                transformed_line.append([z.real, z.imag])  # Store the transformed point
            transformed_line = np.array(transformed_line)  # Convert to numpy array
            transformed_circular_lines.append(transformed_line)  # Save the transformed line
            plt.plot(transformed_line[:, 0], transformed_line[:, 1], color='black')  # Plot the transformed circular line

        # Plot the geometry of the shape in the z-plane
        plt.plot(cyl.upper_coords[:, 0], cyl.upper_coords[:, 1], color='black')
        plt.plot(cyl.lower_coords[:, 0], cyl.lower_coords[:, 1], color='black')

        plt.show()

    def conv_accel_using_comp_conj(self,point_r_theta_in_Chi,Gamma):
        V_inf, epsilon, R, alpha, zeta_0 = self.freestream_velocity, self.epsilon, self.cylinder_radius, self.angle_of_attack, self.zeta_center
        xi_0, eta_0 = zeta_0.real, zeta_0.imag
        r_0, theta_0 = hlp.xy_to_r_theta(xi_0, eta_0)
        r, theta = point_r_theta_in_Chi[0], point_r_theta_in_Chi[1]
        term1 = np.exp(-1j * alpha) + 1j * (Gamma / (2 * np.pi * V_inf)) * (1 / (r * np.exp(1j * theta))) - (R**2 * np.exp(1j * alpha)) / ((r * np.exp(1j * theta))**2)
        # print("term1", term1)
        term2 = (-1j * (Gamma / (2 * np.pi * V_inf)) * (1 / ((r * np.exp(1j * theta))**2))) + (2 * R**2 * np.exp(1j * alpha) / ((r * np.exp(1j * theta))**3))
        # print("term2", term2)
        term3 = 2 * (R - epsilon)**2 / (((r * np.exp(1j * theta)) + (r_0 * np.exp(1j * theta_0)))**3)
        # print("term3", term3)
        term4 = (1 - ((R - epsilon)**2 / ((r * np.exp(1j * theta)) + (r_0 * np.exp(1j * theta_0)))**2))
        # # print("term4", term4)
        # term1_conj = np.exp(1j * alpha) - 1j * (Gamma / (2 * np.pi * V_inf)) * (1 / (r * np.exp(-1j * theta))) - (R**2 * np.exp(-1j * alpha)) / ((r * np.exp(-1j * theta))**2)
        # term2_conj = (1j * (Gamma / (2 * np.pi * V_inf)) * (1 / ((r * np.exp(-1j * theta))**2))) + (2 * R**2 * np.exp(-1j * alpha) / ((r * np.exp(-1j * theta))**3))
        # term3_conj = 2 * (R - epsilon)**2 / (((r * np.exp(-1j * theta)) + (r_0 * np.exp(-1j * theta_0)))**3)
        # term4_conj = (1 - ((R - epsilon)**2 / ((r * np.exp(-1j * theta)) + (r_0 * np.exp(-1j * theta_0)))**2))
        conv_accel = V_inf**2 * term1 / term4 * (term2*term4 - (term1 * term3)) / term4**3
        # conv_accel_comp_conj = V_inf**2 * term1_conj / term4_conj * (term2_conj*term4_conj - (term1_conj * term3_conj)) / term4_conj**3
        conv_accel_comp_conj = np.conj(conv_accel)
        conv_accel_squared = conv_accel * conv_accel_comp_conj
        return conv_accel_squared
    
    def pressure_gradient(self, point_xi_eta_in_z, Gamma):
        """Takes in a point in the zeta plane and returns the pressure gradient"""
        z = point_xi_eta_in_z[0] + 1j*point_xi_eta_in_z[1]
        xi, eta = point_xi_eta_in_z[0], point_xi_eta_in_z[1]
        zeta = self.z_to_zeta(z, self.epsilon)
        point_xi_eta_in_zeta_plane = [zeta.real, zeta.imag]
        V_inf, epsilon, R, alpha, zeta0 = self.freestream_velocity, self.epsilon, self.cylinder_radius, self.angle_of_attack, self.zeta_center      
        first = (-2*(V_inf*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf*(zeta-zeta0)) - np.exp(1j*alpha)*R**2/(zeta-zeta0)**2) / (1 - (R-epsilon)**2/(zeta)**2)))/V_inf**2
        # second_1 = -1j*Gamma/(2*np.pi*V_inf)*1/(zeta-zeta0)**2 + (2*R**2*np.exp(1j*alpha))/(zeta-zeta0)**3
        # second_2 = np.exp(-1j*alpha)+1j*Gamma/(2*np.pi*V_inf)*1/(zeta-zeta0) - (R**2*np.exp(1j*alpha))/(zeta-zeta0)**2
        # second_3 = (2*(R-epsilon)**2/zeta**3)/(1-(R-epsilon)**2/zeta**2)
        # second_4 = (1-(R-epsilon)**2/zeta**2)**2
        # second_d2 = V_inf*(second_1-second_2*second_3)/second_4
        second_d2 = V_inf*((-1j*Gamma/(2*np.pi*V_inf)*1/(zeta-zeta0)**2 + (2*R**2*np.exp(1j*alpha))/(zeta-zeta0)**3)-(np.exp(-1j*alpha)+1j*Gamma/(2*np.pi*V_inf)*1/(zeta-zeta0) - (R**2*np.exp(1j*alpha))/(zeta-zeta0)**2)*((2*(R-epsilon)**2/zeta**3)/(1-(R-epsilon)**2/zeta**2)))/((1-(R-epsilon)**2/zeta**2)**2)
        Dcp = -first*second_d2
        pressure_gradient_complex = np.array([Dcp.real, -Dcp.imag])
        # pressure_gradient_polar = hlp.polar_vector(np.arctan2(point_xi_eta_in_z[1], point_xi_eta_in_z[0]), pressure_gradient_complex)
        # magnitude_pressure_gradient_complex = np.linalg.norm(pressure_gradient_polar)
        # theta = np.arctan2(point_xi_eta_in_z[1], point_xi_eta_in_z[0])
        # pressure_gradient_polar = hlp.polar_vector(theta, pressure_gradient_complex)
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
    # plt.axhline(0, color='gray')
    # plt.axvline(0, color='gray')
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

    # cyl.compare_correct_and_incorrect_z_selection()
    # cyl.grid_plots()

    test_point_in_z = [-25.5,25.5]
    test_point_in_zeta = cyl.z_to_zeta(test_point_in_z[0] + 1j*test_point_in_z[1], cyl.epsilon)
    test_point_in_Chi = cyl.zeta_to_Chi(test_point_in_zeta)
    r_chi, theta_chi = hlp.xy_to_r_theta(test_point_in_Chi.real, test_point_in_Chi.imag)
    r_0, theta_0 = hlp.xy_to_r_theta(cyl.zeta_center.real, cyl.zeta_center.imag)
    test_point_in_Chi = [test_point_in_Chi.real, test_point_in_Chi.imag]

    # conv_accel, conv_accel_comp_conj = conv_accel_using_comp_conj(cyl.circulation, r_chi, cyl.cylinder_radius, theta_chi, r_0, theta_0, cyl.freestream_velocity, cyl.epsilon, cyl.angle_of_attack)
    # conv_accel_squared = conv_accel*conv_accel_comp_conj
    # print("Analytic Convective Acceleration squared at Chi = ", test_point_in_Chi, " is ", conv_accel_squared)
    # # cyl.is_z = False
    # # cyl.is_chi = True
    # # print("test point in chi:                                ", test_point_in_Chi)
    # numerical_cartesian_convective_acceleration_in_chi = cyl.numerical_cartesian_convective_acceleration(test_point_in_Chi, cyl.circulation)
    # # print("numerical_cartesian_convective_acceleration_in_chi: ", numerical_cartesian_convective_acceleration_in_chi)
    # numerical_cartesian_convective_acceleration_in_chi_squared = np.dot(numerical_cartesian_convective_acceleration_in_chi, numerical_cartesian_convective_acceleration_in_chi)
    # # print("Numerical Convective Acceleration squared at Chi =", test_point_in_Chi, " is  ", numerical_cartesian_convective_acceleration_in_chi_squared)
    # # cyl.is_chi = False
    # # cyl.is_z = True
    # numerical_cartesian_convective_acceleration_in_z = cyl.numerical_cartesian_convective_acceleration(test_point_in_z, cyl.circulation, step=1e-6)
    # # print("numerical_cartesian_convective_acceleration_in_z: ", numerical_cartesian_convective_acceleration_in_z)
    # numerical_cartesian_convective_acceleration_in_z_squared = np.dot(numerical_cartesian_convective_acceleration_in_z, numerical_cartesian_convective_acceleration_in_z)
    # print("Numerical Convective Acceleration squared at Z =            ", test_point_in_z, " is                     ", numerical_cartesian_convective_acceleration_in_z_squared)
    # print("\n")

    # convective_acceleration_squared_comp_conj = conv_accel_using_comp_conj_squared(cyl.circulation, r_chi, cyl.cylinder_radius, theta_chi, r_0, theta_0, cyl.freestream_velocity, cyl.epsilon, cyl.angle_of_attack)
    # print("Analytic Convective Acceleration squared at Chi = ", test_point_in_Chi, " is ", convective_acceleration_squared_comp_conj)


    # step = 1e-6
    # Chi = test_point_in_Chi[0] + 1j*test_point_in_Chi[1]
    # Chi_plus = Chi + step
    # Chi_minus = Chi - step
    # test_point_in_Chi_plus = [Chi_plus.real, Chi_plus.imag]
    # test_point_in_Chi_minus = [Chi_minus.real, Chi_minus.imag]
    # chi_velocity = cyl.velocity_chi(test_point_in_Chi, cyl.circulation)[0] + 1j*cyl.velocity_chi(test_point_in_Chi, cyl.circulation)[1]
    # chi_velocity_plus = cyl.velocity_chi(test_point_in_Chi_plus, cyl.circulation)[0] + 1j*cyl.velocity_chi(test_point_in_Chi_plus, cyl.circulation)[1]
    # chi_velocity_minus = cyl.velocity_chi(test_point_in_Chi_minus, cyl.circulation)[0] + 1j*cyl.velocity_chi(test_point_in_Chi_minus, cyl.circulation)[1]
    # chi_acceleration = (chi_velocity_plus - chi_velocity_minus)/(2*step)
    # # print("Chi Acceleration:                              ", chi_acceleration)
    # chi_acceleration_for_reals = chi_velocity*chi_acceleration
    # chi_acceleration_squared = chi_acceleration_for_reals*np.conj(chi_acceleration_for_reals)
    # # print("Chi Acceleration squared:                                                                       ", chi_acceleration_squared)
    # print("\n")

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


    


    


