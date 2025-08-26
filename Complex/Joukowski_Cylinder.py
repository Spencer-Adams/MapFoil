import json
import numpy as np # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import scipy.integrate as sp_int # type: ignore
from scipy.integrate import dblquad # type: ignore
from scipy.integrate import quad # type: ignore
import time
from matplotlib.ticker import MaxNLocator, FormatStrFormatter # type: ignore
from matplotlib.offsetbox import AnchoredText # type: ignore
from matplotlib.lines import Line2D # type: ignore
# import bisection from scipy.optimize
import helper as hlp
from complex_potential_flow_class import potential_flow_object
# import tqdm for progress bar
from tqdm import tqdm # type: ignore
import os # type: ignore
import matplotlib.patches as patches  # type: ignore
import time
from functools import lru_cache
from multiprocessing import Pool

class cylinder(potential_flow_object):
    """This is a class that creates a cylinder object and performs calculations specific to a cylinder"""
    def __init__(self, json_file):
        self.cyl_json_file = json_file
        self.parse_cylinder_json()
        potential_flow_object.__init__(self, json_file) # this is doing self.() of all the variables we want to in the super class

    def parse_cylinder_json(self):
        """This function reads the json file and stores the cylinder specific data in a dictionary"""
        with open(self.cyl_json_file, 'r') as json_handle:
            input = json.load(json_handle)
            # appellian stuff
            self.is_compute_appellian = hlp.parse_dictionary_or_return_default(input, ["appellian", "is_compute_appellian"], False)
            self.is_compute_minimum_appellian = hlp.parse_dictionary_or_return_default(input, ["appellian", "is_compute_minimum_appellian"], False)
            self.integration_method = hlp.parse_dictionary_or_return_default(input, ["appellian", "integration_method"], "trapezoidal")
            # Select integration method
            if self.integration_method == "left_riemann":
                self.integration_func = self.riemann_integration
            elif self.integration_method == "trapezoidal":
                self.integration_func = self.trapezoidal_integration
            elif self.integration_method == "simpson13":
                self.integration_func = self.simpson_13_integration
            elif self.integration_method == "romberg":
                self.integration_func = self.romberg_integration
            else:
                raise ValueError("Invalid integration method specified. Use 'left_riemann' 'trapezoidal'.")
            self.is_plot_appellian = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_appellian"], False)
            self.is_plot_appellian_for_varying_circulation = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_appellian_for_varying_circulation"], False)
            self.is_calc_circulation_for_varying_shape = hlp.parse_dictionary_or_return_default(input, ["appellian", "is_calc_circulation_for_varying_shape"], False)
            self.is_plot_circulation_for_varying_shape = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_circulation_for_varying_shape"], False)
            self.is_single_romberg_theta = hlp.parse_dictionary_or_return_default(input, ["appellian", "is_single_romberg_theta"], True)
            self.is_single_romberg_r = hlp.parse_dictionary_or_return_default(input, ["appellian", "is_single_romberg_r"], False)
            self.num_thetas_start = hlp.parse_dictionary_or_return_default(input, ["appellian", "num_thetas_start"], 10)
            self.num_times_thetas_doubled = hlp.parse_dictionary_or_return_default(input, ["appellian", "num_times_thetas_doubled"], 18)
            self.num_times_r_doubled = hlp.parse_dictionary_or_return_default(input, ["appellian", "num_times_r_doubled"], 18)
            self.one_appellian_theta_points = hlp.parse_dictionary_or_return_default(input, ["appellian", "one_appellian_theta_points"], 1000)
            self.one_appellian_r_points = hlp.parse_dictionary_or_return_default(input, ["appellian", "one_appellian_r_points"], 1000)
            self.r_growth_factor = hlp.parse_dictionary_or_return_default(input, ["appellian", "r_growth_factor"], 2.0)
            self.radial_distance = hlp.parse_dictionary_or_return_default(input, ["appellian", "radial_distance"], 10.0)
            self.is_calc_convergence_to_kutta = hlp.parse_dictionary_or_return_default(input, ["appellian", "is_calc_convergence_to_kutta"], False)
            self.is_plot_convergence_to_kutta = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_convergence_to_kutta"], False)
            self.is_calc_theta_convergence = hlp.parse_dictionary_or_return_default(input, ["appellian", "is_calc_theta_convergence"], False)
            self.is_calc_Gamma_convergence = hlp.parse_dictionary_or_return_default(input, ["appellian", "is_calc_Gamma_convergence"], False)
            self.is_plot_Gamma_convergence = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_Gamma_convergence"], False)
            self.is_plot_theta_convergence = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_theta_convergence"], False)
            self.is_calc_r_convergence = hlp.parse_dictionary_or_return_default(input, ["appellian", "is_calc_r_convergence"], False)
            self.is_plot_r_convergence = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_r_convergence"], False)
            self.grid_conv_is_relative_error = hlp.parse_dictionary_or_return_default(input, ["appellian", "grid_conv_is_relative_error"], False)
            self.appellian_is_line_integral = hlp.parse_dictionary_or_return_default(input, ["appellian", "appellian_is_line_integral"], True)
            self.is_numerical_integrand = hlp.parse_dictionary_or_return_default(input, ["appellian", "is_numerical_integrand"], False)
            if self.appellian_is_line_integral:
                self.appellian_is_area_integral = False
                if self.is_numerical_integrand:
                    self.accel_function = self.numerical_line_int_conv_accel
                else:
                    self.accel_function = self.analytic_conv_accel_for_line_int_comp_conj
            else:
                self.appellian_is_area_integral = True
                if self.is_numerical_integrand:
                    self.accel_function = self.numerical_convective_acceleration
                else:
                    self.accel_function = self.analytic_conv_accel_square_comp_conj
            self.appellian_minimization_method = hlp.parse_dictionary_or_return_default(input, ["appellian", "appellian_minimization_method"], "roots_of_derive_of_poly_fit")
            self.polynomial_order = hlp.parse_dictionary_or_return_default(input, ["appellian", "polynomial_order"], 4)
            self.is_calc_D_sweep_alpha_considerations = hlp.parse_dictionary_or_return_default(input, ["appellian", "is_calc_D_sweep_alpha_considerations"], False)
            self.is_plot_D_sweep_alpha_considerations = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_D_sweep_alpha_considerations"], False)
            self.is_calc_z_plane_surface_pressure_distribution = hlp.parse_dictionary_or_return_default(input, ["appellian", "is_calc_z_plane_surface_pressure_distribution"], False) ######### go through and make this happen! Currently the calculating is tied to the plotting
            self.is_plot_z_plane_surface_pressure_distribution = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_z_plane_surface_pressure_distribution"], False)
            self.is_calc_pressure_alpha_considerations = hlp.parse_dictionary_or_return_default(input, ["appellian", "is_calc_pressure_alpha_considerations"], False)
            self.is_plot_pressure_alpha_considerations = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_pressure_alpha_considerations"], False)
            self.alphas_for_consideration = hlp.parse_dictionary_or_return_default(input, ["plot", "alphas_for_consideration"], [5.0, 15.0])
            for i in range(len(self.alphas_for_consideration)):
                self.alphas_for_consideration[i] = np.radians(self.alphas_for_consideration[i])
            self.is_create_dsweep_gif = hlp.parse_dictionary_or_return_default(input, ["plot", "is_create_dsweep_gif"], False)
            self.use_shape_parameter_D = hlp.parse_dictionary_or_return_default(input, ["geometry", "use_shape_parameter_D"], False)
            self.is_plot_shifted_joukowski_cylinder = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_shifted_joukowski_cylinder"], False)
            self.D = hlp.parse_dictionary_or_return_default(input, ["geometry", "shape_parameter_D"], 1.0)
            self.save_fig = hlp.parse_dictionary_or_return_default(input, ["plot", "save_fig"], False)
            self.show_fig = hlp.parse_dictionary_or_return_default(input, ["plot", "show_fig"], False)
            self.is_plot_z_selection_comparison = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_z_selection_comparison"], False)
            self.is_plot_streamlines = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_streamlines"], False)
            self.cylinder_radius = hlp.parse_dictionary_or_return_default(input, ["geometry", "cylinder_radius"], 1.0) # this is the radius of the cylinder. We are normalizing everything by this value. This is used in the geometry function to calculate the geometry of the cylinder.
            self.original_cylinder_radius = self.cylinder_radius # this is the original cylinder radius. 
            self.cylinder_radius/=self.original_cylinder_radius # normalize the cylinder radius to 1. This makes the math easier. We are normalizing everything by the cylinder radius. This is used in the geometry function to calculate the geometry of the cylinder.
            self.is_airfoil = False
            self.is_text_file = False
            self.is_cylinder = True
            self.type = hlp.parse_dictionary_or_return_default(input, ["geometry", "type"], "cylinder") # this is the type of geometry we are dealing with. It can be either "cylinder" or "airfoil". Default is cylinder.
            angle_of_attack = hlp.parse_dictionary_or_return_default(input, ["operating", "angle_of_attack[deg]"], 5.0) # this is the angle of attack in degrees. We are converting it to radians later.
            self.angle_of_attack = np.radians(angle_of_attack)
            self.freestream_velocity = hlp.parse_dictionary_or_return_default(input, ["operating", "freestream_velocity"], 1.0) # this is the freestream velocity. This is used as a part of the velocity function to calculate the velocity at a given point in the flow field.
            if np.isclose(self.freestream_velocity, 0.0):
                raise ValueError("The freestream velocity cannot be zero. Please set a non-zero value in the input JSON file.")
            self.design_CL = hlp.parse_dictionary_or_return_default(input, ["geometry", "design_CL"], 1.0) # this is the design lift coefficient. This is used to calculate the zeta center and eccentricity for J-airfoils.
            self.design_thickness = hlp.parse_dictionary_or_return_default(input, ["geometry", "design_thickness"], 0.2) # this is the design thickness. This is used to calculate the zeta center and eccentricity for J-airfoils.
            self.output_points = hlp.parse_dictionary_or_return_default(input, ["geometry", "output_points"], 1000) # this is the number of points we want to output for the Joukowski airfoil geometry
            if self.output_points % 2 == 0:
                self.even_num_points = True
            else:
                self.even_num_points = False
            self.n_geom_points = self.output_points
            self.circulation = hlp.parse_dictionary_or_return_default(input, ["operating", "circulation_strength"], 0.0) # this is the circulation strength. This is used to calculate Lift through the Kutta-Joukowski theorem.
            zeta_center = hlp.parse_dictionary_or_return_default(input, ["geometry", "zeta_0"], [-0.01, 0.0]) # this is a list of two elements, real and imaginary parts of zeta_0. We need to convert it to a complex number and normalize it by the cylinder radius.
            self.zeta_center = (zeta_center[0] + 1j*zeta_center[1])/self.cylinder_radius
            if self.use_shape_parameter_D:
                self.epsilon = self.calc_epsilon_from_D(self.D)
                # print("epsilon from D = " + str(self.D) + " and zeta0 = " + str(self.zeta_center) + " is " + str(self.epsilon))
                self.kutta_circulation = self.calc_circulation_J_airfoil()
                print("Kutta Circulation!", self.kutta_circulation)
                # print("D:        ", self.D)
                # print("kutta circ", self.kutta_circulation)
                # self.general_circulation = self.general_kutta_condition()
            else:
                self.epsilon = hlp.parse_dictionary_or_return_default(input, ["geometry", "epsilon"], self.cylinder_radius)
                self.kutta_circulation = self.calc_circulation_J_airfoil()
                D = self.calc_D_from_epsilon()
            self.ja_epsilon = self.calc_J_airfoil_epsilon()
            xi1 = self.cylinder_radius - self.epsilon - np.sqrt(self.cylinder_radius**2 - self.zeta_center.imag**2) # this is the x-coordinate of the leading edge of the Joukowski airfoil
            xiT1 = np.sqrt(self.cylinder_radius**2 - self.zeta_center.imag**2)
            eta1 = np.sqrt(self.cylinder_radius**2 - xiT1**2) # this is the y-coordinate of the leading edge of the Joukowski airfoil
            self.zeta1 = xi1 + 1j*eta1 # this is the location of the circular cylinder whose trailing edge singularity location is coincident with a cylinder of radius R
            chi_AJA = self.cylinder_radius - self.ja_epsilon - self.zeta_center
            chi_A = self.cylinder_radius - self.epsilon - self.zeta_center
            self.r_AJA, self.theta_AJA = hlp.xy_to_r_theta(chi_AJA.real, chi_AJA.imag)
            self.r_A, self.theta_A = hlp.xy_to_r_theta(chi_A.real, chi_A.imag)

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
    
    def geometry_zeta(self, theta_zeta: float): # A1 on project
        """"""
        zeta_surface = self.zeta_center + self.cylinder_radius*np.exp(1j*theta_zeta)
        return [zeta_surface.real, zeta_surface.imag]

    def get_full_geometry_zeta(self, number_of_points: int = 10, theta_start: float = 2*np.pi, theta_half: float = np.pi, theta_end: float = 0.0):
        """Generates full zeta geometry by evaluating from theta_start to theta_end via theta_half."""
        # if sin(theta_start) == sin(theta_end): then add one to the number of points because the first point in the first half of the array will be equal to the last point in the second half of the array (i.e. theta_start = 2*np.pi and theta_end = 0.0)
        if np.isclose(np.sin(theta_start), np.sin(theta_end), atol=1e-14):
            number_of_points += 1  # Add one to avoid duplication of the first point in the second half of the array
        if number_of_points < 2:
            raise ValueError("Number of points must be at least 2.")
        percentage_distance = abs((theta_half - theta_start) / (theta_end - theta_start)) # This gives the fraction of the total angle that the first segment covers
        n1 = int(number_of_points * percentage_distance)  # Number of points in the first segment
        n2 = number_of_points - n1
        # First segment: from theta_start to theta_half (excluding theta_half to avoid duplication)
        theta_coords_1 = np.linspace(theta_start, theta_half, n1, endpoint=False)
        # Second segment: from theta_half to theta_end (including theta_half and theta_end)
        theta_coords_2 = np.linspace(theta_half, theta_end, n2)
        # Concatenate and compute zeta geometry
        theta_coords = np.concatenate((theta_coords_1, theta_coords_2))
        zeta_array = np.array([[*self.geometry_zeta(theta),theta] for theta in theta_coords])
        # Check if the first and last points are the same out to 14 decimal places, then remove the last point
        if np.isclose(zeta_array[0][0], zeta_array[-1][0], atol=1e-14) and np.isclose(zeta_array[0][1], zeta_array[-1][1], atol=1e-14):
            zeta_array = zeta_array[:-1]
        self.zeta_geom_array = zeta_array
        return self.zeta_geom_array

    def plot_geometry_zeta(self):
        """"""
        # make the line type dashed
        linetype = '--'
        color = 'black'
        plt.plot(self.zeta_geom_array[:,0], self.zeta_geom_array[:,1], label = "Cyl", linestyle=linetype, color=color)
        plt.plot([self.zeta_geom_array[0][0], self.zeta_geom_array[-1][0]], [self.zeta_geom_array[0][1], self.zeta_geom_array[-1][1]], linestyle=linetype, color=color)
        size = 15
        plt.scatter(self.zeta_center.real, self.zeta_center.imag, color=color, s = size, label="$\\zeta_0$")

    def geometry(self, theta_zeta: float): # A1 on project
        """This function calculates the geometry of the cylinder at a given x-coordinate"""
        zeta = self.geometry_zeta(theta_zeta)[0] + 1j*self.geometry_zeta(theta_zeta)[1]
        z = self.zeta_to_z(zeta, self.epsilon)
        return [z.real, z.imag] 

    def get_full_geometry(self):
        """This function calculates the geometry of the cylinder at every point"""
        num_points = len(self.zeta_geom_array)  # Use the length of zeta_geom_array to determine the number of points
        # Make sure that x_coords has points at the leading and trailing edges
        full_z_surface = np.zeros((num_points, 2))  # Initialize an array to hold the full z surface coordinates
        for i in range(len(self.zeta_geom_array)): # iterate through the length of the first half of the points
            zeta = self.zeta_geom_array[i][0] + 1j*self.zeta_geom_array[i][1]  # zeta is a complex number
            z = self.zeta_to_z(zeta, self.epsilon)
            full_z_surface[i] = [z.real, z.imag]  # store the x and y coordinates in the full_z_surface array
        self.full_z_surface = full_z_surface
        return self.full_z_surface 
    
    def calc_foci(self):
        """This function calculates the foci of the ellipse using epsilon"""
        self.zeta_trailing_edge_focus = 1*(self.cylinder_radius - self.epsilon)/self.cylinder_radius
        self.zeta_leading_edge_focus = -1*(self.cylinder_radius - self.epsilon)/self.cylinder_radius
        self.z_trailing_edge_focus = self.zeta_to_z(self.zeta_trailing_edge_focus, self.epsilon)
        self.z_leading_edge_focus = self.zeta_to_z(self.zeta_leading_edge_focus, self.epsilon)
        return self.zeta_trailing_edge_focus, self.zeta_leading_edge_focus, self.z_trailing_edge_focus, self.z_leading_edge_focus
    
    def get_and_plot_foci(self):
        """This function calculates and plots the foci of the ellipse using epsilon"""
        self.calc_foci()
        size = 15
        size_tri = 30
        plt.scatter(self.zeta_trailing_edge_focus, 0.0, color='black', marker='^', s = size_tri, label="sing") # list_of_all_possible_markers = ['o', 's', 'D', 'v', ' ^', '<', '>', 'p', 'P', '*', 'X', 'd', 'H', 'h', '+', '|', '_', '1', '2', '3', '4', '8']
        plt.scatter(self.zeta_leading_edge_focus, 0.0, color='black', marker='^', s = size_tri) # D is a diamond, s is a square, o is a circle, v is a triangle pointing down,  ^ is a triangle pointing up
        plt.scatter(self.z_trailing_edge_focus, 0.0, color='black', marker='s', s = size, label = "J-np.sing")
        plt.scatter(self.z_leading_edge_focus, 0.0, color='black', marker='s', s = size)
        return self.zeta_trailing_edge_focus, self.zeta_leading_edge_focus, self.z_trailing_edge_focus, self.z_leading_edge_focus
    
    def plot_geometry_settings(self):
        # include y and x axes 
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
        # epsilon_name = "epsilon_" + str(self.epsilon)
        # zeta_0_name = "zeta_0_" + str(self.zeta_center)
        # if self.show_fig:
        #     if self.type == "cylinder" and self.is_compute_appellian and self.is_plot_appellian_for_varying_circulation:
        #         plt.scatter(xi_eta_vals[:, 0], xi_eta_vals[:, 1], color='red', s=5)
        #         if self.save_fig:
        #             plt.savefig("Joukowski_Cylinder_" + epsilon_name + "_" +  zeta_0_name + "_" + ".svg", dpi=300, bbox_inches=None, pad_inches=0)
        #             plt.close()
    
    def z_to_zeta_incorrect(self, z: complex, epsilon: float):
        """This function returns the opposite root that z_to_zeta returns"""
        zeta_correct = self.z_to_zeta(z, epsilon)
        z_1 = z**2 - 4*(self.cylinder_radius - epsilon)**2
        zeta_option_1 = (z + np.sqrt(z_1))/2
        zeta_option_2 = (z - np.sqrt(z_1))/2
        # check if the zeta coordinate is the same as either of the options
        if np.isclose(zeta_correct, zeta_option_1):
            zeta = zeta_option_2
        elif np.isclose(zeta_correct, zeta_option_2):
            zeta = zeta_option_1
        else:
            raise ValueError("The zeta coordinate does not match either of the options")
        return zeta
    
    def dPhi_dzeta(self, zeta, Gamma):
        """This function calculates the derivative of Phi with respect to zeta for a Joukowski transformation"""
        V_inf, R, alpha, zeta0 = self.freestream_velocity, self.cylinder_radius, self.angle_of_attack, self.zeta_center
        return V_inf*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf*(zeta-zeta0)) - np.exp(1j*alpha)*R**2/(zeta-zeta0)**2)
    
    def d2Phi_dzeta2(self, zeta, Gamma):
        """This function calculates the second derivative of Phi with respect to zeta for a Joukowski transformation"""
        V_inf, R, alpha, zeta0 = self.freestream_velocity, self.cylinder_radius, self.angle_of_attack, self.zeta_center
        return V_inf*(-1j*Gamma/(2*np.pi*V_inf*(zeta-zeta0)**2) + 2*np.exp(1j*alpha)*R**2/((zeta-zeta0)**3))
    
    def dZ_dzeta(self, zeta):
        """This function calculates the derivative of Z with respect to zeta for a Joukowski transformation"""
        epsilon, R, = self.epsilon, self.cylinder_radius
        return 1 - (R-epsilon)**2/(zeta)**2
    
    def d2Z_dzeta2(self, zeta):
        """This function calculates the second derivative of Z with respect to zeta for a Joukowski transformation"""
        epsilon, R = self.epsilon, self.cylinder_radius
        return 2*(R-epsilon)**2/(zeta)**3
    
    def velocity(self, point_xi_eta_in_z_plane, Gamma):
        """This function calculates the velocity at a given point in the flow field in the z plane"""
        z = point_xi_eta_in_z_plane[0] + 1j*point_xi_eta_in_z_plane[1]
        zeta = self.z_to_zeta(z, self.epsilon)
        dZ_dzeta = self.dZ_dzeta(zeta)
        velocity = self.dPhi_dzeta(zeta, Gamma) / dZ_dzeta # eq 107
        velocity_complex = np.array([velocity.real, -velocity.imag])
        return velocity_complex

    def analytic_conv_accel_for_line_int_comp_conj(self, point_r_theta_in_Chi, Gamma):
        """calculates the convective acceleration using the line integral method"""
        V_inf, epsilon, R, alpha, zeta0 = self.freestream_velocity, self.epsilon, self.cylinder_radius, self.angle_of_attack, self.zeta_center
        r, theta = point_r_theta_in_Chi[0], point_r_theta_in_Chi[1]
        xi_chi, eta_chi = hlp.r_theta_to_xy(r, theta)
        chi = xi_chi + 1j*eta_chi
        zeta = self.Chi_to_zeta(chi)
        g, f = self.dZ_dzeta(zeta), self.dPhi_dzeta(zeta, Gamma)
        gbar = np.conj(g)
        fbar = np.conj(f)
        df = V_inf*(-1j*Gamma/(2*np.pi*V_inf*(r**2*np.exp(1j*theta))) + 2*R**2*np.exp(1j*alpha)/(r**3*np.exp(2*1j*theta)))
        dg  = 2*(R-epsilon)**2*np.exp(1j*theta)/(r*np.exp(1j*theta)+zeta0)**3
        dgbar = np.conj(dg)
        dfbar = np.conj(df)
        # g = dz_dzeta
        integrand = 1/(g**4*gbar**4)*((g*gbar)**2*f*fbar*(f*dfbar+fbar*df)- (f*fbar)**2*g*gbar*(g*dgbar+gbar*dg))
        # dzeta_dtheta = 1j*r*np.exp(1j*theta)
        # integrand *= abs(dzeta_dtheta)*abs(g)
        return np.real(integrand)

    def numerical_line_int_conv_accel(self, point_r_theta_in_Chi, Gamma,  stepsize=1e-6):
        """"""
        r, theta = point_r_theta_in_Chi[0], point_r_theta_in_Chi[1]
        xi_chi, eta_chi = hlp.r_theta_to_xy(r, theta)
        chi = xi_chi + 1j*eta_chi
        zeta = self.Chi_to_zeta(chi)
        dz_dzeta = self.dZ_dzeta(zeta)
        dz_dzeta2 = dz_dzeta * np.conj(dz_dzeta)
        z = self.zeta_to_z(zeta, self.epsilon)
        z_point = np.array([z.real, z.imag])
        z_point_plus_x_step = np.array([z.real+stepsize, z.imag])
        z_point_minus_x_step = np.array([z.real-stepsize, z.imag])
        z_point_plus_y_step = np.array([z.real, z.imag+stepsize])
        z_point_minus_y_step = np.array([z.real, z.imag-stepsize])
        velocity_at_z = self.velocity(z_point, Gamma)
        velocity_plus_x_step = self.velocity(z_point_plus_x_step, Gamma)
        velocity_minus_x_step = self.velocity(z_point_minus_x_step, Gamma)
        velocity_plus_y_step = self.velocity(z_point_plus_y_step, Gamma)
        velocity_minus_y_step = self.velocity(z_point_minus_y_step, Gamma)
        ux = velocity_at_z[0]
        dux_dx = self.central_difference_alt(z_point_minus_x_step[0], z_point[0], z_point_plus_x_step[0], velocity_minus_x_step[0], velocity_at_z[0], velocity_plus_x_step[0])
        uy = velocity_at_z[1]
        dux_dy = self.central_difference_alt(z_point_minus_y_step[1], z_point[1], z_point_plus_y_step[1], velocity_minus_y_step[0], velocity_at_z[0], velocity_plus_y_step[0])
        duy_dx = self.central_difference_alt(z_point_minus_x_step[0], z_point[0], z_point_plus_x_step[0], velocity_minus_x_step[1], velocity_at_z[1], velocity_plus_x_step[1])
        duy_dy = self.central_difference_alt(z_point_minus_y_step[1], z_point[1], z_point_plus_y_step[1], velocity_minus_y_step[1], velocity_at_z[1], velocity_plus_y_step[1])
        normal = self.calc_z_normal(theta)
        Ax = 4*ux**3*dux_dx + 4*ux*dux_dx*uy**2 + 4*ux**2*uy*duy_dx + 4*uy**3*duy_dx
        Ay = 4*ux**3*dux_dy + 4*ux*dux_dy*uy**2 + 4*ux**2*uy*duy_dy + 4*uy**3*duy_dy
        #dzeta_dtheta = 1j*r*np.exp(1j*theta)
        Accel4 = 0.5*(Ax*normal[0] + Ay*normal[1])
        # Accel4 = (Ax*normal[0] + Ay*normal[1])
        return np.real(Accel4)

    def calc_z_normal(self, theta_chi):
        tangent = 1j*self.cylinder_radius*np.exp(1j*theta_chi)*(1-((self.cylinder_radius-self.epsilon)**2/((self.cylinder_radius*np.exp(1j*theta_chi) + self.zeta_center)**2)))
        mag = np.sqrt(tangent.real**2 + tangent.imag**2)
        z_normal = [tangent.imag/mag, -tangent.real/mag]
        return z_normal

    def analytic_conv_accel_square_comp_conj(self,point_r_theta_in_Chi,Gamma):
        r, theta = point_r_theta_in_Chi[0], point_r_theta_in_Chi[1]
        xi_chi, eta_chi = hlp.r_theta_to_xy(r, theta)
        chi = xi_chi + 1j*eta_chi
        zeta = self.Chi_to_zeta(chi)
        dphi_dzeta, d2Phi_dzeta2, dz_dzeta, d2z_dzeta2 = self.dPhi_dzeta(zeta, Gamma), self.d2Phi_dzeta2(zeta, Gamma), self.dZ_dzeta(zeta), self.d2Z_dzeta2(zeta)
        dz_dzeta2 = dz_dzeta * np.conj(dz_dzeta)
        dz_dzeta4 = dz_dzeta**2*np.conj(dz_dzeta)**2
        conv_accel = dphi_dzeta*(d2Phi_dzeta2*dz_dzeta - dphi_dzeta*d2z_dzeta2)/dz_dzeta4
        conv_accel_comp_conj = np.conj(conv_accel)
        conv_accel_squared = conv_accel * conv_accel_comp_conj * dz_dzeta2 
        return np.real(conv_accel_squared)
    
    def numerical_convective_acceleration(self, point_r_theta_in_Chi, Gamma, stepsize=1e-8):
        r, theta = point_r_theta_in_Chi[0], point_r_theta_in_Chi[1]
        xi_chi, eta_chi = hlp.r_theta_to_xy(r, theta)
        chi = xi_chi + 1j*eta_chi
        zeta = self.Chi_to_zeta(chi)
        dz_dzeta = self.dZ_dzeta(zeta)
        dz_dzeta2 = dz_dzeta * np.conj(dz_dzeta)
        z = self.zeta_to_z(zeta, self.epsilon)
        z_point = np.array([z.real, z.imag])
        z_point_plus_x_step = np.array([z.real+stepsize, z.imag])
        z_point_minus_x_step = np.array([z.real-stepsize, z.imag])
        z_point_plus_y_step = np.array([z.real, z.imag+stepsize])
        z_point_minus_y_step = np.array([z.real, z.imag-stepsize])
        velocity_at_z = self.velocity(z_point, Gamma)
        velocity_plus_x_step = self.velocity(z_point_plus_x_step, Gamma)
        velocity_minus_x_step = self.velocity(z_point_minus_x_step, Gamma)
        velocity_plus_y_step = self.velocity(z_point_plus_y_step, Gamma)
        velocity_minus_y_step = self.velocity(z_point_minus_y_step, Gamma)
        ux = velocity_at_z[0]
        dux_dx = self.central_difference_alt(z_point_minus_x_step[0], z_point[0], z_point_plus_x_step[0], velocity_minus_x_step[0], velocity_at_z[0], velocity_plus_x_step[0])
        uy = velocity_at_z[1]
        dux_dy = self.central_difference_alt(z_point_minus_y_step[1], z_point[1], z_point_plus_y_step[1], velocity_minus_y_step[0], velocity_at_z[0], velocity_plus_y_step[0])
        duy_dx = self.central_difference_alt(z_point_minus_x_step[0], z_point[0], z_point_plus_x_step[0], velocity_minus_x_step[1], velocity_at_z[1], velocity_plus_x_step[1])
        duy_dy = self.central_difference_alt(z_point_minus_y_step[1], z_point[1], z_point_plus_y_step[1], velocity_minus_y_step[1], velocity_at_z[1], velocity_plus_y_step[1])
        conv_accel = np.array([ux*dux_dx + uy*dux_dy, ux*duy_dx+ uy*duy_dy])
        conv_accel_squared = np.dot(conv_accel, conv_accel) #* dz_dzeta2 
        return np.real(conv_accel_squared)

    def central_difference(self, value_above, value_below, stepsize=1e-6):
        """Calculates central difference based on stepsize"""
        return (value_above - value_below)/(2*stepsize)

    def central_difference_alt(self, x_below, x_at, x_above, value_below, value_at, value_above):
        """Does not require even spacing"""
        Deltaxat = x_at - x_below 
        Deltaxabove = x_above - x_at
        Phi_below = value_below
        Phi_above = value_above
        Phi_at = value_at
        numerator = Phi_above*Deltaxat**2 - Phi_below*Deltaxabove**2 + Phi_at*(Deltaxabove**2-Deltaxat**2)
        denominator = Deltaxabove*Deltaxat*(Deltaxat+Deltaxabove)
        return numerator/denominator # this is the

    def appellian_acceleration_loop(self, Gamma, r_values, theta_values, grid_conv = False, reference_value = None):
        if not grid_conv and not self.is_calc_circulation_for_varying_shape:
            print("Running appellian acceleration loop. Is area integral = " + str(self.appellian_is_area_integral) + ". Integration type = " + str(self.integration_method))
        if self.appellian_is_area_integral:
            factor = 0.5
        else:
            factor = -1/16
        if (self.integration_func == self.romberg_integration or self.integration_func == self.romberg_integration_dOmega) and not grid_conv:
            func = self.integration_func(r_values, theta_values, Gamma, is_theta = self.is_single_romberg_theta, is_r = self.is_single_romberg_r, reference_value=reference_value)
            answer, epsilon, length_trap_2 = func[0], func[1], func[2]
            return factor*answer, epsilon, length_trap_2
        elif (self.integration_func == self.romberg_integration or self.integration_func == self.romberg_integration_dOmega) and grid_conv:
            func = self.integration_func(r_values, theta_values, Gamma, is_theta = self.is_calc_theta_convergence, is_r = self.is_calc_r_convergence, reference_value=reference_value)
            answer, epsilon, length_trap_2 = func[0], func[1], func[2]
            return factor*answer, epsilon, length_trap_2
        else:
            return factor*self.integration_func(r_values, theta_values, Gamma)

    def riemann_integration(self, r_values, theta_values, Gamma):
        """Left Riemann rule over full domain."""
        is_area = len(r_values) > 1
        if is_area:
            total = 0.0
            for i, r in enumerate(r_values):
                r_next = r_values[i + 1] if i < len(r_values) - 1 else r  # Handle last element
                dr = r_next - r if r_next != r else dr * self.r_growth_factor  # Use growth factor if needed
                for j, theta in enumerate(theta_values[:-1]): # the -1 means we are excluding the last element
                    theta_next = theta_values[j + 1]
                    dtheta = theta_next - theta
                    f = self.accel_function([r, theta], Gamma)
                    total += f * r * dr * dtheta
        else:
            r0 = r_values[0]
            total = 0.0
            for j, theta in enumerate(theta_values[:-1]): # the -1 means we are excluding the last element
                theta_next = theta_values[j + 1]
                dtheta = theta_next - theta
                f = self.accel_function([r0, theta], Gamma)
                total += f * r0 * dtheta
        return total

    def riemann_integration_dOmega(self, r_values, theta_values, Gamma):
        """
        Left Riemann integration with mapped geometry.
        - For area cells: use function value at lower-left corner * quadrilateral area.
        - For line segments: use left endpoint * mapped arclength.
        """
        is_area = len(r_values) > 1
        total = 0.0
        if is_area:
            # iterate over cells
            for i in range(len(r_values) - 1):
                r0, r1 = r_values[i], r_values[i + 1]
                for j in range(len(theta_values) - 1):
                    th0, th1 = theta_values[j], theta_values[j + 1]
                    # map corners
                    chi00 = complex(*hlp.r_theta_to_xy(r0, th0))
                    chi01 = complex(*hlp.r_theta_to_xy(r0, th1))
                    chi11 = complex(*hlp.r_theta_to_xy(r1, th1))
                    chi10 = complex(*hlp.r_theta_to_xy(r1, th0))
                    z00 = self.zeta_to_z(self.Chi_to_zeta(chi00), self.epsilon)
                    z01 = self.zeta_to_z(self.Chi_to_zeta(chi01), self.epsilon)
                    z11 = self.zeta_to_z(self.Chi_to_zeta(chi11), self.epsilon)
                    z10 = self.zeta_to_z(self.Chi_to_zeta(chi10), self.epsilon)
                    # numerical area
                    dOmega = self.quad_area_from_complex(z00, z01, z11, z10)
                    # left corner value only
                    f00 = self.accel_function([r0, th0], Gamma)
                    total += f00 * dOmega
            return total
        else:
            r0 = r_values[0]
            for j in range(len(theta_values) - 1):
                th0, th1 = theta_values[j], theta_values[j + 1]
                chi0 = complex(*hlp.r_theta_to_xy(r0, th0))
                chi1 = complex(*hlp.r_theta_to_xy(r0, th1))
                z0 = self.zeta_to_z(self.Chi_to_zeta(chi0), self.epsilon)
                z1 = self.zeta_to_z(self.Chi_to_zeta(chi1), self.epsilon)
                dS = abs(z1 - z0)  # mapped length
                f0 = self.accel_function([r0, th0], Gamma)  # left value only
                total += f0 * dS
            return total

    def quad_area_from_complex(self, z00, z01, z11, z10):
        """
        Area of the quadrilateral with vertices ordered around the cell:
        z00=(r,theta), z01=(r,theta+deltatheta), z11=(r+deltar,theta+deltatheta), z10=(r+deltar,theta).
        Uses shoelace formula; works for convex/concave; orientation-safe.
        """
        xs = np.array([z00.real, z01.real, z11.real, z10.real])
        ys = np.array([z00.imag, z01.imag, z11.imag, z10.imag])
        # close polygon by repeating first vertex
        x_next = np.roll(xs, -1) # roll does a circular shift, for example, if xs = [1, 2, 3] then x_next = [2, 3, 1]
        y_next = np.roll(ys, -1) 
        area = 0.5 * abs(np.sum(xs * y_next - ys * x_next))
        return area

    def trapezoidal_integration(self, r_values, theta_values, Gamma):
        """Trapezoidal rule over full domain."""
        is_area = len(r_values) > 1
        if is_area:
            total = 0.0
            # loop over radial strips
            for i, r in enumerate(r_values[:-1]):
                r_next = r_values[i + 1]
                dr = r_next - r
                # --- preload the bottom edge (r) ---
                f0 = self.accel_function([r, theta_values[0]], Gamma) * r
                f1 = self.accel_function([r_next, theta_values[0]], Gamma) * r_next
                # loop over theta cells
                for j, theta in enumerate(theta_values[:-1]):
                    theta_next = theta_values[j + 1]
                    dtheta = theta_next - theta
                    # reuse bottom edge, compute only the new "top-right" values
                    f2 = self.accel_function([r, theta_next], Gamma) * r
                    f3 = self.accel_function([r_next, theta_next], Gamma) * r_next
                    avg = 0.25 * (f0 + f1 + f2 + f3)
                    total += avg * dr * dtheta
                    # slide the window in theta: right edge becomes left edge
                    f0, f1 = f2, f3
        else:
            # 1D case along theta
            r0 = r_values[0]
            total = 0.0
            f0 = self.accel_function([r0, theta_values[0]], Gamma)
            for j, theta in enumerate(theta_values[:-1]):
                theta_next = theta_values[j + 1]
                dtheta = theta_next - theta
                f1 = self.accel_function([r0, theta_next], Gamma)
                total += 0.5 * (f0 + f1) * r0 * dtheta 
                f0 = f1
        return total

    def trapezoidal_integration_dOmega(self, r_values, theta_values, Gamma):
        """
        Trapezoidal rule using the *mapped* area (for area case) or arclength (for line case).
        For area cells, the area is computed numerically from the four mapped corners.
        """
        is_area = len(r_values) > 1
        total = 0.0
        if is_area:
            # iterate over r,θ cells
            for i in range(len(r_values) - 1):
                r0, r1 = r_values[i], r_values[i + 1]
                for j in range(len(theta_values) - 1):
                    th0, th1 = theta_values[j], theta_values[j + 1]
                    # corners in χ → ζ → z
                    chi00 = complex(*hlp.r_theta_to_xy(r0, th0))
                    chi01 = complex(*hlp.r_theta_to_xy(r0, th1))
                    chi11 = complex(*hlp.r_theta_to_xy(r1, th1))
                    chi10 = complex(*hlp.r_theta_to_xy(r1, th0))
                    z00 = self.zeta_to_z(self.Chi_to_zeta(chi00), self.epsilon)
                    z01 = self.zeta_to_z(self.Chi_to_zeta(chi01), self.epsilon)
                    z11 = self.zeta_to_z(self.Chi_to_zeta(chi11), self.epsilon)
                    z10 = self.zeta_to_z(self.Chi_to_zeta(chi10), self.epsilon)
                    # numerical area of the mapped quadrilateral
                    dOmega = self.quad_area_from_complex(z00, z01, z11, z10)
                    # trapezoidal average of f at the four corners
                    f00 = self.accel_function([r0, th0],  Gamma)
                    f01 = self.accel_function([r0, th1],  Gamma)
                    f11 = self.accel_function([r1, th1],  Gamma)
                    f10 = self.accel_function([r1, th0],  Gamma)
                    avg = 0.25 * (f00 + f01 + f11 + f10)
                    total += avg * dOmega
            return total
        # ---- line integral (mapped arclength along θ at fixed r) ----
        r0 = r_values[0]
        for j in range(len(theta_values) - 1):
            th0, th1 = theta_values[j], theta_values[j + 1]
            chi0 = complex(*hlp.r_theta_to_xy(r0, th0))
            chi1 = complex(*hlp.r_theta_to_xy(r0, th1))
            z0 = self.zeta_to_z(self.Chi_to_zeta(chi0), self.epsilon)
            z1 = self.zeta_to_z(self.Chi_to_zeta(chi1), self.epsilon)
            dS = abs(z1 - z0)  # segment length in physical plane
            f0 = self.accel_function([r0, th0], Gamma)
            f1 = self.accel_function([r0, th1], Gamma)
            total += 0.5 * (f0 + f1) * dS
        return total

    def simpson_13_integration(self, r_values, theta_values, Gamma):
        """Simpson's 1/3 rule over full domain (line integral only)."""
        if len(r_values) > 1:
            raise ValueError("Simpson's 1/3 rule is implemented for line integrals only.")
        if (len(theta_values)-1) % 2 != 0:
            print("length theta", len(theta_values))
            raise ValueError("Simpson's 1/3 rule requires an even number of subintervals.")
        r0 = r_values[0]
        dtheta = theta_values[1] - theta_values[0]
        total = 0.0
        for j in range(0, len(theta_values) - 1, 2):
            theta0 = theta_values[j]
            theta1 = theta_values[j + 1]
            theta2 = theta_values[j + 2]
            f0 = self.accel_function([r0, theta0], Gamma)
            f1 = self.accel_function([r0, theta1], Gamma)
            f2 = self.accel_function([r0, theta2], Gamma)
            total += (f0 + 4 * f1 + f2) * r0 * (dtheta * 2) / 6
        return total

    # def romberg_integration(self, r_values, theta_values, Gamma, is_theta = True, is_r = True, reference_value = None): #####
    #     """Romberg integration over full domain using trapezoidal as base method, with memoization."""
    #     # Local cache for accel_function results
    #     @lru_cache(maxsize=None)
    #     def cached_accel(r, theta):
    #         return self.accel_function([r, theta], Gamma)
    #     # Temporary wrapper to use the cache
    #     def cached_trapezoidal(r_vals, theta_vals): # memoization
    #         dtheta = theta_vals[1] - theta_vals[0]
    #         is_area = len(r_vals) > 1
    #         if is_area:
    #             dr = r_vals[1] - r_vals[0]
    #             total = 0.0
    #             for i, r in enumerate(r_vals[:-1]):
    #                 r_next = r_vals[i + 1]
    #                 dr = r_next - r
    #                 for j, theta in enumerate(theta_vals[:-1]):
    #                     theta_next = theta_vals[j + 1]
    #                     dtheta = theta_next - theta
    #                     f00 = cached_accel(r, theta) * r
    #                     f01 = cached_accel(r, theta_next) * r
    #                     f10 = cached_accel(r_next, theta) * r_next
    #                     f11 = cached_accel(r_next, theta_next) * r_next
    #                     avg = 0.25 * (f00 + f01 + f10 + f11)
    #                     total += avg * dr * dtheta
    #         else:
    #             r0 = r_vals[0]
    #             total = 0.0
    #             for j, theta in enumerate(theta_vals[:-1]):
    #                 theta_next = theta_vals[j + 1]
    #                 dtheta = theta_next - theta
    #                 f0 = cached_accel(r0, theta)
    #                 f1 = cached_accel(r0, theta_next)
    #                 total += 0.5 * (f0 + f1) * r0 * dtheta
    #         return total
    #     # Compute Romberg levels with cached trapezoidal
    #     Trap_0 = cached_trapezoidal(r_values, theta_values)
    #     if is_theta and not is_r:
    #         length_trap_2 = len(double_nodes_periodic(double_nodes_periodic(theta_values)))
    #         Trap_1 = cached_trapezoidal(r_values, double_nodes_periodic(theta_values))
    #         Trap_2 = cached_trapezoidal(r_values, double_nodes_periodic(double_nodes_periodic(theta_values)))
    #     elif is_r and not is_theta:
    #         length_trap_2 = len(double_nodes_linear(double_nodes_linear(r_values)))
    #         Trap_1 = cached_trapezoidal(double_nodes_linear(r_values), theta_values)
    #         Trap_2 = cached_trapezoidal(double_nodes_linear(double_nodes_linear(r_values)), theta_values)
    #     else:
    #         raise ValueError("Romberg integration currently requires either theta or r to be True, not both.")
    #     # Richardson extrapolation
    #     S0 = (4/3) * Trap_1 - (1/3) * Trap_0
    #     S1 = (4/3) * Trap_2 - (1/3) * Trap_1
    #     S2 = (16/15) * S1 - (1/15) * S0
    #     if reference_value is None:
    #         error = abs((S2 - S1)/S2)*100
    #     else:
    #         error = abs((S2 - reference_value)/reference_value)*100
    #     return S2, error, length_trap_2

    def romberg_integration(self, r_values, theta_values, Gamma, is_theta=True, is_r=False, reference_value=None):
        """Romberg integration over full domain using trapezoidal as base method, with memoization."""
        # Local cache for accel_function results
        @lru_cache(maxsize=None)
        def cached_accel(r, theta):
            return self.accel_function([r, theta], Gamma)
        # Optimized trapezoidal (reuses function calls across cells)
        def cached_trapezoidal(r_vals, theta_vals):
            is_area = len(r_vals) > 1
            if is_area:
                total = 0.0
                # loop over radial strips
                for i, r in enumerate(r_vals[:-1]):
                    r_next = r_vals[i + 1]
                    dr = r_next - r
                    # preload bottom edge
                    f0 = cached_accel(r, theta_vals[0]) * r
                    f1 = cached_accel(r_next, theta_vals[0]) * r_next
                    # loop over theta cells
                    for j, theta in enumerate(theta_vals[:-1]):
                        theta_next = theta_vals[j + 1]
                        dtheta = theta_next - theta
                        # only compute new right edge
                        f2 = cached_accel(r, theta_next) * r
                        f3 = cached_accel(r_next, theta_next) * r_next
                        avg = 0.25 * (f0 + f1 + f2 + f3)
                        total += avg * dr * dtheta
                        # slide in theta
                        f0, f1 = f2, f3
            else:
                # 1D case along theta
                r0 = r_vals[0]
                total = 0.0
                f0 = cached_accel(r0, theta_vals[0])
                for j, theta in enumerate(theta_vals[:-1]):
                    theta_next = theta_vals[j + 1]
                    f1 = cached_accel(r0, theta_next)
                    total += 0.5 * (f0 + f1) * r0 * (theta_next - theta)
                    f0 = f1
            return total
        # Compute Romberg levels with cached trapezoidal
        Trap_0 = cached_trapezoidal(r_values, theta_values)
        if is_theta and not is_r:
            length_trap_2 = len(double_nodes_periodic(double_nodes_periodic(theta_values)))
            Trap_1 = cached_trapezoidal(r_values, double_nodes_periodic(theta_values))
            Trap_2 = cached_trapezoidal(r_values, double_nodes_periodic(double_nodes_periodic(theta_values)))
        elif is_r and not is_theta:
            length_trap_2 = len(double_nodes_linear(double_nodes_linear(r_values)))
            Trap_1 = cached_trapezoidal(double_nodes_linear(r_values), theta_values)
            Trap_2 = cached_trapezoidal(double_nodes_linear(double_nodes_linear(r_values)), theta_values)
        else:
            raise ValueError("Romberg integration currently requires either theta or r to be True, not both.")
        # Richardson extrapolation
        S0 = (4/3) * Trap_1 - (1/3) * Trap_0
        S1 = (4/3) * Trap_2 - (1/3) * Trap_1
        S2 = (16/15) * S1 - (1/15) * S0
        if reference_value is None:
            error = abs((S2 - S1) / S2)
        else:
            error = abs((S2 - reference_value) / reference_value)
        return S2, error, length_trap_2
    
    def romberg_integration_dOmega(self, r_values, theta_values, Gamma, is_theta=True, is_r=True, reference_value=None):
        """
        Romberg integration using corrected trapezoidal_dOmega (with mapped areas/arclengths).
        """
        @lru_cache(maxsize=None)
        def cached_accel(r, theta):
            return self.accel_function([r, theta], Gamma)

        def cached_trapezoidal_dOmega(r_vals, theta_vals):
            is_area = len(r_vals) > 1
            total = 0.0
            if is_area:
                for i in range(len(r_vals) - 1):
                    r0, r1 = r_vals[i], r_vals[i + 1]
                    for j in range(len(theta_vals) - 1):
                        th0, th1 = theta_vals[j], theta_vals[j + 1]
                        # map corners
                        chi00 = complex(*hlp.r_theta_to_xy(r0, th0))
                        chi01 = complex(*hlp.r_theta_to_xy(r0, th1))
                        chi11 = complex(*hlp.r_theta_to_xy(r1, th1))
                        chi10 = complex(*hlp.r_theta_to_xy(r1, th0))
                        z00 = self.zeta_to_z(self.Chi_to_zeta(chi00), self.epsilon)
                        z01 = self.zeta_to_z(self.Chi_to_zeta(chi01), self.epsilon)
                        z11 = self.zeta_to_z(self.Chi_to_zeta(chi11), self.epsilon)
                        z10 = self.zeta_to_z(self.Chi_to_zeta(chi10), self.epsilon)
                        # cell area
                        dOmega = self.quad_area_from_complex(z00, z01, z11, z10)
                        # average of 4 corners (standard trapezoid)
                        f00 = cached_accel(r0, th0)
                        f01 = cached_accel(r0, th1)
                        f10 = cached_accel(r1, th0)
                        f11 = cached_accel(r1, th1)
                        avg = 0.25 * (f00 + f01 + f10 + f11)
                        total += avg * dOmega
                return total
            else:
                r0 = r_vals[0]
                for j in range(len(theta_vals) - 1):
                    th0, th1 = theta_vals[j], theta_vals[j + 1]
                    chi0 = complex(*hlp.r_theta_to_xy(r0, th0))
                    chi1 = complex(*hlp.r_theta_to_xy(r0, th1))
                    z0 = self.zeta_to_z(self.Chi_to_zeta(chi0), self.epsilon)
                    z1 = self.zeta_to_z(self.Chi_to_zeta(chi1), self.epsilon)
                    dS = abs(z1 - z0)
                    f0 = cached_accel(r0, th0)
                    f1 = cached_accel(r0, th1)
                    total += 0.5 * (f0 + f1) * dS
                return total

        # ---- Romberg extrapolation part unchanged ----
        Trap_0 = cached_trapezoidal_dOmega(r_values, theta_values)
        if is_theta and not is_r:
            length_trap_2 = len(double_nodes_periodic(double_nodes_periodic(theta_values)))
            Trap_1 = cached_trapezoidal_dOmega(r_values, double_nodes_periodic(theta_values))
            Trap_2 = cached_trapezoidal_dOmega(r_values, double_nodes_periodic(double_nodes_periodic(theta_values)))
        elif is_r and not is_theta:
            length_trap_2 = len(double_nodes_linear(double_nodes_linear(r_values)))
            Trap_1 = cached_trapezoidal_dOmega(double_nodes_linear(r_values), theta_values)
            Trap_2 = cached_trapezoidal_dOmega(double_nodes_linear(double_nodes_linear(r_values)), theta_values)
        else:
            raise ValueError("Romberg integration currently requires either theta or r to be True, not both.")

        S0 = (4/3) * Trap_1 - (1/3) * Trap_0
        S1 = (4/3) * Trap_2 - (1/3) * Trap_1
        S2 = (16/15) * S1 - (1/15) * S0
        if reference_value is None:
            error = abs((S2 - S1)/S2)
        else:
            error = abs((S2 - reference_value)/reference_value)
        return S2, error, length_trap_2

    def theta_convergence(self, Gamma, reference_value,fixed_r=1.0, theta_start = 10, theta_range=(0, 2*np.pi),output_dir="Grid_conv", tol=1e-3,relative_error=False,num_double_points = 8, is_compare_integrations = False):
        os.makedirs(output_dir, exist_ok=True)
        theta_converged_dict = {}
        best_N_theta_dict = {}
        if relative_error:
            header_2 = "GridPoints,RelativeError,Result,NormalizedResult"
        else:
            header_2 = "GridPoints,TrueError,Result,NormalizedResult"
        if self.is_numerical_integrand:
            if is_compare_integrations:
                method_dict = {"Riemann": self.riemann_integration_dOmega,"Trapezoidal": self.trapezoidal_integration_dOmega,"Romberg": self.romberg_integration_dOmega}#,"Simpson": self.simpson_13_integration}
            else:
                method_dict = {"Trapezoidal": self.trapezoidal_integration_dOmega}
        else:
            if is_compare_integrations:
                method_dict = {"Riemann": self.riemann_integration,"Trapezoidal": self.trapezoidal_integration,"Romberg": self.romberg_integration} 
            else:
                method_dict = {"Trapezoidal": self.trapezoidal_integration}
        print("Running theta-direction convergence study...")
        for method_name, integrator in method_dict.items():
            self.integration_func = integrator
            gridpoints, errors, results, normalized_results = [], [], [], []
            theta_converged = None
            best_N_theta = None
            error = 100.0
            previous_result = None
            theta_values = np.linspace(theta_range[0], theta_range[1], theta_start, endpoint=True)
            start_time = time.time()
            for _ in tqdm(range(num_double_points+1), desc=f"{method_name} theta convergence"):
                r_values = np.array([fixed_r])
                if method_name == "Romberg":
                    if relative_error:
                        result, error, length_trap_2 = self.appellian_acceleration_loop(Gamma, r_values, theta_values, grid_conv = True)
                    else:
                        result, error, length_trap_2 = self.appellian_acceleration_loop(Gamma, r_values, theta_values, grid_conv = True, reference_value=reference_value)
                    gridpoints.append(length_trap_2)
                    errors.append(error)
                    results.append(result)
                else:
                    result = self.appellian_acceleration_loop(Gamma, r_values, theta_values, grid_conv = True)
                    if relative_error:
                        if previous_result is not None:
                            error = abs((result - previous_result) / result) 
                    else:
                        error = abs((reference_value-result) / reference_value) 
                    gridpoints.append(len(theta_values))
                    errors.append(error)
                    results.append(result)
                    normalized_result = result/(self.freestream_velocity**4)
                    normalized_results.append(normalized_result)
                previous_result = result
                theta_values = double_nodes_periodic(theta_values)
            theta_converged_dict[method_name] = theta_converged
            best_N_theta_dict[method_name] = best_N_theta
            end_time = time.time()
            total_time = end_time - start_time
            arr = np.column_stack((gridpoints, errors, results, normalized_results))
            filename = os.path.join(output_dir,f"{method_name.lower()}_thetaconv_D_{round(self.D, 3)}"f"_zeta0_r_{round(self.zeta_center.real, 3)}"f"_zeta0_i_{round(self.zeta_center.imag, 3)}_time_{round(total_time, 3)}s_.csv")
            np.savetxt(filename, arr, delimiter=",", header=header_2, comments='')
       
    def r_convergence(self, Gamma, reference_value, num_r_points_start, theta_start, theta_converged_doubles, theta_range = (0.0, 2*np.pi), r_range=(1.0, 350.0), output_dir="Grid_conv", tol=1e-3, relative_error=False, num_double_points=8, is_compare_integrations=False):
        os.makedirs(output_dir, exist_ok=True)
        if relative_error:
            header_2 = "GridPoints,RelativeError,Result,NormalizedResult"
        else:
            header_2 = "GridPoints,TrueError,Result,NormalizedResult"
        if self.is_numerical_integrand:
            if is_compare_integrations:
                method_dict = {"Riemann": self.riemann_integration_dOmega,"Trapezoidal": self.trapezoidal_integration_dOmega,"Romberg": self.romberg_integration_dOmega}#,"Simpson": self.simpson_13_integration}
            else:
                method_dict = {"Trapezoidal": self.trapezoidal_integration_dOmega}
        else:
            if is_compare_integrations:
                method_dict = {"Riemann": self.riemann_integration,"Trapezoidal": self.trapezoidal_integration,"Romberg": self.romberg_integration}#,"Simpson": self.simpson_13_integration}
            else:
                method_dict = {"Trapezoidal": self.trapezoidal_integration}
        print("Running radial-direction convergence study...")
        theta_values = np.linspace(theta_range[0], theta_range[1], theta_start, endpoint=True)
        for i in range(theta_converged_doubles):
            theta_intermediate = double_nodes_periodic(theta_values)
            theta_values = theta_intermediate
        thetas = theta_values
        print("len thetas", len(thetas))
        for method_name, integrator in method_dict.items():
            self.integration_func = integrator
            gridpoints, errors, results, normalized_results = [], [], [], []
            converged = False
            error = 100.0
            previous_result = None
            # r_values = np.linspace(r_range[0], r_range[1], 10, endpoint=True)
            first_r_spacing = self.calc_deltaD_0(num_r_points_start, self.r_growth_factor, total_distance=self.radial_distance-self.cylinder_radius)
            # Compute r_values via cumulative sum of geometrically growing intervals
            intervals = first_r_spacing * cyl.r_growth_factor ** np.arange(num_r_points_start - 1)
            r_values = self.cylinder_radius + np.concatenate(([0.0], np.cumsum(intervals)))
            time_start = time.time()
            for _ in tqdm(range(num_double_points), desc=f"{method_name} radial convergence"):
                if method_name == "Romberg":
                    if relative_error:
                        result, error, length_trap_2 = self.appellian_acceleration_loop(Gamma, r_values, thetas, grid_conv = True)
                    else:
                        result, error, length_trap_2 = self.appellian_acceleration_loop(Gamma, r_values, thetas, grid_conv = True, reference_value=reference_value)
                    gridpoints.append(length_trap_2)
                    errors.append(error)
                    results.append(result)
                else:
                    result = self.appellian_acceleration_loop(Gamma, r_values, thetas, grid_conv = True)
                    if relative_error:
                        if previous_result is not None:
                            error = abs((result - previous_result) / previous_result) * 100
                    else:
                        error = abs((result - reference_value) / reference_value) * 100
                    gridpoints.append(len(r_values))
                    errors.append(error)
                    results.append(result)
                    normalized_result = result/(self.freestream_velocity**4)
                    normalized_results.append(normalized_result)
                previous_result = result
                r_values = double_nodes_linear(r_values)
            arr = np.column_stack((gridpoints, errors, results))
            time_end = time.time()
            total_time = time_end-time_start
            filename = os.path.join(output_dir,f"{method_name.lower()}_radialconv_growth_{round(self.r_growth_factor, 3)}_radial_distance_{round(self.radial_distance-self.cylinder_radius, 3)}_D_{round(self.D, 3)}"f"_zeta0_r_{round(self.zeta_center.real, 3)}"f"_zeta0_i_{round(self.zeta_center.imag, 3)}_time_{round(total_time, 3)}s_.csv")
            np.savetxt(filename, arr, delimiter=",", header=header_2, comments='')

    def calculate_aft_stagnation_theta_in_Chi_from_Gamma(self, Gamma: float): # from Nate's paper
        """This function calculates the aft stagnation point angle in Chi from the circulation"""
        return self.angle_of_attack - np.arcsin(Gamma/(4*np.pi*self.freestream_velocity*self.cylinder_radius))
    
    def calculate_forward_stag_theta_in_Chi_from_aft_stag_theta(self, aft_stag_theta: float):
        """This function calculates the forward stagnation point angle in Chi from the aft stagnation point angle"""
        return np.pi - aft_stag_theta + 2*self.angle_of_attack
        
    def calculate_forward_and_aft_stag_locations_in_z(self, aft_stag_angle: float):
        """This function calculates the forward and aft stagnation points in z coordinates given the aft stagnation angle in Chi"""
        forward_stag_angle = self.calculate_forward_stag_theta_in_Chi_from_aft_stag_theta(aft_stag_angle)
        # convert the stagnation angles to z coordinates 
        r = self.cylinder_radius
        # convert r and theta in Chi to x and y coordinates
        x_aft, y_aft = hlp.r_theta_to_xy(r, aft_stag_angle)
        x_forward, y_forward = hlp.r_theta_to_xy(r, forward_stag_angle)
        # convert the x and y coordinates to zeta coordinates using the Chi_to_zeta function
        Chi_aft = x_aft + 1j*y_aft
        Chi_forward = x_forward + 1j*y_forward
        zeta_aft = self.Chi_to_zeta(Chi_aft)
        zeta_forward = self.Chi_to_zeta(Chi_forward)
        z_aft = self.zeta_to_z(zeta_aft, self.epsilon)
        z_forward = self.zeta_to_z(zeta_forward, self.epsilon)
        # forward_stag_normal_vector = self.surface_normal(zeta_forward.real)[0]  # get the upper surface normal vector at the forward stagnation point
        forward_stag = np.array([z_forward.real, z_forward.imag]) 
        aft_stag = np.array([z_aft.real, z_aft.imag])
        return forward_stag, aft_stag
    
    def numerically_integrate_appellian(self, Gamma: float, r_values: np.array, theta_vals):
        """This function calculates the Appellian function numerically requires evenly spaced ranges for Gamma, r, and theta in Chi"""
        # calculate the area element 
        if len(r_values) == 1:
            dr = 0.01
        else:        
            dr = r_values[1] - r_values[0]
        # now these are the dr and dtheta values in the z plane for the numerical integration in after the else statement
        if self.appellian_is_area_integral:
            Appellian_array = self.appellian_acceleration_loop(Gamma, r_values, theta_vals)            
        else: # line integral
            r_values = np.array([self.cylinder_radius])
            Appellian_array = self.appellian_acceleration_loop(Gamma, r_values, theta_vals)
        return Appellian_array

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

    def z_to_zeta(self, z: complex, epsilon: float, D=0.0): # eq 104 
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

    def calc_J_airfoil_epsilon(self):
        """This function calculates the epsilon of the airfoil"""
        epsilon = self.cylinder_radius - 1*np.sqrt(self.cylinder_radius**2-self.zeta_center.imag**2)-self.zeta_center.real # eq 113 in complex variables
        return epsilon
    
    def calc_circulation_J_airfoil(self):
        """This function calculates the circulation of the airfoil based on the Kutta condition"""
        gamma = 4*np.pi*self.freestream_velocity*(np.sin(self.angle_of_attack)*np.sqrt(self.cylinder_radius**2 - self.zeta_center.imag**2) + self.zeta_center.imag*np.cos(self.angle_of_attack)) # eq 122 in complex variables
        return gamma

    def general_kutta_condition(self): ### chat based equation
        """This function calculates circulation based on general aft singularity location"""
        R, V_inf, alpha, pie, epsilon, xi0, eta0 = self.cylinder_radius, self.freestream_velocity, self.angle_of_attack, np.pi, self.epsilon, self.zeta_center.real, self.zeta_center.imag
        xi1 = R-epsilon-np.sqrt(R**2-eta0**2)
        eta1 = eta0
        zeta0 = xi0 + 1j*eta0
        zeta1 = xi1 + 1j*eta1
        zeta_point = R-epsilon 
        e_ialpha = np.exp(1j*alpha)
        e_neg_ialpha = np.exp(-1j*alpha)
        Gamma = 2*pie*V_inf/1j * ((R**2*e_ialpha)/(zeta_point-zeta1)-e_neg_ialpha*(zeta_point-zeta1))
        gamma_real = np.real(Gamma)
        print("gamma_real", gamma_real)
        gamma_imag = np.imag(Gamma)
        print("gamma_imag", gamma_imag)
        gamma_total = np.sqrt(gamma_real**2 + gamma_imag**2)
        print("gamma_total", gamma_total)
        return gamma_total

    def run_appellian_roots(self, r_vals, theta_vals, D):
        """This function runs the Appellian stuff"""
        Gamma_vals = np.linespace(0, self.kutta_circulation, 6, endpoint=True) # if kutta circulation is 10, gamma vals will be [0, 2, 4, 6, 8, 10]
        poly_order = 4
        appellian_root, appellian_value = hlp.polyfit(self.numerically_integrate_appellian, r_vals, theta_vals, Gamma_vals, poly_order, self.is_plot_appellian, "$\\Gamma$", "S", "Appellian Function", D)
        return appellian_root, appellian_value
    
    def calc_gamma_app_over_gamma_kutta(self, D_values, r_values, theta_values):
        kutta_appellian_array = np.zeros((len(D_values), 2))
        for i in tqdm(range(len(D_values)), desc="Calculating Appellian Roots"):  
            self.D = D_values[i]
            self.epsilon = self.calc_epsilon_from_D(D_values[i]) ##### before
            kutta_appellian_array[i, 0] = D_values[i]
            root = self.run_appellian_roots(r_values,theta_values, self.D)[0]
            kutta_appellian_array[i, 1] = root / self.kutta_circulation  
        text_file_name = "text_files/dsweep/dsweep_zeta0_" + str(cyl.zeta_center.real) + "_" + str(cyl.zeta_center.imag) + "_alpha_" + str(round(cyl.angle_of_attack*180/np.pi, 1)) + ".txt"
        np.savetxt(text_file_name, kutta_appellian_array, delimiter=",", header="D, Gamma_A/Gamma_K", comments="")
        return kutta_appellian_array
    
    def plot_gamma_over_gamma_kutta(self, kutta_appellian_array, is_plot_current_D=False, kutta_and_current_D=[0,0], label = "$\\Gamma_A$/$\\Gamma_K$", linestyle = "solid", is_first_file = True):
        alpha_name = str(round(self.angle_of_attack*180/np.pi, 1)) + " [deg]"
        if not self.is_create_dsweep_gif:
            plt.plot(kutta_appellian_array[:,0], kutta_appellian_array[:,1], label=label, color = "black", linestyle=linestyle)
            # plot a line at y = 1
            tightly_dash_dotted = (0, (3, 1, 1, 1))  # tightly dashdotted
            plt.axhline(1, color='gray', linestyle=tightly_dash_dotted)
            # plt.plot(kutta_appellian_array[:,0], kutta_appellian_array[:,1], label="$\\alpha$ = " + alpha_name, color = "black")
        else:
            plt.plot(kutta_appellian_array[:,0], kutta_appellian_array[:,1], label="$\\alpha$ = " + alpha_name, color="black")
        plt.xlabel("$D$")
        # plt.ylabel("$\\frac{\\Gamma_A}{\\Gamma_K}$")
        # change ylabel size
        plt.ylabel("$\\frac{\\Gamma_A}{\\Gamma_K}$", fontsize=22)
        # Set the y-axis label rotation
        plt.gca().yaxis.label.set_rotation(0)
        # plt.xscale("log")
        # Adjust axis limits
        plt.axhline(0, color='gray')
        # plt.xlim(0.001, 1.0)  # Lower limit should be slightly less than 0.005
        xticks = np.array([0.0,0.2,0.4,0.6,0.8,1.0])
        xtick_labels = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
        plt.xticks(xticks, xtick_labels)
        plt.xlim(0.0, 1.0)  # Lower limit should be slightly less than 0.005
        # insert text in the middle of the plot right below y = 1.0 that says "$\\Gamma_K$/$\\Gamma_k$"
        if is_first_file:
            plt.text(0.5, 0.9, "$\\Gamma_K/\\Gamma_K$", fontsize=17, ha='center', va='bottom')
        # insert an arrow pointing to the line at y = 1
        # Move the first x-tick label to the right by 0.1
        x_tick_labels = [f"{(tick)}" for tick in xtick_labels]  # Format tick labels as integers
        x_tick_labels[0] = f"    {x_tick_labels[0]}"  # Add spaces to move the first label
        plt.gca().set_xticklabels(x_tick_labels)
        yticks = np.array([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        ytick_labels = ["-0.2", "0.0", "0.2", "0.4", "0.6", "0.8", "1.0", "1.2"]
        plt.yticks(yticks, ytick_labels)
        plt.ylim(-0.2, 1.2)
        # Adjust label padding
        plt.gca().xaxis.labelpad = 1
        plt.gca().yaxis.labelpad = 5
        #include text in the top middle
        y_range = plt.ylim()[1] - plt.ylim()[0]
        y_offset = 0.05 * y_range
        if is_plot_current_D:
            plt.scatter(kutta_and_current_D[0], kutta_and_current_D[1], color="red", s = 40)
        figure_name = "figures/dsweep_zeta0_" + str(cyl.zeta_center.real) + "_" + str(cyl.zeta_center.imag) + "_alpha_" + str(round(cyl.angle_of_attack*180/np.pi, 1)) + ".svg"
        return figure_name
    
    def calc_convergence_to_kutta(self, D_values, r_values, theta_values, zeta_center):
        """"""
        original_zeta_center = self.zeta_center
        original_kutta_circulation = self.kutta_circulation
        self.zeta_center = zeta_center
        kutta_appellian_array = np.zeros((len(D_values), 2))
        for i in tqdm(range(len(D_values)), desc="Calculating Appellian Roots"):  
            self.D = D_values[i]
            self.epsilon = self.calc_epsilon_from_D(D_values[i])
            self.kutta_circulation = self.calc_circulation_J_airfoil()
            kutta_appellian_array[i, 0] = D_values[i]
            kutta_appellian_array[i, 1] = abs((self.run_appellian_roots(r_values,theta_values, D_values[i])[0] - self.kutta_circulation)/self.kutta_circulation)*100 # percent error 
        self.zeta_center = original_zeta_center
        self.kutta_circulation = original_kutta_circulation
        zeta_center_name = str(zeta_center.real) + "_" + str(zeta_center.imag) + "_"
        file_name = "text_files/convergence_to_kutta/zeta0_" + zeta_center_name + "_alpha_" + str(round(self.angle_of_attack*180/np.pi, 1))+".txt"
        np.savetxt(file_name, kutta_appellian_array, delimiter = ",", header = "D, Appellian Root")
        return kutta_appellian_array

    def plot_convergence_to_kutta(self, filenames):
        """This function plots the difference between the Appellian root and the Kutta circulation for different D values and different zeta centers"""
        linestyles = ['solid','dashed', (0, (3, 1, 1, 1)), 'solid','dashed', (0, (3, 1, 1, 1))]
        colors = ['black', 'black', 'gray', 'gray', 'black', 'black']
        index = 0
        for filename in filenames:
            data = np.loadtxt(filename, delimiter=",", skiprows=1)
            plt.plot(data[1:, 0], data[1:, 1], 
                    label="$\\zeta_0 /R$ = " + str(filenames[index]).replace("(", "").replace(")", "").replace("text_files/convergence_to_kutta\zeta0_","").replace("_alpha_5.0.txt", "").replace("0.0_", "0.0i").replace("_", "+"),
                    linestyle=linestyles[index % len(linestyles)], 
                    color=colors[index % len(colors)], 
                    linewidth=1.5)
            index += 1
        plt.gca().invert_xaxis() # invert x-axis
        plt.xscale("log")
        xticks = np.array([0.0001, 0.001, 0.01, 0.1])
        xtick_labels = ["1e-4", "1e-3", "1e-2", "1e-1"]
        plt.xticks(xticks, xtick_labels)
        plt.xlim(0.1, 0.0001)  # Lower limit should be slightly less than 0.005
        x_tick_labels = [f"{(tick)}" for tick in xtick_labels]  # Format tick labels as integers
        x_tick_labels[-1] = f"      {x_tick_labels[-1]}"  # Add spaces to move the first label
        plt.gca().set_xticklabels(x_tick_labels)
        plt.yscale("log")
        plt.yticks([1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2], ["1e-8", "1e-6", "1e-4", "1e-2", "1e0", "1e2"])
        plt.ylim(1e-8, 1e2)
        plt.xlabel("$D$")
        plt.ylabel("$\\Gamma  \\%$ Error", labelpad=15)
        plt.axhline(0, color='gray')
        plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.15), ncol=2,frameon=False)

    def shift_joukowski_cylinder(self):
        """This function finds shifts all the points back to where zeta_center = 0, finds how far to the left the leading edge is, and shifts all points by that much (so the leading edge is at 0)
        Then it finds the right most point, and finds the length from the right most point to the left most point, and scales all points by that much"""
        self.shifted_z_surface = np.copy(self.full_z_surface)
        # shift the coordinates so that the zeta_center is at the origin
        self.shifted_z_surface[:,0], self.shifted_z_surface[:,1] = self.full_z_surface[:,0] - self.zeta_center.real, self.full_z_surface[:,1] - self.zeta_center.imag
        # now find the left most point in the upper coords and shift all points by that much
        self.leading_edge = np.min(self.shifted_z_surface[:,0])
        self.shifted_z_surface[:,0] = self.shifted_z_surface[:,0] - self.leading_edge
        # now find the right most point in the upper coords and scale all points by that much
        self.trailing_edge = np.max(self.shifted_z_surface[:,0])
        self.shifted_z_surface = self.shifted_z_surface/self.trailing_edge
        return self.shifted_z_surface
    
    def export_shifted_joukowski_cylinder(self):
        """This function exports the shifted Joukowski cylinder coordinates to a text file which starts at the trailing edge, wraps from the bottom surface over the upper surface and back to the trailing edge"""
        # ensure the output directory exists
        os.makedirs("text_files", exist_ok=True)
        # save the shifted upper coords, lower coords, and camber coords to a text file
        np.savetxt("text_files/shifted_z_coords.txt", self.shifted_z_surface, delimiter=",", header="x,y")

    def plot_shifted_joukowski_cylinder(self):
        """This function plots the shifted Joukowski cylinder"""
        # plot the upper and lower coords
        plt.scatter(self.shifted_z_surface[:,0], self.shifted_z_surface[:,1], color='black', s = 2)
        # close the plot

    def calculate_surface_pressure(self, point_xi_eta_in_zplane, Gamma):
        """This function calculates the surface pressure at a point using the velocity at that point"""
        velocity = self.velocity(point_xi_eta_in_zplane, Gamma)
        V_squared = np.dot(velocity, velocity)
        pressure = 1-(V_squared/self.freestream_velocity**2)
        return pressure
    
    def given_geom_and_stag_thetas_calc_top_and_bottom_coords(self, theta_aft, theta_forward, zeta_geom_array):
        """This function calculates the upper and lower coordinates of the Joukowski cylinder given the geometry in zeta coordinates"""
        # any points between theta_start and theta_half that are within the first half of zeta_geom_array are upper coords, the rest are lower coords
        self.upper_coords = zeta_geom_array[(zeta_geom_array[:, 2] >= theta_aft) & (zeta_geom_array[:, 2] <= theta_forward), :2] # the theta aft is included, the theta forward is not included
        self.lower_coords = zeta_geom_array[(zeta_geom_array[:, 2] > theta_forward) & (zeta_geom_array[:, 2] <= 2*np.pi + theta_aft), :2] # theta aft is not included, theta forward is included
        return self.upper_coords, self.lower_coords
    
    def given_geom_zeta_upper_lower_get_z_upper_lower(self, upper_coords, lower_coords):
        """"""
        z_upper = np.zeros((len(upper_coords), 2))
        z_lower = np.zeros((len(lower_coords), 2))
        for i in range(len(upper_coords)):
            zeta_upper = upper_coords[i, 0] + 1j* upper_coords[i, 1]  # Convert to complex number
            z_upper_comp = cyl.zeta_to_z(zeta_upper, self.epsilon)
            z_upper[i] = [z_upper_comp.real, z_upper_comp.imag]
        # order z_upper in increasing order of x-coordinate
        z_upper = z_upper[np.argsort(z_upper[:, 0])]  # Sort by x-coordinate
        for j in range(len(lower_coords)):
            zeta_lower = lower_coords[j, 0] + 1j* lower_coords[j, 1]
            z_lower_comp = cyl.zeta_to_z(zeta_lower, self.epsilon)
            z_lower[j] = [z_lower_comp.real, z_lower_comp.imag]
        # order z_lower in decreasing order of x-coordinate
        z_lower = z_lower[np.argsort(z_lower[:, 0])[::-1]]  # Sort by x-coordinate in reverse order
        self.z_upper = z_upper
        self.z_lower = z_lower
        for i in range(len(z_upper)):
            print("z_upper", i, ":", z_upper[i, 0], z_upper[i, 1])
        for j in range(len(z_lower)):
            print("z_lower", j, ":", z_lower[j, 0], z_lower[j, 1])
        return z_upper, z_lower

    def calc_deltaD_0(self, N, radial_growth, total_distance):
        """
        Calculate the initial radial spacing deltaD_0 such that N radial points
        (i.e., N-1 intervals) with geometric growth span a given total distance.
        """
        if np.isclose(total_distance, 0.0):
            print("Distance is zero, returning 0.0 for deltaD_0.")
            return 0.0

        if np.isclose(radial_growth, 1.0):
            return total_distance / (N - 1)

        # Sum of geometric series with N-1 terms (i.e., N points → N-1 intervals)
        sum_geom = (1 - radial_growth ** (N - 1)) / (1 - radial_growth)
        return total_distance / sum_geom

    def run_appellian_vals_for_Gammas(self, theta_range: np.ndarray = np.array([0.0, 2 * np.pi]), theta_converged_doubles: int = 5, folder_location: str = "Grid_conv/figures/Gamma_selection/", relative_error: str = True, reference_value: float = None):
        if self.appellian_is_line_integral == False:
            raise ValueError("The run_appellian_roots_for_Gammas function is only equipped to handle line integrals. Adjust input accordingly")
        kutta_Gamma = self.calc_circulation_J_airfoil()  # calculate the circulation of the airfoil
        Gamma_vals = np.linspace(0, kutta_Gamma, 6, endpoint=True) # if kutta_gamma = 10, Gamma_vals would be [0, 2, 4, 6, 8, 10]
        r_values = np.array([self.cylinder_radius])
        plt.figure()
        for i in range(len(Gamma_vals)):
            Gamma = Gamma_vals[i]
            theta_values = np.linspace(theta_range[0], theta_range[1], self.num_thetas_start, endpoint=True)
            # make an array which fits all of the theta densities in the x column, the relative error in the y column, and the result from the appellian in the z column
            appellian_array = np.zeros((theta_converged_doubles, 3))
            previous_result = None
            error = 100.0
            for j in range(theta_converged_doubles):
                if self.integration_func == self.romberg_integration:
                    if relative_error:
                        result, error, length_trap_2 = self.appellian_acceleration_loop(Gamma, r_values, theta_values, grid_conv = True)
                    else:
                        result, error, length_trap_2 = self.appellian_acceleration_loop(Gamma, r_values, theta_values, grid_conv = True, reference_value=reference_value)
                    # print("result, error, length_trap_2", result, error, length_trap_2)
                    appellian_array[j, 0] = length_trap_2 # number of theta points
                    appellian_array[j, 1] = error
                    appellian_array[j, 2] = result
                else:
                    result = self.appellian_acceleration_loop(Gamma, r_values, theta_values, grid_conv = True)
                    if relative_error:
                        if previous_result is not None:
                            error = abs((result - previous_result) / result) * 100
                    else:
                        error = abs((reference_value-result) / reference_value) * 100
                    appellian_array[j, 0] = len(theta_values)  # number of theta points
                    appellian_array[j, 1] = error
                    appellian_array[j, 2] = result
                theta_intermediate = double_nodes_periodic(theta_values)
                theta_values = theta_intermediate
                previous_result = result
            # write results to a file
            file_name = folder_location + "Gamma_" + str(round(Gamma_vals[i], 3)) + "_D_" + str(self.D) + "_int_method_" + str(self.integration_method) + "_zeta0_" + str(self.zeta_center.real) + "_" + str(self.zeta_center.imag) + "_alpha_" + str(round(self.angle_of_attack*180/np.pi, 1)) + ".txt"
            np.savetxt(file_name, appellian_array, header="points, relative_error, appellian")

    def plot_appellian_vals_for_Gammas(
        self,
        folder_location: str = "Grid_conv/figures/Gamma_selection/",
        D_value: float = None,
        integration_method: str = None,
    ):
        import glob
        import re

        # Pattern to match all files for this D value (any Gamma)
        pattern = f"Gamma_*_D_{D_value}_*.txt"
        files = sorted(glob.glob(os.path.join(folder_location, pattern)))
        if not files:
            print("No files found for pattern:", pattern)
            return

        # Extract Gamma values and sort
        gamma_file_pairs = []
        for file in files:
            match = re.search(r"Gamma_([0-9\.]+)_D_", os.path.basename(file))
            if match:
                gamma_val = float(match.group(1))
                gamma_file_pairs.append((gamma_val, file))
            else:
                gamma_file_pairs.append((float('inf'), file))  # Put unparseable at end

        gamma_file_pairs.sort()  # Sort by Gamma value

        plt.figure()
        for gamma_val, file in gamma_file_pairs:
            data = np.loadtxt(file, skiprows=1)
            label = f"$\\Gamma$={gamma_val}" if gamma_val != float('inf') else "?"
            plt.plot(data[:, 0], data[:, 1], label=label)

        plt.xlabel("Theta Points")
        plt.ylabel("Relative Error (%)")
        plt.title(f"D: {D_value}")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(title="Gamma")
        plt.grid()
        plot_name = os.path.join(folder_location, f"D_{D_value}_allGammas.svg")
        plt.savefig(plot_name)
        plt.close()
        print("Saved plot:", plot_name)


def double_nodes_periodic(theta_vals: np.ndarray) -> np.ndarray:
    """
    Doubles node count for theta from 0 to 2π by inserting midpoints.
    Assumes:
      - Input is sorted ascending
      - First value is 0
      - Last value is 2π
    """
    thetas_new = []
    for i in range(len(theta_vals) - 1):
        t0 = theta_vals[i]
        t1 = theta_vals[i + 1]
        midpoint = 0.5 * (t0 + t1)
        thetas_new.extend([t0, midpoint])
    thetas_new.append(theta_vals[-1])  # ensure final 2π is included
    return np.array(thetas_new)

def double_nodes_linear(x_vals: np.ndarray) -> np.ndarray:
    """Doubles the number of linear nodes by adding midpoints."""
    x_new = []
    for i in range(len(x_vals) - 1):
        x0, x1 = x_vals[i], x_vals[i + 1]
        midpoint = 0.5 * (x0 + x1)
        x_new.extend([x0, midpoint])
    x_new.append(x_vals[-1])  # Add last point
    return np.array(sorted(set(x_new)))

if __name__ == "__main__":
    print("")
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
    subdict = {"figsize" : (3.25,3.5),"constrained_layout" : True,"sharex" : True}
    cyl = cylinder("Joukowski_Cylinder.json")
    if cyl.appellian_is_line_integral:
        r_values = np.array([cyl.cylinder_radius])
    else:
        r_values = np.linspace(cyl.cylinder_radius, cyl.radial_distance*cyl.cylinder_radius, cyl.one_appellian_r_points, endpoint=True)  # r range from cyl.cylinder_radius to 1.01*cyl.cylinder_radius with num_r_steps steps
        if cyl.one_appellian_r_points > 1:
            first_r_spacing = cyl.calc_deltaD_0(cyl.one_appellian_r_points, cyl.r_growth_factor, total_distance=cyl.radial_distance-cyl.cylinder_radius)
            # Compute r_values via cumulative sum of geometrically growing intervals
            intervals = first_r_spacing * cyl.r_growth_factor ** np.arange(cyl.one_appellian_r_points - 1)
            r_values = cyl.cylinder_radius + np.concatenate(([0.0], np.cumsum(intervals)))
        else:
            r_values = np.array([cyl.cylinder_radius])
    theta_start = 0.0#cyl.theta_AJA  # this is the angle in Chi where the geometry starts
    theta_end = 2*np.pi #+ theta_start  # this is the angle in Chi where the geometry ends
    theta_values = np.linspace(theta_start, theta_end, cyl.one_appellian_theta_points, endpoint=True)  # theta range from theta_start to theta_end with num_theta_steps steps
    if cyl.is_compute_minimum_appellian:
        Gamma = cyl.calc_circulation_J_airfoil()  # calculate the circulation of the airfoil
        time_start = time.time()
        appellian_root, appellian_value = cyl.run_appellian_roots(r_values, theta_values, cyl.D) # calculate the minimum Appellian value
        print("Gamma Appellian: ", appellian_root)
        print("Appellian value", appellian_value)
    if cyl.is_compute_appellian:
        # print("running Appellian roots for cylinder with D = ", cyl.D)
        if cyl.is_compute_minimum_appellian:
            Gamma = appellian_root
        else:
            Gamma = cyl.calc_circulation_J_airfoil()
        time_start = time.time()
        appellian_value = cyl.appellian_acceleration_loop(Gamma, r_values, theta_values)
        time_end = time.time()
        print("D: ", cyl.D)
        print("Gamma_used_to_compute_Appellian: ", Gamma)
        print("num_r_vals: ", len(r_values))
        print("num_theta_vals: ", len(theta_values))
        print("Appellian_calculation_time: ", time_end - time_start, " seconds")
        print("Appellian_Value: ", appellian_value)
    if cyl.is_calc_circulation_for_varying_shape:
        D_range_first = [0.0001,0.0009, 0.0001]
        D_values_first = hlp.list_to_range(D_range_first)
        D_range_second = [0.001, 0.009, 0.001]
        D_values_second = hlp.list_to_range(D_range_second)
        D_range_third = [0.01, 1.00, 0.01]
        D_values_third = hlp.list_to_range(D_range_third)
        D_values = np.concatenate((D_values_second, D_values_third), axis=0)
        print("number of D values: ", len(D_values))
        kutta_appellian_array = cyl.calc_gamma_app_over_gamma_kutta(D_values, r_values, theta_values)
        if cyl.is_plot_circulation_for_varying_shape:
            cyl.plot_gamma_over_gamma_kutta(kutta_appellian_array, is_plot_current_D = False, kutta_and_current_D = [kutta_appellian_array[0,0], kutta_appellian_array[0,1]])
            plt.savefig("figures/dsweep_zeta0_" + str(cyl.zeta_center.real) + "_" + str(cyl.zeta_center.imag) + "_alpha_" + str(round(cyl.angle_of_attack*180/np.pi, 1)) + ".svg", dpi=300, bbox_inches=None, pad_inches=0)
        if cyl.is_plot_D_sweep_alpha_considerations:
            for i in range(len(cyl.alphas_for_consideration)):
                cyl.angle_of_attack = cyl.alphas_for_consideration[i]
                cyl.kutta_circulation = cyl.calc_circulation_J_airfoil()
                kutta_appellian_array = cyl.calc_gamma_app_over_gamma_kutta(D_values, r_values, theta_values)
                cyl.plot_gamma_over_gamma_kutta(kutta_appellian_array, is_plot_current_D = False, kutta_and_current_D = [kutta_appellian_array[0,0], kutta_appellian_array[0,1]])
            plt.legend()
            plt.savefig("figures/dsweep_zeta0_" + str(cyl.zeta_center.real) + "_" + str(cyl.zeta_center.imag) + "_alpha_" + str(round(cyl.angle_of_attack*180/np.pi, 1)) + ".svg", dpi=300, bbox_inches=None, pad_inches=0)
        plt.figure()
        with open(cyl.cyl_json_file, 'r') as json_handle:
            input = json.load(json_handle)
            cyl.D = hlp.parse_dictionary_or_return_default(input, ["geometry", "shape_parameter_D"], cyl.D)
            if cyl.use_shape_parameter_D:
                cyl.epsilon = cyl.calc_epsilon_from_D(cyl.D)
            else:
                cyl.epsilon = input["geometry"]["epsilon"]
    if cyl.is_plot_circulation_for_varying_shape and not cyl.is_calc_circulation_for_varying_shape:
        file_location = "text_files/dsweep/"
        filename_index = 0
        for filename in os.listdir(file_location):
            if filename.endswith(".txt"):
                kutta_appellian_array = np.loadtxt(os.path.join(file_location, filename), delimiter=",", skiprows=1)
                # if the file name ends with line_int.txt, make the label "Line Integral"
                if filename.endswith("line_int.txt"):
                    label = "Line Integral"
                    linestyle = "solid"
                else:
                    label = "Area Integral"
                    linestyle = "dashed"
                appellian_array = np.loadtxt(os.path.join(file_location, filename), delimiter=",", skiprows=1)
                if filename_index == 0:
                    figure_name = cyl.plot_gamma_over_gamma_kutta(kutta_appellian_array, is_plot_current_D = False, kutta_and_current_D = [kutta_appellian_array[0,0], kutta_appellian_array[0,1]], label = label, linestyle = linestyle, is_first_file = True)
                else:
                    figure_name = cyl.plot_gamma_over_gamma_kutta(kutta_appellian_array, is_plot_current_D = False, kutta_and_current_D = [kutta_appellian_array[0,0], kutta_appellian_array[0,1]], label = label, linestyle = linestyle, is_first_file = False)
                filename_index += 1
        plt.legend()
        plt.savefig(figure_name, dpi=300, bbox_inches=None, pad_inches=0)
        print("Curvature sweep figure saved as:", figure_name)
        plt.close()
    cyl.get_full_geometry_zeta(number_of_points=cyl.output_points, theta_start = 2*np.pi, theta_half = np.pi, theta_end = 0.0)
    cyl.plot_geometry_settings()
    # cyl.plot_geometry_zeta() ####
    cyl.get_full_geometry()
    cyl.plot_geometry()
    # zeta_trailing_edge_focus, zeta_leading_edge_focus, z_leading_edge_focus, z_trailing_edge_focus = cyl.get_and_plot_foci() ####
    if cyl.is_plot_streamlines:
        streamlines = cyl.calc_streamlines()
        cyl.plot(streamlines)
        plt.gca().set_aspect('equal', adjustable='box')
        if cyl.show_fig:
            plt.show()
    elif not cyl.is_plot_streamlines and cyl.show_fig:
        plt.show()
    if cyl.is_plot_shifted_joukowski_cylinder:
        cyl.shift_joukowski_cylinder()
        cyl.plot_shifted_joukowski_cylinder()
    cyl.plot_geometry_settings()
    # calculate theta and r convergence ####

    # running Gamma convergence 
    if cyl.is_calc_Gamma_convergence:
        print("Calculating Gamma convergence for a sweep of D values")
        D_vals = np.array([0.001, 0.2, 0.4, 0.6, 0.8, 1.0])
        for i in tqdm(range(len(D_vals)), desc="D values (Gamma conv)"):
            cyl.D = D_vals[i]
            cyl.epsilon = cyl.calc_epsilon_from_D(cyl.D)
            appellian_values = cyl.calculate_appellian_vals_for_Gammas()
    
    if cyl.is_plot_Gamma_convergence:
        plt.close()
        D_vals = np.array([0.001, 0.2, 0.4, 0.6, 0.8, 1.0])
        kutta_Gamma = cyl.calc_circulation_J_airfoil()
        Gamma_vals = np.linspace(0, kutta_Gamma, 6, endpoint=True)
        for i in range(len(D_vals)):
            cyl.D = D_vals[i]
            cyl.epsilon = cyl.calc_epsilon_from_D(cyl.D)
            appellian_values = cyl.plot_appellian_vals_for_Gammas(D_value = cyl.D, integration_method = cyl.integration_method)

    if cyl.is_calc_theta_convergence:
        print("Running theta convergence")
        # Conv_d_values = np.array([cyl.D])
        Conv_d_values = np.array([0.001, 0.01, 0.1, 0.2, 0.4, 0.8, 1.0])
        romberg_reference_value = 90572.7333392366 # for 17 doubles area int romberg
        Gamma_value = cyl.kutta_circulation
        reference_value = romberg_reference_value # for D=0.001 trap 17 and area int analytic with mult
        for i in range(len(Conv_d_values)):
            cyl.D = Conv_d_values[i]
            cyl.epsilon = cyl.calc_epsilon_from_D(cyl.D)
            theta_start = 0.0
            theta_end = 2*np.pi
            theta_range = (theta_start, theta_end)
            tol = 1e-20
            is_relative_error = cyl.grid_conv_is_relative_error
            is_compare_integrations = False
            cyl.theta_convergence(Gamma_value, reference_value, theta_start=cyl.num_thetas_start, theta_range=theta_range, tol = tol, relative_error=is_relative_error, num_double_points=cyl.num_times_thetas_doubled, is_compare_integrations=is_compare_integrations)
    if cyl.is_calc_r_convergence:
        if cyl.appellian_is_line_integral:
            raise ValueError("Appellian is line integral, cannot compute r convergence")
        print("USING A DEFAULT CONVERGED THETA COUNT FOR R CONVERGENCE")
        print("USING A DEFAULT CONVERGED THETA COUNT FOR R CONVERGENCE")
        print("USING A DEFAULT CONVERGED THETA COUNT FOR R CONVERGENCE")
        print("USING A DEFAULT CONVERGED THETA COUNT FOR R CONVERGENCE")
        theta_converged = 2305
        tol = 1e-20
        cyl.D = 0.1
        cyl.epsilon = cyl.calc_epsilon_from_D(cyl.D)
        romberg_gamma_values = np.array([7.67148849088136]) # for D = 0.1
        romberg_reference_values = np.array([90572.7333392366]) # for 17 doubles area int romberg
        is_relative_error = cyl.grid_conv_is_relative_error
        Gamma_values = np.array([2.0]) #np.array([10.0])  # for D=0.001 trap 17 and area int analytic with mult
        reference_values = romberg_reference_values
        num_converged_thetas_double = 13
        num_r_points_start = 10
        # radial distances should go from 2 to 20 in steps of 1
        # radial_distances = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        # radial distances from 21 to 40
        radial_distances = [23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,31.0,32.0,33.0,34.0,35.0,36.0,37.0,38.0,39.0,40.0]
        is_compare_integrations = False
        for i in range(len(radial_distances)):
            cyl.radial_distance = radial_distances[i]
            cyl.r_convergence(Gamma_values[0], reference_values[0], num_r_points_start = num_r_points_start, theta_start = cyl.num_thetas_start, theta_converged_doubles=num_converged_thetas_double, theta_range = (0.0, 2*np.pi), r_range=(1.0, 350.0), output_dir="Grid_conv", tol=tol, relative_error=is_relative_error, num_double_points=cyl.num_times_r_doubled, is_compare_integrations=is_compare_integrations)




    # if cyl.is_calc_convergence_to_kutta:
    #     print("r_values: ", r_values)
    #     D_range_first = [0.0001, 0.00099, 0.00001]
    #     D_values_first = hlp.list_to_range(D_range_first)
    #     D_range_second = [0.001, 0.009, 0.001]
    #     D_values_second = hlp.list_to_range(D_range_second)
    #     D_range_third = [0.01, 0.1, 0.01]
    #     D_values_third = hlp.list_to_range(D_range_third)
    #     r_range = [1.0*cyl.cylinder_radius, 7*cyl.cylinder_radius, 0.01*cyl.cylinder_radius]
    #     r_values = hlp.list_to_range(r_range)
    #     D_values = np.concatenate((D_values_first, D_values_second, D_values_third), axis=0) # instead, make D_values on a logarithmic scale from 0.0001 to 0.1
    #     D_values = np.logspace(-4, -1, num=100)  # From 10^-4 (0.0001) to 10^-1 (0.1) in 6 steps (in a logarithmic scale)
    #     zeta_center_values = np.array([-0.20 + 0.0j])
    #     for i in range(len(zeta_center_values)):
    #         print("zeta center: ", zeta_center_values[i])
    #         cyl.calc_convergence_to_kutta(D_values, r_values, theta_values, zeta_center_values[i])
    # if cyl.is_plot_convergence_to_kutta:
    #     plt.figure()
    #     folder_name = "text_files/convergence_to_kutta"
    #     file_name_array = []
    #     for filename in os.listdir(folder_name):
    #         if filename.endswith(".txt"):
    #             file_path = os.path.join(folder_name, filename)
    #             file_name_array.append(file_path)
    #     cyl.plot_convergence_to_kutta(file_name_array)
    #     filename = "figures/convergence_to_kutta_alpha_" + str(round(cyl.angle_of_attack*180/np.pi, 1)) + ".pdf"
    #     plt.savefig(filename, format='pdf', bbox_inches='tight')
    #     print("Convergence to Kutta figure saved as:", filename)


    # cyl.zeta_center = -0.09 + 0.0j
    # cyl.D = 0.1
    # cyl.epsilon = cyl.calc_epsilon_from_D(cyl.D)
    # cyl.freestream_velocity = 10.0
    # cyl.angle_of_attack = np.deg2rad(5.0)
    # cyl.circulation = 2.0

    # r_chi = 1.0#cyl.cylinder_radius
    # theta_chi = 0.1
    # print("                      D = ", cyl.D)
    # print("                epsilon = ", cyl.epsilon)
    # print("                  zeta0 = ", cyl.zeta_center)
    # print("                   Vinf = ", cyl.freestream_velocity)
    # print("            alpha [deg] = ", np.rad2deg(cyl.angle_of_attack))
    # print("                  Gamma = ", cyl.circulation)
    # print("")
    # print("                  theta = ", theta_chi)
    # print("                      r = ", r_chi)
    # # print("              chi theta = ", theta_chi)
    # print("")
    # cyl.appellian_is_area_integral = False
    # cyl.appellian_is_line_integral = True
    # cyl.integration_method = "trapezoidal"
    # #     # Select integration method
    # if cyl.integration_method == "left_riemann":
    #     cyl.integration_func = cyl.riemann_integration
    # elif cyl.integration_method == "trapezoidal":
    #     cyl.integration_func = cyl.trapezoidal_integration
    # elif cyl.integration_method == "simpson13":
    #     cyl.integration_func = cyl.simpson_13_integration
    # elif cyl.integration_method == "romberg":
    #     cyl.integration_func = cyl.romberg_integration
    # else:
    #     raise ValueError("Invalid integration method specified. Use 'left_riemann' 'trapezoidal'.")
    # analytic_conv_accel = cyl.analytic_conv_accel_for_line_int_comp_conj([r_chi, theta_chi], cyl.circulation)
    # print("  Analytic Line Integrand = ", analytic_conv_accel)
    # # now integrate around the circle using thetas from 0 to 2*pi with 1000 points
    # points = 10000
    # print("          Num theta points = ", points)
    # theta_values = np.linspace(0, 2*np.pi, points, endpoint=True)
    # r_values = np.array([r_chi])
    # cyl.accel_function = cyl.numerical_line_int_conv_accel
    # appellian = cyl.appellian_acceleration_loop(cyl.circulation, r_values, theta_values)
    # print("Trapezoidal Numerical Line Integral", appellian)
    # # r_values = np.linspace(r_chi, 2.0, 2)
    # # for i in range(len(r_values)):
    # #     print("  r", i, ": ", r_values[i])


    # analytic_conv_accel = cyl.analytic_conv_accel_for_line_int_comp_conj([r_chi, theta_chi], cyl.circulation)
    # # print("  Analytic Line Integrand = ", analytic_conv_accel)
    # numerical_conv_accel = cyl.numerical_line_int_conv_accel([r_chi, theta_chi], cyl.circulation)
    # # print("  Numerical Line Integrand  ", numerical_conv_accel)
    # Spencer_accel_squared_comp_conj = cyl.analytic_conv_accel_square_comp_conj([r_chi,theta_chi], cyl.circulation)
    # # print("  Analytic Area Integrand = ", Spencer_accel_squared_comp_conj)
    # numerical_double_int = cyl.numerical_convective_acceleration([r_chi, theta_chi], cyl.circulation)
    # # print("  Numerical Area Integrand  ", numerical_double_int/2)
    # cyl.accel_function = cyl.analytic_conv_accel_square_comp_conj
    # # appellian = cyl.appellian_acceleration_loop(cyl.circulation, r_values, theta_values)
    # # print("   appellian = ", appellian)


    # print("")

    # run
    # cyl.plot_geometry_settings() ######################
    # cyl.plot_geometry_in_every_plane() ###################
    # plt.show() #############################################
    # cyl.numerically_integrate_appellian_chi1(cyl.circulation, r_values, cyl.D)
    # step_size = cyl.kutta_circulation/10
    # Gamma_range = [0.0, cyl.kutta_circulation, step_size]
    # Gamma_vals = hlp.list_to_range(Gamma_range)
    # # print("Gamma_vals", Gamma_vals)
    # poly_order = 4
    # r_vals = np.array([cyl.cylinder_radius])
    # appellian_root, appellian_value = hlp.polyfit(cyl.numerically_integrate_appellian_chi1, r_vals, Gamma_vals, poly_order, cyl.is_plot_appellian, "$\\Gamma$", "S", "Appellian Function", cyl.D)
    # print("\n\n\n")
    # print("Appellian Circulation: ", appellian_root)
    # print("kutta Circulation: ", cyl.kutta_circulation)
    # print("Appellian Value: ", appellian_value)
    # print("\n\n\n")
    # # cyl.plot_geometry_settings()
    # # D_range_first = [0.0, 0.00099, 0.00001]
    # # D_values_first = hlp.list_to_range(D_range_first)
    # D_range_second = [0.0, 0.009, 0.001]
    # D_values_second = hlp.list_to_range(D_range_second)
    # D_range_third = [0.01, 1.00, 0.01]
    # D_values_third = hlp.list_to_range(D_range_third)
    # D_values = np.concatenate((D_values_second, D_values_third), axis=0)
    # print("number of D values: ", len(D_values))
    # # kutta_appellian_array = cyl.calc_gamma_app_over_gamma_kutta(D_values, r_values, theta_values)
    # kutta_appellian_array = np.zeros((len(D_values), 2))
    # rounded_kutta_appellian_array = np.zeros((len(D_values), 2))
    # # self.D
    # # self.tau = 0.1 ######## remove
    # # C = 1/(1+epsilon_held)
    # for i in range(len(D_values)):
    #     cyl.D = D_values[i]
    #     cyl.epsilon = cyl.calc_epsilon_from_D(cyl.D) ##### before
    #     # print("D value: ", cyl.D, "epsilon: ", cyl.epsilon, "kutta circulation: ", cyl.kutta_circulation, "Cylinder Radius: ", cyl.cylinder_radius)  
    #     # cyl.zeta_trailing_edge_focus, cyl.zeta_leading_edge_focus,cyl.z_trailing_edge_focus, cyl.z_leading_edge_focus = cyl.calc_foci()
    #     kutta_appellian_array[i, 0] = cyl.D
    #     kutta_appellian_array[i, 1] = hlp.polyfit(cyl.numerically_integrate_appellian_chi1, r_vals, Gamma_vals, poly_order, cyl.is_plot_appellian, "$\\Gamma$", "S", "Appellian Function", D_values[i])[0] / cyl.kutta_circulation 
    #     print("Kutta_appellian_array_i:", kutta_appellian_array[i, 0], kutta_appellian_array[i, 1])
    #     print("Kutta Circulation: ", cyl.kutta_circulation)
    #     # break
    #     # rounded_kutta_appellian_array[i, 0] = D_values[i]
    #     # rounded_kutta_appellian_array[i, 1] = cyl.kutta_for_rounded_TE(D_values[i])/cyl.kutta_circulation
    # plt.plot(kutta_appellian_array[:,0], kutta_appellian_array[:,1], color = "red", label = "Around Aft Sing")
    # # plt.plot(rounded_kutta_appellian_array[:,0], rounded_kutta_appellian_array[:,1], color = "blue", label = "Kutta for Rounded TE")
    # # plot a gray axis line at y=1 and y=0
    # plt.axhline(1, color='gray', linestyle='--', linewidth=0.5)
    # plt.axhline(0, color='gray', linewidth=0.5)
    # plt.xlabel("$D$", labelpad=5)
    # plt.ylabel("$\\Gamma / \\Gamma_{Kutta}$", labelpad=5)
    # # change the color of the plot to red
    # kutta_appellian_array = np.zeros((len(D_values), 2))
    # # kutta_appellian_array = cyl.calc_gamma_app_over_gamma_kutta(D_values, r_values, theta_values)
    # # cyl.plot_gamma_over_gamma_kutta(kutta_appellian_array, is_plot_current_D = False, kutta_and_current_D = [kutta_appellian_array[0,0], kutta_appellian_array[0,1]])
    # plt.legend()
    # plt.show()
    # ## run D sweep in this way. 


