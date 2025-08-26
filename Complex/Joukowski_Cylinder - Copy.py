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
                self.integration_func = self.single_romberg_integration
            else:
                raise ValueError("Invalid integration method specified. Use 'left_riemann' 'trapezoidal'.")
            self.is_plot_appellian = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_appellian"], False)
            self.is_plot_appellian_for_varying_circulation = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_appellian_for_varying_circulation"], False)
            self.is_calc_circulation_for_varying_shape = hlp.parse_dictionary_or_return_default(input, ["appellian", "is_calc_circulation_for_varying_shape"], False)
            self.is_plot_circulation_for_varying_shape = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_circulation_for_varying_shape"], False)
            self.num_thetas_start = hlp.parse_dictionary_or_return_default(input, ["appellian", "num_thetas_start"], 10)
            self.num_times_thetas_doubled = hlp.parse_dictionary_or_return_default(input, ["appellian", "num_times_thetas_doubled"], 18)
            num_thetas_end = self.num_thetas_start*2**self.num_times_thetas_doubled
            self.one_appellian_r_points = hlp.parse_dictionary_or_return_default(input, ["appellian", "num_r_steps"], 1000)
            self.dtheta = 2*np.pi/num_thetas_end
            self.is_calc_rb_comparisons = hlp.parse_dictionary_or_return_default(input, ["appellian", "is_calc_rb_comparisons"], False)
            self.is_plot_rb_comparisons = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_rb_comparisons"], False)
            self.is_calc_convergence_to_kutta = hlp.parse_dictionary_or_return_default(input, ["appellian", "is_calc_convergence_to_kutta"], False)
            self.is_plot_convergence_to_kutta = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_convergence_to_kutta"], False)
            self.is_calc_theta_convergence = hlp.parse_dictionary_or_return_default(input, ["appellian", "is_calc_theta_convergence"], False)
            self.is_plot_theta_convergence = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_theta_convergence"], False)
            self.is_calc_r_convergence = hlp.parse_dictionary_or_return_default(input, ["appellian", "is_calc_r_convergence"], False)
            self.is_plot_r_convergence = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_r_convergence"], False)
            self.appellian_is_line_integral = hlp.parse_dictionary_or_return_default(input, ["appellian", "appellian_is_line_integral"], True)
            self.is_numerical_conv_accel_for_area_int = hlp.parse_dictionary_or_return_default(input, ["appellian", "is_numerical_conv_accel_for_area_int"], False)
            if self.appellian_is_line_integral:
                self.appellian_is_area_integral = False
                self.memoized_accel_function = memoize_acceleration_func(self.analytic_conv_accel_for_line_int_comp_conj) # for dramatic speed improvements
                self.accel_function = self.analytic_conv_accel_for_line_int_comp_conj
            else:
                self.appellian_is_area_integral = True
                if self.is_numerical_conv_accel_for_area_int:
                    print("Numerical Convective acceleration!!")
                    print("Numerical Convective acceleration!!")
                    print("Numerical Convective acceleration!!")
                    self.memoized_accel_function = memoize_acceleration_func(self.numerical_convective_acceleration) # for dramatic speed improvements
                    self.accel_function = self.numerical_convective_acceleration
                else:
                    self.memoized_accel_function = memoize_acceleration_func(self.analytic_conv_accel_square_comp_conj) # for dramatic speed improvements
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
        plt.axhline(0, color='gray')
        plt.axvline(0, color='gray')
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
        epsilon_name = "epsilon_" + str(self.epsilon)
        zeta_0_name = "zeta_0_" + str(self.zeta_center)
        if self.show_fig:
            if self.type == "cylinder" and self.is_compute_appellian and self.is_plot_appellian_for_varying_circulation:
                plt.scatter(xi_eta_vals[:, 0], xi_eta_vals[:, 1], color='red', s=5)
                if self.save_fig:
                    plt.savefig("Joukowski_Cylinder_" + epsilon_name + "_" +  zeta_0_name + "_" + ".svg", dpi=300, bbox_inches=None, pad_inches=0)
                    plt.close()
    
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
        V_inf, epsilon, R, alpha, zeta_0 = self.freestream_velocity, self.epsilon, self.cylinder_radius, self.angle_of_attack, self.zeta_center
        r, theta = point_r_theta_in_Chi[0], point_r_theta_in_Chi[1]
        xi_chi, eta_chi = hlp.r_theta_to_xy(r, theta)
        chi = xi_chi + 1j*eta_chi
        zeta = self.Chi_to_zeta(chi)
        dz_dzeta, dphi_dzeta = self.dZ_dzeta(zeta), self.dPhi_dzeta(zeta, Gamma)
        dz_dzeta_conj = np.conj(dz_dzeta)
        dphi_dzeta_conj = np.conj(dphi_dzeta)
        dd_phi_dzeta_dr = V_inf*(-1j*Gamma/(2*np.pi*V_inf*(r**2*np.exp(1j*theta))) + 2*R**2*np.exp(1j*alpha)/(r**3*np.exp(2*1j*theta)))
        dd_phi_dzeta_dr_conj = np.conj(dd_phi_dzeta_dr)
        ddz_dzeta_dr = 2*(R-epsilon)**2*np.exp(1j*theta)/(r*np.exp(1j*theta))**3
        ddz_dzeta_dr_conj = np.conj(ddz_dzeta_dr)
        integrand = 1/(dz_dzeta**4*dz_dzeta_conj**4)*((dz_dzeta*dz_dzeta_conj)**2*dphi_dzeta*dphi_dzeta_conj*(dphi_dzeta*dd_phi_dzeta_dr_conj+dphi_dzeta_conj*dd_phi_dzeta_dr)- (dphi_dzeta*dphi_dzeta_conj)**2*dz_dzeta*dz_dzeta_conj*(dz_dzeta*ddz_dzeta_dr_conj+dz_dzeta_conj*ddz_dzeta_dr))
        dzeta_dtheta = 1j*r*np.exp(1j*theta)
        integrand *= abs(dzeta_dtheta)*abs(dz_dzeta)
        return np.real(integrand)
    
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
    
    def numerical_convective_acceleration(self, point_r_theta_in_Chi, Gamma, stepsize=1e-6):
        r, theta = point_r_theta_in_Chi[0], point_r_theta_in_Chi[1]
        xi_chi, eta_chi = hlp.r_theta_to_xy(r, theta)
        chi = xi_chi + 1j*eta_chi
        zeta = self.Chi_to_zeta(chi)
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
        dux_dx = self.central_difference(velocity_plus_x_step[0], velocity_minus_x_step[0], stepsize=stepsize)
        uy = velocity_at_z[1]
        dux_dy = self.central_difference(velocity_plus_y_step[0], velocity_minus_y_step[0], stepsize=stepsize)
        duy_dx = self.central_difference(velocity_plus_x_step[1], velocity_minus_x_step[1], stepsize=stepsize)
        duy_dy = self.central_difference(velocity_plus_y_step[1], velocity_minus_y_step[1], stepsize=stepsize)
        conv_accel = np.array([ux*dux_dx + uy*dux_dy, ux*duy_dx+ uy*duy_dy])
        conv_accel_squared = np.dot(conv_accel, conv_accel)
        return conv_accel_squared

    def central_difference(self, value_above, value_below, stepsize=1e-6):
        """Calculates central difference based on stepsize"""
        return (value_above - value_below)/(2*stepsize)

    def memoize_integration_scheme(self, integration_func):
        cache = {}
        def wrapper(point_pair_r, point_pair_theta, Gamma):
            key = (tuple(round(r, 12) for r in point_pair_r),tuple(round(theta, 12) for theta in point_pair_theta),Gamma)
            if key not in cache:
                cache[key] = integration_func(point_pair_r, point_pair_theta, Gamma)
            return cache[key]
        wrapper.clear_cache = lambda: cache.clear()
        return wrapper

    def appellian_acceleration_loop(self, Gamma: float, r_values: np.array, theta_values: np.array, grid_conv = False):
        Appellian_value = 0.0
        theta_values = np.unique(theta_values)
        theta_pairs = [(theta_values[k], theta_values[k + 1]) for k in range(len(theta_values) - 1)]
        if self.integration_func != self.single_romberg_integration:
            if not grid_conv:
                memoized_int_func = self.memoize_integration_scheme(self.integration_func)
            else:
                memoized_int_func = self.memoized_int_func
            if self.appellian_is_area_integral:
                # === Area integral ===
                if len(r_values) <= 1:  # Fixed radius → line in theta
                    r0 = r_values[0]
                    if self.is_compute_appellian:
                        print("Computing Appellian Using An Area Integral Applied on the Surface")
                    for theta0, theta1 in theta_pairs:
                        Appellian_value += memoized_int_func([r0, r0], [theta0, theta1], Gamma)
                else:  # Full polar area integration (r, theta)
                    if self.is_compute_appellian:
                        print("Computing Appellian Using An Area Integral Applied on and around the Surface")
                    for j in range(len(r_values) - 1):
                        r0, r1 = r_values[j], r_values[j + 1]
                        for theta0, theta1 in theta_pairs:
                            Appellian_value += memoized_int_func([r0, r1], [theta0, theta1], Gamma)
                return 0.5 * Appellian_value
            else:
                # === Line integral ===
                r0 = r_values[0]
                if self.is_compute_appellian:
                    for theta0, theta1 in tqdm(theta_pairs, desc="Computing Appellian Line Integral", unit="segment"):
                        Appellian_value += memoized_int_func([r0, r0], [theta0, theta1], Gamma)
                else:
                    for theta0, theta1 in theta_pairs:
                        Appellian_value += memoized_int_func([r0, r0], [theta0, theta1], Gamma)
                return Appellian_value * (-0.03125)
        else:  # O(h^6) Romberg integration
            if not grid_conv:
                memoized_int_func = self.memoize_integration_scheme(self.trapezoidal_integration)
            else:
                memoized_int_func = self.memoized_int_func
            thetas_half = double_nodes_periodic(theta_values)
            # thetas_half = np.unique(thetas_half)
            thetas_quarter = double_nodes_periodic(thetas_half)
            # thetas_quarter = np.unique(thetas_quarter)
            theta_pairs = [(theta_values[k], theta_values[k + 1]) for k in range(len(theta_values) - 1)]
            theta_pairs_half = [(thetas_half[l], thetas_half[l + 1]) for l in range(len(thetas_half) - 1)]
            theta_pairs_quarter = [(thetas_quarter[m], thetas_quarter[m + 1]) for m in range(len(thetas_quarter) - 1)]
            if self.appellian_is_area_integral:
                if len(r_values) <= 1:  # Fixed radius → line in theta
                    r0 = r_values[0]
                    if self.is_compute_appellian:
                        print("Computing Appellian Using An Area Integral Applied on the Surface")
                    Appellian_value = 0.0
                    Appellian_value_half = 0.0
                    Appellian_value_quarter = 0.0
                    for theta0, theta1 in theta_pairs:
                        Appellian_value += memoized_int_func([r0, r0], [theta0, theta1], Gamma)
                    for theta0half, theta1half in theta_pairs_half:
                        Appellian_value_half += memoized_int_func([r0, r0], [theta0half, theta1half], Gamma)
                    for theta0quarter, theta1quarter in theta_pairs_quarter:
                        Appellian_value_quarter += memoized_int_func([r0, r0], [theta0quarter, theta1quarter], Gamma)
                    I_l = (4/3) * Appellian_value_half - (1/3) * Appellian_value
                    I_m = (4/3) * Appellian_value_quarter - (1/3) * Appellian_value_half
                    I_final = (16/15) * I_m - (1/15) * I_l
                    return 0.5 * I_final
                else:  # Full area integral with r0 ≠ r1
                    if self.is_compute_appellian:
                        print("Computing Appellian Using a 2D Romberg Area Integral")
                    r_vals = r_values
                    r_half = double_nodes_linear(r_vals)
                    r_quarter = double_nodes_linear(r_half)
                    theta_half = double_nodes_periodic(theta_values)
                    print("theta_half", theta_half)
                    theta_quarter = double_nodes_periodic(theta_half)
                    def get_integral(r_grid, theta_grid):
                        total = 0.0
                        r_pairs = [(r_grid[i], r_grid[i+1]) for i in range(len(r_grid)-1)]
                        theta_pairs = [(theta_grid[j], theta_grid[j+1]) for j in range(len(theta_grid)-1)]
                        for r0, r1 in r_pairs:
                            for theta0, theta1 in theta_pairs:
                                total += memoized_int_func([r0, r1], [theta0, theta1], Gamma)
                        return total
                    # Romberg in theta direction (r fixed)
                    I_0theta = get_integral(r_vals, theta_values)
                    I_1theta = get_integral(r_vals, theta_half)
                    I_2theta = get_integral(r_vals, theta_quarter)
                    I_ltheta = (4/3) * I_1theta - (1/3) * I_0theta
                    I_mtheta = (4/3) * I_2theta - (1/3) * I_1theta
                    I_final_θ = (16/15) * I_mtheta - (1/15) * I_ltheta
                    # Romberg in r direction (theta fixed)
                    I_0r = I_0theta
                    I_1r = get_integral(r_half, theta_half)
                    I_2r = get_integral(r_quarter, theta_quarter)
                    I_lr = (4/3) * I_1r - (1/3) * I_0r
                    I_mr = (4/3) * I_2r - (1/3) * I_1r
                    I_final_r = (16/15) * I_mr - (1/15) * I_lr
                    # Final result: average both extrapolations for symmetry
                    I_final = 0.5 * (I_final_θ + I_final_r)
                    return 0.5 * I_final
            else:
                # === Line integral with Romberg ===
                r0 = r_values[0]
                Appellian_value = 0.0
                Appellian_value_half = 0.0
                Appellian_value_quarter = 0.0
                if self.is_compute_appellian:
                    print("Computing Appellian Line Integral Using Romberg")
                for theta0, theta1 in theta_pairs:
                    Appellian_value += memoized_int_func([r0, r0], [theta0, theta1], Gamma)
                for theta0, theta1 in theta_pairs_half:
                    Appellian_value_half += memoized_int_func([r0, r0], [theta0, theta1], Gamma)
                for theta0, theta1 in theta_pairs_quarter:
                    Appellian_value_quarter += memoized_int_func([r0, r0], [theta0, theta1], Gamma)
                I_l = (4/3) * Appellian_value_half - (1/3) * Appellian_value
                I_m = (4/3) * Appellian_value_quarter - (1/3) * Appellian_value_half
                I_final = (16/15) * I_m - (1/15) * I_l
                return I_final * (-0.03125)

    def riemann_integration(self, r_values, theta_values, Gamma):
        """Left Riemann integration for a segment (1D or 2D).Evaluates function at lower corner and multiplies by r * dtheta or r * dr * dtheta."""
        r0, r1 = r_values
        theta0, theta1 = theta_values
        xi0_chi, eta0_chi = hlp.r_theta_to_xy(r0, theta0)
        xi1_chi, eta1_chi = hlp.r_theta_to_xy(r1, theta1)
        chi0 = xi0_chi + 1j*eta0_chi
        chi1 = xi1_chi + 1j*eta1_chi
        zeta0 = self.Chi_to_zeta(chi0)
        zeta1 = self.Chi_to_zeta(chi1)
        z0 = self.zeta_to_z(zeta0, self.epsilon)
        z1 = self.zeta_to_z(zeta1, self.epsilon)
        dx = abs(z1.real-z0.real)
        dy = abs(z1.imag-z0.imag)
        d_Omega = np.sqrt(dx**2 + dy**2)
        dtheta = theta1 - theta0
        dr = r1 - r0
        # if dtheta <= 0:
            # dtheta += 2 * np.pi
        if r0 == r1:
            # 1D over theta at fixed r
            f = self.accel_function([r0, theta0], Gamma)
            return f * d_Omega # r0 * dtheta
        else:
            # 2D over (r, theta)
            f = self.accel_function([r0, theta0], Gamma)
            return f * r0 * dr * dtheta
        
    def single_romberg_integration(self, r_values, theta_values, Gamma):
        """Takes the two fidelities of integration to make a better estimate"""
        # self.integration_func = self.trapezoidal_integration

    def trapezoidal_integration(self, r_values, theta_values, Gamma):
        """r_values is either 1 repeated value or 2 distinct values. theta_values is 2 distinct values. This function calculates the integral of a function over a single line or area segment depending on if r0=r1"""
        r0, r1 = r_values
        theta0, theta1 = theta_values
        xi0_chi, eta0_chi = hlp.r_theta_to_xy(r0, theta0)
        xi1_chi, eta1_chi = hlp.r_theta_to_xy(r1, theta1)
        chi0 = xi0_chi + 1j*eta0_chi
        chi1 = xi1_chi + 1j*eta1_chi
        zeta0 = self.Chi_to_zeta(chi0)
        zeta1 = self.Chi_to_zeta(chi1)
        z0 = self.zeta_to_z(zeta0, self.epsilon)
        z1 = self.zeta_to_z(zeta1, self.epsilon)
        dx = abs(z1.real-z0.real)
        dy = abs(z1.imag-z0.imag)
        d_Omega = np.sqrt(dx**2 + dy**2)
        dtheta = abs(theta1 - theta0)
        # if dtheta <= 0:
            # dtheta += 2 * np.pi
        if r0 == r1:  # 1D
            f0 = self.accel_function([r0, theta0], Gamma)
            f1 = self.accel_function([r0, theta1], Gamma)
            return 0.5 * (f0 + f1) * d_Omega# r0 * dtheta
        else:  # 2D with midpoints
            dr = r1 - r0
            r_mid = 0.5 * (r0 + r1)
            theta_mid = 0.5 * (theta0 + theta1)
            # Corner evaluations
            f00 = self.accel_function([r0, theta0], Gamma) * r0
            f01 = self.accel_function([r0, theta1], Gamma) * r0
            f10 = self.accel_function([r1, theta0], Gamma) * r1
            f11 = self.accel_function([r1, theta1], Gamma) * r1
            fmm = self.accel_function([r_mid, theta_mid], Gamma) * r_mid
            # Average with midpoint
            avg = (f00 + f01 + f10 + f11 + 4 * fmm) / 8
            return avg * dr * dtheta
        
    def simpson_13_integration(self, r_values, theta_values, Gamma):
        """Simpson's 1/3 rule for 1D or single 2D cell in polar coords."""
        r0, r1 = r_values
        theta0, theta1 = theta_values
        xi0_chi, eta0_chi = hlp.r_theta_to_xy(r0, theta0)
        xi1_chi, eta1_chi = hlp.r_theta_to_xy(r1, theta1)
        chi0 = xi0_chi + 1j*eta0_chi
        chi1 = xi1_chi + 1j*eta1_chi
        zeta0 = self.Chi_to_zeta(chi0)
        zeta1 = self.Chi_to_zeta(chi1)
        z0 = self.zeta_to_z(zeta0, self.epsilon)
        z1 = self.zeta_to_z(zeta1, self.epsilon)
        dx = abs(z1.real-z0.real)
        dy = abs(z1.imag-z0.imag)
        d_Omega = np.sqrt(dx**2 + dy**2)
        # Ensure positive theta difference
        dtheta = abs(theta1 - theta0)
        if dtheta <= 0:
            dtheta += 2 * np.pi
        def safe_eval(x): # Safely evaluate function, returning 0 on error
            try:
                return self.accel_function(x, Gamma)
            except:
                return 0.0  # or np.nan or eps
        if r0 == r1:
            # 1D case: integrate over theta at fixed r
            theta_mid = 0.5 * (theta0 + theta1)
            if np.isclose(theta_mid % (2*np.pi), self.theta_AJA % (2*np.pi)):
                theta_mid += 1e-10  # bump away from singularity
            f0 = safe_eval([r0, theta0])
            f1 = safe_eval([r0, theta_mid])
            f2 = safe_eval([r0, theta1])
            return (f0 + 4*f1 + f2) * d_Omega / 6
        else:
            # 2D case
            dr = r1 - r0
            r_mid = 0.5 * (r0 + r1)
            theta_mid = 0.5 * (theta0 + theta1)
            if np.isclose(theta_mid % (2*np.pi), self.theta_AJA % (2*np.pi)):
                theta_mid += 1e-10  # avoid singularity
            f00 = safe_eval([r0, theta0]) * r0
            f01 = safe_eval([r0, theta1]) * r0
            f10 = safe_eval([r1, theta0]) * r1
            f11 = safe_eval([r1, theta1]) * r1
            fmm = safe_eval([r_mid, theta_mid]) * r_mid
            f0m = safe_eval([r0, theta_mid]) * r0
            f1m = safe_eval([r1, theta_mid]) * r1
            fm0 = safe_eval([r_mid, theta0]) * r_mid
            fm1 = safe_eval([r_mid, theta1]) * r_mid
            return (f00 + f01 + f10 + f11 + 4*fmm + 2*(f0m + f1m + fm0 + fm1)) * dr * dtheta / 36

    def run_two_stage_convergence(self, Gamma, reference_value,fixed_r=1.0, r_range=(1.0, 10.0), theta_range=(0, 2*np.pi),output_dir="Grid_conv", tol=1e-3,run_theta_stage=True, run_r_stage=True,relative_error=False,num_double_points = 8):
        os.makedirs(output_dir, exist_ok=True)
        theta_converged_dict = {}
        best_N_theta_dict = {}
        if relative_error:
            header_2 = "GridPoints,RelativeError"
        else:
            header_2 = "GridPoints,TrueError"
        method_dict = {"Riemann": self.riemann_integration,"Trapezoidal": self.trapezoidal_integration,"Romberg": self.single_romberg_integration,"Simpson": self.simpson_13_integration}
        # method_dict = {"Riemann": self.riemann_integration}
        # method_dict = {"Trapezoidal": self.trapezoidal_integration}
        # method_dict = {"Trapezoidal": self.trapezoidal_integration,"Romberg": self.single_romberg_integration}
        # method_dict = {"Simpson": self.simpson_13_integration}
        if run_theta_stage:
            print("Running theta-direction convergence study...")
            for method_name, integrator in method_dict.items():
                self.integration_func = integrator
                if method_name != "Romberg":
                    self.memoized_int_func = self.memoize_integration_scheme(self.integration_func)
                gridpoints, errors = [], []
                theta_converged = None
                best_N_theta = None
                error = 100.0
                previous_result = None
                theta_values = np.linspace(theta_range[0], theta_range[1], 10, endpoint=False)
                start_time = time.time()
                for _ in tqdm(range(num_double_points+1), desc=f"{method_name} theta convergence"):
                    r_values = np.array([fixed_r])
                    result = self.appellian_acceleration_loop(Gamma, r_values, theta_values, grid_conv = True)
                    if relative_error:
                        if previous_result is not None:
                            error = abs((result - previous_result) / result) * 100
                    else:
                        error = abs((reference_value-result) / reference_value) * 100
                    gridpoints.append(len(theta_values))
                    errors.append(error)
                    if error < tol and theta_converged is None:
                        theta_converged = np.copy(theta_values)
                        best_N_theta = len(theta_values)
                        break
                    previous_result = result
                    theta_values = double_nodes_periodic(theta_values)
                theta_converged_dict[method_name] = theta_converged
                best_N_theta_dict[method_name] = best_N_theta
                end_time = time.time()
                total_time = end_time - start_time
                arr = np.column_stack((gridpoints, errors))
                filename = os.path.join(output_dir,f"{method_name.lower()}_theta_convergence_D_{round(self.D, 3)}"f"_zeta0_r_{round(self.zeta_center.real, 3)}"f"_zeta0_i_{round(self.zeta_center.imag, 3)}_time_{round(total_time, 3)}s_.csv")
                np.savetxt(filename, arr, delimiter=",", header=header_2, comments='')
        if run_r_stage:
            print("Running radial-direction convergence study...")
            for method_name, integrator in method_dict.items():
                self.integration_func = integrator
                if method_name != "Romberg":
                    self.memoized_int_func = self.memoize_integration_scheme(self.integration_func)
                gridpoints, errors = [], []
                converged = False
                error = 100.0
                index = 0
                previous_result = None
                theta_converged = theta_converged_dict.get(method_name)

                if theta_converged is None:
                    print(f"Skipping radial convergence for {method_name} as theta did not converge.")
                    continue
                
                r_values = np.linspace(r_range[0], r_range[1], 10)
                time_start = time.time()
                for _ in tqdm(range(num_double_points), desc=f"{method_name} radial convergence"):
                    result = self.appellian_acceleration_loop(Gamma, r_values, theta_converged, grid_conv = True)
                    if relative_error:
                        if previous_result is not None:
                            error = abs((result - previous_result) / previous_result) * 100
                    else:
                        error = abs((result - reference_value) / reference_value) * 100
                    gridpoints.append(len(r_values))
                    errors.append(error)
                    if error < tol and not converged:
                        converged = True
                        break
                    previous_result = result
                    r_values = double_nodes_linear(r_values)
                arr = np.column_stack((gridpoints, errors))
                time_end = time.time()
                total_time = time_end-time_start
                filename = os.path.join(output_dir,f"{method_name.lower()}_radial_convergence_D_{round(self.D, 3)}"f"_zeta0_r_{round(self.zeta_center.real, 3)}"f"_zeta0_i_{round(self.zeta_center.imag, 3)}_time_{round(total_time, 3)}s_.csv")
                np.savetxt(filename, arr, delimiter=",", header="GridPoints,RelativeError", comments='')

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
    
    def numerically_integrate_appellian(self, Gamma: float, r_values: np.array, D):
        """This function calculates the Appellian function numerically requires evenly spaced ranges for Gamma, r, and theta in Chi"""
        # create a meshgrid of r and theta values, the first value in r_range is the lower bound the second value is the upper bound, the third value is the increment size
        # calculate the area element 
        if len(r_values) == 1:
            dr = 0.01
        else:        
            dr = r_values[1] - r_values[0]
        # theta start is the location of the geo
        theta_start = 0.0#self.theta_AJA
        theta_end = 2*np.pi #+ theta_start
        theta_range = [theta_start, theta_end, self.dtheta]
        theta_values = hlp.list_to_range(theta_range)
        # add dtheta to all values in theta_values
        #theta_values = [theta + self.dtheta/2 for theta in theta_values]
        dr_original = dr
        dtheta_original = self.dtheta
        # now get dr and dtheta in z plane using the zeta to z function 
        d_Chi = hlp.r_theta_to_xy(dr, self.dtheta)[0] + 1j*hlp.r_theta_to_xy(dr, self.dtheta)[1]
        d_zeta = self.Chi_to_zeta(d_Chi)
        d_z = self.zeta_to_z(d_zeta, self.epsilon)
        d_xi_z, d_eta_z = d_z.real, d_z.imag
        # now these are the dr and dtheta values in the z plane for the numerical integration in after the else statement
        dr, dtheta = hlp.xy_to_r_theta(d_xi_z, d_eta_z)
        if self.appellian_is_area_integral:
            Appellian_array = self.appellian_acceleration_loop(Gamma, r_values, theta_values)            
        else: # line integral
            r_values = np.array([self.cylinder_radius])
            Appellian_array = self.appellian_acceleration_loop(Gamma, r_values, theta_values)
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

    def run_appellian_roots(self, r_vals, theta_vals, D):
        """This function runs the Appellian stuff"""
        xi_eta_array = np.zeros((len(r_vals)*len(theta_vals), 2))
        step_size = 2*self.kutta_circulation/6
        Gamma_range = [-self.kutta_circulation, self.kutta_circulation, step_size]
        Gamma_vals = hlp.list_to_range(Gamma_range)
        poly_order = 4
        appellian_root, appellian_value = hlp.polyfit(self.numerically_integrate_appellian, r_vals, Gamma_vals, poly_order, self.is_plot_appellian, "$\\Gamma$", "S", "Appellian Function", D)
        return appellian_root, xi_eta_array, appellian_value
    
    def calc_gamma_app_over_gamma_kutta(self, D_values, r_values, theta_values):
        kutta_appellian_array = np.zeros((len(D_values), 2))
        for i in tqdm(range(len(D_values)), desc="Calculating Appellian Roots"):  
            self.D = D_values[i]
            self.epsilon = self.calc_epsilon_from_D(D_values[i]) ##### before
            kutta_appellian_array[i, 0] = D_values[i]
            kutta_appellian_array[i, 1] = self.run_appellian_roots(r_values,theta_values, D_values[i])[0] / self.kutta_circulation  
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

    def calc_theta_convergence(self, D, r_values, theta_range_array):
        D = round(D, 4)
        # Ensure output directory exists
        output_dir = "text_files/theta_grid_convergence"
        os.makedirs(output_dir, exist_ok=True)
        # Storage for [num_theta_steps, % error in root, % error in value]
        appellian_roots = np.zeros((len(theta_range_array), 3))
        # Backup original attributes
        self.original_epsilon = self.epsilon
        self.original_D = self.D
        self.epsilon = self.calc_epsilon_from_D(D)
        original_dtheta = self.dtheta
        # To store previous values for percent error comparison
        previous_root = None
        previous_value = None
        for i in tqdm(range(len(theta_range_array)), desc=f"Calculating Appellian Roots at Varying theta refinement at D = {D}"):
            theta_values = hlp.list_to_range(theta_range_array[i])
            num_theta = theta_range_array[i][2]
            self.dtheta = num_theta # Update dtheta for current iteration
            appellian_root, xi_eta_vals, appellian_value = self.run_appellian_roots(r_values, theta_values, D)
            appellian_roots[i, 0] = num_theta
            if i == 0:
                # First iteration: can't compute percent error
                appellian_roots[i, 1] = 100.0
                appellian_roots[i, 2] = 100.0
            else:
                # Safeguards against division by zero
                if previous_root != 0:
                    root_error = abs((appellian_root - previous_root) / previous_root) * 100
                else:
                    root_error = np.nan
                if previous_value != 0:
                    value_error = abs((appellian_value - previous_value) / previous_value) * 100
                else:
                    value_error = np.nan
                appellian_roots[i, 1] = root_error
                appellian_roots[i, 2] = value_error
            previous_root = appellian_root
            previous_value = appellian_value
        # Save results to file
        file_name = f"{output_dir}/theta_convergence_{self.zeta_center.real}_alpha_{round(self.angle_of_attack * 180 / np.pi, 1)}_D_{D}.txt"
        np.savetxt(file_name, appellian_roots, delimiter=",", header="number_theta_steps,Appellian Root %,Appellian Value %", comments='')
        # Restore original parameters
        self.epsilon = self.original_epsilon 
        self.D = self.original_D
        self.dtheta = original_dtheta
        return appellian_roots
    
    def plot_theta_convergence(self, file_name_array: list):
        """This function plots the theta convergence of the Appellian roots and values for varying theta refinements"""
        linestyles = ['solid','dashed', (0, (3, 1, 1, 1)), 'solid','dashed', (0, (3, 1, 1, 1))]
        colors = ['black', 'gray', 'black', 'gray', 'black', 'gray']
        index = 0
        for filename in file_name_array:
            D_val = filename.rsplit("_", 1)[-1].rsplit(".", 1)[0]
            label = f"D = {round(float(D_val), 4)}"
            data = np.loadtxt(filename, delimiter=",", skiprows=2)
            plt.plot(data[:, 0], data[:, 1], 
                    label=label,
                    linestyle=linestyles[index % len(linestyles)], 
                    color=colors[index % len(colors)], 
                    linewidth=1.5)
            index += 1
        # invert x-axis
        plt.gca().invert_xaxis()
        plt.xscale("log")
        xticks = np.array([0.0001,0.001, 0.01, 0.1, 1.0])
        xtick_labels = ["1e-4","1e-3", "1e-2", "1e-1", "1e0"]
        plt.yscale("log")
        plt.xlabel("$\\theta$ Stepsize")
        plt.ylabel("$\\epsilon_{a}$ for $\\Gamma_A$", labelpad=5)
        plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.15), ncol=2,frameon=False)
        plt.tight_layout()
        file1 = f"figures/Appellian_Roots_theta_convergence_{self.zeta_center.real}_alpha_{round(self.angle_of_attack * 180 / np.pi, 1)}.svg"
        plt.savefig(file1, format='svg', bbox_inches='tight')
        print("theta convergence figure saved as:", file1)
        plt.show()
        plt.close()

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

def memoize_acceleration_func(func):
    cache = {}
    def wrapper(point, Gamma):
        key = (round(point[0], 12), round(point[1], 12), round(Gamma, 12))
        if key not in cache:
            # print(f"New cache entry for key: {key}")
            cache[key] = func(point, Gamma)
        # else:
            # print(f"Cache hit for key: {key}")
        return cache[key]
    wrapper.clear_cache = lambda: cache.clear()
    return wrapper

def double_nodes_periodic(x_vals: np.ndarray, full_period=2 * np.pi) -> np.ndarray:
    """Doubles periodic node count by inserting proper midpoints between each pair, wrapping around at the end."""
    x_new = []
    for i in range(len(x_vals)):
        x0 = x_vals[i]
        x1 = x_vals[(i + 1) % len(x_vals)]
        delta = (x1 - x0) % full_period
        midpoint = (x0 + delta / 2) % full_period
        x_new.extend([x0, midpoint])
    x_final = np.array(x_new)
    return np.unique(x_final)

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
    print("\n")
    if cyl.appellian_is_line_integral:
        r_range = [cyl.cylinder_radius, cyl.cylinder_radius, 0.01*cyl.cylinder_radius]
    else:
        r_range = [cyl.cylinder_radius, cyl.cylinder_radius, 0.5*cyl.cylinder_radius]
    r_values = hlp.list_to_range(r_range)
    theta_start = 0.0#cyl.theta_AJA  # this is the angle in Chi where the geometry starts
    theta_end = 2*np.pi #+ theta_start  # this is the angle in Chi where the geometry ends
    # theta_range = [0.0, 2*np.pi, cyl.one_appellian_theta_points]  # theta range from 0 to 2*pi with num_theta_steps steps
    theta_values1 = np.linspace(theta_start, theta_end, cyl.num_thetas_start, endpoint=False)  # create theta values from theta_start to theta_end with num_theta_steps steps
    # rotate the theta_values by dtheta/2 to avoid the singularity at theta = 0
    for i in range(cyl.num_times_thetas_doubled):
        theta_values1 = double_nodes_periodic(theta_values1)
    theta_values = theta_values1
    if cyl.is_compute_minimum_appellian:
        # print("running minimum Appellian roots for cylinder with D = ", cyl.D)
        Gamma = cyl.calc_circulation_J_airfoil()  # calculate the circulation of the airfoil
        time_start = time.time()
        appellian_root, xi_eta_array, appellian_value = cyl.run_appellian_roots(r_values, theta_values, cyl.D) # calculate the minimum Appellian value
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
        print("zeta0: ", cyl.zeta_center)
        print("angle_of_attack: ", np.rad2deg(cyl.angle_of_attack))
        print("velocity: ", cyl.freestream_velocity)
        print("num_r_vals: ", len(r_values))
        print("num_theta_vals: ", len(theta_values))
        print("Appellian_is_line_integral:", cyl.appellian_is_line_integral)
        print("Appellian_integration_method: ", cyl.integration_method)
        print("Appellian_calculation_time: ", time_end - time_start, " seconds")
        print("Appellian_Value: ", appellian_value)
        # appellian_root, xi_eta_vals, appelian_value = cyl.run_appellian_roots(r_values, theta_values, cyl.D)
        # cyl.circulation = appellian_root
        # print("theta steps: ", theta_range[2])
        # print("Appellian Circulation: ", appellian_root)
        # print("Appellian Value: ", appelian_value)
        # print("kutta circulation: ", cyl.kutta_circulation)
        # print("percent difference between Appellian and Kutta: ", 100*(appellian_root - cyl.kutta_circulation)/cyl.kutta_circulation)
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
    cyl.plot_geometry_zeta()
    cyl.get_full_geometry()
    cyl.plot_geometry()
    zeta_trailing_edge_focus, zeta_leading_edge_focus, z_leading_edge_focus, z_trailing_edge_focus = cyl.get_and_plot_foci()
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
    if cyl.is_calc_theta_convergence and cyl.is_calc_r_convergence:
        # Conv_d_values = np.array([1.0,0.0])
        Conv_d_values = np.array([0.1])
        # Conv_d_values = np.array([1.0])
        # reference_values = np.array([251327.4123, 186345.459117856])
        # reference_values = np.array([251327.4123]) # this is the value for D = 1.0
        left_riemann_reference_values = np.array([90572.51950146342])
        left_riemann_gamma_values = 7.671488490881828 # for D = 0.1
        trapezoidal_reference_values = np.array([90572.51954043655])
        trapezoidal_gamma_values = 7.671488490880593 # for D = 0.1
        romberg_reference_values = np.array([90572.70008770275])
        romberg_gamma_values = 7.671488490881888 # for D = 0.1
        simpson_reference_values = np.array([90572.51954044093])
        simpson_gamma_values = 7.671488490881632 # for D = 0.1
        reference_values = romberg_reference_values
        Gamma_values = np.array([romberg_gamma_values]) # this is the value for D = 0.1
        # Gamma_values = np.array([0.1]) # this is the value for D = 0.0
        for i in range(len(Conv_d_values)):
            cyl.D = Conv_d_values[i]
            cyl.epsilon = cyl.calc_epsilon_from_D(cyl.D)
            theta_start = cyl.theta_AJA 
            theta_end = cyl.theta_AJA + 2*np.pi
            theta_range = (theta_start, theta_end)
            tol = 1e-12
            is_relative_error = True
            cyl.run_two_stage_convergence(Gamma_values[i], reference_values[i], theta_range=theta_range, tol = tol, run_theta_stage = True, run_r_stage = False, relative_error=is_relative_error, num_double_points=cyl.num_times_thetas_doubled)


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




    
    # test_point_in_z = cyl.full_z_surface[12]
    # test_point_in_zeta = cyl.z_to_zeta(test_point_in_z[0] + 1j*test_point_in_z[1], cyl.epsilon)
    # test_point_in_Chi = cyl.zeta_to_Chi(test_point_in_zeta)
    # r_chi, theta_chi = hlp.xy_to_r_theta(test_point_in_Chi.real, test_point_in_Chi.imag)
    # r_0, theta_0 = hlp.xy_to_r_theta(cyl.zeta_center.real, cyl.zeta_center.imag)
    # analytic_conv_accel = cyl.analytic_conv_accel_for_line_int_comp_conj([r_chi, theta_chi], cyl.circulation)
    # print("Line Int Convective Acceleration is                 ", -1/32*analytic_conv_accel)
    # Spencer_accel_squared_comp_conj = cyl.analytic_conv_accel_square_comp_conj([r_chi,theta_chi], cyl.circulation)
    # print("Spencer Alternat Convective Acceleration squared is ", Spencer_accel_squared_comp_conj/2)
    # numerical_double_int = cyl.numerical_convective_acceleration([r_chi, theta_chi], cyl.circulation)
    # print("Numerical Convective Acceleration squared is        ", numerical_double_int/2)
    
    
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


