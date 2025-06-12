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

class cylinder(potential_flow_object):
    """This is a class that creates a cylinder object and performs calculations specific to a cylinder"""
    def __init__(self, json_file):
        self.cyl_json_file = json_file
        self.parse_cylinder_json()
        # print("MRO:", [cls.__name__ for cls in self.__class__.mro()])
        potential_flow_object.__init__(self, json_file) # this is doing self.() of all the variables we want to in the super class

    def parse_cylinder_json(self):
        """This function reads the json file and stores the cylinder specific data in a dictionary"""
        with open(self.cyl_json_file, 'r') as json_handle:
            input = json.load(json_handle)
            print("\n")
            # use the parse_dictionary_or_return_default function from the helper file to get the values from the json file. the function inputs are as follows: parse_dictionary_or_return_default(input, ["section", "key"], default_value)
            # appellian stuff
            self.is_compute_appellian = hlp.parse_dictionary_or_return_default(input, ["appellian", "is_compute_appellian"], False)
            self.is_plot_appellian = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_appellian"], False)
            self.is_plot_appellian_for_varying_circulation = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_appellian_for_varying_circulation"], False)
            self.is_calc_circulation_for_varying_shape = hlp.parse_dictionary_or_return_default(input, ["appellian", "is_calc_circulation_for_varying_shape"], False)
            self.is_plot_circulation_for_varying_shape = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_circulation_for_varying_shape"], False)
            self.dtheta = 2*np.pi/hlp.parse_dictionary_or_return_default(input, ["appellian", "num_theta_steps"], 1000)
            self.is_calc_rb_comparisons = hlp.parse_dictionary_or_return_default(input, ["appellian", "is_calc_rb_comparisons"], False)
            self.is_plot_rb_comparisons = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_rb_comparisons"], False)
            self.is_calc_convergence_to_kutta = hlp.parse_dictionary_or_return_default(input, ["appellian", "is_calc_convergence_to_kutta"], False)
            self.is_plot_convergence_to_kutta = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_convergence_to_kutta"], False)
            self.is_calc_theta_convergence = hlp.parse_dictionary_or_return_default(input, ["appellian", "is_calc_theta_convergence"], False)
            self.is_plot_theta_convergence = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_theta_convergence"], False)
            self.is_calc_r_convergence = hlp.parse_dictionary_or_return_default(input, ["appellian", "is_calc_r_convergence"], False)
            self.is_plot_r_convergence = hlp.parse_dictionary_or_return_default(input, ["plot", "is_plot_r_convergence"], False)
            self.appellian_is_line_integral = hlp.parse_dictionary_or_return_default(input, ["appellian", "appellian_is_line_integral"], True)
            if self.appellian_is_line_integral:
                self.appellian_is_area_integral = False
            else:
                self.appellian_is_area_integral = True
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
            if self.type == "cylinder":
                self.circulation = hlp.parse_dictionary_or_return_default(input, ["operating", "circulation_strength"], 0.0) # this is the circulation strength. This is used to calculate Lift through the Kutta-Joukowski theorem.
                # print("self.circulation", self.circulation)
                zeta_center = hlp.parse_dictionary_or_return_default(input, ["geometry", "zeta_0"], [-0.01, 0.0]) # this is a list of two elements, real and imaginary parts of zeta_0. We need to convert it to a complex number and normalize it by the cylinder radius.
                self.zeta_center = (zeta_center[0] + 1j*zeta_center[1])/self.cylinder_radius
                if self.use_shape_parameter_D:
                    self.epsilon = self.calc_epsilon_from_D(self.D)
                    # print("epsilon from D = " + str(self.D) + " and zeta0 = " + str(self.zeta_center) + " is " + str(self.epsilon))
                    self.kutta_circulation = self.calc_circulation_J_airfoil()
                    # print("Kutta circulation", self.kutta_circulation)
                else:
                    self.epsilon = hlp.parse_dictionary_or_return_default(input, ["geometry", "epsilon"], self.cylinder_radius)
                    self.kutta_circulation = self.calc_circulation_J_airfoil()
                    # print("Kutta circulation", self.kutta_circulation)
                    D = self.calc_D_from_epsilon()
                    # print("D from epsilon = " + str(self.epsilon) + " and zeta0 = " + str(self.zeta_center) + " is " + str(D))
            
            else:
                self.zeta_center = self.calc_J_airfoil_zeta_center() # uses design CL and thickness to calculate zeta center
                # print("J airfoil zeta center", self.zeta_center)
                self.epsilon = self.calc_J_airfoil_epsilon() # uses the cylinder radius, and zeta center to calculate epsilon
                # print("J airfoil epsilon", self.epsilon)
                self.circulation = self.calc_circulation_J_airfoil() # uses the Kutta condition to calculate circulation
                # print("J airfoil circulation", self.circulation)
                # calculate gamma so that it satisfies the Kutta condition for the airfoil
            
            # self.transformation_type = input["geometry"]["transformation_type"]
            self.transformation_type = hlp.parse_dictionary_or_return_default(input, ["geometry", "transformation_type"], "joukowski") # this is the type of transformation we are using. It can be either "joukowski" or "taha". Default is Joukowski.
            self.zeta_to_z, self.z_to_zeta, self.dZ_dzeta, self.d2Z_dzeta2 = self.general_zeta_to_z_transformation()
            if self.transformation_type == "taha":
                self.tau = hlp.parse_dictionary_or_return_default(input, ["geometry", "taha", "tau"], 0.1) # this is the assymetry parameter in the Taha refining Kutta paper (see Eqs 8-9)
                self.D = hlp.parse_dictionary_or_return_default(input, ["geometry", "taha", "D"], 1.0) # this is the D parameter in the Taha refining Kutta paper (see Eqs 8-9)
                self.epsilon = self.calc_taha_epsilon_from_tau() # this is the epsilon parameter in the Taha refining Kutta paper (see Eqs 8-9)
                self.C = self.calc_taha_C_from_epsilon() + 1j*0.0 # this is the C parameter in the Taha refining Kutta paper (see Eqs 8-9)
                self.zeta_center = self.calc_taha_zeta_center_from_epsilon() + 1j*0.0 # this is the zeta center in the Taha refining Kutta paper (see Eqs 8-9)
                print("\nTaha zeta center", self.zeta_center)
                print("Taha epsilon", self.epsilon)
                print("Taha C", self.C)
                print("Taha D", self.D)
                print("Taha tau", self.tau)
                equivalent_Joukowski_epsilon = self.equivalent_Joukowski_epsilon_from_taha_C_and_D(self.C, self.D)
                print("Equivalent Joukowski epsilon", equivalent_Joukowski_epsilon)

    def general_zeta_to_z_transformation(self):
        """"""
        if self.transformation_type == 'joukowski':
            return self.Joukowski_zeta_to_z, self.Joukowski_z_to_zeta, self.dZ_dzeta_Joukowski, self.d2Z_dzeta2_Joukowski
        elif self.transformation_type == 'taha':
            return self.Taha_zeta_to_z, self.Taha_z_to_zeta, self.dZ_dzeta_Taha, self.d2Z_dzeta2_Taha
        else:
            raise ValueError("Invalid transformation type. Choose 'joukowski' or 'taha'.")
        
    def equivalent_Joukowski_epsilon_from_taha_C_and_D(self, C: float, D: float):
        """This function calculates the equivalent Joukowski epsilon from Taha C and D"""
        return self.cylinder_radius-C*np.sqrt((1-D)/(1+D))

    def calc_epsilon_from_D(self, D: float):
        """This function calculates epsilon from D"""
        epsilon_o = self.calc_J_airfoil_epsilon()
        epsilon = D*(1-epsilon_o)+epsilon_o
        return epsilon
    
    def calc_taha_epsilon_from_tau(self):
        """This function calculates epsilon from tau"""
        return 4*self.tau/(3*np.sqrt(3))
    
    def calc_taha_C_from_epsilon(self):
        """This function calculates C from cylinder radius, D and epsilon"""
        return 1/(1+self.epsilon) # right after Eq. 13 in The Principle of Minimum Pressure Gradient as a Selection Criterion for Weak Solutions of Euler’s Equation paper by Taha
    
    def calc_taha_zeta_center_from_epsilon(self):
        """This function calculates zeta center from epsilon"""
        return -self.epsilon*self.C
    
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
        # assuming theta_start to theta_end is a closed loop, we can see the percent distance between theta_start and theta_half. That percentage will determine the percentage of points allotted to the first half of the array and the remaining percentage will be allotted to the second half of the array.
        # Calculate the number of points in each segment based on the percentage distance
        percentage_distance = abs((theta_half - theta_start) / (theta_end - theta_start)) # This gives the fraction of the total angle that the first segment covers
        n1 = int(number_of_points * percentage_distance)  # Number of points in the first segment
        n2 = number_of_points - n1
        # # Ensure the total points include only one instance of theta_half
        # n1 = number_of_points // 2
        # n2 = number_of_points - n1  # ensures total = n1 + n2
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
        # connect the first and last points to make a closed shape
        plt.plot([self.zeta_geom_array[0][0], self.zeta_geom_array[-1][0]], [self.zeta_geom_array[0][1], self.zeta_geom_array[-1][1]], linestyle=linetype, color=color)
        size = 15
        # plt.scatter(self.zeta_center.real, self.zeta_center.imag, color=color, marker='o', s = size, facecolors='none', label="$\\zeta_0$")
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
    
    def get_and_plot_foci(self):
        """This function calculates and plots the foci of the ellipse using epsilon"""
        self.zeta_trailing_edge_focus = 1*(self.cylinder_radius- self.epsilon)/self.cylinder_radius
        self.zeta_leading_edge_focus = -1*(self.cylinder_radius - self.epsilon)/self.cylinder_radius
        self.z_trailing_edge_focus = self.zeta_to_z(self.zeta_trailing_edge_focus, self.epsilon)
        self.z_leading_edge_focus = self.zeta_to_z(self.zeta_leading_edge_focus, self.epsilon)
        # reduce size of markers 
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
        plt.plot(upper[:,0], upper[:,1], label="$z$", color="black")
        plt.plot(lower[:,0], lower[:,1], color="black")
        plt.plot(upper_zeta[:,0], upper_zeta[:,1],linestyle="--", label="$\\zeta$", color="black")
        plt.plot(lower_zeta[:,0], lower_zeta[:,1],linestyle="--", color="black")
        plt.plot(upper_zeta_incorrect[:,0], upper_zeta_incorrect[:,1], label="$\\zeta$ incorrect", linestyle="dotted", color="black")
        plt.plot(lower_zeta_incorrect[:,0], lower_zeta_incorrect[:,1], linestyle="dotted", color="black")
        self.plot_geometry_settings()
    
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
    
    def dZ_dzeta_Joukowski(self, zeta):
        """This function calculates the derivative of Z with respect to zeta for a Joukowski transformation"""
        epsilon, R, = self.epsilon, self.cylinder_radius
        return 1 - (R-epsilon)**2/(zeta)**2
    
    def d2Z_dzeta2_Joukowski(self, zeta):
        """This function calculates the second derivative of Z with respect to zeta for a Joukowski transformation"""
        epsilon, R = self.epsilon, self.cylinder_radius
        return 2*(R-epsilon)**2/(zeta)**3
    
    def dZ_dzeta_Taha(self, zeta):
        """""This function calculates the derivative of Z with respect to zeta for a Taha transformation"""
        return 1 - (1-self.D)/(1+self.D)*(self.C**2)/(zeta**2) 
    
    def d2Z_dzeta2_Taha(self, zeta):
        """"""
        return 2*(1-self.D)/(1+self.D)*(self.C**2)/(zeta)**3
    
    def velocity(self, point_xi_eta_in_z_plane, Gamma):
        """This function calculates the velocity at a given point in the flow field in the z plane"""
        z = point_xi_eta_in_z_plane[0] + 1j*point_xi_eta_in_z_plane[1]
        zeta = self.z_to_zeta(z, self.epsilon)
        dZ_dzeta = self.dZ_dzeta(zeta)
        velocity = self.dPhi_dzeta(zeta, Gamma) / dZ_dzeta # eq 107
        velocity_complex = np.array([velocity.real, -velocity.imag])
        return velocity_complex
    
    def acceleration(self, point_xi_eta_in_z_plane, Gamma):
        """This function calculates the acceleration at a given point in the flow field in the z plane"""
        z = point_xi_eta_in_z_plane[0] + 1j*point_xi_eta_in_z_plane[1]
        zeta = self.z_to_zeta(z, self.epsilon)
        dphi_dzeta, d2Phi_dzeta2, dz_dzeta, d2z_dzeta2 = self.dPhi_dzeta(zeta, Gamma), self.d2Phi_dzeta2(zeta, Gamma), self.dZ_dzeta(zeta), self.d2Z_dzeta2(zeta)
        acceleration = (dz_dzeta*d2Phi_dzeta2- dphi_dzeta*d2z_dzeta2)/(dz_dzeta**3)
        acceleration_complex = np.array([acceleration.real, -acceleration.imag])
        return acceleration_complex

    def analytic_conv_accel_for_line_int_comp_conj(self, point_r_theta_in_Chi, Gamma, step_size = 1e-10):
        """calculates the convective acceleration using the line integral method"""
        V_inf, epsilon, R, alpha, zeta_0 = self.freestream_velocity, self.epsilon, self.cylinder_radius, self.angle_of_attack, self.zeta_center
        r, theta = point_r_theta_in_Chi[0], point_r_theta_in_Chi[1]
        r_plus_dr = r + step_size
        r_0, theta_0 = hlp.xy_to_r_theta(zeta_0.real, zeta_0.imag)
        xi_chi, eta_chi = hlp.r_theta_to_xy(r, theta)
        xi_chi_plus_dr, eta_chi_plus_dr = hlp.r_theta_to_xy(r_plus_dr, theta)
        chi = xi_chi + 1j*eta_chi
        chi_plus_dr = xi_chi_plus_dr + 1j*eta_chi_plus_dr
        zeta = self.Chi_to_zeta(chi)
        zeta_plus_dr = self.Chi_to_zeta(chi_plus_dr)
        dz_dzeta, omega_of_zeta = self.dZ_dzeta(zeta), self.dPhi_dzeta(zeta, Gamma)
        G_of_zeta = 1/dz_dzeta
        dz_dzeta_plus_dr, omega_of_zeta_plus_dr = self.dZ_dzeta(zeta_plus_dr), self.dPhi_dzeta(zeta_plus_dr, Gamma)
        G_of_zeta_plus_dr = 1/dz_dzeta_plus_dr
        integrand = G_of_zeta**2*np.conj(G_of_zeta)**2*omega_of_zeta**2*np.conj(omega_of_zeta)**2
        integrand_plus_dr = G_of_zeta_plus_dr**2*np.conj(G_of_zeta_plus_dr)**2*omega_of_zeta_plus_dr**2*np.conj(omega_of_zeta_plus_dr)**2
        # find the partial derivative of the integrand with respect to r
        line_int = (integrand_plus_dr - integrand)/step_size
        return np.real(line_int)

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
    
    def taha_analytic_conv_accel_square_comp_conj(self, point_r_theta_in_Chi, Gamma): # need to adjust so that it uses the Taha transformation
        """This function calculates the pressure gradient at a given point in the flow field in the z plane"""
        r, theta = point_r_theta_in_Chi[0], point_r_theta_in_Chi[1]
        xi_chi, eta_chi = hlp.r_theta_to_xy(r, theta)
        chi = xi_chi + 1j*eta_chi
        zeta = self.Chi_to_zeta(chi)
        dphi_dzeta, d2Phi_dzeta2, dz_dzeta, d2z_dzeta2 = self.dPhi_dzeta(zeta, Gamma), self.d2Phi_dzeta2(zeta, Gamma), self.dZ_dzeta(zeta), self.d2Z_dzeta2(zeta)
        G_of_zeta = 1/dz_dzeta
        G_of_zeta_comp_conj = np.conj(G_of_zeta)
        G_of_zeta_squared = G_of_zeta * G_of_zeta_comp_conj
        R, epsilon = self.cylinder_radius, self.epsilon
        dG_dzeta = -2*(R-epsilon)**2/((1-(R-epsilon)**2/zeta**2)**2*zeta**3)
        dG_dzeta_comp_conj = np.conj(dG_dzeta)
        omega_of_zeta = dphi_dzeta
        omega_of_zeta_comp_conj = np.conj(omega_of_zeta)
        omega_of_zeta_squared = omega_of_zeta * omega_of_zeta_comp_conj 
        a_of_zeta = d2Phi_dzeta2
        integrand = G_of_zeta_comp_conj * a_of_zeta + omega_of_zeta_squared*dG_dzeta_comp_conj
        integrand_comp_conj = np.conj(integrand)
        conv_accel_squared =  G_of_zeta_squared * integrand * integrand_comp_conj 
        return np.real(conv_accel_squared)
    
    def numerical_cartesian_convective_acceleration(self, point_xi_eta, Gamma, step=1e-10):
        """Calculates the convective acceleration numerically in the Cartesian (x, y) plane."""
        point_xy = [point_xi_eta[0], point_xi_eta[1]]
        velocity_func = self.velocity
        omega_xi, omega_eta = velocity_func(point_xy, Gamma)
        derivs = self.function_plus_minus_step_variable(point_xy, Gamma, step, velocity_func)
        partials = [hlp.central_difference(derivs[i], derivs[i+1], step) for i in range(0, 8, 2)]  
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
        perturbations = [(xi + step, eta), (xi - step, eta),  (xi, eta + step), (xi, eta - step)]
        velocities = [vel_func(p, Gamma) for p in perturbations]  # [(u,v) at each perturbed point]
        omega_xi_plus_dxi, omega_eta_plus_dxi = velocities[0]
        omega_xi_minus_dxi, omega_eta_minus_dxi = velocities[1]
        omega_xi_plus_d_eta, omega_eta_plus_d_eta = velocities[2]
        omega_xi_minus_d_eta, omega_eta_minus_d_eta = velocities[3]
        return (omega_xi_plus_dxi, omega_xi_minus_dxi, omega_xi_plus_d_eta, omega_xi_minus_d_eta,omega_eta_plus_dxi, omega_eta_minus_dxi, omega_eta_plus_d_eta, omega_eta_minus_d_eta)  # For du/dx, du/dy, dv/dx, dv/dy

    def appellian_acceleration_loop(self, Gamma: float, r_values: np.array, theta_values: np.array, dr: float, dtheta: float, D, acceleration: callable):
        """"""
        Appellian_value = 0.0
        if self.appellian_is_area_integral: # using complex conjugate
            for j in range(len(r_values)):
                for k in range(len(theta_values)):
                    area_element = r_values[j]*dr*dtheta
                    accel = acceleration([r_values[j], theta_values[k]], Gamma)
                    Appellian_value += accel*area_element
            return Appellian_value*(0.5)
                # Define the integration bounds
        elif self.appellian_is_line_integral:
            for k in range(len(theta_values)):
                area_element = r_values[0]*dtheta
                accel = acceleration([r_values[0], theta_values[k]], Gamma)
                Appellian_value += accel*area_element
            return Appellian_value*(-0.03125) # 1/32
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
        return Appellian_value*0.5
    
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
        theta_start = self.calculate_aft_stagnation_theta_in_Chi_from_Gamma(Gamma)
        theta_end = 2*np.pi + theta_start
        theta_range = [theta_start, theta_end, self.dtheta]
        theta_values = hlp.list_to_range(theta_range)
        # add dtheta to all values in theta_values
        theta_values = [theta + self.dtheta/2 for theta in theta_values]
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
            Appellian_array = self.appellian_acceleration_loop(Gamma, r_values, theta_values, dr_original, dtheta_original, D, self.analytic_conv_accel_square_comp_conj)            
        elif self.appellian_is_line_integral:
            r_values = np.array([self.cylinder_radius])
            Appellian_array = self.appellian_acceleration_loop(Gamma, r_values, theta_values, dr_original, dtheta_original, D, self.analytic_conv_accel_for_line_int_comp_conj)
        else: 
            Appellian_array = self.appellian_acceleration_loop(Gamma, r_values, theta_values, dr, self.dtheta, D, self.numerical_cartesian_convective_acceleration)
        return Appellian_array
    
    def Joukowski_zeta_to_z(self, zeta: complex, epsilon: float):
        """This function takes in a zeta coordinate and returns the z coordinate"""
        if np.isclose(zeta.real, 0.0) and np.isclose(zeta.imag, 0.0):
            z = zeta
        else:
            z = zeta + (self.cylinder_radius - epsilon)**2/zeta # eq 96
        return z
    
    def Taha_zeta_to_z(self, zeta: complex, epsilon: float):
        """This function takes in a zeta coordinate and returns the z coordinate"""
        if np.isclose(zeta.real, 0.0) and np.isclose(zeta.imag, 0.0):
            z = zeta  
        else:
            z = zeta + ((1 - self.D)/(1 + self.D)) * (self.C**2 / zeta)
        return z
    
    def zeta_to_Chi(self, zeta: complex):
        """This function takes in a zeta coordinate and returns the Chi coordinate"""
        Chi = zeta - self.zeta_center
        return Chi

    def Chi_to_zeta(self, Chi: complex):
        """This function takes in a Chi coordinate and returns the zeta coordinate"""
        zeta = Chi + self.zeta_center
        return zeta

    def Joukowski_z_to_zeta(self, z: complex, epsilon: float, D=0.0): # eq 104 
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
    
    def Taha_z_to_zeta(self, z: complex, D: float = 0.0):
        """Given a z coordinate, return the correct zeta coordinate for the transformation:
        z = zeta + A*C**2/zeta, where A = (1-D)/(1+D)
        """
        A = (1 - D) / (1 + D)
        AC2 = A * self.C**2
        z_1 = z**2 - 4 * AC2

        if z_1.real > 0:
            zeta = (z + np.sqrt(z_1)) / 2
            zeta_2 = (z - np.sqrt(z_1)) / 2
        elif z_1.real < 0:
            zeta = (z - 1j * np.sqrt(-z_1)) / 2
            zeta_2 = (z + 1j * np.sqrt(-z_1)) / 2
        elif z_1.imag >= 0:
            zeta = (z + np.sqrt(z_1)) / 2
            zeta_2 = (z - np.sqrt(z_1)) / 2
        else:
            zeta = (z - 1j * np.sqrt(-z_1)) / 2
            zeta_2 = (z + 1j * np.sqrt(-z_1)) / 2

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
        z_leading_edge = self.Joukowski_zeta_to_z(zeta_leading_edge, self.epsilon)
        z_trailing_edge = self.Joukowski_zeta_to_z(zeta_trailing_edge, self.epsilon)
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
        self.surface_points = np.copy(unique_points)
        
    def output_J_airfoil(self):
        # flip the lower coordinates so that they go from the trailing edge to the leading edge
        self.surface_points = self.full_z_surface
        self.make_surface_points_go_from_lower_trailing_to_upper_trailing()
        self.surface_points = self.shift_joukowski_airfoil()
        # if self.export_geometry is True, then export the geometry to a txt file
        print("Exporting geometry to txt file...")
        file_name = "Joukowski" + self.type + "_" + str(self.n_geom_points) + ".txt"
        # save text file with the geometry in double precision
        np.savetxt(file_name, self.surface_points, delimiter = ",", header = "x, y", comments = "")
        print("Geometry exported to", file_name)

    def run_J_airfoil_stuff(self):
        """"""
        lift = self.calc_J_airfoil_CL()
        print("CL Joukowski: ", lift)
        Cmo = self.calc_J_airfoil_Cmz((0+1j*0))
        print("Cm0 Joukowski: ", Cmo)
        c4 = self.calc_J_airfoil_c4()
        print("C4 location Joukowski: ", c4)
        # now find the moment coefficient at c4
        Cmc4 = self.calc_J_airfoil_Cmz(c4)
        print("Cmc4 Joukowski: ", Cmc4)
        # now create the full geometry of the airfoil and export it to a text file
        self.output_J_airfoil()

    def run_appellian_roots(self, r_vals, theta_vals, D):
        """This function runs the Appellian stuff"""
        xi_eta_array = np.zeros((len(r_vals)*len(theta_vals), 2))
        if self.appellian_minimization_method != "roots_of_derive_of_poly_fit":
            Gamma_start = self.kutta_circulation
            appellian_root, appellian_value = hlp.newtons_method(self.numerically_integrate_appellian, Gamma_start, r_values = r_vals, theta_values = theta_vals, D=D)
            return appellian_root, xi_eta_array, appellian_value
        else: # poly fit
            if self.kutta_circulation > 0: # if self.kutta_circulation is positive, then the gamma range goes from 0 to the kutta circulation in a step size that makes it so there are 10 points in the range
                step_size = self.kutta_circulation/10
                Gamma_range = [0.0, self.kutta_circulation, step_size]
            else: # if the kutta circulation is negative, then the gamma range goes from the kutta circulation to 0 in steps of 0.1
                step_size = abs(self.kutta_circulation)/10
                Gamma_range = [self.kutta_circulation, 0.0, step_size]
            Gamma_vals = hlp.list_to_range(Gamma_range)
            # print("Gamma_vals", Gamma_vals)
            poly_order = 4
            appellian_root, appellian_value = hlp.polyfit(self.numerically_integrate_appellian, r_vals, Gamma_vals, poly_order, self.is_plot_appellian, "$\\Gamma$", "S", "Appellian Function", D)
            # appellian_root, appellian_value = hlp.polyfit(self.numerically_integrate_appellian, r_vals, Gamma_vals, poly_order, self.is_plot_appellian_for_varying_circulation, "$\\Gamma$", "S", "Appellian Function", D)
            return appellian_root, xi_eta_array, appellian_value
        
    def calc_cL_from_appellian_root_gamma(self, Gamma_root):
        """CL = Gamma/(0.5*V_inf*L)"""
        x_trailing_minus_leading = self.x_trailing_edge - self.x_leading_edge
        # transform this purely real number to a complex number
        x_trailing_minus_leading = x_trailing_minus_leading + 0j
        # now find the length of the airfoil in the z plane
        cbar = self.zeta_to_z(x_trailing_minus_leading, self.epsilon)
        # now calculate CL
        CL = Gamma_root/(0.5*self.freestream_velocity*cbar.real)
        return CL
    
    def calc_gamma_app_over_gamma_kutta(self, D_values, r_values, theta_values):
        kutta_appellian_array = np.zeros((len(D_values), 2))
        # self.D
        # self.tau = 0.1 ######## remove
        # epsilon_held = self.calc_taha_epsilon_from_tau()
        # C = 1/(1+epsilon_held)
        if self.transformation_type == "joukowski":
            for i in tqdm(range(len(D_values)), desc="Calculating Appellian Roots"):  
                self.D = D_values[i]
                # self.epsilon = self.equivalent_Joukowski_epsilon_from_taha_C_and_D(C, D_values[i])
                self.epsilon = self.calc_epsilon_from_D(D_values[i]) ##### before
                kutta_appellian_array[i, 0] = D_values[i]
                kutta_appellian_array[i, 1] = self.run_appellian_roots(r_values,theta_values, D_values[i])[0] / self.kutta_circulation  
        else:
            for i in tqdm(range(len(D_values)), desc="Calculating Appellian Roots"):  
                self.D = D_values[i]
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
        # Manually set the tick locations to match the taha figure
        # plt.xticks([0.005, 0.05, 1.0], ["0.005", "0.050", "1.000"])
        # Adjust axis limits
        plt.axhline(0, color='gray')
        # plt.xlim(0.001, 1.0)  # Lower limit should be slightly less than 0.005
        xticks = np.array([0.0,0.2,0.4,0.6,0.8,1.0])
        xtick_labels = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
        plt.xticks(xticks, xtick_labels)
        # move the first tick label to the right slightly
        # plt.xticks([0.0001, 0.001, 0.01, 0.1], ["1e-4", "1e-3", "1e-2", "1e-1"])
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
        # plt.legend(loc="center right")
        return figure_name
    
    def create_dsweep_gif(self, kutta_appellian_array, r_values, theta_values):
        """This function creates a gif of the D sweep"""
        if self.is_calc_circulation_for_varying_shape:
            if self.transformation_type == "joukowski":
                for i in range(len(kutta_appellian_array)):
                    plot_names_list = []
                    self.D = kutta_appellian_array[i,0]
                    print("D: ", self.D)
                    self.epsilon = self.calc_epsilon_from_D(self.D)
                    self.circulation = kutta_appellian_array[i,1] * self.kutta_circulation
                    gamma_over_kutta_name = self.plot_gamma_over_gamma_kutta(kutta_appellian_array, is_plot_current_D = True, kutta_and_current_D = [kutta_appellian_array[i,0], kutta_appellian_array[i,1]])
                    # plt.gcf().set_size_inches(3.25, 3.5)  # force figure size in inches
                    plt.savefig(gamma_over_kutta_name, dpi=300, bbox_inches=None, pad_inches=0)
                    plot_names_list.append(gamma_over_kutta_name)
                    plt.close()
                    plt.figure()
                    if self.is_plot_z_plane_surface_pressure_distribution:
                        # circulation = self.run_appellian_roots(r_values, theta_values, self.D)[0]
                        pressure_name_one = self.plot_surface_pressures(self.circulation, self.D)
                        pressure_name_one = pressure_name_one.replace(".png", "_firstpressure.png")
                        plt.legend(loc="upper right")
                        # plt.gcf().set_size_inches(3.25, 3.5)  # force figure size in inches
                        plt.savefig(pressure_name_one, dpi=300, bbox_inches=None, pad_inches=0)
                        plot_names_list.append(pressure_name_one)
                        plt.close()
                        plt.figure()
                        self.angle_of_attack = self.angle_of_attack + 10*np.pi/180
                        self.kutta_circulation = self.calc_circulation_J_airfoil()
                        circulation = self.run_appellian_roots(r_values, theta_values, self.D)[0]
                        pressure_name_two = self.plot_surface_pressures(circulation, self.D)
                        plt.legend(loc="upper right")
                        pressure_name_two = pressure_name_two.replace(".png", "_secondpressure.png")
                        # plt.gcf().set_size_inches(3.25, 3.5)  # force figure size in inches
                        plt.savefig(pressure_name_two, dpi=300, bbox_inches=None, pad_inches=0)
                        plot_names_list.append(pressure_name_two)
                        plt.close()
                        # set the angle of attack back to the original value
                        self.angle_of_attack = self.angle_of_attack - 10*np.pi/180
                        self.kutta_circulation = self.calc_circulation_J_airfoil()
                    self.get_full_geometry_zeta(number_of_points = self.output_points)
                    self.plot_geometry_zeta()
                    self.get_full_geometry()
                    self.plot_geometry()
                    self.plot()
                    self.plot_geometry_settings()
                    geometry_name = gamma_over_kutta_name.replace(".png", "_geometry.png")
                    # plt.gcf().set_size_inches(3.25, 3.5)  # force figure size in inches
                    plt.savefig(geometry_name, dpi=300, bbox_inches=None, pad_inches=0)
                    plot_names_list.append(geometry_name)
                    plt.close()
                    # plt.figure()
                    combined_plot_name = gamma_over_kutta_name.replace(".png", "_combined__"+str(i)+".png")
                    plot_names_array = np.array(plot_names_list)
                    hlp.combine_plots(plot_names_array, combined_plot_name)
            else:
                for i in range(len(kutta_appellian_array)):
                    self.D = kutta_appellian_array[i,0]
                    print("D: ", self.D)
                    self.circulation = kutta_appellian_array[i,1] * self.kutta_circulation
                    gamma_over_kutta_name = self.plot_gamma_over_gamma_kutta(kutta_appellian_array, is_plot_current_D = True, kutta_and_current_D = [kutta_appellian_array[i,0], kutta_appellian_array[i,1]])
                    # plt.gcf().set_size_inches(3.25, 3.5)  # force figure size in inches
                    plt.savefig(gamma_over_kutta_name, dpi=300, bbox_inches=None, pad_inches=0)
                    plt.close()
                    plt.figure()
                    self.get_full_geometry_zeta(number_of_points=self.output_points)
                    self.plot_geometry_zeta()
                    self.get_full_geometry()
                    self.plot_geometry()
                    self.plot()
                    self.plot_geometry_settings()
                    geometry_name = gamma_over_kutta_name.replace(".png", "_geometry.png")
                    # plt.gcf().set_size_inches(3.25, 3.5)  # force figure size in inches
                    plt.savefig(geometry_name, dpi=300, bbox_inches=None, pad_inches=0)
                    plt.close()
                    plt.figure()
                    combined_plot_name = gamma_over_kutta_name.replace(".png", "_combined__"+str(i)+".png")
                    hlp.combine_two_plots(gamma_over_kutta_name, geometry_name, combined_plot_name)
                    # Delete all non-combined figures
        for filename in os.listdir("figures"):
            # if filename does not contain double underscore 
            if "__" not in filename:
                file_path = os.path.join("figures", filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
            # if i == 0:
            #     break
        hlp.create_animation_from_all_figs_in_folder("figures", "figures/Joukowski_Cylinder_animation.gif")

    def run_r_b_comparisons(self, r_values, theta_values, D):
        """"""
        # make an array of zeros to hold the values of the Appellian roots
        appellian_roots = np.zeros((len(r_values), 3))
        # print("r_array", r_array)
        self.epsilon = self.calc_epsilon_from_D(D)
        for i in tqdm(range(1, len(r_values)), desc="Calculating Appellian Roots at Varying r/R at D = " + str(D)):
            # get current r values array by cutting the r_values array off after the current index (but include everything before and including the current index)
            current_r_values = r_values[:i+1] # include everything before and including the current index
            appellian_root, xi_eta_vals, appellian_value = self.run_appellian_roots(current_r_values, theta_values, D)
            appellian_roots[i, 0] = r_values[i]
            appellian_roots[i, 1] = appellian_root
            appellian_roots[i, 2] = appellian_value
        # save the Appellian roots to a text file whose name contains zeta_0, the angle of attack, theta_stepsize, and D
        file_name = "text_files/Appellian_Roots_r_b_" + str(self.zeta_center.real) + "_alpha_" + str(round(self.angle_of_attack*180/np.pi, 1)) + "_theta_stepsize_" + str(round(theta_values[1]-theta_values[0], 4)) + "_D_" + str(D) + ".txt"
        np.savetxt(file_name, appellian_roots, delimiter = ",", header = "D, Appellian Root")

    def plot_rb_comparisons(self, file_name_array: list):
        tightly_dash_dotted = (0, (3, 1, 1, 1))  # tightly dashdotted
        linestyles = ['solid','dashed',tightly_dash_dotted, 'solid','dashed',tightly_dash_dotted]
        colors = ['black', 'gray', 'black', 'gray', 'black', 'gray']
        os.makedirs("figures", exist_ok=True)

        ### First Plot: Appellian Root vs r/R ###
        plt.figure(figsize=(8, 6))
        all_y_values = []

        for idx, file_name in enumerate(file_name_array):
            D_val = file_name.rsplit("_", 1)[-1].rsplit(".", 1)[0]
            label = f"D = {round(float(D_val), 2)}"

            data = np.loadtxt(file_name, delimiter=",", skiprows=2) # skips the first two lines of the file
            all_y_values.append(data[:, 1])

            linestyle = linestyles[idx % len(linestyles)]
            color = colors[idx % len(colors)]

            plt.plot(data[:, 0], data[:, 1], label=label,
                    linestyle=linestyle, color=color, linewidth=1.5)

            y_all = np.concatenate(all_y_values)
            y_min = y_all.min()
            y_max = y_all.max()

            # Compute y_max rounded up to next tick within its order of magnitude
            if y_max != 0:
                y_max_order = np.floor(np.log10(abs(y_max)))
                y_max_tick = np.ceil(y_max / (10**y_max_order)) * (10**y_max_order)
            else:
                y_max_tick = 0.0

            # Compute y_min rounded down to next tick within its order of magnitude
            if y_min != 0:
                y_min_order = np.floor(np.log10(abs(y_min)))
                y_min_tick = np.floor(y_min / (10**y_min_order)) * (10**y_min_order)
            else:
                y_min_tick = 0.0

            # Create 8 evenly spaced ticks between these bounds
            yticks = np.linspace(y_min_tick, y_max_tick, 8)

            # Round ticks to a reasonable number of decimals for display (based on smallest order of magnitude)
            smallest_order = min(y_min_order if y_min != 0 else 0, y_max_order if y_max != 0 else 0)
            decimal_places = int(max(0, -smallest_order + 1))
            # yticks = np.round(yticks, decimal_places)
            yticks = np.array([-10.0, 0.0, 10.0, 20.0, 30.0])

            # Apply to plot
            plt.ylim(yticks[0], yticks[-1])
            plt.yticks(yticks)

            plt.xlabel("$r/R$")
            plt.ylabel("$\\Gamma_A$", rotation=0, labelpad=5)
            x_ticks = np.array([1, 2, 4, 6, 8])  # Define x-ticks as integers
            plt.xlim(x_ticks[0], x_ticks[-1])
            plt.xticks(x_ticks)

            # Move the first x-tick label to the right by 0.1
            x_tick_labels = [f"{int(tick)}" for tick in x_ticks]  # Format tick labels as integers
            x_tick_labels[0] = f" {x_tick_labels[0]}"  # Add spaces to move the first label
            plt.gca().set_xticklabels(x_tick_labels)
    

        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,
            frameon=False
        )

        plt.tight_layout()
        file1 = f"figures/Appellian_Roots_r_b_{self.zeta_center.real}_alpha_{round(self.angle_of_attack * 180 / np.pi, 1)}.svg"
        plt.savefig(file1, format='svg', bbox_inches='tight')
        print("r/b comparison root figure saved as:", file1)
        plt.close()

        ### Second Plot: Appellian Value vs r/R ###
        plt.figure(figsize=(8, 6))
        all_y_values = []

        for idx, file_name in enumerate(file_name_array):
            D_val = file_name.rsplit("_", 1)[-1].rsplit(".", 1)[0]
            label = f"D = {round(float(D_val), 2)}"

            data = np.loadtxt(file_name, delimiter=",", skiprows=2) # skips the first two lines of the file
            data[:, 2] = data[:, 2] / data[0, 2]
            all_y_values.append(data[:, 2])

            linestyle = linestyles[idx % len(linestyles)]
            color = colors[idx % len(colors)]

            plt.plot(data[:, 0], data[:, 2], label=label,
                    linestyle=linestyle, color=color, linewidth=1.5)

            y_all = np.concatenate(all_y_values)
            y_min = y_all.min()
            y_max = y_all.max()

            # # Compute y_max rounded up to next tick within its order of magnitude
            # if y_max != 0:
            #     y_max_order = np.floor(np.log10(abs(y_max)))
            #     y_max_tick = np.ceil(y_max / (10**y_max_order)) * (10**y_max_order)
            # else:
            #     y_max_tick = 0.0

            # # Compute y_min rounded down to next tick within its order of magnitude
            # if y_min != 0:
            #     y_min_order = np.floor(np.log10(abs(y_min)))
            #     y_min_tick = np.floor(y_min / (10**y_min_order)) * (10**y_min_order)
            # else:
            #     y_min_tick = 0.0

            # # Create 8 evenly spaced ticks between these bounds
            # yticks = np.linspace(y_min_tick, y_max_tick, 8)

            # # Round ticks to a reasonable number of decimals for display (based on smallest order of magnitude)
            # smallest_order = min(y_min_order if y_min != 0 else 0, y_max_order if y_max != 0 else 0)
            # decimal_places = int(max(0, -smallest_order + 1))
            # # yticks = np.round(yticks, decimal_places)
            yticks = np.array([500.0, 1000.0, 1500.0, 2000.0])

            # Apply to plot
            plt.ylim(1.0, yticks[-1])
            plt.yticks(yticks)
            

        # plt.title("Appellian Value vs r/R")
        plt.xlabel("$r/R$")
        plt.ylabel("$\\frac{S}{S_0}$", fontsize=28, rotation=0, labelpad=20)
        x_ticks = np.array([1.0, 2.0, 4.0, 6.0, 8.0])
        plt.xlim(x_ticks[0], x_ticks[-1])
        plt.xticks(x_ticks)

        # Move the first x-tick label to the left
        x_tick_labels = [f"{int(tick)}" for tick in x_ticks]  # Format tick labels as integers
        x_tick_labels[0] = f"{x_tick_labels[0]}  "  # Add spaces to move the first label to the left
        plt.gca().set_xticklabels(x_tick_labels)
        # move the first x-tick to the left by 0.1
        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,
            frameon=False
        )

        plt.tight_layout()
        file2 = f"figures/Appellian_Value_r_b_{self.zeta_center.real}_alpha_{round(self.angle_of_attack * 180 / np.pi, 1)}.svg"
        plt.savefig(file2, format='svg', bbox_inches='tight')
        print("r/b comparison value figure saved as:", file2)
        plt.close()

    def calc_convergence_to_kutta(self, D_values, r_values, theta_values, zeta_center):
        """"""
        original_zeta_center = self.zeta_center
        original_kutta_circulation = self.kutta_circulation
        self.zeta_center = zeta_center
        kutta_appellian_array = np.zeros((len(D_values), 2))
        for i in tqdm(range(len(D_values)), desc="Calculating Appellian Roots"):  
            self.D = D_values[i]
            # self.epsilon = self.equivalent_Joukowski_epsilon_from_taha_C_and_D(C, D_values[i])
            self.epsilon = self.calc_epsilon_from_D(D_values[i])
            self.kutta_circulation = self.calc_circulation_J_airfoil()
            kutta_appellian_array[i, 0] = D_values[i]
            kutta_appellian_array[i, 1] = abs((self.run_appellian_roots(r_values,theta_values, D_values[i])[0] - self.kutta_circulation)/self.kutta_circulation)*100 # percent error 
            # kutta_appellian_array[i, 1] = abs(self.run_appellian_roots(r_values,theta_values, D_values[i])[0] - self.kutta_circulation) # difference
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
        # invert x-axis
        plt.gca().invert_xaxis()
        # include x ticks from 0.00001, 0.0001, 0.001, 0.01, and 0.1 named as 1e-5, 1e-4, 1e-3, 1e-2, and 1e-1
        plt.xscale("log")
        xticks = np.array([0.0001, 0.001, 0.01, 0.1])
        xtick_labels = ["1e-4", "1e-3", "1e-2", "1e-1"]
        plt.xticks(xticks, xtick_labels)
        # plt.xticks([0.0001, 0.001, 0.01, 0.1], ["1e-4", "1e-3", "1e-2", "1e-1"])
        plt.xlim(0.1, 0.0001)  # Lower limit should be slightly less than 0.005
                    # Move the first x-tick label to the right by 0.1
        x_tick_labels = [f"{(tick)}" for tick in xtick_labels]  # Format tick labels as integers
        x_tick_labels[-1] = f"      {x_tick_labels[-1]}"  # Add spaces to move the first label
        plt.gca().set_xticklabels(x_tick_labels)

        plt.yscale("log")
        # plt.yticks([0.000000001,0.00000001,0.0000001,0.000001,0.00001,0.0001, 0.001, 0.01, 0.1, 1.0, 10.0], ["1e-9","1e-8","1e-7","1e-6","1e-5","1e-4","1e-3", "1e-2", "1e-1", "1e0", "1e1"])
        plt.yticks([1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2], ["1e-8", "1e-6", "1e-4", "1e-2", "1e0", "1e2"])
        # plt.yticks([0.0,1.0,2.0,3.0,4.0,5.0,6.0, 7.0], ["0.0", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0"])
        plt.ylim(1e-8, 1e2)
        plt.xlabel("$D$")
        # plt.ylabel("$\\Delta \\Gamma$", rotation=0, labelpad=15)
        # percent error y-axis label
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
            # approximate_error = np.zeros((len(data), 3))
            # first row absolute error = 100
            # approximate_error[0, 1] = 100 
            # plot data_current-data_previous / data_previous*100
            # for i in range(1, len(data)):
                # if i==0:
                #     approximate_error[i, 0] = data[i, 0]
                #     approximate_error[i, 1] = 100
                # else:
                #     approximate_error[i, 0] = data[i, 0]
                #     approximate_error[i, 1] = abs((data[i, 1] - data[i-1, 1])/data[i-1, 1])*100
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
        # plt.xticks(xticks, xtick_labels)
        # plt.xlim(1.0, 0.00001)  # Lower limit should be slightly less than 0.005  
        
        # Move the first x-tick label to the right by 0.1
        # x_tick_labels = [f"{(tick)}" for tick in xtick_labels]
        # x_tick_labels[0] = f"    {x_tick_labels[0]}"  # Add spaces to move the first label
        # plt.gca().set_xticklabels(x_tick_labels)
        plt.yscale("log")
        # yticks = np.array([1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2])
        # plt.yticks(yticks, ["1e-8", "1e-6", "1e-4", "1e-2", "1e0", "1e2"])
        # plt.ylim(-10, 10)
        plt.xlabel("$\\theta$ Stepsize")
        # plt.ylabel("$\\epsilon_{approx}$", rotation=0, labelpad=5)
        plt.ylabel("$\\epsilon_{a}$ for $\\Gamma_A$", labelpad=5)
        plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.15), ncol=2,frameon=False)
        plt.tight_layout()
        file1 = f"figures/Appellian_Roots_theta_convergence_{self.zeta_center.real}_alpha_{round(self.angle_of_attack * 180 / np.pi, 1)}.svg"
        plt.savefig(file1, format='svg', bbox_inches='tight')
        print("theta convergence figure saved as:", file1)
        plt.show()
        plt.close()
        # now plot the Appellian value convergence

        # plt.figure(figsize=(8, 6))
        # for filename in file_name_array:
        #     D_val = filename.rsplit("_", 1)[-1].rsplit(".", 1)[0]
        #     label = f"D = {round(float(D_val), 2)}"
        #     data = np.loadtxt(filename, delimiter=",", skiprows=1)
        #     # data[:, 2] = data[:, 2] / data[0, 2]
        #     approximate_error = np.zeros((len(data), 3))
        #     # plot data_current-data_previous / data_previous*100
        #     for i in range(1, len(data)):
        #         if i==0:
        #             approximate_error[i, 0] = data[i, 0]
        #             approximate_error[i, 1] = 100
        #         else:
        #             approximate_error[i, 0] = data[i, 0]
        #             approximate_error[i, 1] = abs((data[i, 1] - data[i-1, 1])/data[i-1, 1])*100
        #     plt.plot(approximate_error[:, 0], approximate_error[:, 2], 
        #             label=label,
        #             linestyle=linestyles[index % len(linestyles)], 
        #             color=colors[index % len(colors)], 
        #             linewidth=1.5)
        #     index += 1
        # plt.xscale("log")
        # xticks = np.array([10, 100, 1000, 10000])
        # xtick_labels = ["1e1", "1e2", "1e3", "1e4"]
        # plt.xticks(xticks, xtick_labels)
        # plt.xlim(10, 10000)
        # # Move the first x-tick label to the right by 0.1
        # x_tick_labels = [f"{(tick)}" for tick in xtick_labels]
        # x_tick_labels[0] = f" {x_tick_labels[0]}"  # Add spaces to move the first label
        # plt.gca().set_xticklabels(x_tick_labels)
        # # plt.yscale("log")
        # # yticks = np.array([1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2])
        # # plt.yticks(yticks, ["1e-8", "1e-6", "1e-4", "1e-2", "1e0", "1e2"])
        # # plt.ylim(1e-8, 1e2)
        # plt.xlabel("Number of $\\theta$ Steps")
        # plt.ylabel("$\\epsilon_{approx}$ for $S$", labelpad=5)
        # plt.axhline(0, color='gray')
        # plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.15), ncol=2,frameon=False)
        # plt.tight_layout() 
        # file2 = f"figures/Appellian_Value_theta_convergence_{self.zeta_center.real}_alpha_{round(self.angle_of_attack * 180 / np.pi, 1)}.svg"
        # plt.savefig(file2, format='svg', bbox_inches='tight')
        # print("theta convergence value figure saved as:", file2)
        # plt.close()

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
        # plt.close(fig)

    def calculate_surface_pressure(self, point_xi_eta_in_zplane, Gamma):
        """This function calculates the surface pressure at a point using the velocity at that point"""
        velocity = self.velocity(point_xi_eta_in_zplane, Gamma)
        V_squared = np.dot(velocity, velocity)
        pressure = 1-(V_squared/self.freestream_velocity**2)
        return pressure
    
    def given_geom_zeta_calc_top_and_bottom_coords(self, theta_aft, theta_forward, zeta_geom_array):
        """This function calculates the upper and lower coordinates of the Joukowski cylinder given the geometry in zeta coordinates"""
        # any points between theta_start and theta_half that are within the first half of zeta_geom_array are upper coords, the rest are lower coords
        self.upper_coords = zeta_geom_array[(zeta_geom_array[:, 2] >= theta_aft) & (zeta_geom_array[:, 2] <= theta_forward), :2] # spits out the upper zeta_geom_array coordinates which satisfy the conditions of being greater than or equal to theta_aft and less than or equal to theta_forward
        self.lower_coords = zeta_geom_array[(zeta_geom_array[:, 2] > theta_forward) & (zeta_geom_array[:, 2] <= 2*np.pi + theta_aft), :2] # spits out the lower zeta_geom_array coordinates which satisfy the conditions of being greater than theta_forward and less than or equal to 2*np.pi + theta_aft
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
    
    def calculate_surface_pressures(self, Gamma):
        """This function calculates the surface pressures at all the points on the airfoil."""
        # Generate the geometry of the Joukowski cylinder using the stagnation points as start and end thetas
        # first get forward and aft stagnation points 
        if not self.is_compute_appellian:
            r_values = np.array([self.cylinder_radius])
            theta_array = np.array([0.0, 2*np.pi, np.pi/1000])  # theta values for the full circle
            theta_values = hlp.list_to_range(theta_array)
            print("calculating the Circulation for the Joukowski cylinder with D =", self.D, "and r =", self.cylinder_radius, "in order to calculate the surface pressures")
            appellian_root, xi_eta_vals, appelian_value = cyl.run_appellian_roots(r_values, theta_values, cyl.D)
            cyl.circulation = appellian_root
        theta_aft = self.calculate_aft_stagnation_theta_in_Chi_from_Gamma(self.circulation)
        theta_forward = self.calculate_forward_stag_theta_in_Chi_from_aft_stag_theta(theta_aft)
        self.get_full_geometry_zeta(number_of_points = self.output_points, theta_start = theta_aft, theta_half = theta_forward, theta_end = 2*np.pi + theta_aft)
        zeta_geom_array = self.zeta_geom_array
        # any points between theta_start and theta_half that are within the first half of zeta_geom_array are upper coords, the rest are lower coords
        self.upper_coords, self.lower_coords = self.given_geom_zeta_calc_top_and_bottom_coords(theta_aft, theta_forward, zeta_geom_array)
        pressures_upper = np.zeros((len(self.upper_coords), 2))
        pressures_lower = np.zeros((len(self.lower_coords), 2))
        self.shifted_z_surface = self.shift_joukowski_cylinder()
        for i in range(len(self.upper_coords)):
            x, y = self.upper_coords[i]
            # Handle the tangent vector calculation for the last point
            if i < len(self.upper_coords) - 1:
                node_i_plus_one = self.upper_coords[i + 1]
            else:
                node_i_plus_one = self.upper_coords[0]  # Wrap around to the first point for the last node
            node_i = self.upper_coords[i]
            # Calculate tangent vector by subtracting the two nodes
            tangent_vector = node_i_plus_one - node_i
            # Calculate normal vector by rotating the tangent vector 90 degrees
            normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])
            # Normalize the normal vector
            normal_vector = normal_vector / np.linalg.norm(normal_vector)
            # Step off the point in the normal direction
            point_xy = np.array([x, y]) + 1e-6 * normal_vector
            # Calculate the pressure at the stepped-off point
            pressure = self.calculate_surface_pressure(point_xy, Gamma)
            # pressures_upper[i, 0] = point_xy[0] # x-coordinate of the point
            pressures_upper[i, 0] = self.shifted_z_surface[i, 0]  # x-coordinate of the point
            pressures_upper[i, 1] = pressure  # Pressure coefficient at the point
        for j in range(len(self.lower_coords)):
            x, y = self.lower_coords[j]
            # Handle the tangent vector calculation for the last point
            if j < len(self.lower_coords) - 1:
                node_j_plus_one = self.lower_coords[j + 1]
            else:
                node_j_plus_one = self.lower_coords[0]
            node_j = self.lower_coords[j]
            # Calculate tangent vector by subtracting the two nodes
            tangent_vector = node_j_plus_one - node_j
            # Calculate normal vector by rotating the tangent vector 90 degrees
            normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])
            # Normalize the normal vector
            normal_vector = normal_vector / np.linalg.norm(normal_vector)
            #step off the point in the normal direction
            point_xy = np.array([x, y]) + 1e-6 * normal_vector
            # calculate the pressure at the stepped-off point
            pressure = self.calculate_surface_pressure(point_xy, Gamma)
            # pressures_lower[j, 0] = point_xy[0] # x-coordinate of the point
            pressures_lower[j, 0] = self.shifted_z_surface[j+len(self.upper_coords), 0]  # x-coordinate of the point
            pressures_lower[j, 1] = pressure  # Pressure coefficient at the point
        return pressures_upper, pressures_lower

    def plot_surface_pressures(self, Gamma, D):
        """This function plots the surface pressures"""
        pressures_upper, pressures_lower = self.calculate_surface_pressures(Gamma)
        alpha_label = "$\\alpha$ = " + str(round(self.angle_of_attack*180/np.pi, 1)) + " [deg]"
        if self.is_create_dsweep_gif or not self.is_plot_pressure_alpha_considerations:
            line, = plt.plot(pressures_upper[:, 0], pressures_upper[:, 1], label = alpha_label, color = "black")  # Plot the first line and capture its properties
        else:
            line, = plt.plot(pressures_upper[:, 0], pressures_upper[:, 1], label = alpha_label)  # Plot the first line and capture its properties
        plt.plot(pressures_lower[:, 0], pressures_lower[:, 1], color=line.get_color())  # Use the same color as the first line
        plt.xlabel("$\\xi/\\bar{c}$")
        plt.ylabel("$C_p$")
        # rotate the y axis label to be horizontal
        plt.gca().yaxis.label.set_rotation(0)
        # plt.title("Surface Pressure Coefficient")
        # label padding
        plt.gca().yaxis.labelpad = 15 # distance between label and axis
        plt.gca().xaxis.labelpad = 1
        plt.xlim(0, 1.0)
        plt.ylim(-6.0, 1.0)
        # Make the plot decrease in the positive y direction
        plt.gca().invert_yaxis()
        # set x-axis ticks to be 0.2, 0.4, 0.6, 0.8, 1.0
        plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ["    0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
        # Save the figure
        filename = "figures/Appellian_D_sweep_at_zeta_center_" + str(self.zeta_center.real) + "_alpha_" + str(round(self.angle_of_attack*180/np.pi, 1)) + "_D_" + str(round(D, 2)) + ".png"
        if self.save_fig:
            # plt.gcf().set_size_inches(3.25, 3.5)  # force figure size in inches
            plt.savefig(str(filename) + "_surface_pressure_at_" + str(self.angle_of_attack) + "_degrees.png", dpi=300, bbox_inches=None, pad_inches=0)
        # if self.show_fig:
            # plt.show()
        # plt.close()
        return filename
    
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
    if cyl.appellian_is_line_integral:
        r_range = [cyl.cylinder_radius, cyl.cylinder_radius, 0.01*cyl.cylinder_radius]
    else:
        # r_range = [cyl.cylinder_radius, 7*cyl.cylinder_radius, 0.01*cyl.cylinder_radius]
        r_range = [cyl.cylinder_radius, 7*cyl.cylinder_radius, 0.007*cyl.cylinder_radius]
    r_values = hlp.list_to_range(r_range)
    # r_values = np.array([cyl.cylinder_radius])
    theta_range = [0.0, 2*np.pi, np.pi/1000]
    theta_values = hlp.list_to_range(theta_range)
    if cyl.is_compute_appellian and cyl.type == "cylinder":
        appellian_root, xi_eta_vals, appelian_value = cyl.run_appellian_roots(r_values, theta_values, cyl.D)
        cyl.circulation = appellian_root
        print("theta steps: ", theta_range[2])
        print("Appellian Circulation: ", appellian_root)
        print("Appellian Value: ", appelian_value)
        print("percent difference between Appellian and Kutta: ", 100*(appellian_root - cyl.kutta_circulation)/cyl.kutta_circulation)
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
            # save the figure
            plt.savefig("figures/dsweep_zeta0_" + str(cyl.zeta_center.real) + "_" + str(cyl.zeta_center.imag) + "_alpha_" + str(round(cyl.angle_of_attack*180/np.pi, 1)) + ".svg", dpi=300, bbox_inches=None, pad_inches=0)
        plt.figure()
        if cyl.transformation_type == "joukowski":
            with open(cyl.cyl_json_file, 'r') as json_handle:
                input = json.load(json_handle)
                cyl.D = hlp.parse_dictionary_or_return_default(input, ["geometry", "shape_parameter_D"], cyl.D)
                if cyl.use_shape_parameter_D:
                    cyl.epsilon = cyl.calc_epsilon_from_D(cyl.D)
                else:
                    cyl.epsilon = input["geometry"]["epsilon"]
        else:
            with open(cyl.cyl_json_file, 'r') as json_handle:
                input = json.load(json_handle)
                cyl.D = input["geometry"]["taha"]["D"] # this is the D parameter in the Taha refining Kutta paper (see Eqs 8-9)
                # cyl.epsilon = input["geometry"]["epsilon"]
    if cyl.is_plot_circulation_for_varying_shape and not cyl.is_calc_circulation_for_varying_shape:
        # plt.figure()
        # plot all of the files in the text_files/dsweep directory
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
        # kutta_appellian_txt_file = "text_files/dsweep/dsweep_zeta0_" + str(cyl.zeta_center.real) + "_" + str(cyl.zeta_center.imag) + "_alpha_" + str(round(cyl.angle_of_attack*180/np.pi, 1)) + ".txt"
        # kutta_appellian_array = np.loadtxt(kutta_appellian_txt_file, delimiter=",", skiprows=1)
        # figure_name = cyl.plot_gamma_over_gamma_kutta(kutta_appellian_array, is_plot_current_D = False, kutta_and_current_D = [kutta_appellian_array[0,0], kutta_appellian_array[0,1]])
        # save
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
        # appellian_root, xi_eta_vals, appelian_value = cyl.run_appellian_roots(r_values, theta_values, cyl.D)
        # print("appellian circulation: ", appellian_root)
        # cyl.circulation = appellian_root
        streamlines = cyl.calc_streamlines()
        cyl.plot(streamlines)
        plt.gca().set_aspect('equal', adjustable='box')
        if cyl.show_fig:
            plt.show()
        # plt.close()
    else:
        plt.show()
    if cyl.is_plot_shifted_joukowski_cylinder:
        cyl.shift_joukowski_cylinder()
        cyl.plot_shifted_joukowski_cylinder()
    cyl.plot_geometry_settings()
    if cyl.is_plot_z_selection_comparison:
        plt.figure()
        cyl.compare_correct_and_incorrect_z_selection()
    # zeta_to_z_test_point = 1 + 1j
    if cyl.type == "airfoil":
        cyl.run_J_airfoil_stuff()
    if cyl.is_plot_z_plane_surface_pressure_distribution:
        plt.figure()
        cyl.plot_surface_pressures(cyl.circulation, cyl.D)
        if cyl.is_plot_pressure_alpha_considerations:
            for i in range(len(cyl.alphas_for_consideration)):
                cyl.angle_of_attack = cyl.alphas_for_consideration[i]
                cyl.kutta_circulation = cyl.calc_circulation_J_airfoil()
                appellian_circulation = cyl.run_appellian_roots(r_values, theta_values, cyl.D)[0]
                cyl.circulation = appellian_circulation
                cyl.plot_surface_pressures(cyl.circulation, cyl.D)
        plt.legend()
        if cyl.show_fig:
            plt.show()
        plt.close()
    plt.close()
    if cyl.is_create_dsweep_gif and cyl.is_calc_circulation_for_varying_shape:
        plt.show()
        plt.figure()
        cyl.create_dsweep_gif(kutta_appellian_array, r_values, theta_values)
    
    # if cyl.is_grid_plots:
    #     plt.figure()
    #     cyl.grid_plots()

    # #run rb comparisons
    if cyl.is_plot_rb_comparisons:
        r_ranges = [cyl.cylinder_radius, 8*cyl.cylinder_radius, 0.01]
        r_vals = hlp.list_to_range(r_ranges)
        theta_ranges = [0, 2*np.pi, np.pi/1000]
        theta_vals = hlp.list_to_range(theta_ranges) 
        D = 0.0
        r_range_first = [cyl.cylinder_radius, 1.0001*cyl.cylinder_radius, 0.0001*cyl.cylinder_radius]
        r_values_first = hlp.list_to_range(r_range_first)
        r_range_second = [1.05*cyl.cylinder_radius, 2.0*cyl.cylinder_radius, 0.05*cyl.cylinder_radius]
        r_values_second = hlp.list_to_range(r_range_second)
        r_range_third = [2.1*cyl.cylinder_radius, 2.9*cyl.cylinder_radius, 0.1*cyl.cylinder_radius]
        r_values_third = hlp.list_to_range(r_range_third)
        r_range_fourth = [3.0*cyl.cylinder_radius, 8.0*cyl.cylinder_radius, 1.0*cyl.cylinder_radius]
        r_values_fourth = hlp.list_to_range(r_range_fourth)
        r_values = np.concatenate((r_values_first, r_values_second, r_values_third, r_values_fourth), axis=0)
        if cyl.is_calc_rb_comparisons:
            r_range_first = [1.0*cyl.cylinder_radius, 1.0001*cyl.cylinder_radius, 1.0001*cyl.cylinder_radius]
            r_values_first = hlp.list_to_range(r_range_first)
            r_range_second = [1.0002*cyl.cylinder_radius, 8.0*cyl.cylinder_radius, 7.9998*cyl.cylinder_radius]
            r_values_second = hlp.list_to_range(r_range_second)
            r_values = np.concatenate((r_values_first, r_values_second), axis=0)
            print("r_values: ", r_values)
            D_array = np.array([0.6])
            D_array = np.logspace(-2, 0, num=6)  # From 10^-2 (0.01) to 10^0 (1.0) in 6 steps (in a logarithmic scale)
            for j in range(len(D_array)):
                D = D_array[j]
                cyl.run_r_b_comparisons(r_values, theta_values, D)
                ### break
        folder_name = "text_files"
        file_name_array = []
        for filename in os.listdir(folder_name):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_name, filename)
                file_name_array.append(file_path)
        # now plot the rb comparisons
        cyl.plot_rb_comparisons(file_name_array)
    
    if cyl.is_calc_convergence_to_kutta:
        print("r_values: ", r_values)
        D_range_first = [0.0001, 0.00099, 0.00001]
        D_values_first = hlp.list_to_range(D_range_first)
        # D_range_second = [0.0001, 0.0009, 0.0001]
        # D_values_second = hlp.list_to_range(D_range_second)
        D_range_second = [0.001, 0.009, 0.001]
        D_values_second = hlp.list_to_range(D_range_second)
        D_range_third = [0.01, 0.1, 0.01]
        D_values_third = hlp.list_to_range(D_range_third)
        r_range = [1.0*cyl.cylinder_radius, 7*cyl.cylinder_radius, 0.01*cyl.cylinder_radius]
        r_values = hlp.list_to_range(r_range)
        D_values = np.concatenate((D_values_first, D_values_second, D_values_third), axis=0)
        # instead, make D_values on a logarithmic scale from 0.0001 to 0.1
        D_values = np.logspace(-4, -1, num=100)  # From 10^-4 (0.0001) to 10^-1 (0.1) in 6 steps (in a logarithmic scale)
        # zeta_center_values = np.array([-0.05+0.0j])
        # zeta_center_values = np.array([-0.1+0.0j])
        # zeta_center_values = np.array([-0.15+0.0j])
        zeta_center_values = np.array([-0.20 + 0.0j])
        # zeta_center_values = np.array([-0.05+0.0j, -0.1+0.0j,-0.15+0.0j,-0.20 + 0.0j])
        # zeta_center = cyl.zeta_center
        for i in range(len(zeta_center_values)):
            print("zeta center: ", zeta_center_values[i])
            cyl.calc_convergence_to_kutta(D_values, r_values, theta_values, zeta_center_values[i])
    if cyl.is_plot_convergence_to_kutta:
        plt.figure()
        folder_name = "text_files/convergence_to_kutta"
        file_name_array = []
        for filename in os.listdir(folder_name):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_name, filename)
                file_name_array.append(file_path)
        cyl.plot_convergence_to_kutta(file_name_array)
        filename = "figures/convergence_to_kutta_alpha_" + str(round(cyl.angle_of_attack*180/np.pi, 1)) + ".pdf"
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print("Convergence to Kutta figure saved as:", filename)

    # theta convergence
    if cyl.is_calc_theta_convergence:
        r_values = np.array([cyl.cylinder_radius])
        D_values = np.logspace(-4, 0, num=6)  # From 10^-2 (0.01) to 10^0 (1.0) in 6 steps (in a logarithmic scale)
        # D_values = np.array([0.0001])
        # theta_steps = np.logspace(0, 4, num=100)  # From 10^1 (10) to 10^4 (10000) in 100 steps (in a logarithmic scale)
        theta_steps = np.logspace(1, 4, num=100)  # From 10^1 (10) to 10^4 (10000) in 100 steps (in a logarithmic scale)
        theta_range_array = np.zeros((len(theta_steps), 3))
        for i in range(len(theta_steps)):
            theta_range_array[i, 0] = 0.0
            theta_range_array[i, 1] = 2*np.pi
            theta_range_array[i, 2] = np.pi/theta_steps[i]
        for i in range(len(D_values)):
            # print("D value: ", D_values[i])
            appellian_roots_theta_convergence = cyl.calc_theta_convergence(D_values[i], r_values, theta_range_array)
    # # plot theta convergence
    if cyl.is_plot_theta_convergence:
        plt.figure()
        folder_name = "text_files/theta_grid_convergence"
        file_name_array = []
        for filename in os.listdir(folder_name):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_name, filename)
                file_name_array.append(file_path)
        cyl.plot_theta_convergence(file_name_array)
        # filename = "figures/theta_convergence_alpha_" + str(round(cyl.angle_of_attack*180/np.pi, 1)) + ".pdf"
        # plt.savefig(filename, format='pdf', bbox_inches='tight')
        # print("Theta convergence figure saved as:", filename)
    theta_aft = cyl.calculate_aft_stagnation_theta_in_Chi_from_Gamma(cyl.circulation)
    theta_forward = cyl.calculate_forward_stag_theta_in_Chi_from_aft_stag_theta(theta_aft)
    print("Aft Stagnation Theta in Chi: ", np.rad2deg(theta_aft))
    print("Forward Stagnation Theta in Chi: ", np.rad2deg(theta_forward))
    # now get the upper and lower zeta coordinates
    # upper_coords, lower_coords = cyl.given_geom_zeta_calc_top_and_bottom_coords(theta_aft, theta_forward, cyl.zeta_geom_array)
    # now get the upper and lower z coordinates
    # z_upper, z_lower = cyl.given_geom_zeta_upper_lower_get_z_upper_lower(upper_coords, lower_coords)

    # ## test the line integral convective acceleration
    # # print("circulation: ", cyl.circulation)
    # # cyl.numerically_integrate_appellian(cyl.circulation, r_values, cyl.D)
    # ## print()
    # test_point_in_z = cyl.full_z_surface[5]
    # # test_point_in_z = [0.5, 0.5]
    # test_point_in_zeta = cyl.Joukowski_z_to_zeta(test_point_in_z[0] + 1j*test_point_in_z[1], cyl.epsilon)
    # test_point_in_Chi = cyl.zeta_to_Chi(test_point_in_zeta)
    # r_chi, theta_chi = hlp.xy_to_r_theta(test_point_in_Chi.real, test_point_in_Chi.imag)
    # r_0, theta_0 = hlp.xy_to_r_theta(cyl.zeta_center.real, cyl.zeta_center.imag)
    # test_point_in_Chi = [test_point_in_Chi.real, test_point_in_Chi.imag]
    # test_point_in_zeta = [test_point_in_zeta.real, test_point_in_zeta.imag]

    # line_int_conv_accel_squared_comp_conj = cyl.analytic_conv_accel_for_line_int_comp_conj([r_chi, theta_chi], cyl.circulation)
    # print("Line Int Convective Acceleration squared is         ", r_chi*line_int_conv_accel_squared_comp_conj/(-32))
    # numerical_cartesian_convective_acceleration_in_z = cyl.numerical_cartesian_convective_acceleration(test_point_in_zeta, cyl.circulation, step=1e-12)
    # numerical_cartesian_convective_acceleration_in_z_squared = np.dot(numerical_cartesian_convective_acceleration_in_z, numerical_cartesian_convective_acceleration_in_z)
    # # print("Numerical Convective Acceleration squared is        ", numerical_cartesian_convective_acceleration_in_z_squared/2)

    # Taha_conv_accel_squared_comp_conj = cyl.taha_analytic_conv_accel_square_comp_conj([r_chi,theta_chi], cyl.circulation)
    # print("Taha Analytic Convective Acceleration squared is    ", Taha_conv_accel_squared_comp_conj/2)
    # Spencer_accel_squared_comp_conj = cyl.analytic_conv_accel_square_comp_conj([r_chi,theta_chi], cyl.circulation)
    # print("Spencer Alternat Convective Acceleration squared is ", Spencer_accel_squared_comp_conj/2)
    # print("\n")
