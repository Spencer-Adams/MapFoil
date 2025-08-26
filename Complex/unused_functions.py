import json
import numpy as np # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import scipy.integrate as spi # this is for numerical integration
# import bisection from scipy.optimize
import helper as hlp
from complex_potential_flow_class import potential_flow_object  
import os
import tqdm # type: ignore

class extras(potential_flow_object):
    """This is a class that creates a cylinder object and performs calculations specific to a cylinder"""
    def __init__(self, json_file):
        self.cyl_json_file = json_file
        self.parse_cylinder_json()
        super().__init__(json_file) # this is doing self.() of all the variables we want to in the super class

    def calc_z_G_values(self, xi, eta, alpha, epsilon, R, xio, etao):
        """Cartesian G values for the z plane"""
        G1 = (eta-etao)/((xi-xio)**2 + (eta-etao)**2)
        G2 = (xi-xio)/((xi-xio)**2 + (eta-etao)**2)
        G3 = R**2*(np.cos(alpha)*((xi-xio)**2-(eta-etao)**2)+2*np.sin(alpha)*((xi-xio)*(eta-etao)))/(((xi-xio)**2 -(eta-etao)**2)**2 + (2*(xi-xio)*(eta-etao))**2)
        G4 = R**2*(np.sin(alpha)*((xi-xio)**2-(eta-etao)**2)-2*np.cos(alpha)*((xi-xio)*(eta-etao)))/(((xi-xio)**2 -(eta-etao)**2)**2 + (2*(xi-xio)*(eta-etao))**2)
        G5 = 1 - ((xi**2 - eta**2)*(R-epsilon)**2)/(((xi**2 - eta**2)**2 + (2*xi*eta)**2))
        G6 = ((2*xi*eta)*(R-epsilon)**2)/(((xi**2 - eta**2)**2 + (2*xi*eta)**2))
        return G1, G2, G3, G4, G5, G6
    
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

    def calculate_surface_pressures(self, Gamma):
        """This function calculates the surface pressures at all the points on the airfoil."""
        # Generate the geometry of the Joukowski cylinder using the stagnation points as start and end thetas
        # first get forward and aft stagnation points 
        if not self.is_compute_appellian:
            r_values = np.array([self.cylinder_radius])
            theta_array = np.array([0.0, 2*np.pi, np.pi/1000])  # theta values for the full circle
            theta_values = hlp.list_to_range(theta_array)
            print("calculating the Circulation for the Joukowski cylinder with D =", self.D, "and r =", self.cylinder_radius, "in order to calculate the surface pressures")
            appellian_root, xi_eta_vals, appelian_value = cyl.run_appellian_roots(r_values, theta_values, self.D)
            cyl.circulation = appellian_root
        theta_aft = self.calculate_aft_stagnation_theta_in_Chi_from_Gamma(self.circulation)
        theta_forward = self.calculate_forward_stag_theta_in_Chi_from_aft_stag_theta(theta_aft)
        self.get_full_geometry_zeta(number_of_points = self.output_points, theta_start = theta_aft, theta_half = theta_forward, theta_end = 2*np.pi + theta_aft)
        zeta_geom_array = self.zeta_geom_array
        # any points between theta_start and theta_half that are within the first half of zeta_geom_array are upper coords, the rest are lower coords
        self.upper_coords, self.lower_coords = self.given_geom_and_stag_thetas_calc_top_and_bottom_coords(theta_aft, theta_forward, zeta_geom_array)
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

    def velocity_zeta(self, point_xy, Gamma): # A4 on project
        """
        Calculates the velocity at a given point in the flow field in cartesian coordinates using cylindrical velocity equations.
        Parameters:
        - point_xy (list): An xy coordinate.
        Returns:
        - velocity (list): The velocity at the given point.
        """
        z = point_xy[0] + 1j*point_xy[1]
        w_z = self.freestream_velocity*((np.exp(-1j*self.angle_of_attack)) + 1j*Gamma/(2*np.pi*self.freestream_velocity*(z-self.zeta_center)) + -1*(np.exp(1j*self.angle_of_attack))*self.cylinder_radius**2/(z-self.zeta_center)**2)  
        velocity_complex = np.array([w_z.real, -w_z.imag])     
        return velocity_complex

    def all_partials(self, r, theta, alpha, epsilon, R, r0, theta0, Gamma):
        """This function calculates all the partial derivatives of omega r and omega theta"""
        G1, G2, G3, G4, G5, G6 = self.calc_Chi_G_values(r, theta, alpha, epsilon, R, r0, theta0)
        A_omega_rG, B_omega_rG, C_omega_rG, D_omega_rG, E_omega_rG, F_omega_rG, G_omega_rG, H_omega_rG = self.partial_coefficients_of_omega_r_wrt_G_values(self.freestream_velocity, theta, self.angle_of_attack, G1, G2, G3, G4, G5, G6)
        A_omega_thetaG, B_omega_thetaG, C_omega_thetaG, D_omega_thetaG, E_omega_thetaG, F_omega_thetaG, G_omega_thetaG, H_omega_thetaG = self.partial_coefficients_of_omega_theta_wrt_G_values(self.freestream_velocity, theta, self.angle_of_attack, G1, G2, G3, G4, G5, G6)
        AGr, BGr, CGr, DGr, EGr, FGr = self.partial_coefficients_of_G_wrt_r(r, theta, alpha, epsilon, R, r0, theta0)
        AGtheta, BGtheta, CGtheta, DGtheta, EGtheta, FGtheta = self.partial_coefficients_of_G_wrt_theta(r, theta, alpha, epsilon, R, r0, theta0)
        # omega r partials
        d_omega_r_dr = Gamma*A_omega_rG*AGr + Gamma*B_omega_rG*BGr + C_omega_rG*CGr + D_omega_rG*DGr + EGr*(Gamma*E_omega_rG + F_omega_rG) + FGr*(Gamma*G_omega_rG + H_omega_rG)
        d_omega_r_dtheta = Gamma*A_omega_rG*AGtheta + Gamma*B_omega_rG*BGtheta + C_omega_rG*CGtheta + D_omega_rG*DGtheta + EGtheta*(Gamma*E_omega_rG + F_omega_rG) + FGtheta*(Gamma*G_omega_rG + H_omega_rG)
        # omega theta partials
        d_omega_theta_dr = Gamma*A_omega_thetaG*AGr + Gamma*B_omega_thetaG*BGr + C_omega_thetaG*CGr + D_omega_thetaG*DGr + EGr*(Gamma*E_omega_thetaG + F_omega_thetaG) + FGr*(Gamma*G_omega_thetaG + H_omega_thetaG)
        d_omega_theta_dtheta = Gamma*A_omega_thetaG*AGtheta + Gamma*B_omega_thetaG*BGtheta + C_omega_thetaG*CGtheta + D_omega_thetaG*DGtheta + EGtheta*(Gamma*E_omega_thetaG + F_omega_thetaG) + FGtheta*(Gamma*G_omega_thetaG + H_omega_thetaG)
        print("d_omega_r_dr:                    ", d_omega_r_dr)
        print("d_omega_r_dtheta:                    ", d_omega_r_dtheta)
        print("d_omega_theta_dr                     ", d_omega_theta_dr)
        print("d_omega_theta_dtheta                     ", d_omega_theta_dtheta)
        return d_omega_r_dr, d_omega_r_dtheta, d_omega_theta_dr, d_omega_theta_dtheta
    
    def partial_coefficients_of_omega_r_wrt_G_values(self, V_inf, theta, alpha, G1, G2, G3, G4, G5, G6):
        """This function retrieves the partial derivatives of omega r with respect to G values"""
        # G1, G2, G3, G4, G5, G6 = self.calc_Chi_G_values(r, theta, alpha, epsilon, R, r0, theta0)
        # omega r partials with respect to G1, G2, G3, G4, G5, G6
        A_omega_rG = 1/(2*np.pi)*((G5*np.cos(theta))/(G5**2 + G6**2)+ (G6*np.sin(theta))/(G5**2 + G6**2))
        B_omega_rG = 1/(2*np.pi)*((G6*np.cos(theta))/(G5**2 + G6**2) - (G5*np.sin(theta))/(G5**2 + G6**2))
        C_omega_rG = - V_inf*(G5*np.cos(theta)/(G5**2 + G6**2) + G6*np.sin(theta)/(G5**2 + G6**2))
        D_omega_rG = V_inf*(G5*np.sin(theta)/(G5**2 + G6**2) - G6*np.cos(theta)/(G5**2 + G6**2))
        E_omega_rG = (G1*np.cos(theta))/(2*np.pi*(G5**2 + G6**2)) - (G2*np.sin(theta))/(2*np.pi*(G5**2 + G6**2)) - (G5*np.cos(theta)*(G1*G5 + G2*G6))/(np.pi*(G5**2 + G6**2)**2) - (G5*np.sin(theta)*(G1*G6 - G2*G5))/(np.pi*(G5**2 + G6**2)**2)
        F_omega_rG = V_inf*((np.sin(theta)*(G4+np.sin(alpha)))/(G5**2+G6**2)-(np.cos(theta)*(G3-np.cos(alpha)))/(G5**2+G6**2)+(2*G5*np.cos(theta)*(G6*np.sin(alpha)-G5*np.cos(alpha)+G3*G5+G4*G6))/(G5**2+G6**2)**2-(2*G5*np.sin(theta)*(G6*np.cos(alpha)+G5*np.sin(alpha)-G3*G6+G4*G5))/(G5**2+G6**2)**2)
        G_omega_rG = (G2*np.cos(theta))/(2*np.pi*(G5**2 + G6**2)) + (G1*np.sin(theta)/(2*np.pi*(G5**2+G6**2))) - (G6*np.cos(theta)*(G1*G5+G2*G6))/(np.pi*(G5**2+G6**2)**2) - (G6*np.sin(theta)*(G1*G6-G2*G5))/(np.pi*(G5**2+G6**2)**2)
        H_omega_rG = V_inf*((2*G6*np.cos(theta)*(G6*np.sin(alpha)-G5*np.cos(alpha)+G3*G5+G4*G6))/(G5**2+G6**2)**2-(np.sin(theta)*(G3-np.cos(alpha)))/(G5**2+G6**2)-(np.cos(theta)*(G4+np.sin(alpha)))/(G5**2+G6**2)-(2*G6*np.sin(theta)*(G6*np.cos(alpha)+G5*np.sin(alpha)-G3*G6+G4*G5))/(G5**2+G6**2)**2)
        return A_omega_rG, B_omega_rG, C_omega_rG, D_omega_rG, E_omega_rG, F_omega_rG, G_omega_rG, H_omega_rG

    def partial_coefficients_of_omega_theta_wrt_G_values(self, V_inf, theta, alpha, G1, G2, G3, G4, G5, G6):
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

    def partial_coefficients_of_G_wrt_r(self, r, theta, alpha, epsilon, R, r0, theta0):
        """This function retrieves the partial derivatives of G values with respect to r"""
        AGr = -np.sin(theta)/r**2
        BGr = -np.cos(theta)/r**2
        CGr = (R**2*(r**2*np.cos(2*theta)*np.cos(alpha) + r**2*np.sin(2*theta)*np.sin(alpha))*(2*r**3*(np.cos(4*theta) - 1) - 4*r**3*np.cos(2*theta)**2))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2)**2 - (R**2*(2*r*np.cos(2*theta)*np.cos(alpha) + 2*r*np.sin(2*theta)*np.sin(alpha)))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2)
        DGr = (R**2*(r**2*np.cos(2*theta)*np.sin(alpha) - r**2*np.sin(2*theta)*np.cos(alpha))*(2*r**3*(np.cos(4*theta) - 1) - 4*r**3*np.cos(2*theta)**2))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2)**2 - (R**2*(2*r*np.cos(2*theta)*np.sin(alpha) - 2*r*np.sin(2*theta)*np.cos(alpha)))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2)
        EGr = ((R - epsilon)**2*(2*(2*r*np.cos(2*theta) + 2*r0*np.cos(theta + theta0))*(np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2) + 2*(2*r*np.sin(2*theta) + 2*r0*np.sin(theta + theta0))*(np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))*(np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2)**2 - ((R - epsilon)**2*(2*r*np.cos(2*theta) + 2*r0*np.cos(theta + theta0)))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2)
        FGr= ((2*r*np.sin(2*theta) + 2*r0*np.sin(theta + theta0))*(R - epsilon)**2)/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2) - ((R - epsilon)**2*(2*(2*r*np.cos(2*theta) + 2*r0*np.cos(theta + theta0))*(np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2) + 2*(2*r*np.sin(2*theta) + 2*r0*np.sin(theta + theta0))*(np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))*(np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2)**2
        return AGr, BGr, CGr, DGr, EGr, FGr

    def partial_coefficients_of_G_wrt_theta(self, r, theta, alpha, epsilon, R, r0, theta0):
        AGtheta = np.cos(theta)/r
        BGtheta  = -np.sin(theta)/r
        CGtheta = -(R**2*(2*r**2*np.cos(2*theta)*np.sin(alpha) - 2*r**2*np.sin(2*theta)*np.cos(alpha)))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2) - (R**2*(2*r**4*np.sin(4*theta) - 4*r**4*np.cos(2*theta)*np.sin(2*theta))*(r**2*np.cos(2*theta)*np.cos(alpha) + r**2*np.sin(2*theta)*np.sin(alpha)))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2)**2
        DGtheta = (R**2*(2*r**2*np.cos(2*theta)*np.cos(alpha) + 2*r**2*np.sin(2*theta)*np.sin(alpha)))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2) - (R**2*(2*r**4*np.sin(4*theta) - 4*r**4*np.cos(2*theta)*np.sin(2*theta))*(r**2*np.cos(2*theta)*np.sin(alpha) - r**2*np.sin(2*theta)*np.cos(alpha)))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2)**2
        EGtheta = ((R - epsilon)**2*(2*np.sin(2*theta)*r**2 + 2*r0*np.sin(theta + theta0)*r))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2) - ((2*(2*np.sin(2*theta)*r**2 + 2*r0*np.sin(theta + theta0)*r)*(np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2) - 2*(2*np.cos(2*theta)*r**2 + 2*r0*np.cos(theta + theta0)*r)*(np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))*(R - epsilon)**2*(np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2)**2
        FGtheta = ((R - epsilon)**2*(2*np.cos(2*theta)*r**2 + 2*r0*np.cos(theta + theta0)*r))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2) + ((2*(2*np.sin(2*theta)*r**2 + 2*r0*np.sin(theta + theta0)*r)*(np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2) - 2*(2*np.cos(2*theta)*r**2 + 2*r0*np.cos(theta + theta0)*r)*(np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))*(R - epsilon)**2*(np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2)**2
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
        A_omega_rG, B_omega_rG, C_omega_rG, D_omega_rG, E_omega_rG, F_omega_rG, G_omega_rG, H_omega_rG = self.partial_coefficients_of_omega_r_wrt_G_values(V_inf, theta, alpha, G1, G2, G3, G4, G5, G6)
        A_omega_thetaG, B_omega_thetaG, C_omega_thetaG, D_omega_thetaG, E_omega_thetaG, F_omega_thetaG, G_omega_thetaG, H_omega_thetaG = self.partial_coefficients_of_omega_theta_wrt_G_values(V_inf, theta, alpha, G1, G2, G3, G4, G5, G6)
        AGr, BGr, CGr, DGr, EGr, FGr = self.partial_coefficients_of_G_wrt_r(r, theta, alpha, epsilon, R, r0, theta0)
        AGtheta, BGtheta, CGtheta, DGtheta, EGtheta, FGtheta = self.partial_coefficients_of_G_wrt_theta(r, theta, alpha, epsilon, R, r0, theta0)
        # calculate the A1 through A12 values
        A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12 = self.calc_A1_through_A12(V_inf, theta, alpha, G1, G2, G3, G4, G5, G6, A_omega_rG, B_omega_rG, C_omega_rG, D_omega_rG, E_omega_rG, F_omega_rG, G_omega_rG, H_omega_rG, A_omega_thetaG, B_omega_thetaG, C_omega_thetaG, D_omega_thetaG, E_omega_thetaG, F_omega_thetaG, G_omega_thetaG, H_omega_thetaG, AGr, BGr, CGr, DGr, EGr, FGr, AGtheta, BGtheta, CGtheta, DGtheta, EGtheta, FGtheta)
        # omega_r, omega_theta = Gamma*A1 + A2, Gamma*A3 + A4
        # partial_omega_r_wrt_r, partial_omega_r_wrt_theta, partial_omega_theta_wrt_theta, partial_omega_theta_wrt_r = self.chi_partial_derivatives(Gamma, A5, A6, A7, A8, A9, A10, A11, A12)
        B1, B2, B3, B4, B5, B6 = self.calc_B1_through_B6(r, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12)
        convective_acceleration = np.array([Gamma**2*B1 + Gamma*B2 + B3, Gamma**2*B4 + Gamma*B5 + B6])
        C1, C2, C3, C4, C5 = self.calc_C1_through_C5(B1, B2, B3, B4, B5, B6)
        return C1, C2, C3, C4, C5, convective_acceleration
    
    def convective_acceleration_squared(self, point_xy_in_Chi_plane, Gamma):
        """This function calculates the convective acceleration squared"""
        C1, C2, C3, C4, C5, chi_convective_acceleration = self.chi_convective_acceleration(point_xy_in_Chi_plane, Gamma)
        convective_acceleration_squared = np.dot(chi_convective_acceleration, chi_convective_acceleration)
        # convective_acceleration_squared = Gamma**4*C1 + Gamma**3*C2 + Gamma**2*C3 + Gamma*C4 + C5
        return convective_acceleration_squared










    def split_velocity_z(self, point_xy_in_z_plane, Gamma):
        """This function calculates the velocity at a given point in the flow field in the z plane"""
        z = point_xy_in_z_plane[0] + 1j*point_xy_in_z_plane[1]
        zeta = self.z_to_zeta(z)
        xi, eta = zeta.real, zeta.imag
        G1, G2, G3, G4, G5, G6 = self.calc_z_G_values(xi, eta, self.angle_of_attack, self.epsilon, self.cylinder_radius, self.zeta_center.real, self.zeta_center.imag)
        omega_real = (Gamma/(2*np.pi))*((G1*G5+G2*G6)/(G5**2 + G6**2)) + self.freestream_velocity*((G5*np.cos(self.angle_of_attack)-G6*np.sin(self.angle_of_attack)-G3*G5-G4*G6)/(G5**2 + G6**2))
        omega_imag = -(Gamma/(2*np.pi))*((G2*G5-G1*G6)/(G5**2 + G6**2)) + self.freestream_velocity*((G6*np.cos(self.angle_of_attack)+G5*np.sin(self.angle_of_attack)+G4*G5-G3*G6)/(G5**2 + G6**2))
        return omega_real, omega_imag


    
    def chi_acceleration(self, point_xy_in_Chi_plane, Gamma):
        """This function calculates the acceleration at a given point in the flow field in the Chi plane"""
        chi = point_xy_in_Chi_plane[0] + 1j*point_xy_in_Chi_plane[1]
        xi, eta = point_xy_in_Chi_plane[0], point_xy_in_Chi_plane[1]
        r, theta = hlp.xy_to_r_theta(xi, eta)
        xio, etao = self.zeta_center.real, self.zeta_center.imag
        r0, theta0 = hlp.xy_to_r_theta(xio, etao)
        V_inf, R, alpha, epsilon = self.freestream_velocity,  self.cylinder_radius, self.angle_of_attack, self.epsilon
        acceleration = V_inf*(-1j*Gamma/(2*np.pi*V_inf*chi**2) + 2*np.exp(1j*alpha)*R**2/(chi**3)) / (1 - (R-epsilon)**2/(chi+self.zeta_center)**2)**2
        acceleration_complex = np.array([acceleration.real, -acceleration.imag])
        acceleration_r = np.cos(theta)*acceleration_complex[0] + np.sin(theta)*acceleration_complex[1]
        acceleration_theta = np.cos(theta)*acceleration_complex[1] - np.sin(theta)*acceleration_complex[0]
        acceleration_polar = np.array([acceleration_r, acceleration_theta])
        # print("\n\nradial acceleration!", acceleration_r)
        # print("theta acceleration!", acceleration_theta)
        # print("acceleration squared!", polar_acceleration_squared)
        return acceleration_polar
    
    def chi_acceleration_squared(self, point_xy_in_Chi_plane, Gamma):
        """This function calculates the acceleration squared at a given point in the flow field in the Chi plane"""
        acceleration = self.chi_acceleration(point_xy_in_Chi_plane, Gamma)
        acceleration_squared = np.dot(acceleration, acceleration)
        return acceleration_squared
    

    def velocity(self, point_xi_eta_in_z_plane, Gamma):
        """This function calculates the velocity at a given point in the flow field in the z plane"""
        z = point_xi_eta_in_z_plane[0] + 1j*point_xi_eta_in_z_plane[1]
        zeta = self.z_to_zeta(z)
        V_inf, epsilon, R, alpha, zeta0 = self.freestream_velocity, self.epsilon, self.cylinder_radius, self.angle_of_attack, self.zeta_center
        velocity = V_inf*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf*(zeta-zeta0)) - np.exp(1j*alpha)*R**2/(zeta-zeta0)**2) / (1 - (R-epsilon)**2/(zeta)**2) # eq 107
        velocity_complex = np.array([velocity.real, -velocity.imag])
        return velocity_complex

    def polar_vector(self, theta, cartesian_vector):
        """This function converts the cartesian velocity to polar velocity (can go from z, zeta, or chi plane to polar velocity)"""
        r = cartesian_vector[0]*np.cos(theta) + cartesian_vector[1]*np.sin(theta)
        # print("\nradial velocity", velocity_r)
        theta = cartesian_vector[1]*np.cos(theta) - cartesian_vector[0]*np.sin(theta)
        # print("theta velocity", velocity_theta)
        polar_velocity = np.array([r, theta])
        return polar_velocity
    
    def numerical_convective_acceleration(self, point_xi_eta_z, Gamma, step):
        """calculates the convective acceleration numerically in the z plane"""
        r, theta = hlp.xy_to_r_theta(point_xi_eta_z[0], point_xi_eta_z[1])
        cartesian_z_velocity = self.velocity(point_xi_eta_z, Gamma)
        polar_z_velocity = hlp.polar_vector(theta, cartesian_z_velocity)
        omega_r, omega_theta = polar_z_velocity[0], polar_z_velocity[1]
        step_point = point_xi_eta_z
        
        omega_r_plus_dr, omega_r_minus_dr, omega_r_plus_dtheta, omega_r_minus_dtheta, omega_theta_plus_dr, omega_theta_minus_dr, omega_theta_plus_dtheta, omega_theta_minus_dtheta = self.function_plus_minus_step_variable(step_point, Gamma, step)
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
        
    def function_plus_minus_step_variable(self, point_xy, Gamma, stepsize):
        """takes in a function and returns the function plus and minus a stepsize"""
        r, theta = hlp.xy_to_r_theta(point_xy[0], point_xy[1])
        r_plus, r_minus = r + stepsize, r - stepsize
        theta_plus, theta_minus = theta + stepsize, theta - stepsize

        x_r_plus, x_r_minus, y_r_plus, y_r_minus = hlp.r_theta_to_xy(r_plus, theta)[0], hlp.r_theta_to_xy(r_minus, theta)[0], hlp.r_theta_to_xy(r_plus, theta)[1], hlp.r_theta_to_xy(r_minus, theta)[1]
        x_theta_plus, x_theta_minus, y_theta_plus, y_theta_minus = hlp.r_theta_to_xy(r, theta_plus)[0], hlp.r_theta_to_xy(r, theta_minus)[0], hlp.r_theta_to_xy(r, theta_plus)[1], hlp.r_theta_to_xy(r, theta_minus)[1]
        omega_xy_plus_r, omega_xy_minus_r = self.velocity([x_r_plus, y_r_plus], Gamma), self.velocity([x_r_minus, y_r_minus], Gamma)
        omega_xy_plus_theta, omega_xy_minus_theta = self.velocity([x_theta_plus, y_theta_plus], Gamma), self.velocity([x_theta_minus, y_theta_minus], Gamma)
        omega_r_plus_dr, omega_r_minus_dr = hlp.polar_vector(theta, omega_xy_plus_r)[0], hlp.polar_vector(theta, omega_xy_minus_r)[0]
        omega_r_plus_dtheta, omega_r_minus_dtheta = hlp.polar_vector(theta, omega_xy_plus_theta)[0], hlp.polar_vector(theta, omega_xy_minus_theta)[0]
        omega_theta_plus_dr, omega_theta_minus_dr = hlp.polar_vector(theta, omega_xy_plus_r)[1], hlp.polar_vector(theta, omega_xy_minus_r)[1]
        omega_theta_plus_dtheta, omega_theta_minus_dtheta = hlp.polar_vector(theta, omega_xy_plus_theta)[1], hlp.polar_vector(theta, omega_xy_minus_theta)[1]
        return omega_r_plus_dr, omega_r_minus_dr, omega_r_plus_dtheta, omega_r_minus_dtheta, omega_theta_plus_dr, omega_theta_minus_dr, omega_theta_plus_dtheta, omega_theta_minus_dtheta

    def convective_acceleration_squared_numerical(self, point_xi_eta, Gamma, step):
        """Calculates the convective acceleration squared numerically xi and eta are the coordinates in the plane of interest"""
        convective_acceleration = self.numerical_convective_acceleration(point_xi_eta, Gamma, step)
        convective_acceleration_squared = np.dot(convective_acceleration, convective_acceleration)
        return convective_acceleration_squared
    
    def numerical_appellian(self, Gamma_range: list, r_range: list, theta_range: list, is_analytic_conv_accel: bool, step):
        """This function calculates the Appellian function numerically requires evenly spaced ranges for Gamma, r, and theta in Chi"""
        # create a meshgrid of r and theta values, the first value in r_range is the lower bound the second value is the upper bound, the third value is the increment size
        Gamma_values = hlp.list_to_range(Gamma_range)
        r_values = hlp.list_to_range(r_range)
        theta_values = hlp.list_to_range(theta_range)
        # calculate the area element 
        if len(r_values) == 1:
            dr = 1.0
        else:        
            dr = r_values[1] - r_values[0]
        if len(theta_values) == 1:
            dtheta = 1.0
        else:
            dtheta = theta_values[1] - theta_values[0]
        # now get dr and dtheta in z plane using the zeta to z function 
        d_zeta = hlp.r_theta_to_xy(dr, dtheta)[0] + 1j*hlp.r_theta_to_xy(dr, dtheta)[1]
        d_z = self.zeta_to_z(d_zeta)
        d_xi_z, d_eta_z = d_z.real, d_z.imag
        # now these are the dr and dtheta values in the z plane
        dr, dtheta = hlp.xy_to_r_theta(d_xi_z, d_eta_z)
        # The appellian sums the squared convective acceleration at each r and theta value. Plot the Appellian function with respect to each change in Gamma.
        Appellian_array = np.zeros((len(Gamma_values), 2))
        xi_eta_values = np.zeros((len(r_values)*len(theta_values), 2))
        Appellian_value = 0.0
        index = 0
        print("NUMERICAL CONVECTIVE ACCELERATION")
        for i in range(len(Gamma_values)):
            for j in range(len(r_values)):
                for k in range(len(theta_values)):
                    Chi = hlp.r_theta_to_xy(r_values[j], theta_values[k])[0] + 1j*hlp.r_theta_to_xy(r_values[j], theta_values[k])[1] # the r, theta point in the Chi plane is converted to a complex number
                    zeta = self.Chi_to_zeta(Chi)
                    z = self.zeta_to_z(zeta)
                    r_z = hlp.xy_to_r_theta(z.real, z.imag)[0]
                    area_element = r_z * dr * dtheta
                    convective_acceleration = self.numerical_convective_acceleration([z.real, z.imag], Gamma_values[i], step)
                    Appellian_value += np.dot(convective_acceleration, convective_acceleration)*area_element
                    if i == 0:  # Save xi and eta values only once
                        xi_eta_values[index] = [z.real, z.imag]
                        index += 1
            Appellian_array[i] = [Gamma_values[i], 0.5*Appellian_value]
            Appellian_value = 0.0
        return Appellian_array, xi_eta_values
    
    def zeta_to_Chi(self, zeta: complex):
        """This function takes in a zeta coordinate and returns the Chi coordinate"""
        Chi = zeta - self.zeta_center
        return Chi

    def Chi_to_zeta(self, Chi: complex):
        """This function takes in a Chi coordinate and returns the zeta coordinate"""
        zeta = Chi + self.zeta_center
        return zeta
    
    def zeta_to_z(self, zeta: complex):
        """This function takes in a zeta coordinate and returns the z coordinate"""
        if np.isclose(zeta.real, 0.0) and np.isclose(zeta.imag, 0.0):
            z = zeta
        else:
            z = zeta + (self.cylinder_radius - self.epsilon)**2/zeta # eq 96
        return z
    
    def z_to_zeta(self, z: complex): # eq 104 
        """This function takes in a z coordinate and returns the zeta coordinate"""
        z_1 = z**2 - 4*(self.cylinder_radius - self.epsilon)**2
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
    
    def conv_accel_from_equations(Gamma, r, R, theta, r_0, theta_0, V_inf, epsilon, alpha, velocity):
        term1 = np.exp(-1j * alpha) + 1j * (Gamma / (2 * np.pi * V_inf)) * (1 / (r * np.exp(1j * theta))) - (R**2 * np.exp(1j * alpha)) / ((r * np.exp(1j * theta))**2)
        # print("term1", term1)
        term2 = (-1j * (Gamma / (2 * np.pi * V_inf)) * (1 / ((r * np.exp(1j * theta))**2))) + (2 * R**2 * np.exp(1j * alpha) / ((r * np.exp(1j * theta))**3))
        # print("term2", term2)
        term3 = 2 * (R - epsilon)**2 / ((r * np.exp(1j * theta)) + (r_0 * np.exp(1j * theta_0)))**3
        # print("term3", term3)
        term4 = (1 - ((R - epsilon)**2 / ((r * np.exp(1j * theta)) + (r_0 * np.exp(1j * theta_0)))**2))
        # print("term4", term4)
        
        # Compute grad Cp using the equation in the image
        conv_accel = velocity**2 * term1 / term4 * (term2*term4 - (term1 * term3)) / term4**3
        # conv_accel = velocity**2 * term1 / term4 * (term2*term4 - (term1 * term3)) / term4**3
        
        # Convert to real-valued pressure gradient vector
        conv_accel_complex = np.array([conv_accel.real, -conv_accel.imag])
        # conv_accel_complex = np.linalg.norm(conv_accel_complex)
        return conv_accel_complex
    
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
        z = self.Joukowski_zeta_to_z(point_xi_eta_in_zeta, self.epsilon)
        r_z, theta_z = hlp.xy_to_r_theta(z.real, z.imag)
        return r_z, theta_z
    
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
    
    def d2phi_dzeta2(self, point_r_theta_in_Chi, Gamma):
        V_inf, epsilon, R, alpha, zeta_0 = self.freestream_velocity, self.epsilon, self.cylinder_radius, self.angle_of_attack, self.zeta_center
        xi_0, eta_0 = zeta_0.real, zeta_0.imag
        r_0, theta_0 = hlp.xy_to_r_theta(xi_0, eta_0)
        r, theta = point_r_theta_in_Chi[0], point_r_theta_in_Chi[1]
        d2phi_dzeta2 = V_inf*(-1j*Gamma/(2*np.pi*V_inf*(r*np.exp(1j*theta))**2) + 2*np.exp(1j*alpha)*R**2/((r*np.exp(1j*theta))**3))
        d2phi_dzeta2_comp_conj = np.conj(d2phi_dzeta2)
        d2phi_dzeta2_squared = d2phi_dzeta2 * d2phi_dzeta2_comp_conj
        # print("d2phi_dzeta2\n", d2phi_dzeta2_squared)
        return d2phi_dzeta2_squared
    
    def grid_plots(self):
        # Reset the plot settings
        self.reset_plot_stuff()

        # Parameters for the polar grid
        R = self.cylinder_radius  # Cylinder radius
        zeta_center = self.zeta_center  # Center of the cylinder in the zeta plane
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
        plt.plot(self.upper_zeta_coords[:, 0], self.upper_zeta_coords[:, 1], color='black')
        plt.plot(self.lower_zeta_coords[:, 0], self.lower_zeta_coords[:, 1], color='black')

        # plt.show()

        # Transform the grid to the z-plane
        plt.figure()
        self.reset_plot_stuff()

        # Arrays to store the transformed lines
        transformed_radial_lines = []
        transformed_circular_lines = []

        # Transform radial lines to the z-plane
        for line in radial_lines:
            transformed_line = []
            for point in line:
                zeta = point[0] + 1j * point[1]  # Convert to complex zeta
                z = self.zeta_to_z(zeta, self.epsilon)  # Transform to z-plane
                transformed_line.append([z.real, z.imag])  # Store the transformed point
            transformed_line = np.array(transformed_line)  # Convert to numpy array
            transformed_radial_lines.append(transformed_line)  # Save the transformed line
            plt.plot(transformed_line[:, 0], transformed_line[:, 1], color='black')  # Plot the transformed radial line

        # Transform circular lines to the z-plane
        for line in circular_lines:
            transformed_line = []
            for point in line:
                zeta = point[0] + 1j * point[1]  # Convert to complex zeta
                z = self.zeta_to_z(zeta, self.epsilon)  # Transform to z-plane
                transformed_line.append([z.real, z.imag])  # Store the transformed point
            transformed_line = np.array(transformed_line)  # Convert to numpy array
            transformed_circular_lines.append(transformed_line)  # Save the transformed line
            plt.plot(transformed_line[:, 0], transformed_line[:, 1], color='black')  # Plot the transformed circular line

        # Plot the geometry of the shape in the z-plane
        plt.plot(self.upper_coords[:, 0], self.upper_coords[:, 1], color='black')
        plt.plot(self.lower_coords[:, 0], self.lower_coords[:, 1], color='black')

        plt.show()

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

    # def analytic_conv_accel_for_line_int_comp_conj(self, point_r_theta_in_Chi, Gamma, step_size = 1e-5):
    #     """calculates the convective acceleration using the line integral method"""
    #     V_inf, epsilon, R, alpha, zeta_0 = self.freestream_velocity, self.epsilon, self.cylinder_radius, self.angle_of_attack, self.zeta_center
    #     r, theta = point_r_theta_in_Chi[0], point_r_theta_in_Chi[1]
    #     r_plus_dr = r + step_size
    #     r_0, theta_0 = hlp.xy_to_r_theta(zeta_0.real, zeta_0.imag)
    #     xi_chi, eta_chi = hlp.r_theta_to_xy(r, theta)
    #     xi_chi_plus_dr, eta_chi_plus_dr = hlp.r_theta_to_xy(r_plus_dr, theta)
    #     chi = xi_chi + 1j*eta_chi
    #     chi_plus_dr = xi_chi_plus_dr + 1j*eta_chi_plus_dr
    #     zeta = self.Chi_to_zeta(chi)
    #     zeta_plus_dr = self.Chi_to_zeta(chi_plus_dr)
    #     dz_dzeta, omega_of_zeta = self.dZ_dzeta(zeta), self.dPhi_dzeta(zeta, Gamma)
    #     G_of_zeta = 1/dz_dzeta
    #     dz_dzeta_plus_dr, omega_of_zeta_plus_dr = self.dZ_dzeta(zeta_plus_dr), self.dPhi_dzeta(zeta_plus_dr, Gamma)
    #     G_of_zeta_plus_dr = 1/dz_dzeta_plus_dr
    #     G_of_zeta_comp_conj = np.conj(G_of_zeta)
    #     omega_of_zeta_comp_conj = np.conj(omega_of_zeta)
    #     G_of_zeta_plus_dr_comp_conj = np.conj(G_of_zeta_plus_dr)
    #     omega_of_zeta_plus_dr_comp_conj = np.conj(omega_of_zeta_plus_dr)
    #     integrand = G_of_zeta**2*G_of_zeta_comp_conj**2*omega_of_zeta**2*omega_of_zeta_comp_conj**2
    #     integrand_plus_dr = G_of_zeta_plus_dr**2*G_of_zeta_plus_dr_comp_conj**2*omega_of_zeta_plus_dr**2*omega_of_zeta_plus_dr_comp_conj**2
    #     # find the partial derivative of the integrand with respect to r
    #     line_int = (integrand_plus_dr - integrand)/(step_size)
    #     dzeta_dtheta = 1j*r*np.exp(1j*theta)
    #     line_int *= abs(dzeta_dtheta)*abs(dz_dzeta)
    #     return np.real(line_int)

    # def appellian_acceleration_loop(self, Gamma: float, r_values: np.array, theta_values: np.array, dr: float, dtheta: float, D, acceleration: callable):
    #     """"""
    #     Appellian_value = 0.0
    #     if self.appellian_is_area_integral: # using complex conjugate
    #         for j in range(len(r_values)):
    #             for k in range(len(theta_values)):
    #                 area_element = r_values[j]*dr*dtheta
    #                 accel = acceleration([r_values[j], theta_values[k]], Gamma)
    #                 Appellian_value += accel*area_element
    #         return Appellian_value*(0.5)
    #             # Define the integration bounds
    #     elif self.appellian_is_line_integral:
    #         for k in range(len(theta_values)):
    #             area_element = r_values[0]*dtheta
    #             accel = acceleration([r_values[0], theta_values[k]], Gamma)
    #             Appellian_value += accel*area_element
    #         return Appellian_value*(-0.03125) # 1/32
    #     else:  # using numerical integration
    #         print("Using numerical integration for Appellian acceleration")
    #         return Appellian_value*0.5

        # def appellian_acceleration_loop(self, Gamma: float, r_values: np.array, theta_values: np.array, acceleration: callable):
    #     """"""
    #     Appellian_value = 0.0
    #     if self.appellian_is_area_integral:
    #         if len(r_values) <= 1: # if there is only one r value, then we are dealing with a line integral but with the area integral acceleration formula
    #             if self.integration_method == "left_riemann": # using riemann integration
    #                 for k in range(len(theta_values)):
    #                     if k == len(theta_values) - 1:
    #                         accel = self.riemann_integration([r_values[0], r_values[0]], [theta_values[k], theta_values[0]], Gamma, acceleration)
    #                     else:
    #                         accel = self.riemann_integration([r_values[0], r_values[0]], [theta_values[k], theta_values[k+1]], Gamma, acceleration)
    #                     Appellian_value += accel
    #             elif self.integration_method == "trapezoidal": # using trapezoidal integration
    #                 for k in range(len(theta_values)):
    #                     if k == len(theta_values) - 1:
    #                         accel = self.trapezoidal_integration([r_values[0], r_values[0]], [theta_values[k], theta_values[0]], Gamma, acceleration)
    #                     else:
    #                         accel = self.trapezoidal_integration([r_values[0], r_values[0]], [theta_values[k], theta_values[k+1]], Gamma, acceleration)
    #                     Appellian_value += accel
    #         else:
    #             if self.integration_method == "left_riemann": # using riemann integration  
    #                 for j in range(len(r_values)-1):  
    #                     for k in range(len(theta_values)):
    #                         accel = 0.0
    #                         if k == len(theta_values) - 1:
    #                             accel = self.riemann_integration([r_values[j], r_values[j+1]],[theta_values[k], theta_values[0]],Gamma, acceleration)
    #                         else:
    #                             accel = self.riemann_integration([r_values[j], r_values[j+1]],[theta_values[k], theta_values[k+1]],Gamma, acceleration)
    #                         Appellian_value += accel
    #             elif self.integration_method == "trapezoidal": # using trapezoidal integration
    #                 for j in range(len(r_values) - 1):  # <-- Only go up to len(r_values)-2
    #                     for k in range(len(theta_values)):
    #                         if k == len(theta_values) - 1: 
    #                             accel = self.trapezoidal_integration([r_values[j], r_values[j+1]],[theta_values[k], theta_values[0]],Gamma, acceleration)
    #                         else:
    #                             accel = self.trapezoidal_integration([r_values[j], r_values[j+1]],[theta_values[k], theta_values[k+1]],Gamma, acceleration)
    #                         Appellian_value += accel
    #             else:
    #                 raise ValueError("Invalid integration method specified. Use 'left_riemann' or 'trapezoidal'.")
    #         return Appellian_value*(0.5)
    #     else: # using line integral
    #         if self.integration_method == "left_riemann": # using riemann integration
    #             for k in range(len(theta_values)):
    #                 if k == len(theta_values) - 1: # if k is the last element, then theta values should go from theta_values[k] to theta_values[0] to close the loop
    #                     accel = self.riemann_integration([r_values[0], r_values[0]], [theta_values[k], theta_values[0]], Gamma, acceleration)
    #                 else:
    #                     accel = self.riemann_integration([r_values[0], r_values[0]], [theta_values[k], theta_values[k+1]], Gamma, acceleration)
    #                 Appellian_value += accel
    #         elif self.integration_method == "trapezoidal": # using trapezoidal integration
    #             for k in range(len(theta_values)):
    #                 if k == len(theta_values) - 1:
    #                     accel = self.trapezoidal_integration([r_values[0], r_values[0]], [theta_values[k], theta_values[0]], Gamma, acceleration)
    #                 else:
    #                     accel = self.trapezoidal_integration([r_values[0], r_values[0]], [theta_values[k], theta_values[k+1]], Gamma, acceleration)
    #                 Appellian_value += accel
    #         else:
    #             raise ValueError("Invalid integration method specified. Use 'left_riemann' or 'trapezoidal'.")
    #         return Appellian_value*(-0.03125) # 1/32

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

    def Taha_zeta_to_z(self, zeta: complex, epsilon: float):
        """This function takes in a zeta coordinate and returns the z coordinate"""
        if np.isclose(zeta.real, 0.0) and np.isclose(zeta.imag, 0.0):
            z = zeta  
        else:
            z = zeta + ((1 - self.D)/(1 + self.D)) * (self.C**2 / zeta)
        return z

    def calc_J_airfoil_thickness(self, CLd):
        """This function calculates the thickness of the airfoil"""
        thickness = -3*np.sqrt(3)/4 + (3*np.sqrt(3)*CLd)/(8*np.pi*(self.zeta_center.imag/self.cylinder_radius))  # rearrangement of eq 152 in complex variables
        return thickness

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
        self.surface_points = self.full_z_surface
        self.make_surface_points_go_from_lower_trailing_to_upper_trailing()
        self.surface_points = self.shift_joukowski_airfoil()
        print("Exporting geometry to txt file...")
        file_name = "Joukowski" + self.type + "_" + str(self.n_geom_points) + ".txt"
        np.savetxt(file_name, self.surface_points, delimiter = ",", header = "x, y", comments = "")
        print("Geometry exported to", file_name)


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

    def run_J_airfoil_stuff(self):
        """"""
        lift = self.calc_J_airfoil_CL()
        print("CL Joukowski: ", lift)
        Cmo = self.calc_J_airfoil_Cmz((0+1j*0))
        print("Cm0 Joukowski: ", Cmo)
        c4 = self.calc_J_airfoil_c4()
        print("C4 location Joukowski: ", c4)
        Cmc4 = self.calc_J_airfoil_Cmz(c4)
        print("Cmc4 Joukowski: ", Cmc4)
        self.output_J_airfoil()

    def Chi1_to_zeta(self, Chi1: complex):
        """This function takes in a Chi1 coordinate and returns the zeta coordinate"""
        zeta = Chi1 + self.zeta1
        return zeta

    def reimann_or_trapezoidal_integration(self):
        """This function performs a Riemann or trapezoidal integration over the zeta geometry"""
        if self.integration_method == "left_riemann_sum":
            self.integration_function = self.reimann_integration
        elif self.integration_method == "trapezoidal_rule":
            self.integration_function = self.trapezoidal_integration
    
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
    
    def zeta_to_Chi1(self, zeta: complex):
        """This function takes in a zeta coordinate and returns the Chi1 coordinate"""
        Chi1 = zeta - self.zeta1
        return Chi1

    def plot_geometry_in_every_plane(self):
        """This function plots the geometry in every plane"""
        plt.plot(self.full_z_surface[:,0], self.full_z_surface[:,1], label="$z$", color="black")
        plt.plot(self.zeta_geom_array[:,0], self.zeta_geom_array[:,1], linestyle="--", label="$\\zeta$", color="black")
        self.get_and_plot_foci()
        # convert zeta geometry to chi1 plane 
        chi1_plane = np.zeros((len(self.zeta_geom_array), 2))
        chi1_r = np.linspace(self.cylinder_radius, self.cylinder_radius, len(self.zeta_geom_array)) # this is an array of the same length as zeta_geom_array, with all values equal to the cylinder radius
        chi1_theta = np.linspace(0, 2*np.pi, len(self.zeta_geom_array), endpoint=False) # this is an array of the same length as zeta_geom_array, with values from 0 to 2*pi
        for i in range(len(chi1_plane)):
            chi1_x = chi1_r[i]*np.cos(chi1_theta[i])
            chi1_y = chi1_r[i]*np.sin(chi1_theta[i])
            chi1 = chi1_x + 1j*chi1_y # this is a complex number
            # convert chi1 to the zeta plane
            chi1_in_zeta = self.Chi1_to_zeta(chi1)
            chi1_plane[i][0] = chi1_in_zeta.real
            chi1_plane[i][1] = chi1_in_zeta.imag
        # plot the chi1 plane        
        plt.plot(chi1_plane[:,0], chi1_plane[:,1], linestyle=":", label="$\\chi_1$", color="red")
        # convert the ch1_plane to the z_plane 
        z_plane = np.zeros((len(chi1_plane), 2))
        for i in range(len(chi1_plane)):
            chi1 = chi1_plane[i][0] + 1j*chi1_plane[i][1]
            z = self.zeta_to_z(chi1, self.epsilon)
            z_plane[i][0] = z.real
            z_plane[i][1] = z.imag
        # plot the z plane
        plt.plot(z_plane[:,0], z_plane[:,1], label="$z$", color="red")
        plt.title("Joukowski Cylinder Geometry", fontsize=16)

        
    def acceleration(self, point_xi_eta_in_z_plane, Gamma):
        """This function calculates the acceleration at a given point in the flow field in the z plane"""
        z = point_xi_eta_in_z_plane[0] + 1j*point_xi_eta_in_z_plane[1]
        zeta = self.z_to_zeta(z, self.epsilon)
        dphi_dzeta, d2Phi_dzeta2, dz_dzeta, d2z_dzeta2 = self.dPhi_dzeta(zeta, Gamma), self.d2Phi_dzeta2(zeta, Gamma), self.dZ_dzeta(zeta), self.d2Z_dzeta2(zeta)
        acceleration = (dz_dzeta*d2Phi_dzeta2- dphi_dzeta*d2z_dzeta2)/(dz_dzeta**3)
        acceleration_complex = np.array([acceleration.real, -acceleration.imag])
        return acceleration_complex

    def numerically_integrate_appellian_chi1(self, Gamma: float, r_values: np.array, D):
        """This function calculates the Appellian function numerically requires evenly spaced ranges for Gamma, r, and theta in Chi"""
        # zeta_start is the location of the trailing edge foci
        chi_A = self.cylinder_radius - self.epsilon - self.zeta_center
        self.r_A, self.theta_A = hlp.xy_to_r_theta(chi_A.real, chi_A.imag)
        theta_start = self.theta_A
        theta_end = 2*np.pi + theta_start
        theta_values = np.linspace(theta_start, theta_end, 10)  # ensure we have enough points
        theta_values = theta_values[1:-1]  # remove the first and last points
        dtheta = 0.0000001
        theta_values = np.concatenate(([theta_start + dtheta],theta_values,[theta_end - dtheta]))
        r_values = np.array([self.cylinder_radius])
        xi_chi_array = r_values * np.cos(theta_values)
        eta_chi_array = r_values * np.sin(theta_values)
        Chi_values = xi_chi_array + 1j*eta_chi_array
        zeta_values = np.zeros(len(Chi_values), dtype=complex)
        for i, chi in enumerate(Chi_values):
            zeta_values[i] = self.Chi_to_zeta(chi)
        Chi1_values = np.zeros(len(zeta_values), dtype=complex)
        for i, zeta in enumerate(zeta_values):
            Chi1_values[i] = self.zeta_to_Chi1(zeta)
        Chi1_r, Chi1_theta = np.zeros(len(Chi1_values)), np.zeros(len(Chi1_values))
        for i, chi1 in enumerate(Chi1_values):
            Chi1_r[i], Chi1_theta[i] = hlp.xy_to_r_theta(chi1.real, chi1.imag)
        Appellian_array = self.appellian_acceleration_loop(Gamma, Chi1_r, Chi1_theta, self.analytic_conv_accel_chi1_for_line_int_comp_conj)
        return Appellian_array

    def analytic_conv_accel_chi1_for_line_int_comp_conj(self, point_r_theta_in_Chi1, Gamma, step_size = 1e-5):
        """calculates the convective acceleration using the line integral method"""
        V_inf, epsilon, R, alpha, zeta_0 = self.freestream_velocity, self.epsilon, self.cylinder_radius, self.angle_of_attack, self.zeta_center
        r, theta = point_r_theta_in_Chi1[0], point_r_theta_in_Chi1[1]
        xi_chi1, eta_chi1 = hlp.r_theta_to_xy(r, theta)
        chi1 = xi_chi1 + 1j*eta_chi1
        # leading edge foci
        zeta = self.Chi1_to_zeta(chi1)
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

    def kutta_for_rounded_TE(self, D):
        """"""
        epsilon = self.calc_epsilon_from_D(D)
        V_inf, R, xi0, eta0,alpha = self.freestream_velocity, self.cylinder_radius, self.zeta_center.real, self.zeta_center.imag, self.angle_of_attack
        xi1, eta1 = R-epsilon-np.sqrt(R**2-eta0**2), eta0
        first = 2*np.pi*V_inf*(R-epsilon-xi1-1j*eta1)/1j 
        second = R**2*np.exp(1j*alpha)/((R-epsilon-xi1-1j*eta1)**2)
        third = np.exp(1j*alpha)
        kutta_for_rounded = first*(second-third)
        print("kutta_for_rounded: ", kutta_for_rounded)
        return kutta_for_rounded

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

    def d2Z_dzeta2_Taha(self, zeta):
        return 2*(1-self.D)/(1+self.D)*(self.C**2)/(zeta)**3

    def dZ_dzeta_Taha(self, zeta):
        """""This function calculates the derivative of Z with respect to zeta for a Taha transformation"""
        return 1 - (1-self.D)/(1+self.D)*(self.C**2)/(zeta**2) 

    def calc_gamma_app_over_gamma_kutta(self, D_values, r_values, theta_values):
        kutta_appellian_array = np.zeros((len(D_values), 2))
        if self.transformation_type == "joukowski":
            for i in tqdm(range(len(D_values)), desc="Calculating Appellian Roots"):  
                self.D = D_values[i]
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

    def calc_taha_zeta_center_from_epsilon(self):
        """This function calculates zeta center from epsilon"""
        return -self.epsilon*self.C

    def calc_taha_epsilon_from_tau(self):
        """This function calculates epsilon from tau"""
        return 4*self.tau/(3*np.sqrt(3))
    
    def calc_taha_C_from_epsilon(self):
        """This function calculates C from cylinder radius, D and epsilon"""
        return 1/(1+self.epsilon) # right after Eq. 13 in The Principle of Minimum Pressure Gradient as a Selection Criterion for Weak Solutions of Euler’s Equation paper by Taha

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

    def potential(self, point_xi_eta_in_z_plane, Gamma, C = 0.0):
        """This function calculates the potential at a given point in the flow field in the z plane"""
        z = point_xi_eta_in_z_plane[0] + 1j*point_xi_eta_in_z_plane[1]
        zeta = self.z_to_zeta(z, self.epsilon)
        Potential = self.freestream_velocity*(zeta*np.exp(-1j*self.angle_of_attack) + 1j*Gamma/(2*np.pi*self.freestream_velocity)*np.log(zeta-self.zeta_center) + np.exp(1j*self.angle_of_attack)*self.cylinder_radius**2/(zeta-self.zeta_center)) + C
        Potential_complex = np.array([Potential.real, Potential.imag])  # the minus sign is because we are using the complex potential
        return Potential_complex

    def potential_for_mesh(self, point_xi_eta_in_z_plane, C = 0.0):
        """This function calculates the potential at a given point in the flow field in the z plane"""
        Gamma = self.circulation  # use the circulation from the class variable
        z = point_xi_eta_in_z_plane[0] + 1j*point_xi_eta_in_z_plane[1]
        zeta = self.z_to_zeta(z, self.epsilon)
        Potential = self.freestream_velocity*(zeta*np.exp(-1j*self.angle_of_attack) + 1j*Gamma/(2*np.pi*self.freestream_velocity)*np.log(zeta-self.zeta_center) + np.exp(1j*self.angle_of_attack)*self.cylinder_radius**2/(zeta-self.zeta_center)) + C
        Potential_complex = np.array([Potential.real, Potential.imag])  # the minus sign is because we are using the complex potential
        return Potential_complex

    def stream_function(self, point_xi_eta_in_z_plane):
        """This function calculates the stream function from the potential vector"""
        potential = self.potential_for_mesh(point_xi_eta_in_z_plane)
        stream_function = potential[1]  # the second element is the imaginary part
        return stream_function

    def small_potential_function(self, point_xi_eta_in_z_plane):
        """"""
        potential = self.potential_for_mesh(point_xi_eta_in_z_plane)
        # The small potential function is the real part of the complex potential
        small_potential = potential[0]  # the first element is the real part
        return small_potential

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

    def reset_plot_stuff(self):
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

    def analytic_conv_accel_square_comp_conj_wiki(self, point_r_theta_chi, Gamma):
        """This function calculates the pressure gradient at a given point in the flow field in the z plane"""
        point_xi_eta_chi = hlp.r_theta_to_xy(point_r_theta_chi[0], point_r_theta_chi[1])
        Chi = point_xi_eta_chi[0] + 1j*point_xi_eta_chi[1]
        zeta = self.Chi_to_zeta(Chi)
        z = self.zeta_to_z(zeta, self.epsilon)
        V_inf, epsilon, R, alpha, zeta0 = self.freestream_velocity, self.epsilon, self.cylinder_radius, self.angle_of_attack, self.zeta_center
        plus = (z + np.sqrt(z**2 - 4*(R-epsilon)**2))/2
        minus = (z - np.sqrt(z**2 - 4*(R-epsilon)**2))/2
        omega_z = V_inf*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf*(zeta-zeta0)) - np.exp(1j*alpha)*R**2/(zeta-zeta0)**2) / (1 - (R-epsilon)**2/(zeta)**2)
        dPhi_dzeta = V_inf*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf*(zeta-zeta0)) - np.exp(1j*alpha)*R**2/(zeta-zeta0)**2)
        d2Phi_dzeta2 = V_inf*(-1j*Gamma/(2*np.pi*V_inf*(zeta-zeta0)**2) + 2*np.exp(1j*alpha)*R**2/((zeta-zeta0)**3))
        if np.isclose(zeta, plus, atol=1e-10):
            dzeta_dz_squared = ((1/2)+(z)/(2*np.sqrt(z**2-4*(R-epsilon)**2)))**2
            d2zeta_dz2 = -(2*(R-epsilon)**2)/(z**2-4*(R-epsilon)**2)**(3/2)
        elif np.isclose(zeta, minus, atol=1e-10):
            dzeta_dz_squared = ((1/2)-(z)/(2*np.sqrt(z**2-4*(R-epsilon)**2)))**2
            d2zeta_dz2 = (2*(R-epsilon)**2)/(z**2-4*(R-epsilon)**2)**(3/2)
        else:
            print("zeta: ", zeta)
            print("plus: ", plus)
            print("minus: ", minus)
            raise ValueError("zeta is not equal to plus or minus")
        convective_acceleration = omega_z*(d2Phi_dzeta2*dzeta_dz_squared + dPhi_dzeta*d2zeta_dz2)
        convective_acceleration_comp_conj = np.conj(convective_acceleration)
        convective_acceleration_squared = convective_acceleration * convective_acceleration_comp_conj
        return np.real(convective_acceleration_squared)
    
    def taha_analytic_conv_accel_square_comp_conj(self, point_r_theta_in_Chi, Gamma): # need to adjust so that it uses the Taha transformation
        """This function calculates the pressure gradient at a given point in the flow field in the z plane"""
        V_inf, epsilon, R, alpha, zeta_0 = self.freestream_velocity, self.epsilon, self.cylinder_radius, self.angle_of_attack, self.zeta_center
        xi_0, eta_0 = zeta_0.real, zeta_0.imag
        r_0, theta_0 = hlp.xy_to_r_theta(xi_0, eta_0)
        r, theta = point_r_theta_in_Chi[0], point_r_theta_in_Chi[1]
        xi_chi, eta_chi = hlp.r_theta_to_xy(r, theta)
        chi = xi_chi + 1j*eta_chi
        zeta = self.Chi_to_zeta(chi)
        G_of_zeta = 1/(1-(R-epsilon)**2/(r*np.exp(1j*theta)+r_0*np.exp(1j*theta_0))**2)
        # G_of_zeta = 1/self.zeta_to_z(zeta)
        G_of_zeta_comp_conj = np.conj(G_of_zeta)
        G_of_zeta_squared = G_of_zeta * G_of_zeta_comp_conj
        # derivative of G with respect to zeta
        dG_dzeta = 4
        dG_dzeta_comp_conj = np.conj(dG_dzeta)
        omega_of_zeta = V_inf*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf*(r*np.exp(1j*theta))) - np.exp(1j*alpha)*R**2/(r*np.exp(1j*theta))**2)
        omega_of_zeta_comp_conj = np.conj(omega_of_zeta)
        omega_of_zeta_squared = omega_of_zeta * omega_of_zeta_comp_conj 
        d_omega_of_zeta_dzeta = V_inf*(-1j*Gamma/(2*np.pi*V_inf*(r*np.exp(1j*theta))**2) + 2*np.exp(1j*alpha)*R**2/((r*np.exp(1j*theta))**3))
        d_omega_of_zeta_dzeta_comp_conj = np.conj(d_omega_of_zeta_dzeta)
        a_of_zeta = omega_of_zeta*d_omega_of_zeta_dzeta_comp_conj
        integrand = G_of_zeta_squared * (G_of_zeta_comp_conj * a_of_zeta + omega_of_zeta_squared*dG_dzeta_comp_conj)
        integrand_comp_conj = np.conj(integrand)
        conv_accel_squared = integrand * integrand_comp_conj
        # self.tau = 0.1
        # self.epsilon = self.calc_taha_epsilon_from_tau()
        # self.C = self.calc_taha_C_from_epsilon()
        return np.real(conv_accel_squared)
    
    def analytic_conv_accel_for_line_int_comp_conj(self, point_r_theta_in_Chi, Gamma, step = 1e-14):
        """"""
        V_inf, epsilon, R, alpha, zeta_0 = self.freestream_velocity, self.epsilon, self.cylinder_radius, self.angle_of_attack, self.zeta_center
        r, theta = point_r_theta_in_Chi[0], point_r_theta_in_Chi[1]
        xi_Chi, eta_Chi = hlp.r_theta_to_xy(r, theta)
        Chi = xi_Chi + 1j*eta_Chi
        r_0, theta_0 = hlp.xy_to_r_theta(zeta_0.real, zeta_0.imag)
        A = Gamma/(2*np.pi*V_inf)
        B = R**2
        C = (R-epsilon)**2
        conv_accel_squared = ((np.exp(-alpha*1j)+(A*np.exp(-theta*1j)*1j)/r - (B*np.exp(alpha*1j)*np.exp(-theta*2*1j))/r**2)**2*(- np.exp(alpha*1j)+(A*np.exp(theta*1j)*1j)/r \
                            + (B*np.exp(-alpha*1j)*np.exp(theta*2*1j))/r**2)**2)/((C/(r*np.exp(-theta*1j) + r_0*np.exp(-theta_0*1j))**2 - 1)**2*(C/(r*np.exp(theta*1j) + \
                            r_0*np.exp(theta_0*1j))**2 - 1)**2) - (2*r*((A*np.exp(-theta*1j)*1j)/r**2 - (2*B*np.exp(alpha*1j)*np.exp(-theta*2*1j))/r**3)*(np.exp(-alpha*1j) + \
                            (A*np.exp(-theta*1j)*1j)/r - (B*np.exp(alpha*1j)*np.exp(-theta*2*1j))/r**2)*(- np.exp(alpha*1j) + (A*np.exp(theta*1j)*1j)/r + (B*np.exp(-alpha*1j)* \
                            np.exp(theta*2*1j))/r**2)**2)/((C/(r*np.exp(-theta*1j) + r_0*np.exp(-theta_0*1j))**2 - 1)**2*(C/(r*np.exp(theta*1j) + r_0*np.exp(theta_0*1j))**2 - 1)**2) -\
                            (2*r*((A*np.exp(theta*1j)*1j)/r**2 + (2*B*np.exp(-alpha*1j)*np.exp(theta*2*1j))/r**3)*(np.exp(-alpha*1j) + (A*np.exp(-theta*1j)*1j)/r - (B*np.exp(alpha*1j)*\
                            np.exp(-theta*2*1j))/r**2)**2*(- np.exp(alpha*1j) + (A*np.exp(theta*1j)*1j)/r + (B*np.exp(-alpha*1j)*np.exp(theta*2*1j))/r**2))/((C/(r*np.exp(-theta*1j) + \
                            r_0*np.exp(-theta_0*1j))**2 - 1)**2*(C/(r*np.exp(theta*1j) + r_0*np.exp(theta_0*1j))**2 - 1)**2) + (4*C*r*np.exp(-theta*1j)*(np.exp(-alpha*1j) + (A*np.exp(-theta*1j)*1j)/r \
                            - (B*np.exp(alpha*1j)*np.exp(-theta*2*1j))/r**2)**2*(- np.exp(alpha*1j) + (A*np.exp(theta*1j)*1j)/r + (B*np.exp(-alpha*1j)*np.exp(theta*2*1j))/r**2)**2)/((r*np.exp(-theta*1j) \
                            + r_0*np.exp(-theta_0*1j))**3*(C/(r*np.exp(-theta*1j) + r_0*np.exp(-theta_0*1j))**2 - 1)**3*(C/(r*np.exp(theta*1j) + r_0*np.exp(theta_0*1j))**2 - 1)**2) + \
                            (4*C*r*np.exp(theta*1j)*(np.exp(-alpha*1j) + (A*np.exp(-theta*1j)*1j)/r - (B*np.exp(alpha*1j)*np.exp(-theta*2*1j))/r**2)**2*(- np.exp(alpha*1j) + (A*np.exp(theta*1j)*1j)/r \
                            + (B*np.exp(-alpha*1j)*np.exp(theta*2*1j))/r**2)**2)/((r*np.exp(theta*1j) + r_0*np.exp(theta_0*1j))**3*(C/(r*np.exp(-theta*1j) + r_0*np.exp(-theta_0*1j))**2 - 1)**2*\
                            (C/(r*np.exp(theta*1j) + r_0*np.exp(theta_0*1j))**2 - 1)**3)
        # print("conv_accel_squared", conv_accel_squared)
        return np.real(conv_accel_squared)
    

    def appellian_acceleration_loop(self, Gamma: float, r_values: np.array, theta_values: np.array, dr: float, dtheta: float, D, acceleration: callable):
        """"""
        Appellian_value = 0.0
        if self.appellian_is_area_integral: # using complex conjugate
            # if abs(D) >= 0.001 and r_values[0] != r_values[-1]:
            #     r_min, r_max = r_values[0], r_values[-1]
            #     theta_min, theta_max = theta_values[0], theta_values[-1]

            #     # Define the integrand for dblquad
            #     def integrand(theta, r):
            #         accel = acceleration([r, theta], Gamma)
            #         return accel * r  # Include the r term for the area element

            #     # Perform the double integration
            #     integral, _ = dblquad(integrand, r_min, r_max, lambda r: theta_min, lambda r: theta_max)

            #     # Scale the result
            #     Appellian_value = integral * 0.5
            #     return Appellian_value
            # elif abs(D) >= 0.001 and r_values[0] == r_values[-1]:
            #     def integrand(theta, r, Gamma):
            #         """Define the integrand for the line integral."""
            #         accel = acceleration([r, theta], Gamma)
            #         return accel # Include the r term for the area element

            #     # Use the single value of r from r_values
            #     r = r_values[0]#+0.005*self.cylinder_radius  # Assuming r_values contains only one value
            #     theta_start = theta_values[0]
            #     theta_end = theta_values[-1]

            #     # Perform the integration over theta
            #     integral, _ = quad(integrand, theta_start, theta_end, args=(r, Gamma))

            #     # The result of the line integral
            #     Appellian_value = integral*(0.5)
            #     return Appellian_value
            # else:
            for j in range(len(r_values)):
                for k in range(len(theta_values)):
                    area_element = r_values[j]*dr*dtheta
                    accel = acceleration([r_values[j], theta_values[k]], Gamma)
                    Appellian_value += accel*area_element
            return Appellian_value*(0.5)
                # Define the integration bounds
        elif self.appellian_is_line_integral:
        #     if abs(D) > 0.001:
        #         def integrand(theta, r, Gamma):
        #             """Define the integrand for the line integral."""
        #             accel = acceleration([r, theta], Gamma)
        #             return accel # Include the r term for the area element

        #         # Use the single value of r from r_values
        #         r = r_values[0]#+0.005*self.cylinder_radius  # Assuming r_values contains only one value
        #         theta_start = theta_values[0]
        #         theta_end = theta_values[-1]

        #         # Perform the integration over theta
        #         integral, _ = quad(integrand, theta_start, theta_end, args=(r, Gamma))

        #         # The result of the line integral
        #         Appellian_value = integral*(-0.03125) # 1/32
        #         return Appellian_value
        #     else:
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
    

            def run_two_stage_convergence(self, Gamma, acceleration_func, reference_value,
                                fixed_r=1.0, r_range=(1.0, 10.0), theta_range=(0, 2*np.pi),
                                output_dir="Grid_conv", tol=1e-3,
                                run_theta_stage=True, run_r_stage=True):
        # Add memoization to speed up repeated function evaluations
        eval_cache = {}
        def memoized_func(point, Gamma):
            key = (round(point[0], 12), round(point[1], 12))
            if key not in eval_cache:
                eval_cache[key] = acceleration_func(point, Gamma)
            return eval_cache[key]

        os.makedirs(output_dir, exist_ok=True)
        default_N_values = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960, 81920,163840, 327680, 655360, 1310720]
        simpson_N_values = [2**k + 1 for k in range(2, 18)]
        theta_converged_dict = {}
        best_N_theta_dict = {}
        method_dict = {"Riemann": self.riemann_integration,"Trapezoidal": self.trapezoidal_integration,"Simpson": self.simpson_13_integration}

        if run_theta_stage:
            print("Running theta-direction convergence study...")
            for method_name, integrator in method_dict.items():
                N_theta_list = simpson_N_values if method_name == "Simpson" else default_N_values
                gridpoints, rel_errors = [], []
                theta_converged = None
                best_N_theta = None
                time_start = time.time()
                for N_theta in tqdm(N_theta_list, desc=f"{method_name} theta convergence"):
                    r_values = np.array([fixed_r])
                    theta_values = np.linspace(theta_range[0], theta_range[1], N_theta, endpoint=False)
                    dtheta = abs(theta_values[1] - theta_values[0])
                    theta_values += dtheta / 2
                    self.integration_func = integrator
                    result = self.appellian_acceleration_loop(Gamma, r_values, theta_values, memoized_func)
                    error = abs((result - reference_value) / reference_value) * 100
                    gridpoints.append(N_theta)
                    rel_errors.append(error)
                    if error < tol and theta_converged is None:
                        theta_converged = theta_values
                        best_N_theta = N_theta
                        break
                theta_converged_dict[method_name] = theta_converged
                best_N_theta_dict[method_name] = best_N_theta
                arr = np.column_stack((gridpoints, rel_errors))
                total_time = time.time() - time_start
                filename = os.path.join(output_dir,
                    f"{method_name.lower()}_theta_convergence_D_{round(self.D, 3)}_zeta0_r_{round(self.zeta_center.real, 3)}"
                    f"_zeta0_i_{round(self.zeta_center.imag, 3)}_time_{round(total_time, 3)}s_.csv")
                np.savetxt(filename, arr, delimiter=",", header="GridPoints,RelativeError", comments='')

            # Romberg Theta Convergence
            print("Running Romberg theta convergence...")
            gridpoints_rom, rel_errors_rom, refined_int, grids = [], [], [], []
            max_iter = 10  # reduced max_iter for speed
            time_start = time.time()
            initial_N = 10  # smaller initial N for faster startup
            r_values = np.array([fixed_r])
            theta_values = np.linspace(theta_range[0], theta_range[1], initial_N, endpoint=False)
            dtheta = abs(theta_values[1] - theta_values[0])
            theta_values += dtheta / 2
            grids.append((r_values, theta_values))

            pbar = tqdm(range(max_iter), desc="Romberg Integration", dynamic_ncols=True)
            for j in pbar:
                rj, thetaj = grids[-1]
                self.integration_func = self.trapezoidal_integration
                I_j1 = self.appellian_acceleration_loop(Gamma, rj, thetaj, memoized_func)
                refined_int.append([I_j1])

                for k in range(1, j + 1):
                    I_jk = (4**k * refined_int[j][k-1] - refined_int[j-1][k-1]) / (4**k - 1)
                    refined_int[j].append(I_jk)

                I_curr = refined_int[j][j]
                error_t = abs((I_curr - reference_value) / reference_value) * 100
                gridpoints_rom.append(len(thetaj))
                rel_errors_rom.append(error_t)
                pbar.set_postfix({"error_t": f"{error_t:.2e}%", "tol": f"{tol:.2e}"})

                if j > 1:
                    delta = abs(refined_int[j][j] - refined_int[j-1][j-1])
                    if error_t < tol or delta < 1e-10:
                        break

                theta_refined = self.refine_uniform_grid_theta(thetaj)
                grids.append((rj, theta_refined))

            romberg_theta_converged = grids[-1][1]
            romberg_best_N_theta = len(grids[-1][1])
            arr_rom = np.column_stack((gridpoints_rom, rel_errors_rom))
            total_time = time.time() - time_start
            rom_filename = os.path.join(output_dir,
                f"romberg_theta_convergence_D_{round(self.D, 3)}_zeta0_r_{round(self.zeta_center.real, 3)}"
                f"_zeta0_i_{round(self.zeta_center.imag, 3)}_time_{round(total_time, 3)}s_.csv")
            np.savetxt(rom_filename, arr_rom, delimiter=",", header="GridPoints,RelativeError", comments='')
            theta_converged_dict["Romberg"] = romberg_theta_converged
            best_N_theta_dict["Romberg"] = romberg_best_N_theta

        if run_r_stage:
            print("Running radial-direction convergence study...")
            for method_name, integrator in method_dict.items():
                N_r_list = simpson_N_values if method_name == "Simpson" else default_N_values
                gridpoints, rel_errors = [], []
                converged = False
                theta_converged = theta_converged_dict.get(method_name)

                if theta_converged is None:
                    print(f"Skipping radial convergence for {method_name} as theta did not converge.")
                    continue

                time_start = time.time()
                for N_r in tqdm(N_r_list, desc=f"{method_name} radial convergence"):
                    r_values = np.linspace(r_range[0], r_range[1], N_r)
                    self.integration_func = integrator
                    result = self.appellian_acceleration_loop(Gamma, r_values, theta_converged, memoized_func)
                    error = abs((result - reference_value) / reference_value) * 100
                    gridpoints.append(N_r)
                    rel_errors.append(error)

                    if error < tol and not converged:
                        converged = True
                        break

                arr = np.column_stack((gridpoints, rel_errors))
                total_time = time.time() - time_start
                filename = os.path.join(output_dir,
                    f"{method_name.lower()}_radial_convergence_D_{round(self.D, 3)}_zeta0_r_{round(self.zeta_center.real, 3)}"
                    f"_zeta0_i_{round(self.zeta_center.imag, 3)}_time_{round(total_time, 3)}s_.csv")
                np.savetxt(filename, arr, delimiter=",", header="GridPoints,RelativeError", comments='')

            print("Running Romberg radial convergence...")
            theta_converged = theta_converged_dict.get("Romberg")
            if theta_converged is None:
                print("Skipping Romberg radial convergence as theta did not converge.")
                return

            gridpoints_rom, rel_errors_rom, refined_int, grids = [], [], [], []
            max_iter = 10
            time_start = time.time()

            initial_N = 10
            r_values = np.linspace(r_range[0], r_range[1], initial_N)
            grids.append((r_values, theta_converged))

            pbar = tqdm(range(max_iter), desc="Romberg Radial Integration", dynamic_ncols=True)
            for j in pbar:
                rj, thetaj = grids[-1]
                self.integration_func = self.trapezoidal_integration
                I_j1 = self.appellian_acceleration_loop(Gamma, rj, thetaj, memoized_func)
                refined_int.append([I_j1])

                for k in range(1, j + 1):
                    I_jk = (4**k * refined_int[j][k-1] - refined_int[j-1][k-1]) / (4**k - 1)
                    refined_int[j].append(I_jk)

                I_curr = refined_int[j][j]
                error_t = abs((I_curr - reference_value) / reference_value) * 100
                gridpoints_rom.append(len(rj))
                rel_errors_rom.append(error_t)
                pbar.set_postfix({"error_t": f"{error_t:.2e}%", "tol": f"{tol:.2e}"})

                if j > 1:
                    delta = abs(refined_int[j][j] - refined_int[j-1][j-1])
                    if error_t < tol or delta < 1e-10:
                        break

                r_refined = self.refine_uniform_grid_r(rj)
                grids.append((r_refined, thetaj))

            arr_rom = np.column_stack((gridpoints_rom, rel_errors_rom))
            total_time = time.time() - time_start
            rom_filename = os.path.join(output_dir,
                f"romberg_radial_convergence_D_{round(self.D, 3)}_zeta0_r_{round(self.zeta_center.real, 3)}"
                f"_zeta0_i_{round(self.zeta_center.imag, 3)}_time_{round(total_time, 3)}s_.csv")
            np.savetxt(rom_filename, arr_rom, delimiter=",", header="GridPoints,RelativeError", comments='')

    def romberg_integration(self, Gamma, r_values, theta_values, acceleration, eps=1e-3, max_iter=5242880, reference_value=None):
        refined_int = []  # Store results for each iteration
        grids = [(r_values, theta_values)]
        if reference_value is None:
            pbar = tqdm(range(max_iter), desc="Romberg Integration", dynamic_ncols=True)
            epsilon_a = None
            for j in pbar:
                rj, thetaj = grids[-1]
                self.integration_func = self.trapezoidal_integration
                I_j1 = self.appellian_acceleration_loop(Gamma, rj, thetaj, acceleration)
                refined_int.append([I_j1])
                for k in range(1, j + 1):
                    I_jk = (4**k * refined_int[j][k - 1] - refined_int[j - 1][k - 1]) / (4**k - 1)
                    refined_int[j].append(I_jk)
                if j > 0:
                    epsilon_a = abs((refined_int[j][j] - refined_int[j - 1][j - 1]) / refined_int[j][j])
                    pbar.set_postfix({"epsilon_a": f"{epsilon_a:.2e}", "tol": f"{eps:.2e}"})
                    if epsilon_a < eps:
                        pbar.close()
                        return refined_int[j][j], len(rj) * len(thetaj)
                else:
                    pbar.set_postfix({"epsilon_a": "N/A", "tol": f"{eps:.2e}"})
                # Refine grid
                r_refined = self.refine_uniform_grid_r(rj)
                theta_refined = self.refine_uniform_grid_theta(thetaj)
                grids = [(r_refined, theta_refined)]
            pbar.close()
            # If never converged, return None and grid size
            return None, len(grids[-1][0]) * len(grids[-1][1])
        else:
            pbar = tqdm(range(max_iter), desc="Romberg Integration", dynamic_ncols=True)
            error_t = None
            for j in pbar:
                rj, thetaj = grids[-1]
                self.integration_func = self.trapezoidal_integration
                I_j1 = self.appellian_acceleration_loop(Gamma, rj, thetaj, acceleration)
                refined_int.append([I_j1])
                for k in range(1, j + 1):
                    I_jk = (4**k * refined_int[j][k - 1] - refined_int[j - 1][k - 1]) / (4**k - 1)
                    refined_int[j].append(I_jk)
                if j > 0:
                    error_t = abs((refined_int[j][j] - reference_value) / reference_value) * 100
                    pbar.set_postfix({"error_t": f"{error_t:.2e}%", "tol": f"{eps:.2e}"})
                    if error_t < eps:
                        pbar.close()
                        return refined_int[j][j], len(rj) * len(thetaj)
                else:
                    pbar.set_postfix({"error_t": "N/A", "tol": f"{eps:.2e}"})
                # Refine grid
                r_refined = self.refine_uniform_grid_r(rj)
                theta_refined = self.refine_uniform_grid_theta(thetaj)
                grids = [(r_refined, theta_refined)]
            pbar.close()
            # If never converged, return None and grid size
            return None, len(grids[-1][0]) * len(grids[-1][1])
        
        # def appellian_acceleration_loop(self, Gamma: float, r_values: np.array, theta_values: np.array, acceleration: callable):
    #     """
    #     Computes Appellian acceleration using either a line or area integral, and either
    #     left Riemann or trapezoidal integration.
    #     """
    #     Appellian_value = 0.0
    #     # Handle line or area integration
    #     if self.appellian_is_area_integral:
    #         # Area integral
    #         if len(r_values) <= 1: # Single radius → effectively a line over theta
    #             r0 = r_values[0]
    #             if self.is_compute_appellian: # then use tqdm progress bar
    #                 for k in tqdm(range(len(theta_values)), desc="Computing Appellian Area Integral", unit="segment"):
    #                     theta0 = theta_values[k]
    #                     theta1 = theta_values[(k + 1) % len(theta_values)]
    #                     Appellian_value += self.integration_func([r0, r0], [theta0, theta1], Gamma, acceleration)
    #             else: # no progress bar
    #                 for k in range(len(theta_values)):
    #                     theta0 = theta_values[k]
    #                     theta1 = theta_values[(k + 1) % len(theta_values)]
    #                     Appellian_value += self.integration_func([r0, r0], [theta0, theta1], Gamma, acceleration)
    #         else: # Full polar area integration over (r, theta)
    #             if self.is_compute_appellian: # then use tqdm progress bar
    #                 for j in tqdm(range(len(r_values) - 1), desc="Computing Appellian Area Integral", unit="segment"):
    #                     r0, r1 = r_values[j], r_values[j + 1]
    #                     for k in range(len(theta_values)):
    #                         theta0 = theta_values[k]
    #                         theta1 = theta_values[(k + 1) % len(theta_values)]
    #                         Appellian_value += self.integration_func([r0, r1], [theta0, theta1], Gamma, acceleration)
    #         return Appellian_value * 0.5  # Confirm this constant is required
    #     else: # Line integral (theta loop, fixed r)
    #         r0 = r_values[0]
    #         if self.is_compute_appellian: # then use tqdm progress bar
    #             for k in tqdm(range(len(theta_values)), desc="Computing Appellian Line Integral", unit="segment"):
    #                 theta0 = theta_values[k]
    #                 theta1 = theta_values[(k + 1) % len(theta_values)]
    #                 Appellian_value += self.integration_func([r0, r0], [theta0, theta1], Gamma, acceleration)
    #         else: # no progress bar
    #             for k in range(len(theta_values)):
    #                 theta0 = theta_values[k]
    #                 theta1 = theta_values[(k + 1) % len(theta_values)]
    #                 Appellian_value += self.integration_func([r0, r0], [theta0, theta1], Gamma, acceleration)
    #         return Appellian_value * (-0.03125)  # Confirm this constant is required

        # def run_two_stage_convergence(self, Gamma, acceleration_func, reference_value, fixed_r=1.0, r_range=(1.0, 10.0), theta_range=(0, 2*np.pi),output_dir="Grid_conv", tol=1e-3,run_theta_stage=True, run_r_stage=True, relative_error = False):
    #     # Add memoization to speed up repeated function evaluations
    #     memoized_func = memoize_acceleration_func(acceleration_func)
    #     os.makedirs(output_dir, exist_ok=True)
    #     default_N_values = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960, 81920,163840, 327680, 655360, 1310720]
    #     simpson_N_values = [2**k + 1 for k in range(2, 18)]
    #     theta_converged_dict = {}
    #     best_N_theta_dict = {}
    #     # method_dict = {"Riemann": self.riemann_integration,"Trapezoidal": self.trapezoidal_integration,"Romberg": self.single_romberg_integration, "Simpson": self.simpson_13_integration}
    #     method_dict = {"Romberg": self.single_romberg_integration}
    #     if run_theta_stage:
    #         print("Running theta-direction convergence study...")
    #         for method_name, integrator in method_dict.items():
    #             if method_name != "Romberg":
    #                 memoized_func.clear_cache()
    #             N_theta_list = simpson_N_values if method_name == "Simpson" else default_N_values
    #             gridpoints, rel_errors = [], []
    #             theta_converged = None
    #             best_N_theta = None
    #             time_start = time.time()
    #             error = 100.0
    #             index = 0
    #             previous_result = 0
    #             for N_theta in tqdm(N_theta_list, desc=f"{method_name} theta convergence"):
    #                 r_values = np.array([fixed_r])
    #                 theta_values = np.linspace(theta_range[0], theta_range[1], N_theta, endpoint=False)
    #                 dtheta = abs(theta_values[1] - theta_values[0])
    #                 theta_values += dtheta / 2
    #                 self.integration_func = integrator
    #                 result = self.appellian_acceleration_loop(Gamma, r_values, theta_values, memoized_func)
    #                 if relative_error == False:
    #                     error = abs((result - reference_value) / reference_value) * 100
    #                 elif relative_error == True and index >0:
    #                     error = abs((result - previous_result)/previous_result)*100
    #                 else:
    #                     error = error
    #                 index += 1
    #                 previous_result = result
    #                 gridpoints.append(N_theta)
    #                 rel_errors.append(error)
    #                 if error < tol and theta_converged is None:
    #                     theta_converged = theta_values
    #                     best_N_theta = N_theta
    #                     break
    #             theta_converged_dict[method_name] = theta_converged
    #             best_N_theta_dict[method_name] = best_N_theta
    #             arr = np.column_stack((gridpoints, rel_errors))
    #             total_time = time.time() - time_start
    #             filename = os.path.join(output_dir,f"{method_name.lower()}_theta_convergence_D_{round(self.D, 3)}_zeta0_r_{round(self.zeta_center.real, 3)}"f"_zeta0_i_{round(self.zeta_center.imag, 3)}_time_{round(total_time, 3)}s_.csv")
    #             np.savetxt(filename, arr, delimiter=",", header="GridPoints,RelativeError", comments='')
    #     if run_r_stage:
    #         print("Running radial-direction convergence study...")
    #         for method_name, integrator in method_dict.items():
    #             if method_name != "Romberg":
    #                 memoized_func.clear_cache()
    #             N_r_list = simpson_N_values if method_name == "Simpson" else default_N_values
    #             gridpoints, rel_errors = [], []
    #             converged = False
    #             theta_converged = theta_converged_dict.get(method_name)
    #             if theta_converged is None:
    #                 print(f"Skipping radial convergence for {method_name} as theta did not converge.")
    #                 continue
    #             time_start = time.time()
    #             error = 100.0
    #             index = 0
    #             previous_result = 0
    #             for N_r in tqdm(N_r_list, desc=f"{method_name} radial convergence"):
    #                 r_values = np.linspace(r_range[0], r_range[1], N_r)
    #                 self.integration_func = integrator
    #                 result = self.appellian_acceleration_loop(Gamma, r_values, theta_converged, memoized_func)
    #                 if relative_error == False:
    #                     error = abs((result - reference_value) / reference_value) * 100
    #                 elif relative_error == True and index >0:
    #                     error = abs((result - previous_result)/previous_result)*100
    #                 else:
    #                     error = error
    #                 index += 1
    #                 previous_result = result
    #                 gridpoints.append(N_theta)
    #                 rel_errors.append(error)
    #                 gridpoints.append(N_r)
    #                 rel_errors.append(error)
    #                 if error < tol and not converged:
    #                     converged = True
    #                     break
    #             arr = np.column_stack((gridpoints, rel_errors))
    #             total_time = time.time() - time_start
    #             filename = os.path.join(output_dir,f"{method_name.lower()}_radial_convergence_D_{round(self.D, 3)}_zeta0_r_{round(self.zeta_center.real, 3)}"f"_zeta0_i_{round(self.zeta_center.imag, 3)}_time_{round(total_time, 3)}s_.csv")
    #             np.savetxt(filename, arr, delimiter=",", header="GridPoints,RelativeError", comments='')

def appellian_acceleration_loop(self,Gamma: float,r_values: np.array,theta_values: np.array,acceleration: callable,wraparound: bool = False):
        """
        Computes Appellian acceleration using either a line or area integral,
        and supports optional wraparound for periodic integration in theta.
        Parameters:
            Gamma (float): Strength of the vortex sheet or source.
            r_values (np.array): Array of r coordinates.
            theta_values (np.array): Array of theta coordinates.
            acceleration (callable): Vector field to integrate.
            wraparound (bool): Whether to include the final segment from theta[-1] to theta[0].
        """
        Appellian_value = 0.0
        # Construct theta intervals
        theta_pairs = [(theta_values[k], theta_values[k + 1]) for k in range(len(theta_values) - 1)]
        # if wraparound:
            # theta_pairs.append((theta_values[-1], theta_values[0]))
        if self.appellian_is_area_integral:
            # === Area integral ===
            if len(r_values) <= 1:  # Fixed radius → line in theta
                r0 = r_values[0]
                if self.is_compute_appellian:
                    print("Computing Appellian Using An Area Integral Applied on the Surface")
                for theta0, theta1 in theta_pairs:
                        Appellian_value += self.integration_func([r0, r0], [theta0, theta1], Gamma, acceleration)
            else:  # Full polar area integration (r, theta)
                if self.is_compute_appellian:
                        print("Computing Appellian Using An Area Integral Applied on and around the Surface")
                for j in range(len(r_values) - 1):
                    r0, r1 = r_values[j], r_values[j + 1]
                    for theta0, theta1 in theta_pairs:
                        Appellian_value += self.integration_func([r0, r1], [theta0, theta1], Gamma, acceleration)
            return 0.5 * Appellian_value 
        else:
            # === Line integral ===
            r0 = r_values[0]
            if self.is_compute_appellian:
                for theta0, theta1 in tqdm(theta_pairs, desc="Computing Appellian Line Integral", unit="segment"):
                    Appellian_value += self.integration_func([r0, r0], [theta0, theta1], Gamma, acceleration)
            else:
                for theta0, theta1 in theta_pairs:
                    Appellian_value += self.integration_func([r0, r0], [theta0, theta1], Gamma, acceleration)

            return Appellian_value * (-0.03125) 
        

            def appellian_acceleration_loop(self,Gamma: float,r_values: np.array,theta_values: np.array,acceleration: callable,wraparound: bool = False):
        """
        Computes Appellian acceleration using either a line or area integral,
        and supports optional wraparound for periodic integration in theta.
        Parameters:
            Gamma (float): Strength of the vortex sheet or source.
            r_values (np.array): Array of r coordinates.
            theta_values (np.array): Array of theta coordinates.
            acceleration (callable): Vector field to integrate.
            wraparound (bool): Whether to include the final segment from theta[-1] to theta[0].
        """
        Appellian_value = 0.0
        # Construct theta intervals
        theta_pairs = [(theta_values[k], theta_values[k + 1]) for k in range(len(theta_values) - 1)]
        # if wraparound:
            # theta_pairs.append((theta_values[-1], theta_values[0]))
        if self.integration_func != self.single_romberg_integration:
            if self.appellian_is_area_integral:
                # === Area integral ===
                if len(r_values) <= 1:  # Fixed radius → line in theta
                    r0 = r_values[0]
                    if self.is_compute_appellian:
                        print("Computing Appellian Using An Area Integral Applied on the Surface")
                    for theta0, theta1 in theta_pairs:
                            Appellian_value += self.integration_func([r0, r0], [theta0, theta1], Gamma, acceleration)
                else:  # Full polar area integration (r, theta)
                    if self.is_compute_appellian:
                            print("Computing Appellian Using An Area Integral Applied on and around the Surface")
                    for j in range(len(r_values) - 1):
                        r0, r1 = r_values[j], r_values[j + 1]
                        for theta0, theta1 in theta_pairs:
                            Appellian_value += self.integration_func([r0, r1], [theta0, theta1], Gamma, acceleration)
                return 0.5 * Appellian_value 
            else:
                # === Line integral ===
                r0 = r_values[0]
                if self.is_compute_appellian:
                    for theta0, theta1 in tqdm(theta_pairs, desc="Computing Appellian Line Integral", unit="segment"):
                        Appellian_value += self.integration_func([r0, r0], [theta0, theta1], Gamma, acceleration)
                else:
                    for theta0, theta1 in theta_pairs:
                        Appellian_value += self.integration_func([r0, r0], [theta0, theta1], Gamma, acceleration)

                return Appellian_value * (-0.03125) 
            
        else:  # O(h^6) Romberg integration
            thetas_half = double_nodes_periodic(theta_values)
            thetas_quarter = double_nodes_periodic(thetas_half)
            theta_pairs = [(theta_values[k], theta_values[k + 1]) for k in range(len(theta_values) - 1)]
            theta_pairs_half = [(thetas_half[k], thetas_half[k + 1]) for k in range(len(thetas_half) - 1)]
            theta_pairs_quarter = [(thetas_quarter[k], thetas_quarter[k + 1]) for k in range(len(thetas_quarter) - 1)]

            if self.appellian_is_area_integral:
                Appellian_value = 0.0
                Appellian_value_half = 0.0
                Appellian_value_quarter = 0.0
                if len(r_values) <= 1:  # Line in theta at fixed radius
                    r0 = r_values[0]
                    if self.is_compute_appellian:
                        print("Computing Appellian Using An Area Integral Applied on the Surface")
                    for theta0, theta1 in theta_pairs:
                        Appellian_value += self.trapezoidal_integration([r0, r0], [theta0, theta1], Gamma, acceleration)
                    for theta0, theta1 in theta_pairs_half:
                        Appellian_value_half += self.trapezoidal_integration([r0, r0], [theta0, theta1], Gamma, acceleration)
                    for theta0, theta1 in theta_pairs_quarter:
                        Appellian_value_quarter += self.trapezoidal_integration([r0, r0], [theta0, theta1], Gamma, acceleration)
                else:  # Area integral over full domain
                    if self.is_compute_appellian:
                        print("Computing Appellian Using An Area Integral Applied on and around the Surface")
                    for j in range(len(r_values) - 1):
                        r0, r1 = r_values[j], r_values[j + 1]
                        for theta0, theta1 in theta_pairs:
                            Appellian_value += self.trapezoidal_integration([r0, r1], [theta0, theta1], Gamma, acceleration)
                        for theta0, theta1 in theta_pairs_half:
                            Appellian_value_half += self.trapezoidal_integration([r0, r1], [theta0, theta1], Gamma, acceleration)
                        for theta0, theta1 in theta_pairs_quarter:
                            Appellian_value_quarter += self.trapezoidal_integration([r0, r1], [theta0, theta1], Gamma, acceleration)
                I_l = (4/3) * Appellian_value_half - (1/3) * Appellian_value
                I_m = (4/3) * Appellian_value_quarter - (1/3) * Appellian_value_half
                I_final = (16/15) * I_m - (1/15) * I_l
                return 0.5 * I_final
            else:  # Line integral case
                r0 = r_values[0]
                Appellian_value = 0.0
                Appellian_value_half = 0.0
                Appellian_value_quarter = 0.0
                if self.is_compute_appellian:
                    print("Computing Appellian Line Integral Using Romberg")
                for theta0, theta1 in theta_pairs:
                    Appellian_value += self.trapezoidal_integration([r0, r0], [theta0, theta1], Gamma, acceleration)
                for theta0, theta1 in theta_pairs_half:
                    Appellian_value_half += self.trapezoidal_integration([r0, r0], [theta0, theta1], Gamma, acceleration)
                for theta0, theta1 in theta_pairs_quarter:
                    Appellian_value_quarter += self.trapezoidal_integration([r0, r0], [theta0, theta1], Gamma, acceleration)
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
            
 def appellian_acceleration_loop(self, Gamma: float, r_values: np.array, theta_values: np.array, grid_conv = False):
        Appellian_value = 0.0
        # theta_values = np.unique(theta_values)
        theta_pairs = [(theta_values[k], theta_values[k + 1]) for k in range(len(theta_values) - 1)]
        if self.integration_func != self.romberg_integration:
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
            
        def memoize_integration_scheme(self, integration_func):
            cache = {}
            def wrapper(point_pair_r, point_pair_theta, Gamma):
                key = (tuple(round(r, 12) for r in point_pair_r),tuple(round(theta, 12) for theta in point_pair_theta),Gamma)
                if key not in cache:
                    cache[key] = integration_func(point_pair_r, point_pair_theta, Gamma)
                return cache[key]
            wrapper.clear_cache = lambda: cache.clear()
            return wrapper
        
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

        def riemann_integration(self, r_values, theta_values, Gamma):
            """Left Riemann integration for a segment (1D or 2D).Evaluates function at lower corner and multiplies by r * dtheta or r * dr * dtheta."""
            r0, r1 = r_values
            theta0, theta1 = theta_values
            dtheta = theta1 - theta0
            dr = r1 - r0
            if r0 == r1:
                f = self.accel_function([r0, theta0], Gamma)
                return f * r0 * dtheta
            else:
                f = self.accel_function([r0, theta0], Gamma)
                return f * r0 * dr * dtheta
        
    def single_romberg_integration(self, r_values, theta_values, Gamma):
        """Takes the two fidelities of integration to make a better estimate"""
        # self.integration_func = self.trapezoidal_integration

    def trapezoidal_integration(self, r_values, theta_values, Gamma):
        """r_values is either 1 repeated value or 2 distinct values. theta_values is 2 distinct values. This function calculates the integral of a function over a single line or area segment depending on if r0=r1"""
        r0, r1 = r_values
        theta0, theta1 = theta_values
        dtheta = abs(theta1 - theta0)
        # if dtheta <= 0:
            # dtheta += 2 * np.pi
        if r0 == r1:  # 1D
            f0 = self.accel_function([r0, theta0], Gamma)
            f1 = self.accel_function([r0, theta1], Gamma)
            return 0.5 * (f0 + f1) * r0 * dtheta
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
            return (f0 + 4*f1 + f2) * r0 * dtheta / 6
            else:
                raise ValueError("Simpson 1/3 only applies to 1D integration")

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
        dtheta = theta1 - theta0
        dr = r1 - r0
        # if dtheta <= 0:
            # dtheta += 2 * np.pi
        if r0 == r1:
            # 1D over theta at fixed r
            f = self.accel_function([r0, theta0], Gamma)
            return f * r0 * dtheta
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
        dtheta = abs(theta1 - theta0)
        # if dtheta <= 0:
            # dtheta += 2 * np.pi
        if r0 == r1:  # 1D
            f0 = self.accel_function([r0, theta0], Gamma)
            f1 = self.accel_function([r0, theta1], Gamma)
            return 0.5 * (f0 + f1) * r0 * dtheta
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
            return (f0 + 4*f1 + f2) * r0 * dtheta / 6
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
    
    def refine_uniform_grid_r(self, values: np.ndarray):
        """Refines a uniformly spaced 1D grid by inserting midpoints between existing points."""
        midpoints = (values[:-1] + values[1:]) / 2
        refined = np.empty(len(values) + len(midpoints))
        refined[::2] = values
        refined[1::2] = midpoints
        return refined
    
    def refine_uniform_grid_theta(self, values: np.ndarray):
        """Refines a 1D theta grid while avoiding singularity at theta_AJA."""
        theta_start = self.theta_AJA
        theta_end = self.theta_AJA + 2 * np.pi
        new_N = 2 * len(values)
        theta_values = np.linspace(theta_start, theta_end, new_N, endpoint=False)
        dtheta = abs(theta_values[1] - theta_values[0])
        # Shift to avoid singularity
        offset_theta = theta_values + dtheta / 2
        # Ensure singularity is avoided
        epsilon = 1e-10
        offset_theta = np.where(np.isclose((offset_theta - theta_start) % (2 * np.pi), 0.0, atol=epsilon),offset_theta + dtheta / 2, offset_theta)
        return offset_theta


    if __name__ == "__main__":
        four = 4 # placeholder 
        # zeta_test = 4 - 1j*4
        # Gamma = self.circulation
        # # z_test = self.zeta_to_z(zeta_test)
        # chi_test = self.zeta_to_Chi(zeta_test)
        # print("\n")
        # z_test = self.zeta_to_z(zeta_test)
        # complex_zeta_center = self.zeta_center.real + 1j*self.zeta_center.imag
        # z_0 = self.zeta_to_z(complex_zeta_center)
        # # print("z_0: ", z_0)
        # zeta_test = [zeta_test.real, zeta_test.imag]
        # theta_zeta = np.arctan2(zeta_test[1], zeta_test[0])
        # z_test = [z_test.real, z_test.imag]
        # # theta_z = np.arctan2(z_test[1], z_test[0])
        # chi_test = [chi_test.real, chi_test.imag]
        # theta_chi = np.arctan2(chi_test[1], chi_test[0])
        # z_velocity_at_z_test = self.velocity(z_test, Gamma)
        # z_polar = hlp.polar_vector(theta_zeta, z_velocity_at_z_test)
        # # Chi_polar = hlp.polar_vector(theta_chi, velocity_chi_at_chi_test)
        # z_polar_mag = np.sqrt(z_polar[0]**2 + z_polar[1]**2)
        
        # diffs between acceleration methods
            # counter = 0
    # decimal_places = 14
    # num_points = len(r_vals) * len(theta_vals)
    # tolerance = 10**-decimal_places  # Absolute tolerance for comparison
    # print("Number of points to be compared: ", num_points)
    # print("Comparison out to " + str(decimal_places) + " decimal places")
            # for i in range(len(r_vals)):
    #     for j in range(len(theta_vals)):
    #         conv_accel_squared_comp_conj = self.taha_analytic_conv_accel_square_comp_conj([r_vals[i], theta_vals[j]], self.circulation)
    #         alternative_accel_squared_comp_conj = self.analytic_conv_accel_square_comp_conj([r_vals[i], theta_vals[j]], self.circulation)
            
    #         # Check if the values are equal within the specified decimal places
    #         if not np.isclose(conv_accel_squared_comp_conj, alternative_accel_squared_comp_conj, atol=tolerance): # atol is the absolute tolerance which is calculated as atol = alt
    #             # compare the absolute tolerance to the result
    #             # print("conv_accel_squared_comp_conj: ", conv_accel_squared_comp_conj)
    #             # print("alternative_accel_squared_comp_conj: ", alternative_accel_squared_comp_conj)
    #             # print("Difference: ", abs(conv_accel_squared_comp_conj - alternative_accel_squared_comp_conj))
                
    #             print("Conv Accel Squared Comp Conj:        ", conv_accel_squared_comp_conj)
    #             print("Alternative Accel Squared Comp Conj: ", alternative_accel_squared_comp_conj)
    #             print("r_vals[i]: ", r_vals[i])
    #             print("theta_vals[j]: ", theta_vals[j])
    #             print("\n")
    #             counter += 1

    # print("counter: ", counter)
    # if counter == 0:
    #     print("All values are equal between the two acceleration methods out to " + str(decimal_places) + " decimal places")

    # diffs between S values
        # print("\n")
    # r_range = [self.cylinder_radius, 8*self.cylinder_radius, 0.01]
    # r_vals = hlp.list_to_range(r_range)
    # theta_range = [0, 2*np.pi, np.pi/1000]
    # theta_vals = hlp.list_to_range(theta_range)
    # Gamma = 0.376992934959691
    # S = self.numerically_integrate_appellian(Gamma, r_vals, theta_vals)
    # print("Gamma: ", Gamma)
    # print("r_end: ", r_range[1])
    # print("r_step: ", r_range[2])
    # print("theta_step: ", theta_range[2])
    # print("S: ", S)
    # print("\n")
    if cyl.is_calc_theta_convergence:
        r_values = np.array([cyl.cylinder_radius])
        D_values = np.logspace(-4, 0, num=6)  # From 10^-2 (0.01) to 10^0 (1.0) in 6 steps (in a logarithmic scale)
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