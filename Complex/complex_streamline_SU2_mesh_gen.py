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
# from complex_potential_flow_class import potential_flow_object
from Joukowski_Cylinder import cylinder
from vortex_panel import vort_panel
# import tqdm for progress bar
from tqdm import tqdm # type: ignore
import os # type: ignore
from scipy.interpolate import CubicSpline # type: ignore
import matplotlib.patches as patches  # type: ignore

class mesh_generation_object(cylinder, vort_panel):
    """This class uses conformal mapping or NACA airfoil coordinates to generate a mesh for a given domain."""
    def __init__(self, mesh_json_file: str = "complex_streamline_SU2_mesh_gen.json", vort_panel_json_file: str = "vortex_panel_input.json", joukowski_cylinder_json_file: str = "Joukowski_Cylinder.json"):
        """Initialize the mesh generation object."""
        # Initialize the parent classes
        self.mesh_json_file = mesh_json_file
        self.joukowski_cylinder_json_file = joukowski_cylinder_json_file
        self.vort_panel_json_file = vort_panel_json_file
        self.parse_mesh_json(mesh_json_file)
    
    def parse_mesh_json(self, json_file: str):
        """Parse the JSON file and return the parameters."""
        with open(json_file, 'r') as json_handle:
            input = json.load(json_handle)
            self.is_generate_mesh = hlp.parse_dictionary_or_return_default(input, ["mesh", "is_generate_mesh"], True)
            self.is_visualize_mesh = hlp.parse_dictionary_or_return_default(input, ["mesh", "is_visualize_mesh"], True)
            self.is_export_mesh = hlp.parse_dictionary_or_return_default(input, ["mesh", "is_export_mesh"], True)
            self.surface_nodes = hlp.parse_dictionary_or_return_default(input, ["mesh", "surface_nodes"], 100)
            self.wake_nodes = hlp.parse_dictionary_or_return_default(input, ["mesh",  "wake_nodes"], 100)
            self.radial_nodes = hlp.parse_dictionary_or_return_default(input, ["mesh", "radial_nodes"], 100)
            self.radial_distance = hlp.parse_dictionary_or_return_default(input, ["mesh", "radial_distance"], 10)
            self.radial_growth = hlp.parse_dictionary_or_return_default(input, ["mesh", "radial_growth"], 1.0)
            cylinder.__init__(self, self.joukowski_cylinder_json_file) # this initializes the cylinder with the given JSON file
            self.text_file_attribute = "Joukowski_cylinder_zeta0_over_R_is_" + str(self.zeta_center.real) + "+" + str(self.zeta_center.imag) + "i"
            self.type = "cylinder" # default type is cylinder
            self.transformation_type = "joukowski" # default transformation type is Joukowski
            self.use_shape_parameter_D = True
            self.cylinder_radius = hlp.parse_dictionary_or_return_default(input, ["mesh", "cylinder_radius"], 1.0)
            self.D = hlp.parse_dictionary_or_return_default(input, ["mesh", "shape_parameter_D"], 1.0)
            zeta_center = hlp.parse_dictionary_or_return_default(input, ["mesh", "zeta_0"], [-0.09, 0.09])
            self.zeta_center = (zeta_center[0] + 1j*zeta_center[1])/self.cylinder_radius
            angle_of_attack = hlp.parse_dictionary_or_return_default(input, ["operating", "angle_of_attack[deg]"], 5.0)
            self.angle_of_attack = np.deg2rad(angle_of_attack)  # convert to radians
            self.freestream_velocity = hlp.parse_dictionary_or_return_default(input, ["operating", "freestream_velocity"], 1.0)
            self.epsilon = self.calc_epsilon_from_D(self.D)
            self.kutta_circulation = self.calc_circulation_J_airfoil()
            if self.D <= 0.001:
                self.circulation = self.kutta_circulation
            else:
                r_values = np.array([self.cylinder_radius])
                theta_array = np.array([0.0, np.pi/2, np.pi/1000])
                theta_values = hlp.list_to_range(theta_array)
                appellian_root, xi_eta_vals, appelian_value = self.run_appellian_roots(r_values, theta_values, self.D)
                print("appellian circulation: ", appellian_root)
                self.circulation = appellian_root


            plt.xlim(-self.radial_distance-1, self.radial_distance+1)
            plt.ylim(-self.radial_distance-1, self.radial_distance+1)
            plt.xlabel(r"$x/c$")
            plt.ylabel(r"$y/c$")
            plt.gca().set_aspect('equal', adjustable='box')
            # self.plot_geometry()
            # plt.close()

    def calc_vertical_midline_start_points(self):
        self.theta_aft = 0.0  # self.calculate_aft_stagnation_theta_in_Chi_from_Gamma(self.circulation)
        self.theta_forward = np.pi  # self.calculate_forward_stag_theta_in_Chi_from_aft_stag_theta(self.theta_aft)
        self.upper_coords, self.lower_coords = self.given_geom_zeta_calc_top_and_bottom_coords(
            self.theta_aft, self.theta_forward, self.zeta_geom_array
        )
        # Sort the upper and lower coordinates by x-coordinate (left to right)
        upper_sorted = self.upper_coords[np.argsort(self.upper_coords[:, 0])]
        lower_sorted = self.lower_coords[np.argsort(self.lower_coords[:, 0])]
        # Add the node index as the third column
        upper_indices = np.arange(len(upper_sorted)).reshape(-1, 1)  # 0, 1, ..., N_upper-1
        lower_indices = (np.arange(len(lower_sorted)) + len(upper_sorted)).reshape(-1, 1)  # N_upper, N_upper+1, ...
        self.upper_coords = np.hstack((upper_sorted, upper_indices))  # shape (N, 3)
        self.lower_coords = np.hstack((lower_sorted, lower_indices))  # shape (N, 3)

        self.leftmost_point = self.upper_coords[0]  # leftmost point of the entire surface (upper and lower)
        self.rightmost_point = self.lower_coords[-1]  # rightmost point of the entire surface (upper and lower)

        # Map zeta to z, updating only x and y, preserving node index
        for i in range(len(self.upper_coords)):
            zeta_upper = self.upper_coords[i, 0] + 1j * self.upper_coords[i, 1]
            z = self.zeta_to_z(zeta_upper, self.epsilon)
            self.upper_coords[i, 0] = z.real
            self.upper_coords[i, 1] = z.imag
        for i in range(len(self.lower_coords)):
            zeta_lower = self.lower_coords[i, 0] + 1j * self.lower_coords[i, 1]
            z = self.zeta_to_z(zeta_lower, self.epsilon)
            self.lower_coords[i, 0] = z.real
            self.lower_coords[i, 1] = z.imag

        leftmost_upper = np.min(self.upper_coords[:, :2], axis=0)
        rightmost_upper = np.max(self.upper_coords[:, :2], axis=0)
        leftmost_lower = np.min(self.lower_coords[:, :2], axis=0)
        rightmost_lower = np.max(self.lower_coords[:, :2], axis=0)
        # calculate the midline x-coordinate
        midline_x_upper = (leftmost_upper[0] + rightmost_upper[0]) / 2.0
        midline_x_lower = (leftmost_lower[0] + rightmost_lower[0]) / 2.0
        # find the point in the upper_coords that is closest to the upper midline x-coordinate
        closest_upper_points = self.upper_coords
        distances = [abs(z[0] - midline_x_upper) for z in closest_upper_points]
        sorted_indices = np.argsort(distances)
        self.closest_upper_point = closest_upper_points[sorted_indices[0]]
        # find the point in the lower_coords that is closest to the lower midline x-coordinate
        closest_lower_points = self.lower_coords
        distances = [abs(z[0] - midline_x_lower) for z in closest_lower_points]
        sorted_indices = np.argsort(distances)
        self.closest_lower_point = closest_lower_points[sorted_indices[0]]
        if self.is_visualize_mesh:
            plt.scatter(self.upper_coords[:, 0], self.upper_coords[:, 1], color='blue', label='Upper Points', s=0.1)
            plt.scatter(self.lower_coords[:, 0], self.lower_coords[:, 1], color='red', label='Lower Points', s=0.1)
            # for i in range(len(self.upper_coords)):
            #     plt.text(self.upper_coords[i, 0] + 0.05, self.upper_coords[i, 1] - 0.05, str(int(self.upper_coords[i, 2])), fontsize=8, color='black')
            # for j in range(len(self.lower_coords)):
            #     plt.text(self.lower_coords[j, 0] + 0.05, self.lower_coords[j, 1] - 0.05, str(int(self.lower_coords[j, 2])), fontsize=8, color='black')
        self.inner_boundary_mesh = np.concatenate((self.upper_coords, self.lower_coords), axis=0)
        self.last_body_index = self.lower_coords[-1, 2]  # this is the last index of the lower surface
        return self.closest_upper_point, self.closest_lower_point

    def calc_vertical_midlines(self, vertical_midline_start_points):
        """"Calculate the vertical midlines based on the start points."""
        upper_midline_start, lower_midline_start = vertical_midline_start_points
        N = self.radial_nodes + 1 # number of points in the radial direction + 1 because we will remove the very first point in the upper and lower vertical midlines afterwards
        x_upper, y_upper, x_lower, y_lower = upper_midline_start[0], upper_midline_start[1], lower_midline_start[0], lower_midline_start[1]
        deltaD_0_upper = self.calc_deltaD_0(N, self.radial_growth, (x_upper, y_upper), (x_upper - self.radial_distance*np.sin(self.angle_of_attack), y_upper + self.radial_distance*np.cos(self.angle_of_attack)))
        deltaD_0_lower = self.calc_deltaD_0(N, self.radial_growth, (x_lower, y_lower), (x_lower + self.radial_distance*np.sin(self.angle_of_attack), y_lower - self.radial_distance*np.cos(self.angle_of_attack)))
        vertical_midline_upper = np.zeros((N, 3))  # the third column is for the node index
        vertical_midline_lower = np.zeros((N, 3))  # the third column is for the node index
        self.outer_boundary_mesh = []
        start_index = self.last_body_index + 1  # start index for the vertical midlines
        if self.radial_growth == 1.0:
            for i in range(N):
                vertical_midline_upper[i, 0] = x_upper - i * deltaD_0_upper * np.sin(self.angle_of_attack)
                vertical_midline_upper[i, 1] = y_upper + i * deltaD_0_upper * np.cos(self.angle_of_attack)
                vertical_midline_upper[i, 2] = i + start_index
                vertical_midline_lower[i, 0] = x_lower + i * deltaD_0_lower * np.sin(self.angle_of_attack)
                vertical_midline_lower[i, 1] = y_lower - i * deltaD_0_lower * np.cos(self.angle_of_attack)
                vertical_midline_lower[i, 2] = i + start_index + self.radial_nodes # Assign node index
        else:
            vertical_midline_upper[0, 0] = x_upper
            vertical_midline_upper[0, 1] = y_upper
            vertical_midline_upper[0, 2] = 0  # Assign node index
            vertical_midline_lower[0, 0] = x_lower
            vertical_midline_lower[0, 1] = y_lower
            vertical_midline_lower[0, 2] = 0  # Assign node index
            for i in range(1, N):
                vertical_midline_upper[i, 0] = vertical_midline_upper[i-1, 0] - deltaD_0_upper * np.sin(self.angle_of_attack) * self.radial_growth**(i-1)
                vertical_midline_upper[i, 1] = vertical_midline_upper[i-1, 1] + deltaD_0_upper * self.radial_growth**(i-1) * np.cos(self.angle_of_attack)
                vertical_midline_upper[i, 2] = i + start_index # Assign node index
                vertical_midline_lower[i, 0] = vertical_midline_lower[i-1, 0] + deltaD_0_lower * np.sin(self.angle_of_attack) * self.radial_growth**(i-1)
                vertical_midline_lower[i, 1] = vertical_midline_lower[i-1, 1] - deltaD_0_lower * self.radial_growth**(i-1) * np.cos(self.angle_of_attack)
                vertical_midline_lower[i, 2] = i + start_index + self.radial_nodes  # Assign node index
        # remove the first point in the upper and lower vertical midlines
        vertical_midline_upper = vertical_midline_upper[1:]  # shape (N-1, 3), excluding the first point which is the trailing edge
        vertical_midline_lower = vertical_midline_lower[1:]  # shape (N-1, 3), excluding the first point which is the trailing edge
        # subtract each index by 1 to make it start from 0
        vertical_midline_upper[:, 2] -= 1  # Adjust indices to start from 0
        vertical_midline_lower[:, 2] -= 1  # Adjust indices to start from 0
        # get the last value of the 3 column and save it
        self.last_index_vertical_midline_upper = vertical_midline_upper[-1, 2]  # this is the last index of the vertical midline upper
        self.last_index_vertical_midline_lower = vertical_midline_lower[-1, 2]  # this is the last index of the vertical midline lower
        # plot the vertical midlines
        # if self.is_visualize_mesh:
            # plt.scatter(vertical_midline_upper[:, 0], vertical_midline_upper[:, 1], color='green', label='Upper Midline', s=0.1)
            # plt.scatter(vertical_midline_lower[:, 0], vertical_midline_lower[:, 1], color='orange', label='Lower Midline', s=0.1)
            # for i in range(len(vertical_midline_upper)):
            #     plt.text(vertical_midline_upper[i, 0] + 0.05, vertical_midline_upper[i, 1] - 0.05, str(int(vertical_midline_upper[i, 2])), fontsize=8, color='black')
            # for j in range(len(vertical_midline_lower)):
            #     plt.text(vertical_midline_lower[j, 0] + 0.05, vertical_midline_lower[j, 1] - 0.05, str(int(vertical_midline_lower[j, 2])), fontsize=8, color='black')
        # concatenate the last point of the upper and lower midline to the outer boundary mesh
        self.outer_boundary_mesh.append(vertical_midline_upper[-1])  # append the last point of the upper midline
        self.outer_boundary_mesh.append(vertical_midline_lower[-1])  # append the last point of the lower midline
        # save the last index of the lower midline 
        self.last_index_lower_midline = vertical_midline_lower[-1, 2]  # this is the last index of the lower midline
        self.vertical_midline_upper, self.vertical_midline_lower = vertical_midline_upper, vertical_midline_lower
        return vertical_midline_upper, vertical_midline_lower
    
    def generate_spline_streamline(self, start_point, direction, target_point, N_nodes):
        """RK4 + spline + growth sampled streamline."""
        raw_streamline = self.calc_single_streamline(np.array(start_point), direction)[1:]
        sorted_indices = np.argsort(raw_streamline[:, 0])
        x_sorted, y_sorted = raw_streamline[sorted_indices, 0], raw_streamline[sorted_indices, 1]
        spline_func = CubicSpline(x_sorted, y_sorted)
        deltaD_0 = self.calc_deltaD_0(N_nodes, self.radial_growth, (x_sorted[-1 if direction == -1 else 0], y_sorted[-1 if direction == -1 else 0]), target_point)
        return self.generate_growth_sampled_spline(
            spline_func,
            (x_sorted[-1 if direction == -1 else 0], y_sorted[-1 if direction == -1 else 0]),
            N_nodes,
            deltaD_0,
            direction
        )
    
    def generate_growth_sampled_spline(self, spline_func, start_point, N, deltaD_0, direction_sign):
        """Return a streamline of shape (N, 2) sampled using geometric or uniform spacing."""
        result = np.zeros((N, 2))
        result[0] = start_point
        for i in range(1, N):
            dx = deltaD_0 * (self.radial_growth ** (i - 1)) if self.radial_growth != 1.0 else deltaD_0 * i
            dx *= direction_sign
            x = result[i - 1, 0] + dx * np.cos(self.angle_of_attack)
            y = spline_func(x)
            result[i] = [x, y]
        return result
    
        # calc streamlines extending from vertical midline points outwards the radial distance
    def calc_midline_streamlines(self, vertical_midline_upper, vertical_midline_lower):
        """calculate the streamlines extending from the vertical midline points outwards the radial distance."""
        midline_streamlines = []
        if self.angle_of_attack > np.deg2rad(5.0):
            raise ValueError("Angle of attack is too high for this method. Please use a lower angle of attack (<= 5 degrees).")
        #### LE streamline ###
        x_le, y_le = self.upper_coords[0, 0], self.upper_coords[0, 1]
        dist_to_lower = abs(self.closest_lower_point[0] - x_le)
        mesh.plot_x_lower_lim = x_le + dist_to_lower - self.radial_distance
        target_le = (x_le - self.radial_distance * np.cos(self.angle_of_attack), y_le - self.radial_distance * np.sin(self.angle_of_attack))
        streamline_leading_edge = self.generate_spline_streamline((x_le, y_le), -1, target_le, self.radial_nodes)
        normals_streamline_leading_edge = self.compute_normals(streamline_leading_edge)
        midline_streamlines.append(streamline_leading_edge)
        #### TE streamline ###
        x_te, y_te = self.lower_coords[-1, 0], self.lower_coords[-1, 1]
        dist_to_lower = abs(self.closest_lower_point[0] - x_te)
        mesh.plot_x_upper_lim = x_te - dist_to_lower + self.radial_distance
        target_te = (x_te + self.radial_distance * np.cos(self.angle_of_attack), y_te + self.radial_distance * np.sin(self.angle_of_attack))
        streamline_trailing_edge = self.generate_spline_streamline((x_te, y_te), 1, target_te, self.radial_nodes)
        normals_trailing_edge = self.compute_normals(streamline_trailing_edge)
        midline_streamlines.append(streamline_trailing_edge)
        #### Upper midline streamline ###
        x_u, y_u = vertical_midline_upper[-1, 0], vertical_midline_upper[-1, 1]
        mesh.plot_x_upper_lim = x_u + self.radial_distance * np.cos(self.angle_of_attack)
        mesh.plot_x_lower_lim = x_u - self.radial_distance * np.cos(self.angle_of_attack)
        upper_streamline = np.concatenate([self.calc_single_streamline(np.array([x_u, y_u]), -1)[1:], self.calc_single_streamline(np.array([x_u, y_u]), 1)[1:]])
        x_sorted = upper_streamline[np.argsort(upper_streamline[:, 0])]
        spline_func_upper = CubicSpline(x_sorted[:, 0], x_sorted[:, 1])
        N_top = len(streamline_leading_edge) + len(self.upper_coords) + 1 + len(streamline_trailing_edge)
        x_coords = np.linspace(x_sorted[0, 0], x_sorted[-1, 0], N_top)
        streamline_upper_total = np.column_stack([x_coords, spline_func_upper(x_coords)])
        normals_streamline_upper_total = self.compute_normals(streamline_upper_total)
        midline_streamlines.append(streamline_upper_total)
        #### Lower midline streamline ###
        x_l, y_l = vertical_midline_lower[-1, 0], vertical_midline_lower[-1, 1]
        mesh.plot_x_upper_lim = x_l + self.radial_distance * np.cos(self.angle_of_attack)
        mesh.plot_x_lower_lim = x_l - self.radial_distance * np.cos(self.angle_of_attack)
        lower_streamline = np.concatenate([self.calc_single_streamline(np.array([x_l, y_l]), -1)[1:], self.calc_single_streamline(np.array([x_l, y_l]), 1)[1:]])
        x_sorted = lower_streamline[np.argsort(lower_streamline[:, 0])]
        spline_func_lower = CubicSpline(x_sorted[:, 0], x_sorted[:, 1])
        N_bottom = len(streamline_leading_edge) + len(self.lower_coords) + 1 + len(streamline_trailing_edge)
        x_coords = np.linspace(x_sorted[0, 0], x_sorted[-1, 0], N_bottom)
        streamline_lower_total = np.column_stack([x_coords, spline_func_lower(x_coords)])
        normals_streamline_lower_total = self.compute_normals(streamline_lower_total)
        midline_streamlines.append(streamline_lower_total)
        # Now we can fill in the rest of the midline streamlines
        starting_index = self.last_index_lower_midline + 1  # start index for the midline streamlines
        first_midline = np.zeros(len(streamline_leading_edge))
        second_midline = np.zeros(len(self.upper_coords))
        third_midline = np.zeros(1) 
        fourth_midline = np.zeros(len(streamline_trailing_edge))
        
        for i in range(len(vertical_midline_upper)-1):
            x_upper, y_upper = vertical_midline_upper[i, 0], vertical_midline_upper[i, 1]

            # Set RK4 integration limits
            mesh.plot_x_upper_lim = x_upper + self.radial_distance
            mesh.plot_x_lower_lim = x_upper - self.radial_distance

            # Generate streamline (forward + backward)
            streamline_forward = self.calc_single_streamline(np.array([x_upper, y_upper]), 1)[1:]
            streamline_backward = self.calc_single_streamline(np.array([x_upper, y_upper]), -1)[::-1]
            streamline_total = np.concatenate((streamline_backward, streamline_forward), axis=0)

            # Sort points by x-coordinate
            sorted_indices = np.argsort(streamline_total[:, 0])
            x_sorted = streamline_total[sorted_indices, 0]
            y_sorted = streamline_total[sorted_indices, 1]

            if np.any(np.diff(x_sorted) == 0):
                print(f"Skipping streamline {i} due to repeated x values.")
                continue

            # Create spline
            spline_func = CubicSpline(x_sorted, y_sorted)

            # --------- LEADING EDGE BLEND ---------
            x_coords_leading_edge = streamline_leading_edge[:, 0]
            spline_y_le = spline_func(x_coords_leading_edge)

            # --------- UPPER SURFACE BLEND ---------
            x_coords_upper = self.upper_coords[:, 0]
            spline_y_upper = spline_func(x_coords_upper)

            # --------- SINGLE POINT BETWEEN UPPER AND TE ---------
            x_coords_upper_last = np.array([self.lower_coords[-1, 0]])
            spline_y_upper_last = spline_func(x_coords_upper_last)

            # --------- TRAILING EDGE BLEND ---------
            x_coords_trailing_edge = streamline_trailing_edge[:, 0]
            spline_y_trailing = spline_func(x_coords_trailing_edge)

            # --------- STACK RESULTS ---------
            streamline_data = [
                [x_coords_leading_edge, spline_y_le],
                [x_coords_upper, spline_y_upper],
                [x_coords_upper_last, spline_y_upper_last],
                [x_coords_trailing_edge, spline_y_trailing]
            ]
            midline_streamlines.append(streamline_data)

        # do the same for the lower vertical midline points
        for i in range(len(vertical_midline_lower)-1):
            x_lower, y_lower = vertical_midline_lower[i, 0], vertical_midline_lower[i, 1]

            # Set RK4 integration limits
            mesh.plot_x_upper_lim = x_lower + self.radial_distance
            mesh.plot_x_lower_lim = x_lower - self.radial_distance

            # Generate streamline (forward + backward)
            streamline_forward = self.calc_single_streamline(np.array([x_lower, y_lower]), 1)[1:]
            streamline_backward = self.calc_single_streamline(np.array([x_lower, y_lower]), -1)[::-1]
            streamline_total = np.concatenate((streamline_backward, streamline_forward), axis=0)

            # Sort points by x-coordinate
            sorted_indices = np.argsort(streamline_total[:, 0])
            x_sorted = streamline_total[sorted_indices, 0]
            y_sorted = streamline_total[sorted_indices, 1]

            # Skip if duplicate x values cause spline to fail
            if np.any(np.diff(x_sorted) == 0):
                print(f"Skipping streamline {i} due to repeated x values.")
                continue

            # Create spline
            spline_func = CubicSpline(x_sorted, y_sorted)

            # Define x-values to evaluate the spline at
            x_coords_leading_edge = streamline_leading_edge[:, 0]
            x_coords_lower = self.lower_coords[:, 0]
            x_coords_lower_last = np.array([self.upper_coords[0, 0]])
            x_coords_trailing_edge = streamline_trailing_edge[:, 0]

            # Evaluate spline at those x-values
            spline_y_values_leading_edge = spline_func(x_coords_leading_edge)
            spline_y_values_lower = spline_func(x_coords_lower)
            spline_y_values_lower_last = spline_func(x_coords_lower_last)
            spline_y_values_trailing_edge = spline_func(x_coords_trailing_edge)

            # Append [x, y] pairs to midline_streamlines
            streamline_data = [
                [x_coords_leading_edge, spline_y_values_leading_edge],
                [x_coords_lower, spline_y_values_lower],
                [x_coords_lower_last, spline_y_values_lower_last],
                [x_coords_trailing_edge, spline_y_values_trailing_edge]
            ]
            midline_streamlines.append(streamline_data)

        midline_streamlines = np.array(midline_streamlines, dtype=object)  # Convert to numpy array with object type for mixed data    
        if self.is_visualize_mesh:
            for streamline in midline_streamlines:
                for x, y in streamline:
                    plt.scatter(x, y, color='purple', label='Midline Streamline', s=0.1)
                # for j in range(len(streamline)):
                #     plt.text(streamline[j, 0] + 0.05, streamline[j, 1] - 0.05, str(int(streamline[j, 2])), fontsize=8, color='black')

    #         # calc streamlines extending from vertical midline points outwards the radial distance
    # def calc_midline_streamlines(self, vertical_midline_upper, vertical_midline_lower):
    #     """calculate the streamlines extending from the vertical midline points outwards the radial distance."""
    #     midline_streamlines = []
    #     if self.angle_of_attack > np.deg2rad(5.0):
    #         raise ValueError("Angle of attack is too high for this method. Please use a lower angle of attack (<= 5 degrees).")
    #     #### LE streamline ###
    #     x_le, y_le = self.upper_coords[0, 0], self.upper_coords[0, 1]
    #     dist_to_lower = abs(self.closest_lower_point[0] - x_le)
    #     mesh.plot_x_lower_lim = x_le + dist_to_lower - self.radial_distance
    #     target_le = (x_le - self.radial_distance * np.cos(self.angle_of_attack), y_le - self.radial_distance * np.sin(self.angle_of_attack))
    #     streamline_leading_edge = self.generate_spline_streamline((x_le, y_le), -1, target_le, self.radial_nodes)
    #     normals_streamline_leading_edge = self.compute_normals(streamline_leading_edge)
    #     midline_streamlines.append(streamline_leading_edge)
    #     #### TE streamline ###
    #     x_te, y_te = self.lower_coords[-1, 0], self.lower_coords[-1, 1]
    #     dist_to_lower = abs(self.closest_lower_point[0] - x_te)
    #     mesh.plot_x_upper_lim = x_te - dist_to_lower + self.radial_distance
    #     target_te = (x_te + self.radial_distance * np.cos(self.angle_of_attack), y_te + self.radial_distance * np.sin(self.angle_of_attack))
    #     streamline_trailing_edge = self.generate_spline_streamline((x_te, y_te), 1, target_te, self.radial_nodes)
    #     normals_trailing_edge = self.compute_normals(streamline_trailing_edge)
    #     midline_streamlines.append(streamline_trailing_edge)
    #     #### Upper midline streamline ###
    #     x_u, y_u = vertical_midline_upper[-1, 0], vertical_midline_upper[-1, 1]
    #     mesh.plot_x_upper_lim = x_u + self.radial_distance * np.cos(self.angle_of_attack)
    #     mesh.plot_x_lower_lim = x_u - self.radial_distance * np.cos(self.angle_of_attack)
    #     upper_streamline = np.concatenate([self.calc_single_streamline(np.array([x_u, y_u]), -1)[1:], self.calc_single_streamline(np.array([x_u, y_u]), 1)[1:]])
    #     x_sorted = upper_streamline[np.argsort(upper_streamline[:, 0])]
    #     spline_func_upper = CubicSpline(x_sorted[:, 0], x_sorted[:, 1])
    #     N_top = len(streamline_leading_edge) + len(self.upper_coords) + 1 + len(streamline_trailing_edge)
    #     x_coords = np.linspace(x_sorted[0, 0], x_sorted[-1, 0], N_top)
    #     streamline_upper_total = np.column_stack([x_coords, spline_func_upper(x_coords)])
    #     normals_streamline_upper_total = self.compute_normals(streamline_upper_total)
    #     midline_streamlines.append(streamline_upper_total)
    #     #### Lower midline streamline ###
    #     x_l, y_l = vertical_midline_lower[-1, 0], vertical_midline_lower[-1, 1]
    #     mesh.plot_x_upper_lim = x_l + self.radial_distance * np.cos(self.angle_of_attack)
    #     mesh.plot_x_lower_lim = x_l - self.radial_distance * np.cos(self.angle_of_attack)
    #     lower_streamline = np.concatenate([self.calc_single_streamline(np.array([x_l, y_l]), -1)[1:], self.calc_single_streamline(np.array([x_l, y_l]), 1)[1:]])
    #     x_sorted = lower_streamline[np.argsort(lower_streamline[:, 0])]
    #     spline_func_lower = CubicSpline(x_sorted[:, 0], x_sorted[:, 1])
    #     N_bottom = len(streamline_leading_edge) + len(self.lower_coords) + 1 + len(streamline_trailing_edge)
    #     x_coords = np.linspace(x_sorted[0, 0], x_sorted[-1, 0], N_bottom)
    #     streamline_lower_total = np.column_stack([x_coords, spline_func_lower(x_coords)])
    #     normals_streamline_lower_total = self.compute_normals(streamline_lower_total)
    #     midline_streamlines.append(streamline_lower_total)
    #     # Now we can fill in the rest of the midline streamlines
    #     starting_index = self.last_index_lower_midline + 1  # start index for the midline streamlines
    #     first_midline = np.zeros(len(streamline_leading_edge))
    #     second_midline = np.zeros(len(self.upper_coords))
    #     third_midline = np.zeros(1) 
    #     fourth_midline = np.zeros(len(streamline_trailing_edge))
        
    #     for i in range(len(vertical_midline_upper)-1):
    #         x_upper, y_upper = vertical_midline_upper[i, 0], vertical_midline_upper[i, 1]

    #         # Set RK4 integration limits
    #         mesh.plot_x_upper_lim = x_upper + self.radial_distance
    #         mesh.plot_x_lower_lim = x_upper - self.radial_distance

    #         # Generate streamline (forward + backward)
    #         streamline_forward = self.calc_single_streamline(np.array([x_upper, y_upper]), 1)[1:]
    #         streamline_backward = self.calc_single_streamline(np.array([x_upper, y_upper]), -1)[::-1]
    #         streamline_total = np.concatenate((streamline_backward, streamline_forward), axis=0)

    #         # Sort points by x-coordinate
    #         sorted_indices = np.argsort(streamline_total[:, 0])
    #         x_sorted = streamline_total[sorted_indices, 0]
    #         y_sorted = streamline_total[sorted_indices, 1]

    #         if np.any(np.diff(x_sorted) == 0):
    #             print(f"Skipping streamline {i} due to repeated x values.")
    #             continue

    #         # Create spline
    #         spline_func = CubicSpline(x_sorted, y_sorted)

    #         # --------- LEADING EDGE BLEND ---------
    #         x_coords_leading_edge = streamline_leading_edge[:, 0]
    #         spline_y_le = spline_func(x_coords_leading_edge)
    #         pA_le = np.column_stack([x_coords_leading_edge, spline_y_le])
    #         pB_le = np.column_stack([first_midline, spline_y_le])
    #         pX_le = streamline_leading_edge
    #         x_coords_leading_edge_blended = x_coords_leading_edge#self.blend_normals(pA_le, pB_le, pX_le)

    #         # --------- UPPER SURFACE BLEND ---------
    #         x_coords_upper = self.upper_coords[:, 0]
    #         spline_y_upper = spline_func(x_coords_upper)
    #         pA_upper = np.column_stack([x_coords_upper, spline_y_upper])
    #         pB_upper = np.column_stack([second_midline, spline_y_upper])
    #         # Use the portion of streamline_upper_total not used by LE and TE
    #         offset = len(x_coords_leading_edge)
    #         pX_upper = streamline_upper_total[offset:offset + len(x_coords_upper)]
    #         x_coords_upper_blended = x_coords_upper#self.blend_normals(pA_upper, pB_upper, pX_upper)

    #         # --------- SINGLE POINT BETWEEN UPPER AND TE ---------
    #         x_coords_upper_last = np.array([self.lower_coords[-1, 0]])
    #         spline_y_upper_last = spline_func(x_coords_upper_last)
    #         pA_last = np.array([[x_coords_upper_last[0], spline_y_upper_last[0]]])
    #         pB_last = np.array([[third_midline[0], spline_y_upper_last[0]]])
    #         pX_last = np.array([streamline_trailing_edge[0]])
    #         x_coords_upper_last_blended = x_coords_upper_last#self.blend_normals(pA_last, pB_last, pX_last)

    #         # --------- TRAILING EDGE BLEND ---------
    #         x_coords_trailing_edge = streamline_trailing_edge[:, 0]
    #         spline_y_trailing = spline_func(x_coords_trailing_edge)
    #         pA_te = np.column_stack([x_coords_trailing_edge, spline_y_trailing])
    #         pB_te = np.column_stack([fourth_midline, spline_y_trailing])
    #         pX_te = streamline_trailing_edge[:, 0]
    #         x_coords_trailing_edge_blended = pX_te#self.blend_normals(pA_te, pB_te, pX_te)

    #         # --------- STACK RESULTS ---------
    #         streamline_data = [
    #             [x_coords_leading_edge_blended, spline_y_le],
    #             [x_coords_upper_blended, spline_y_upper],
    #             [x_coords_upper_last_blended, spline_y_upper_last],
    #             [x_coords_trailing_edge_blended, spline_y_trailing]
    #         ]
    #         midline_streamlines.append(streamline_data)

    #     # do the same for the lower vertical midline points
    #     for i in range(len(vertical_midline_lower)-1):
    #         x_lower, y_lower = vertical_midline_lower[i, 0], vertical_midline_lower[i, 1]

    #         # Set RK4 integration limits
    #         mesh.plot_x_upper_lim = x_lower + self.radial_distance
    #         mesh.plot_x_lower_lim = x_lower - self.radial_distance

    #         # Generate streamline (forward + backward)
    #         streamline_forward = self.calc_single_streamline(np.array([x_lower, y_lower]), 1)[1:]
    #         streamline_backward = self.calc_single_streamline(np.array([x_lower, y_lower]), -1)[::-1]
    #         streamline_total = np.concatenate((streamline_backward, streamline_forward), axis=0)

    #         # Sort points by x-coordinate
    #         sorted_indices = np.argsort(streamline_total[:, 0])
    #         x_sorted = streamline_total[sorted_indices, 0]
    #         y_sorted = streamline_total[sorted_indices, 1]

    #         # Skip if duplicate x values cause spline to fail
    #         if np.any(np.diff(x_sorted) == 0):
    #             print(f"Skipping streamline {i} due to repeated x values.")
    #             continue

    #         # Create spline
    #         spline_func = CubicSpline(x_sorted, y_sorted)

    #         # Define x-values to evaluate the spline at
    #         x_coords_leading_edge = streamline_leading_edge[:, 0]
    #         x_coords_lower = self.lower_coords[:, 0]
    #         x_coords_lower_last = np.array([self.upper_coords[0, 0]])
    #         x_coords_trailing_edge = streamline_trailing_edge[:, 0]

    #         # Evaluate spline at those x-values
    #         spline_y_values_leading_edge = spline_func(x_coords_leading_edge)
    #         spline_y_values_lower = spline_func(x_coords_lower)
    #         spline_y_values_lower_last = spline_func(x_coords_lower_last)
    #         spline_y_values_trailing_edge = spline_func(x_coords_trailing_edge)

    #         # Append [x, y] pairs to midline_streamlines
    #         streamline_data = [
    #             [x_coords_leading_edge, spline_y_values_leading_edge],
    #             [x_coords_lower, spline_y_values_lower],
    #             [x_coords_lower_last, spline_y_values_lower_last],
    #             [x_coords_trailing_edge, spline_y_values_trailing_edge]
    #         ]
    #         midline_streamlines.append(streamline_data)

    #     midline_streamlines = np.array(midline_streamlines, dtype=object)  # Convert to numpy array with object type for mixed data    
    #     if self.is_visualize_mesh:
    #         for streamline in midline_streamlines:
    #             for x, y in streamline:
    #                 plt.scatter(x, y, color='purple', label='Midline Streamline', s=0.1)
    #             # for j in range(len(streamline)):
    #             #     plt.text(streamline[j, 0] + 0.05, streamline[j, 1] - 0.05, str(int(streamline[j, 2])), fontsize=8, color='black')

    def compute_normals(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute unit normal vectors at each point in a 2D curve.
        
        Parameters:
            coords (np.ndarray): Array of shape (N, 2), representing a 2D path.

        Returns:
            normals (np.ndarray): Array of shape (N, 2), the unit normals.
        """
        tangents = np.gradient(coords, axis=0)
        magnitudes = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents_unit = tangents / magnitudes

        # Rotate tangent vectors 90° counterclockwise to get normals
        normals = np.empty_like(tangents_unit)
        normals[:, 0] = -tangents_unit[:, 1]
        normals[:, 1] = tangents_unit[:, 0]
        return normals
    
    def blend_normals(self, pA, pB, pX):
        """
        Blend the normal vectors between arrays of points pA and pB to estimate
        the normals at points pX, using computed unit normals from pA and pB.

        Parameters:
            pA (np.ndarray): Array of N points A (shape (N, 2))
            pB (np.ndarray): Array of N points B (shape (N, 2))
            pX (np.ndarray): Array of N target points X (shape (N, 2))

        Returns:
            nX (np.ndarray): Array of blended unit normals at pX (shape (N, 2))
            x_corrected (np.ndarray): Array of adjusted x-coordinates (shape (N,))
        """
        nX = 0
        return nX, pX
        assert pA.shape == pB.shape == pX.shape, "All inputs must have shape (N, 2)"

        # Stack for normal computation: shape (N, 2, 2) → reshape to (2N, 2)
        stacked = np.vstack((pA, pB))
        normals = self.compute_normals(stacked)  # shape (2N, 2)
        nA = normals[:len(pA)]
        nB = normals[len(pA):]

        # Compute weights
        d_total = np.linalg.norm(pB - pA, axis=1)
        d_to_B = np.linalg.norm(pB - pX, axis=1)
        weight_A = d_to_B / d_total
        weight_B = 1.0 - weight_A
        weight_A = weight_A[:, np.newaxis]
        weight_B = weight_B[:, np.newaxis]

        # Blend and normalize normals
        nX_raw = weight_A * nA + weight_B * nB
        nX = nX_raw / np.linalg.norm(nX_raw, axis=1, keepdims=True)

        # Adjust x-location of pX
        dot_proj = np.sum(pX * nX, axis=1)
        x_corrected = pX[:, 0] + nX[:, 0] * dot_proj

        return nX, x_corrected


    def calc_unshifted_limits(self):
        """"""
        mesh.plot_x_lower_lim, mesh.plot_x_upper_lim, mesh.plot_y_lower_lim, mesh.plot_y_upper_lim = -mesh.radial_distance+1, mesh.radial_distance+1, -mesh.radial_distance, mesh.radial_distance
        mesh.plot_x_lower_lim, mesh.plot_x_upper_lim = mesh.plot_x_lower_lim + mesh.zeta_center.real + mesh.leading_edge, mesh.plot_x_upper_lim + mesh.zeta_center.real + mesh.leading_edge
        mesh.plot_y_lower_lim, mesh.plot_y_upper_lim = mesh.plot_y_lower_lim + mesh.zeta_center.imag, mesh.plot_y_upper_lim + mesh.zeta_center.imag
        # # now divide the x and y limits by the trailing edge
        mesh.plot_x_lower_lim, mesh.plot_x_upper_lim, mesh.plot_y_lower_lim, mesh.plot_y_upper_lim = mesh.plot_x_lower_lim * mesh.trailing_edge, mesh.plot_x_upper_lim * mesh.trailing_edge, mesh.plot_y_lower_lim * mesh.trailing_edge, mesh.plot_y_upper_lim * mesh.trailing_edge
        mesh.plot_x_start = mesh.plot_x_lower_lim
        # re-define radial_distance to be plot_y_upper_lim 
        mesh.radial_distance = mesh.plot_y_upper_lim
        plt.xlim(-self.radial_distance-5, self.radial_distance+5)
        plt.ylim(-self.radial_distance-5, self.radial_distance+5)
    
    def calc_deltaD_0(self, N, radial_growth, start_xy, end_xy):
        """
        Calculate the initial radial spacing (deltaD_0) such that N points,
        growing by a factor of `radial_growth`, reach from start_xy to end_xy.
        """
        x_start, y_start = start_xy
        x_end, y_end = end_xy
        total_distance = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
        
        if radial_growth == 1.0:
            return total_distance / (N - 1)
        else:
            sum_geom = (1 - radial_growth ** (N - 1)) / (1 - radial_growth)
            if np.isclose(total_distance, 0.0):
                print("Distance is zero, returning 0.0 for deltaD_0.")
            return total_distance / sum_geom
    
    def plot_mesh(self):
        """"""
        # plot the unshifted mesh
        mesh.plot_geometry()
        # plot the midline
        mesh.plot(mesh.unshifted_streamlines)
        plt.show()
        plt.close()

if __name__ == "__main__":

    # initialize the cylinder object
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
    mesh = mesh_generation_object()
    mesh.get_full_geometry_zeta(mesh.surface_nodes)
    mesh.get_full_geometry()
    mesh.shift_joukowski_cylinder()
    print("mesh_trailing_edge: ", mesh.trailing_edge)
    mesh.calc_unshifted_limits()
    midline_start_points = mesh.calc_vertical_midline_start_points()
    vertical_midline_upper, vertical_midline_lower = mesh.calc_vertical_midlines(midline_start_points)
    midline_streamlines = mesh.calc_midline_streamlines(vertical_midline_upper, vertical_midline_lower)
    # mesh.plot_mesh()


    # mesh.surface_mesh_points = mesh.shifted_z_surface
    # mesh.shifted_streamlines = mesh.calc_shifted_streamlines(mesh.unshifted_streamlines)
    # mesh.plot_shifted_joukowski_cylinder()
    # mesh.plot(mesh.shifted_streamlines)
    # mesh.plot_x_lower_lim, mesh.plot_x_upper_lim = mesh.plot_x_lower_lim - mesh.zeta_center.real - mesh.leading_edge, mesh.plot_x_upper_lim - mesh.zeta_center.real - mesh.leading_edge
    # mesh.plot_y_lower_lim, mesh.plot_y_upper_lim = mesh.plot_y_lower_lim - mesh.zeta_center.imag, mesh.plot_y_upper_lim - mesh.zeta_center.imag
    # # # now divide the x and y limits by the trailing edge
    # mesh.plot_x_lower_lim, mesh.plot_x_upper_lim, mesh.plot_y_lower_lim, mesh.plot_y_upper_lim = mesh.plot_x_lower_lim / mesh.trailing_edge, mesh.plot_x_upper_lim / mesh.trailing_edge, mesh.plot_y_lower_lim / mesh.trailing_edge, mesh.plot_y_upper_lim / mesh.trailing_edge
    # plt.xlim(mesh.plot_x_lower_lim, mesh.plot_x_upper_lim)
    # plt.ylim(mesh.plot_y_lower_lim, mesh.plot_y_upper_lim)
    plt.show()
    # plt.close()



    