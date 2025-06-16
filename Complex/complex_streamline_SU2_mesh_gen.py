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
            self.is_visualize_node_integers = hlp.parse_dictionary_or_return_default(input, ["mesh", "is_visualize_node_integers"], False)
            self.is_export_mesh = hlp.parse_dictionary_or_return_default(input, ["mesh", "is_export_mesh"], True)
            self.surface_nodes = hlp.parse_dictionary_or_return_default(input, ["mesh", "surface_nodes"], 100)
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

    def calc_deltaD_0(self, N, radial_growth, start_xy=np.array([0.0,0.0]), end_xy=np.array([1.0,1.0]), total_distance= None,):
        """
        Calculate the initial radial spacing (deltaD_0) such that N points,
        growing by a factor of `radial_growth`, reach from start_xy to end_xy.
        """
        if total_distance is None:
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

    def evaluate_single_stream_or_potential_line(self, start_point: np.array, direction: int, func=None):
        """Calculate a streamline or potential line until the desired arc length is reached."""
        if func is None:
            func = self.unit_velocity
        # Ensure valid start_point
        if not isinstance(start_point, np.ndarray) or start_point.shape != (2,) or not np.issubdtype(start_point.dtype, np.floating):
            raise ValueError("The start_point must be a NumPy array of two float values representing x and y coordinates.")
        point = np.array(start_point)
        streamline_points = [point]
        i = 0
        max_iter = 10000
        total_distance = self.radial_distance
        while True:
            # Take RK4 step
            point_new = hlp.rk4(point, direction, self.plot_delta_s, self.surface_normal, func)
            # Stop if new point is outside x bounds
            if point_new[0] < self.plot_x_lower_lim or point_new[0] > self.plot_x_upper_lim:
                break
            streamline_points.append(point_new)
            # Compute total arc length covered so far
            arc_length_covered = hlp.compute_arc_length(streamline_points)
            # Check termination condition
            if arc_length_covered >= total_distance or i > max_iter:
                break
            point = point_new
            i += 1
        return np.array(streamline_points)
    
    def calc_wrapped_stream_or_potential_line(self, start_point: np.array, direction: int, psi_targets=None, func=None, radial_growth=None):
        """
        Calculates a streamline or potential line radiating from a surface point.
        
        If psi_targets is provided, it steps to match those stream function values.
        Otherwise, it uses geometric spacing defined by self.radial_nodes and radial_growth.
        """
        # Default settings
        if func is None:
            func = self.unit_velocity  # Streamline
            desc = "Calculating Streamline"
        else:
            desc = "Calculating Potential Line"

        if radial_growth is None:
            radial_growth = self.radial_growth

        if not isinstance(start_point, np.ndarray) or start_point.shape != (2,) or not np.issubdtype(start_point.dtype, np.floating):
            raise ValueError("start_point must be a NumPy array of two float values.")

        point = np.array(start_point)
        points = [point]

        if psi_targets is not None:
            # Step outward to match given stream function values
            for target_psi in psi_targets:
                ds = self.find_correct_ds_for_psi(100.0, point, direction, target_psi, func=func)
                point_new = hlp.rk4(point, direction, ds, self.surface_normal, func)
                points.append(point_new)
                point = point_new
        else:
            # Step outward based on geometric spacing
            initial_line = self.evaluate_single_stream_or_potential_line(point, direction, func=func)
            total_distance = hlp.compute_arc_length(initial_line)

            deltaD_0 = self.calc_deltaD_0(self.radial_nodes, radial_growth, total_distance=total_distance)
            deltas = deltaD_0 * radial_growth ** np.arange(self.radial_nodes)

            for delta in deltas:
                ds = self.find_correct_ds(100.0, point, direction, delta, func=func)
                point_new = hlp.rk4(point, direction, ds, self.surface_normal, func)
                points.append(point_new)
                point = point_new

        return np.array(points)

    
    def bisection_function(self, ds, point, direction, delta_position, func=None):
        if func is None:
            func = self.unit_velocity
        new_point = hlp.rk4(point, direction, ds, self.surface_normal, func)
        return np.linalg.norm(new_point - point) - delta_position

    def find_correct_ds_for_psi(self, starting_ds, point, direction, target_psi, func=None, tolerance=1e-4, max_iter=100):
        """
        Find ds such that ψ(point + ds) = target_psi using bisection.
        Works for both positive and negative ψ targets.
        """
        if func is None:
            func = self.unit_normal_velocity

        ds_low = 0.0
        ds_high = starting_ds

        psi_start = self.stream_function(point)

        def psi_error(ds):
            new_point = hlp.rk4(point, direction, ds, self.surface_normal, func)
            return self.stream_function(new_point) - target_psi

        err_low = psi_error(ds_low)
        err_high = psi_error(ds_high)

        # Expand ds_high until it brackets the target
        attempts = 0
        while err_low * err_high > 0 and attempts < 50:
            ds_high *= 2
            err_high = psi_error(ds_high)
            attempts += 1

        if err_low * err_high > 0:
            print("Could not bracket ψ target; returning default step.")
            return starting_ds

        for _ in range(max_iter):
            ds_mid = 0.5 * (ds_low + ds_high)
            err_mid = psi_error(ds_mid)
            if abs(err_mid) < tolerance:
                return ds_mid
            elif err_low * err_mid < 0:
                ds_high = ds_mid
                err_high = err_mid
            else:
                ds_low = ds_mid
                err_low = err_mid

        print("Warning: Maximum iterations reached in find_correct_ds_for_psi.")
        return ds_mid

    def find_correct_ds(self, starting_ds, point, direction, delta_position, tolerance=1e-4, max_iter=10000, func=None):
        """Find ds such that the change in position matches delta_position within tolerance using bisection."""
        # Establish initial bounds
        if func is None:
            func = self.unit_velocity
        ds_low = 0.0
        ds_high = starting_ds
        # Expand upper bound until we bracket the root
        while self.bisection_function(ds_high, point, direction, delta_position, func = func) < 0:
            ds_high *= 2
            if ds_high > 1e6:
                print("Failed to bracket root; returning small ds.")
                return 0.1
        for _ in range(max_iter):
            ds_mid = 0.5 * (ds_low + ds_high)
            f_mid = self.bisection_function(ds_mid, point, direction, delta_position, func = func)
            if abs(f_mid) < tolerance:
                return ds_mid
            elif f_mid < 0:
                ds_low = ds_mid
            else:
                ds_high = ds_mid
        print("Warning: Maximum iterations reached in find_correct_ds.")
        print("Final error:", abs(self.bisection_function(ds_mid, point, direction, delta_position)), "Tolerance:", tolerance)
        input("Press Enter to continue...")
        return ds_mid
    
    def create_body_mesh(self):
        self.upper_coords, self.lower_coords = self.given_geom_and_stag_thetas_calc_top_and_bottom_coords(self.theta_aft, self.theta_forward, self.zeta_geom_array)
        print("length of upper_coords: ", len(self.upper_coords), " length of lower_coords: ", len(self.lower_coords))
        upper_indices = np.arange(len(self.upper_coords)).reshape(-1, 1)  # 0, 1, ..., N_upper-1
        lower_indices = (np.arange(len(self.lower_coords)) + len(self.upper_coords)).reshape(-1, 1)  # N_upper, N_upper+1, ...
        self.upper_coords = np.hstack((self.upper_coords, upper_indices))  # shape (N, 3)
        self.lower_coords = np.hstack((self.lower_coords, lower_indices))  # shape (N, 3)
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
        # find the unit normal vector at the surface stagnation points
        # the forward stagnation point is the last point of the upper surface and the aft stagnation point is the first point of the upper surface.
        # This means that to calculate the tangent in the forward direction, we need the last point in the upper surface, the second to last point in the upper surface, and the first point in the lower surface. 
        forward_stag_tangent = hlp.calc_tangent(self.upper_coords[-1, :2], self.upper_coords[-2, :2], self.lower_coords[0, :2]) # the :2 means we delete the index column
        forward_stag_normal = hlp.calc_normal(forward_stag_tangent) 
        self.forward_stag_unit_normal = self.unit_normal_velocity(forward_stag_normal)
        # to calculate the tangent in the aft direction, we need the first point in the upper surface, the second point in the upper surface, and the last point in the lower surface (IF self.D > 0.001).
        
        if self.D >= 0.001:
            aft_stag_tangent = hlp.calc_tangent(self.upper_coords[0, :2], self.upper_coords[1, :2], self.lower_coords[-1, :2])
            aft_stag_normal = hlp.calc_normal(aft_stag_tangent)
            self.aft_stag_unit_normal = self.unit_normal_velocity(aft_stag_normal)
        else:
            self.aft_stag_unit_normal = np.array([1e-12, 0.0])  # a very small value to avoid numerical issues
        # move the forward and aft stagnation points in their unit normal direction by 1e-12 to avoid numerical issues
        # self.forward_stag += 1e-12 * forward_stag_unit_normal
        # self.aft_stag += 1e-12 * aft_stag_unit_normal
        self.inner_boundary_mesh = np.concatenate((self.upper_coords, self.lower_coords), axis=0)
        self.last_body_index = self.lower_coords[-1, 2]  # this is the last index of the lower surface
    
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
        plt.xlim(-self.radial_distance-10, self.radial_distance+10)
        plt.ylim(-self.radial_distance-10, self.radial_distance+10)
    
    def plot_mesh(self):
        """"""
        # plot the unshifted mesh
        mesh.plot_geometry()
        # plot the midline
        mesh.plot(mesh.unshifted_streamlines)
        # mesh.plot(mesh.unshifted_potential_lines)
        plt.show()
        plt.close()

    def order_points_counterclockwise(self, arr: np.ndarray) -> np.ndarray:
        """
        Orders any number of unique [x, y, index] points counter-clockwise around their centroid.

        Parameters:
            arr (np.ndarray): An (N, 3) array where each row is [x, y, index].

        Returns:
            np.ndarray: A 1D numpy array of indices ordered counter-clockwise.
        """
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("Input must be an (N, 3) array where each row is [x, y, index].")
        if arr.shape[0] < 3:
            raise ValueError("At least three points are required to determine a CCW order.")

        coords = arr[:, :2]
        indices = arr[:, 2].astype(int)

        # Compute centroid
        centroid = coords.mean(axis=0)

        # Compute angle from centroid to each point
        angles = np.arctan2(coords[:,1] - centroid[1], coords[:,0] - centroid[0])

        # Sort by angle to get CCW order
        sorted_idx = np.argsort(angles)
        return indices[sorted_idx]
    
    def make_N_by_2_an_N_by_three(self, array: np.ndarray, starting_index: int) -> np.ndarray:
        """
        Converts an N by 2 array to an N by 3 array by adding an index column.
        The index starts at starting_index and increments by 1 for each row.
        
        Parameters:
            array (np.ndarray): An N by 2 numpy array.
            starting_index (int): The starting index for the new column.
        
        Returns:
            np.ndarray: An N by 3 numpy array with the last column as indices.
        """
        starting_index += 1  # Adjust starting index to start from 1
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError("Input must be an N by 2 numpy array.")
        indices = (np.arange(array.shape[0]) + starting_index).reshape(-1, 1)
        return np.hstack((array, indices))
    
    def plot_cell(self, four_element_array: np.ndarray,  linewidth: float = 0.5):
        """
        Plots a cell defined by 4 points in a 2D array.
        
        Parameters:
            four_element_array (np.ndarray): An array of shape (4, 2) representing the vertices of the cell.
            color (str): Color of the cell.
            alpha (float): Transparency level of the cell.
            linewidth (float): Width of the cell edges.
        """
        # connect the first point to the second point
        plt.plot([four_element_array[0, 0], four_element_array[1, 0]], [four_element_array[0, 1], four_element_array[1, 1]], color="blue", linewidth=linewidth)  # plot the first line as blue
        # connect the second point to the third point
        plt.plot([four_element_array[1, 0], four_element_array[2, 0]], [four_element_array[1, 1], four_element_array[2, 1]], color="green", linewidth=linewidth)  # plot the second line as green
        # connect the third point to the fourth point
        plt.plot([four_element_array[2, 0], four_element_array[3, 0]], [four_element_array[2, 1], four_element_array[3, 1]], color="red", linewidth=linewidth)  # plot the third line as red
        # now connect the fourth point to the first point
        plt.plot([four_element_array[0, 0], four_element_array[-1, 0]], [four_element_array[0, 1], four_element_array[-1, 1]], color="orange", linewidth=linewidth)

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
    mesh.theta_aft = mesh.calculate_aft_stagnation_theta_in_Chi_from_Gamma(mesh.circulation)
    mesh.theta_forward = mesh.calculate_forward_stag_theta_in_Chi_from_aft_stag_theta(mesh.theta_aft)
    mesh.forward_stag, mesh.aft_stag = mesh.calculate_forward_and_aft_stag_locations_in_z(mesh.theta_aft)
    mesh.get_full_geometry_zeta(mesh.surface_nodes, mesh.theta_aft, mesh.theta_forward, 2*np.pi + mesh.theta_aft)
    mesh.get_full_geometry()
    mesh.shift_joukowski_cylinder()
    mesh.calc_unshifted_limits()
    midline_start_points = mesh.create_body_mesh()
    
    marker_size = 0.1
    last_body_index = mesh.last_body_index
    #### streamlines geometrically spaced from the stagnation points
    forward_stag_streamline = mesh.calc_wrapped_stream_or_potential_line(mesh.forward_stag, -1)[1:]
    forward_stag_streamline = mesh.make_N_by_2_an_N_by_three(forward_stag_streamline, last_body_index)  # add index column starting from last_body_index + 1
    last_index_forward_stag = forward_stag_streamline[-1, 2]  # last index of the forward stagnation streamline
    aft_stag_streamline = mesh.calc_wrapped_stream_or_potential_line(mesh.aft_stag, 1)[1:]
    aft_stag_streamline = mesh.make_N_by_2_an_N_by_three(aft_stag_streamline, last_index_forward_stag)  # add index column starting from last_index_forward_stag + 1
    last_index_aft_stag = aft_stag_streamline[-1, 2]  # last index of the aft stagnation streamline
    #### upper midline ####
    upper_midline = mesh.calc_wrapped_stream_or_potential_line(mesh.closest_upper_point[:2],direction=1,psi_targets=None,func=mesh.unit_normal_velocity)[1:]  # drop the first point (which is the surface point)
    upper_stream_function_values = np.array([mesh.stream_function(p) for p in upper_midline])
    matched_upper_lines = []
    last_index_upper_midline = last_index_aft_stag 
    for i in tqdm(range(len(mesh.upper_coords)), desc="Calculating Upper Surface Potential Lines", unit="line"):
        if not np.allclose(mesh.upper_coords[i, :2], mesh.closest_upper_point[:2]): # check if the point is not the closest upper point, the :2 means we only take the first two coordinates (x and y)
            start_point = mesh.upper_coords[i, :2]
            matched_line = mesh.calc_wrapped_stream_or_potential_line(start_point,direction=1,psi_targets=upper_stream_function_values,func=mesh.unit_normal_velocity)[1:]  # skip surface point again
            matched_line = mesh.make_N_by_2_an_N_by_three(matched_line, last_index_upper_midline)  # add index column starting from last_index + 1
            last_index_upper_midline = matched_line[-1, 2]  # update the last index of the upper midline
            matched_upper_lines.append(matched_line)
        else:
            upper_midline = mesh.make_N_by_2_an_N_by_three(upper_midline, last_index_upper_midline)  # add index column starting from last_index_aft_stag + 1
            matched_upper_lines.append(upper_midline)
            last_index_upper_midline = upper_midline[-1, 2]  # update the last index of the upper midline
    last_upper_midline_index = last_index_upper_midline 
    #### lower midline ####
    lower_midline = mesh.calc_wrapped_stream_or_potential_line(mesh.closest_lower_point[:2],direction=-1,psi_targets=None,func=mesh.unit_normal_velocity)[1:]  # drop the first point (which is the surface point)
    lower_stream_function_values = np.array([mesh.stream_function(p) for p in lower_midline])
    matched_lower_lines = []
    last_index_lower_midline = last_upper_midline_index  # start from the last index of the upper midline
    for j in tqdm(range(len(mesh.lower_coords)), desc="Calculating Lower Surface Potential Lines", unit="line"):
        if not np.allclose(mesh.lower_coords[j, :2], mesh.closest_lower_point[:2]):
            start_point = mesh.lower_coords[j, :2]
            matched_line = mesh.calc_wrapped_stream_or_potential_line(start_point,direction=-1,psi_targets=lower_stream_function_values,func=mesh.unit_normal_velocity)[1:]
            matched_line = mesh.make_N_by_2_an_N_by_three(matched_line, last_index_lower_midline)  # add index column starting from last_index_lower_midline + 1
            last_index_lower_midline = matched_line[-1, 2]  # update the last index of the lower midline
            matched_lower_lines.append(matched_line)
        else:
            lower_midline = mesh.make_N_by_2_an_N_by_three(lower_midline, last_index_lower_midline)  # add index column starting from last_index_lower_midline + 1
            matched_lower_lines.append(lower_midline)
            last_index_lower_midline = lower_midline[-1, 2]
    last_lower_midline_index = last_index_lower_midline  # this is the last index of the lower midline
    #### calculate normal lines extending downward from the stagnation points ####
    lower_stag_potential_forward = mesh.calc_wrapped_stream_or_potential_line(mesh.forward_stag, -1, psi_targets=lower_stream_function_values, func=mesh.unit_normal_velocity, radial_growth=mesh.radial_growth)[1:]  # skip surface point
    lower_stag_potential_forward = mesh.make_N_by_2_an_N_by_three(lower_stag_potential_forward, last_lower_midline_index) 
    last_index_lower_potential_forward = lower_stag_potential_forward[-1, 2]  # last index of the lower stagnation potential line
    lower_stag_potential_aft = mesh.calc_wrapped_stream_or_potential_line(mesh.aft_stag, -1, psi_targets=lower_stream_function_values, func=mesh.unit_normal_velocity, radial_growth=mesh.radial_growth)[1:]  # skip surface point
    lower_stag_potential_aft = mesh.make_N_by_2_an_N_by_three(lower_stag_potential_aft, last_index_lower_potential_forward)  # add index column starting from last_index_lower_potential_forward + 1
    last_index_lower_potential_aft = lower_stag_potential_aft[-1, 2]  # last index of the lower stagnation potential line
    #### forward stagnation potential lines extending from the forward stagnation streamline ####
    leading_edge_upper_potential_lines = []
    index_leading_edge_upper = last_index_lower_potential_aft
    for i in tqdm(range(len(forward_stag_streamline)), desc="Calculating Leading Edge Upper Potential Lines", unit="streamline"):
        x_start, y_start = forward_stag_streamline[i, 0], forward_stag_streamline[i, 1]
        start_point = np.array([x_start, y_start])
        potential_line_leading_edge_upper = mesh.calc_wrapped_stream_or_potential_line(start_point, 1,psi_targets=upper_stream_function_values, func=mesh.unit_normal_velocity, radial_growth=mesh.radial_growth)[1:]  # skip surface normal point
        potential_line_leading_edge_upper = mesh.make_N_by_2_an_N_by_three(potential_line_leading_edge_upper, index_leading_edge_upper)  # add index column starting from first_index_leading_edge_upper + 1
        index_leading_edge_upper = potential_line_leading_edge_upper[-1, 2]  # update the last index of the leading edge upper potential line
        leading_edge_upper_potential_lines.append(potential_line_leading_edge_upper)
    last_index_leading_edge_upper = index_leading_edge_upper  # this is the last index of the leading edge upper potential lines
    #### upper normal lines extending from the aft stagnation streamline ####
    trailing_edge_upper_potential_lines = []
    index_trailing_edge_upper = last_index_leading_edge_upper
    for i in tqdm(range(len(aft_stag_streamline)), desc="Calculating Trailing Edge Upper Potential Lines", unit="streamline"):
        x_start, y_start = aft_stag_streamline[i, 0], aft_stag_streamline[i, 1]
        start_point = np.array([x_start, y_start])
        potential_line_trailing_edge_upper = mesh.calc_wrapped_stream_or_potential_line(start_point, 1,psi_targets=upper_stream_function_values, func=mesh.unit_normal_velocity, radial_growth=mesh.radial_growth)[1:]
        potential_line_trailing_edge_upper = mesh.make_N_by_2_an_N_by_three(potential_line_trailing_edge_upper, index_trailing_edge_upper)  # add index column starting from first_index_trailing_edge_upper + 1
        index_trailing_edge_upper = potential_line_trailing_edge_upper[-1, 2]  # last index of the trailing edge upper potential line
        trailing_edge_upper_potential_lines.append(potential_line_trailing_edge_upper)
    last_index_trailing_edge_upper = index_trailing_edge_upper  # this is the last index of the trailing edge upper potential lines
    #### lower normal lines extending from the stagnation streamlines starting with the forward stagnation streamline ####
    leading_edge_lower_potential_lines = []
    index_leading_edge_lower = last_index_trailing_edge_upper
    for i in tqdm(range(len(forward_stag_streamline)), desc="Calculating Leading Edge Lower Potential Lines", unit="streamline"):
        x_start, y_start = forward_stag_streamline[i, 0], forward_stag_streamline[i, 1]
        start_point = np.array([x_start, y_start])
        potential_line_leading_edge_lower = mesh.calc_wrapped_stream_or_potential_line(start_point, -1,psi_targets=lower_stream_function_values, func=mesh.unit_normal_velocity, radial_growth=mesh.radial_growth)[1:]
        potential_line_leading_edge_lower = mesh.make_N_by_2_an_N_by_three(potential_line_leading_edge_lower, index_leading_edge_lower)  # add index column starting from first_index_leading_edge_lower + 1
        index_leading_edge_lower = potential_line_leading_edge_lower[-1, 2]  # update the last index of the leading edge lower potential line
        leading_edge_lower_potential_lines.append(potential_line_leading_edge_lower)
    last_index_leading_edge_lower = index_leading_edge_lower  # this is the last index of the leading edge lower potential lines
    ####lower normal lines extending from the aft stagnation streamline
    trailing_edge_lower_potential_lines = []
    index_trailing_edge_lower = last_index_leading_edge_lower
    for i in tqdm(range(len(aft_stag_streamline)), desc="Calculating Trailing Edge Lower Potential Lines", unit="streamline"):
        x_start, y_start = aft_stag_streamline[i, 0], aft_stag_streamline[i, 1]
        start_point = np.array([x_start, y_start])
        potential_line_trailing_edge_lower = mesh.calc_wrapped_stream_or_potential_line(start_point, -1,psi_targets=lower_stream_function_values, func=mesh.unit_normal_velocity, radial_growth=mesh.radial_growth)[1:]
        potential_line_trailing_edge_lower = mesh.make_N_by_2_an_N_by_three(potential_line_trailing_edge_lower, index_trailing_edge_lower)  # add index column starting from first_index_trailing_edge_lower + 1
        index_trailing_edge_lower = potential_line_trailing_edge_lower[-1, 2]  # last index of the trailing edge lower potential line
        trailing_edge_lower_potential_lines.append(potential_line_trailing_edge_lower)
    last_index_trailing_edge_lower = index_trailing_edge_lower  # this is the last index of the trailing edge lower potential lines


    #### now we have all the points, start connecting them into cells starting with the upper surface which connects to forward_stag_streamline, matched_upper_lines, and the first lines in leading_edge_upper_potential_lines and trailing_edge_upper_potential_lines
    cells = [] # will contain the cells (4 indices each) on (directly touching) or right next to, the body mesh
    cell_one = np.array(([mesh.upper_coords[0], aft_stag_streamline[0], trailing_edge_upper_potential_lines[0][0], matched_upper_lines[0][0]]))
    cells.append(cell_one)  # first cell is the first point of the upper surface, the first point of the aft stagnation streamline, the first point of the matched upper line, and the first point of the leading edge upper potential line

    #### now loop through the upper surface points and connect them to the adjacent surface point and the appropriate matched upper line 
    for i in range(1, len(mesh.upper_coords)):
        # connect the upper surface point to the next point in the upper surface, the next point in the aft stagnation streamline, the next point in the matched upper line, and the next point in the leading edge upper potential line
        cell = np.array(([mesh.upper_coords[i], mesh.upper_coords[i-1], matched_upper_lines[i-1][0], matched_upper_lines[i][0]]))
        cells.append(cell)
    
    #### now connect the last point of the upper surface to the first point of the forward stagnation streamline, the first point of the matched upper line, and the first point of the leading edge upper potential line
    cell_last_upper = np.array(([forward_stag_streamline[0], mesh.upper_coords[-1],  matched_upper_lines[-1][0], leading_edge_upper_potential_lines[0][0]]))
    cells.append(cell_last_upper)  # last cell is the first point of the forward stagnation streamline, the last point of the upper surface, the last point of the matched upper line, and the first point of the leading edge upper potential line
    #### now connect the lower surface points 
    cell_one_lower = np.array(([leading_edge_lower_potential_lines[0][0], lower_stag_potential_forward[0], mesh.upper_coords[-1], forward_stag_streamline[0]]))
    cells.append(cell_one_lower)  # first cell is the first point of the lower stagnation potential line, the first point of the trailing edge lower potential line, the first point of the aft stagnation streamline, and the first point of the upper surface
    cell_two_lower = np.array(([lower_stag_potential_forward[0], matched_lower_lines[0][0],  mesh.lower_coords[0], mesh.upper_coords[-1]]))
    cells.append(cell_two_lower)  # second cell is the first point of the matched lower line, the first point of the lower stagnation potential line, the first point of the upper surface, and the first point of the lower surface
    
    #### now loop through the lower surface points and connect them to the adjacent surface point and the appropriate matched lower line
    for i in range(1, len(mesh.lower_coords)):
        # connect the lower surface point to the next point in the lower surface, the next point in the matched lower line, and the next point in the trailing edge lower potential line
        cell = np.array(([matched_lower_lines[i-1][0], matched_lower_lines[i][0], mesh.lower_coords[i], mesh.lower_coords[i-1]]))
        cells.append(cell)

    #### now get the second to last and last cells in the lower surface
    cell_second_last_lower = np.array(([matched_lower_lines[-1][0], lower_stag_potential_aft[0], mesh.upper_coords[0], mesh.lower_coords[-1]]))
    cells.append(cell_second_last_lower)  # second to last cell is the last point of the matched lower line, the first point of the lower stagnation potential line, the first point of the upper surface, and the last point of the lower surface

    #### now for the last body cell
    cell_last_lower = np.array(([lower_stag_potential_aft[0], trailing_edge_lower_potential_lines[0][0], aft_stag_streamline[0], mesh.upper_coords[0]]))
    cells.append(cell_last_lower)  # last cell is the first point of the lower stagnation potential line, the first point of the trailing edge lower potential line, the first point of the aft stagnation streamline, and the first point of the upper surface

    #### now create cells on the forward stagnation streamline (above and below the streamline)
    for i in range(len(forward_stag_streamline)-1):
        upper_cell = np.array(([forward_stag_streamline[i + 1], forward_stag_streamline[i], leading_edge_upper_potential_lines[i][0], leading_edge_upper_potential_lines[i + 1][0]]))
        cells.append(upper_cell)  # upper cell is the next point in the forward stagnation streamline, the current point in the forward stagnation streamline, the next point in the leading edge upper potential line, and the next point in the leading edge upper potential line
        lower_cell = np.array(([leading_edge_lower_potential_lines[i+1][0], leading_edge_lower_potential_lines[i][0], forward_stag_streamline[i], forward_stag_streamline[i + 1]]))
        cells.append(lower_cell)  # lower cell is the next point in the leading edge lower potential line, the current point in the leading edge lower potential line, the current point in the forward stagnation streamline, and the next point in the forward stagnation streamline
    #### now create cells on the aft stagnation streamline (above and below the streamline)
    for i in range(len(aft_stag_streamline)-1):
        upper_cell = np.array(([aft_stag_streamline[i], aft_stag_streamline[i + 1], trailing_edge_upper_potential_lines[i+1][0], trailing_edge_upper_potential_lines[i][0]]))
        cells.append(upper_cell)  # upper cell is the current point in the aft stagnation streamline, the next point in the aft stagnation streamline, the next point in the trailing edge upper potential line, and the current point in the trailing edge upper potential line
        lower_cell = np.array(([trailing_edge_lower_potential_lines[i][0], trailing_edge_lower_potential_lines[i + 1][0], aft_stag_streamline[i + 1], aft_stag_streamline[i]]))
        cells.append(lower_cell)  # lower cell is the current point in the trailing edge lower potential line, the next point in the trailing edge lower potential line, the next point in the aft stagnation streamline, and the current point in the aft stagnation streamline
    #### create cells on the upper and lower potential lines. Start with the leading edge upper potential lines
    for i in range(len(leading_edge_upper_potential_lines)-1):
        for j in range(len(leading_edge_upper_potential_lines[i])-1):
            upper_cell = np.array(([leading_edge_upper_potential_lines[i+1][j], leading_edge_upper_potential_lines[i][j], leading_edge_upper_potential_lines[i][j+1], leading_edge_upper_potential_lines[i+1][j+1]]))
            cells.append(upper_cell)  # upper cell is the next point in the leading edge upper potential line, the current point in the leading edge upper potential line, the current point in the next point in the leading edge upper potential line, and the next point in the next point in the leading edge upper potential line
    #### make cells in a similar fashion for the leading edge lower potential lines
    for i in range(len(leading_edge_lower_potential_lines)-1):
        for j in range(len(leading_edge_lower_potential_lines[i])-1):
            lower_cell = np.array(([leading_edge_lower_potential_lines[i+1][j+1], leading_edge_lower_potential_lines[i][j+1], leading_edge_lower_potential_lines[i][j], leading_edge_lower_potential_lines[i+1][j]]))
            cells.append(lower_cell)
    #### now do the same for the trailing edge upper potential lines
    for i in range(len(trailing_edge_upper_potential_lines)-1):
        for j in range(len(trailing_edge_upper_potential_lines[i])-1):
            upper_cell = np.array(([trailing_edge_upper_potential_lines[i][j], trailing_edge_upper_potential_lines[i+1][j], trailing_edge_upper_potential_lines[i+1][j+1], trailing_edge_upper_potential_lines[i][j+1]]))
            cells.append(upper_cell)     
    #### do the same for the trailing edge lower potential lines
    for i in range(len(trailing_edge_lower_potential_lines)-1):
        for j in range(len(trailing_edge_lower_potential_lines[i])-1):
            lower_cell = np.array(([trailing_edge_lower_potential_lines[i][j+1], trailing_edge_lower_potential_lines[i+1][j+1], trailing_edge_lower_potential_lines[i+1][j], trailing_edge_lower_potential_lines[i][j]]))
            cells.append(lower_cell)   
    #### finally, create cells for the matched upper and lower lines, starting with the first matched upper line (the one associated with the aft stagnation streamline)
    print("length of matched upper lines:", len(matched_upper_lines[0]))
    for i in range(len(matched_upper_lines[0])-1):
        upper_cell = np.array(([matched_upper_lines[0][i], trailing_edge_upper_potential_lines[0][i], trailing_edge_upper_potential_lines[0][i+1], matched_upper_lines[0][i+1]]))
        cells.append(upper_cell)  # upper cell is the current point in the matched upper line, the current point in the trailing edge upper potential line, the next point in the trailing edge upper potential line, and the next point in the matched upper line
    #### now do the last matched upper line (the one associated with the leading edge upper potential line)
    for i in range(len(matched_upper_lines[-1])-1):
        upper_cell = np.array(([leading_edge_upper_potential_lines[0][i], matched_upper_lines[-1][i], matched_upper_lines[-1][i+1], leading_edge_upper_potential_lines[0][i+1]]))
        cells.append(upper_cell)  # upper cell is the current point in the leading edge upper potential line, the current point in the matched upper line, the next point in the matched upper line, and the next point in the leading edge upper potential line
    #### now loop through the matched_upper_lines and create cells for each matched upper line (that doesn't border with the leading or trailing edge upper potential lines)
    for i in range(len(matched_upper_lines)-1):
        for j in range(len(matched_upper_lines[i])-1):
            upper_cell = np.array(([matched_upper_lines[i+1][j], matched_upper_lines[i][j], matched_upper_lines[i][j+1], matched_upper_lines[i+1][j+1]]))
            cells.append(upper_cell)
    #### now loop through the lower stagnation potential lines and create cells that meet with the leading and trailing edge lower potential lines
    for i in range(len(lower_stag_potential_forward)-1): ##### adjust to hit either side of the leading edge lower potential line
        lower_stag_cell = np.array(([leading_edge_lower_potential_lines[0][i+1], lower_stag_potential_forward[i+1],  lower_stag_potential_forward[i], leading_edge_lower_potential_lines[0][i]]))
        cells.append(lower_stag_cell)  # lower cell is the next point in the leading edge lower potential line, the next point in the lower stagnation potential line, the current point in the lower stagnation potential line, and the current point in the leading edge lower potential line
    #### now loop through the lower stagnation potential line and connect it to the trailing edge lower potential line
    for i in range(len(lower_stag_potential_aft)-1): ##### adjust to hit either side of the trailing edge lower potential line
        lower_stag_cell = np.array(([lower_stag_potential_aft[i+1], trailing_edge_lower_potential_lines[0][i+1], trailing_edge_lower_potential_lines[0][i], lower_stag_potential_aft[i]]))
        cells.append(lower_stag_cell)
        # print("lower stagnation cell :", lower_stag_cell[:, 2])
    #### finally loop through the matched lower lines and create cells for each matched lower line (that doesn't border with the leading or trailing edge lower potential lines)
    for i in range(len(matched_lower_lines)-1):
        for j in range(len(matched_lower_lines[i])-1):
            lower_cell = np.array(([matched_lower_lines[i+1][j], matched_lower_lines[i][j], matched_lower_lines[i][j+1], matched_lower_lines[i+1][j+1]]))
            cells.append(lower_cell)

    


    

    if mesh.is_visualize_mesh:
        plt.scatter(mesh.upper_coords[:, 0], mesh.upper_coords[:, 1], color='blue', label='Upper Points', s=marker_size)
        plt.scatter(mesh.lower_coords[:, 0], mesh.lower_coords[:, 1], color='red', label='Lower Points', s=marker_size)
        plt.scatter(upper_midline[:, 0], upper_midline[:, 1], color='black', s=marker_size)  # blue color and circle marker
        plt.scatter(lower_midline[:, 0], lower_midline[:, 1], color='black', s=marker_size)  # red color and circle marker
        for i in range(len(matched_upper_lines)):
            plt.scatter(matched_upper_lines[i][:, 0], matched_upper_lines[i][:, 1], color='black', s=marker_size)
        for j in range(len(matched_lower_lines)):
            plt.scatter(matched_lower_lines[j][:, 0], matched_lower_lines[j][:, 1], color='black', s=marker_size)
        plt.scatter(lower_stag_potential_forward[:, 0], lower_stag_potential_forward[:, 1], color='black', s=marker_size)  # red color and circle marker
        plt.scatter(lower_stag_potential_aft[:, 0], lower_stag_potential_aft[:, 1], color='black', s=marker_size)  # red color and circle marker
        plt.scatter(forward_stag_streamline[:, 0], forward_stag_streamline[:, 1], color='black', s=marker_size)  # blue color and circle marker
        plt.scatter(aft_stag_streamline[:, 0], aft_stag_streamline[:, 1], color='black', s=marker_size)  # blue color and circle marker
        for i in range(len(leading_edge_upper_potential_lines)):
            plt.scatter(leading_edge_upper_potential_lines[i][:, 0], leading_edge_upper_potential_lines[i][:, 1], color='black', s=marker_size)
        for i in range(len(trailing_edge_upper_potential_lines)):
            plt.scatter(trailing_edge_upper_potential_lines[i][:, 0], trailing_edge_upper_potential_lines[i][:, 1], color='black', s=marker_size)
        for i in range(len(leading_edge_lower_potential_lines)):
            plt.scatter(leading_edge_lower_potential_lines[i][:, 0], leading_edge_lower_potential_lines[i][:, 1], color='black', s=marker_size)
        for i in range(len(trailing_edge_lower_potential_lines)):
            plt.scatter(trailing_edge_lower_potential_lines[i][:, 0], trailing_edge_lower_potential_lines[i][:, 1], color='black', s=marker_size)
        for i in range(len(cells)):
            mesh.plot_cell(cells[i], linewidth=0.5)  # plot the cells on the upper surface
        if mesh.is_visualize_node_integers:
            for i in range(len(mesh.upper_coords)):
                plt.text(mesh.upper_coords[i, 0] + 0.05, mesh.upper_coords[i, 1] - 0.05, str(int(mesh.upper_coords[i, 2])), fontsize=8, color='black')
            for j in range(len(mesh.lower_coords)):
                plt.text(mesh.lower_coords[j, 0] + 0.05, mesh.lower_coords[j, 1] - 0.05, str(int(mesh.lower_coords[j, 2])), fontsize=8, color='black')
            for i in range(len(forward_stag_streamline)):
                plt.text(forward_stag_streamline[i, 0] + 0.05, forward_stag_streamline[i, 1] - 0.05, str(int(forward_stag_streamline[i, 2])), fontsize=8, color='black')
            for j in range(len(aft_stag_streamline)):
                plt.text(aft_stag_streamline[j, 0] + 0.05, aft_stag_streamline[j, 1] - 0.05, str(int(aft_stag_streamline[j, 2])), fontsize=8, color='black')
            for line in matched_upper_lines:
                    for k in range(line.shape[0]):
                        plt.text(line[k, 0] + 0.05, line[k, 1] - 0.05, str(int(line[k, 2])), fontsize=8, color='black')
            for line in matched_lower_lines:
                for k in range(line.shape[0]):
                    plt.text(line[k, 0] + 0.05, line[k, 1] - 0.05, str(int(line[k, 2])), fontsize=8, color='black')
            # lower stagnation potential lines
            for i in range(len(lower_stag_potential_forward)):
                plt.text(lower_stag_potential_forward[i, 0] + 0.05, lower_stag_potential_forward[i, 1] - 0.05, str(int(lower_stag_potential_forward[i, 2])), fontsize=8, color='black')
            for j in range(len(lower_stag_potential_aft)):
                plt.text(lower_stag_potential_aft[j, 0] + 0.05, lower_stag_potential_aft[j, 1] - 0.05, str(int(lower_stag_potential_aft[j, 2])), fontsize=8, color='black')
            # leading_edge_upper_potential_lines
            for i in range(len(leading_edge_upper_potential_lines)):
                for k in range(leading_edge_upper_potential_lines[i].shape[0]):
                    plt.text(leading_edge_upper_potential_lines[i][k, 0] + 0.05, leading_edge_upper_potential_lines[i][k, 1] - 0.05, str(int(leading_edge_upper_potential_lines[i][k, 2])), fontsize=8, color='black')
            # trailing_edge_upper_potential_lines
            for i in range(len(trailing_edge_upper_potential_lines)):
                for k in range(trailing_edge_upper_potential_lines[i].shape[0]):
                    plt.text(trailing_edge_upper_potential_lines[i][k, 0] + 0.05, trailing_edge_upper_potential_lines[i][k, 1] - 0.05, str(int(trailing_edge_upper_potential_lines[i][k, 2])), fontsize=8, color='black')
            # leading_edge_lower_potential_lines
            for i in range(len(leading_edge_lower_potential_lines)):
                for k in range(leading_edge_lower_potential_lines[i].shape[0]):
                    plt.text(leading_edge_lower_potential_lines[i][k, 0] + 0.05, leading_edge_lower_potential_lines[i][k, 1] - 0.05, str(int(leading_edge_lower_potential_lines[i][k, 2])), fontsize=8, color='black')
            # trailing_edge_lower_potential_lines
            for i in range(len(trailing_edge_lower_potential_lines)):
                for k in range(trailing_edge_lower_potential_lines[i].shape[0]):
                    plt.text(trailing_edge_lower_potential_lines[i][k, 0] + 0.05, trailing_edge_lower_potential_lines[i][k, 1] - 0.05, str(int(trailing_edge_lower_potential_lines[i][k, 2])), fontsize=8, color='black')


        plt.show()

        

    # n_upper   = len(mesh.upper_coords)          # len(upper) + 1 cells on upper side
    # n_lower   = len(mesh.lower_coords)          # len(lower) + 3 cells on lower side
    # n_cells   = n_upper + n_lower + 4           # total cells (derived from your loops)
    # n_coords  = mesh.upper_coords.shape[1]      # 2 or 3 depending on your mesh

    # cells = np.empty((n_cells, 4, n_coords), dtype=mesh.upper_coords.dtype)

    # k = 0  # running index -------------------------------------------------
    # cells[k] = np.array([mesh.upper_coords[0],
    #                         aft_stag_streamline[0],
    #                         trailing_edge_upper_potential_lines[0][0],
    #                         matched_upper_lines[0][0]])
    # k += 1

    # for i in range(1, len(mesh.upper_coords)):
    #     cells[k] = np.array([mesh.upper_coords[i],
    #                             mesh.upper_coords[i-1],
    #                             matched_upper_lines[i-1][0],
    #                             matched_upper_lines[i][0]])
    #     k += 1

    # cells[k] = np.array([forward_stag_streamline[0],
    #                         mesh.upper_coords[-1],
    #                         matched_upper_lines[-1][0],
    #                         leading_edge_upper_potential_lines[0][0]])
    # k += 1

    # cells[k] = np.array([leading_edge_lower_potential_lines[0][0],
    #                         lower_stag_potential_forward[0],
    #                         mesh.upper_coords[-1],
    #                         forward_stag_streamline[0]])
    # k += 1

    # cells[k] = np.array([lower_stag_potential_forward[0],
    #                         matched_lower_lines[0][0],
    #                         mesh.lower_coords[0],
    #                         mesh.upper_coords[-1]])
    # k += 1

    # for i in range(1, len(mesh.lower_coords)):
    #     cells[k] = np.array([matched_lower_lines[i-1][0],
    #                             matched_lower_lines[i][0],
    #                             mesh.lower_coords[i],
    #                             mesh.lower_coords[i-1]])
    #     k += 1

    # cells[k] = np.array([matched_lower_lines[-1][0],
    #                         lower_stag_potential_aft[0],
    #                         mesh.upper_coords[0],
    #                         mesh.lower_coords[-1]])
    # k += 1

    # cells[k] = np.array([lower_stag_potential_aft[0],
    #                         trailing_edge_lower_potential_lines[0][0],
    #                         aft_stag_streamline[0],
    #                         mesh.upper_coords[0]])
    # # k += 1  # (optional)

    # # --------------------------------------------------------------------
    # for i in range(len(cells)):
    #     mesh.plot_cell(cells[i], linewidth=0.5)