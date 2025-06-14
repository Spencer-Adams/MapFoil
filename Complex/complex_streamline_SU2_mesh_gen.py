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
            desc = "Calculating Streamline"
        else:
            desc = "Calculating Potential Line"
        # Ensure valid start_point
        if not isinstance(start_point, np.ndarray) or start_point.shape != (2,) or not np.issubdtype(start_point.dtype, np.floating):
            raise ValueError("The start_point must be a NumPy array of two float values representing x and y coordinates.")
        point = np.array(start_point)
        streamline_points = [point]
        i = 0
        max_iter = 10000
        # Use absolute arc length as the total distance to travel
        total_distance = self.radial_distance
        with tqdm(total=total_distance, desc=desc, unit="arc-length", dynamic_ncols=True) as pbar:
            while True:
                # Take RK4 step
                point_new = hlp.rk4(point, direction, self.plot_delta_s, self.surface_normal, func)
                # Stop if new point is outside x bounds
                if point_new[0] < self.plot_x_lower_lim or point_new[0] > self.plot_x_upper_lim:
                    break
                streamline_points.append(point_new)
                # Compute total arc length covered so far
                arc_length_covered = hlp.compute_arc_length(streamline_points)
                # Update progress bar with how much new arc length was added
                remaining_distance = total_distance - pbar.n
                update_distance = min(arc_length_covered - pbar.n, remaining_distance)
                pbar.update(update_distance)
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
        if self.is_visualize_mesh:
            plt.scatter(self.upper_coords[:, 0], self.upper_coords[:, 1], color='blue', label='Upper Points', s=0.1)
            plt.scatter(self.lower_coords[:, 0], self.lower_coords[:, 1], color='red', label='Lower Points', s=0.1)
            # for i in range(len(self.upper_coords)):
            #     plt.text(self.upper_coords[i, 0] + 0.05, self.upper_coords[i, 1] - 0.05, str(int(self.upper_coords[i, 2])), fontsize=8, color='black')
            # for j in range(len(self.lower_coords)):
            #     plt.text(self.lower_coords[j, 0] + 0.05, self.lower_coords[j, 1] - 0.05, str(int(self.lower_coords[j, 2])), fontsize=8, color='black')
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
        plt.xlim(-self.radial_distance-5, self.radial_distance+5)
        plt.ylim(-self.radial_distance-5, self.radial_distance+5)
    
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

    # upper midline
    upper_midline = mesh.calc_wrapped_stream_or_potential_line(mesh.closest_upper_point[:2],direction=1,psi_targets=None,func=mesh.unit_normal_velocity)[1:]  # drop the first point (which is the surface point)
    # Plot the midline potential line
    # plt.plot(upper_midline[:, 0], upper_midline[:, 1], 'bo', markersize=marker_size)  # blue color and circle marker
    plt.scatter(upper_midline[:, 0], upper_midline[:, 1], color='black', s=marker_size)  # blue color and circle marker
    # Compute ψ values along this midline
    upper_stream_function_values = np.array([mesh.stream_function(p) for p in upper_midline])
    # For each other upper surface point, generate potential lines that match the ψ levels found above
    matched_upper_lines = []
    for i in range(len(mesh.upper_coords)):
        if not np.allclose(mesh.upper_coords[i, :2], mesh.closest_upper_point[:2]):
            start_point = mesh.upper_coords[i, :2]
            matched_line = mesh.calc_wrapped_stream_or_potential_line(start_point,direction=1,psi_targets=upper_stream_function_values,func=mesh.unit_normal_velocity)[1:]  # skip surface point again
            matched_upper_lines.append(matched_line)
            plt.scatter(matched_line[:, 0], matched_line[:, 1], color="black", s=marker_size)
            # plt.plot(matched_line[:, 0], matched_line[:, 1], 'go', markersize=marker_size)

    # lower midline
    lower_midline = mesh.calc_wrapped_stream_or_potential_line(mesh.closest_lower_point[:2],direction=-1,psi_targets=None,func=mesh.unit_normal_velocity)[1:]  # drop the first point (which is the surface point)
    # Plot the midline potential line
    # plt.plot(lower_midline[:, 0], lower_midline[:, 1], 'ro', markersize=marker_size)  # red color and circle marker
    plt.scatter(lower_midline[:, 0], lower_midline[:, 1], color='black', s=marker_size)  # blue color and circle marker
    # Compute ψ values along this midline
    lower_stream_function_values = np.array([mesh.stream_function(p) for p in lower_midline])
    # For each other lower surface point, generate potential lines that match the ψ levels found above
    matched_lower_lines = []
    for j in range(len(mesh.lower_coords)):
        if not np.allclose(mesh.lower_coords[j, :2], mesh.closest_lower_point[:2]):
            start_point = mesh.lower_coords[j, :2]
            matched_line = mesh.calc_wrapped_stream_or_potential_line(start_point,direction=-1,psi_targets=lower_stream_function_values,func=mesh.unit_normal_velocity)[1:]
            matched_lower_lines.append(matched_line)
            # plt.plot(matched_line[:, 0], matched_line[:, 1], 'go', markersize=marker_size)  # green color and circle marker
            plt.scatter(matched_line[:, 0], matched_line[:, 1], color='black', s=marker_size)

    # create geometrically spaced streamlines from the stagnation points
    forward_stag_streamline = mesh.calc_wrapped_stream_or_potential_line(mesh.forward_stag, -1)[1:]
    aft_stag_streamline = mesh.calc_wrapped_stream_or_potential_line(mesh.aft_stag, 1)[1:]
    # plot the forward stagnation streamline
    # plt.plot(forward_stag_streamline[:, 0], forward_stag_streamline[:, 1], 'bo', markersize=marker_size)  # red color and circle marker
    plt.scatter(forward_stag_streamline[:, 0], forward_stag_streamline[:, 1], color='black', s=marker_size)  # blue color and circle marker
    # plot the aft stagnation streamline
    # plt.plot(aft_stag_streamline[:, 0], aft_stag_streamline[:, 1], 'bo', markersize=marker_size)  # red color and circle marker
    plt.scatter(aft_stag_streamline[:, 0], aft_stag_streamline[:, 1], color='black', s=marker_size)  # blue color and circle marker
    
    # plot upper normal lines extending from the stagnation streamlines
    # start with the forward stagnation streamline
    leading_edge_upper_streamlines = []
    for i in range(len(forward_stag_streamline)):
        x_start, y_start = forward_stag_streamline[i, 0], forward_stag_streamline[i, 1]
        start_point = np.array([x_start, y_start])
        streamline_leading_edge_upper = mesh.calc_wrapped_stream_or_potential_line(start_point, 1,psi_targets=upper_stream_function_values, func=mesh.unit_normal_velocity, radial_growth=1.1)[1:]  # skip surface normal point
        leading_edge_upper_streamlines.append(streamline_leading_edge_upper)
        # plt.plot(streamline_leading_edge_upper[:, 0], streamline_leading_edge_upper[:, 1], 'ro', markersize=marker_size)  # blue color and circle marker
        plt.scatter(streamline_leading_edge_upper[:, 0], streamline_leading_edge_upper[:, 1], color='black', s=marker_size)  # blue color and circle marker
    
    # # plot upper normal lines extending from the aft stagnation streamline
    trailing_edge_upper_streamlines = []
    for i in range(len(aft_stag_streamline)):
        x_start, y_start = aft_stag_streamline[i, 0], aft_stag_streamline[i, 1]
        start_point = np.array([x_start, y_start])
        streamline_trailing_edge_upper = mesh.calc_wrapped_stream_or_potential_line(start_point, 1,psi_targets=upper_stream_function_values, func=mesh.unit_normal_velocity, radial_growth=1.1)[1:]
        trailing_edge_upper_streamlines.append(streamline_trailing_edge_upper)
        # plt.plot(streamline_trailing_edge_upper[:, 0], streamline_trailing_edge_upper[:, 1], 'ro', markersize=marker_size)  # blue color and circle marker
        plt.scatter(streamline_trailing_edge_upper[:, 0], streamline_trailing_edge_upper[:, 1], color='black', s=marker_size)  # blue color and circle marker

    # plot lower normal lines extending from the stagnation streamlines
    # start with the forward stagnation streamline
    leading_edge_lower_streamlines = []
    for i in range(len(forward_stag_streamline)):
        x_start, y_start = forward_stag_streamline[i, 0], forward_stag_streamline[i, 1]
        start_point = np.array([x_start, y_start])
        streamline_leading_edge_lower = mesh.calc_wrapped_stream_or_potential_line(start_point, -1,psi_targets=lower_stream_function_values, func=mesh.unit_normal_velocity, radial_growth=1.1)[1:]
        leading_edge_lower_streamlines.append(streamline_leading_edge_lower)
        # plt.plot(streamline_leading_edge_lower[:, 0], streamline_leading_edge_lower[:, 1], 'ro', markersize=marker_size)  # red color and circle marker
        plt.scatter(streamline_leading_edge_lower[:, 0], streamline_leading_edge_lower[:, 1], color='black', s=marker_size)  # red color and circle marker

    # # plot lower normal lines extending from the aft stagnation streamline
    trailing_edge_lower_streamlines = []
    for i in range(len(aft_stag_streamline)):
        x_start, y_start = aft_stag_streamline[i, 0], aft_stag_streamline[i, 1]
        start_point = np.array([x_start, y_start])
        streamline_trailing_edge_lower = mesh.calc_wrapped_stream_or_potential_line(start_point, -1,psi_targets=lower_stream_function_values, func=mesh.unit_normal_velocity, radial_growth=1.1)[1:]
        trailing_edge_lower_streamlines.append(streamline_trailing_edge_lower)
        # plt.plot(streamline_trailing_edge_lower[:, 0], streamline_trailing_edge_lower[:, 1], 'ro', markersize=marker_size)
        plt.scatter(streamline_trailing_edge_lower[:, 0], streamline_trailing_edge_lower[:, 1], color='black', s=marker_size)  # red color and circle marker

    plt.show()



    