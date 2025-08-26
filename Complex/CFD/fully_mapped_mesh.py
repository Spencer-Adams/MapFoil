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
from scipy.optimize import root_scalar # type: ignore
from scipy.optimize import least_squares # type: ignore

class mesh_generation_object(cylinder, vort_panel):
    """This class uses conformal mapping or NACA airfoil coordinates to generate a mesh for a given domain."""
    def __init__(self, mesh_json_file: str = "fully_mapped_mesh.json", joukowski_cylinder_json_file: str = "Joukowski_Cylinder.json"):
        """Initialize the mesh generation object."""
        # Initialize the parent classes
        self.mesh_json_file = mesh_json_file
        self.joukowski_cylinder_json_file = joukowski_cylinder_json_file
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
            self.circulation = self.kutta_circulation
            self.mesh_filename = f"mesh_zeta0_{self.zeta_center.real:.3f}_{self.zeta_center.imag:.3f}_D_{self.D:.3f}_AoA_{np.rad2deg(self.angle_of_attack):.1f}deg_radial_nodes_{self.radial_nodes}_surface_nodes_{self.surface_nodes}_radial_distance_{self.radial_distance}_radial_growth_{self.radial_growth}.su2"
            plt.xlim(-self.radial_distance-1, self.radial_distance+1)
            plt.ylim(-self.radial_distance-1, self.radial_distance+1)
            plt.xlabel(r"$x/c$")
            plt.ylabel(r"$y/c$")
            plt.gca().set_aspect('equal', adjustable='box')

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

    # def calc_deltaD_0(self, N, radial_growth, start_xy=np.array([0.0, 0.0]), end_xy=np.array([1.0, 1.0]), total_distance=None):
    #     """
    #     Calculate the initial radial spacing (deltaD_0) such that N points,
    #     growing by a factor of `radial_growth`, span a total radial distance.
    #     """
    #     if total_distance is None:
    #         x_start, y_start = start_xy
    #         x_end, y_end = end_xy
    #         total_distance = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
    #     if np.isclose(total_distance, 0.0):
    #         print("Distance is zero, returning 0.0 for deltaD_0.")
    #         return 0.0
    #     if np.isclose(radial_growth, 1.0):
    #         return total_distance / N  # N intervals of equal spacing
    #     sum_geom = (1 - radial_growth ** N) / (1 - radial_growth)
    #     return total_distance / sum_geom
    
    def sort_points_ccw_by_centroid(self, arr: np.ndarray) -> np.ndarray:
        """
        Sorts an (N, 3) array of [x, y, index] points in counterclockwise order around the centroid.

        Parameters:
            arr (np.ndarray): An (N, 3) array where each row is [x, y, index].

        Returns:
            np.ndarray: The input rows reordered in counterclockwise order.
        """
        # Ensure input is correct shape
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("Input must be an (N, 3) array where each row is [x, y, index].")
        
        # Need at least three points to define a polygon
        if arr.shape[0] < 3:
            raise ValueError("At least three points are required to determine order.")

        # Extract coordinates
        coords = arr[:, :2]
        centroid = coords.mean(axis=0)

        # Compute angle from centroid to each point
        angles = np.arctan2(coords[:, 1] - centroid[1], coords[:, 0] - centroid[0])

        # Sort the original array by these angles
        sorted_idx = np.argsort(angles)
        return arr[sorted_idx]
    
    def plot_colored_ccw_lines(self, points_ccw: np.ndarray, shrink_factor: float = 0.9, ax=None) -> None:
        """
        Plot CCW-connected lines between given points, colored in order and shrunk toward the centroid.

        Parameters:
            points_ccw (np.ndarray): An (N, 3) array of 2D points ordered counterclockwise. The third dimension is an index.
            shrink_factor (float): Factor to shrink each point toward the centroid (0 < shrink_factor <= 1).
            ax (matplotlib.axes.Axes): Optional matplotlib axis. If None, uses current axis.
        """
        # each point in points_ccw is currently N,3. Make them Nx2 first
        points_n_by_2 = points_ccw[:, :2]

        centroid = points_n_by_2.mean(axis=0)
        # Shrink all points toward centroid
        shrunk_points = centroid + shrink_factor * (points_n_by_2 - centroid)
        # Color cycle
        colors = ['red', 'green', 'blue', 'black']
        N = len(shrunk_points)
        # Plot edges
        for i in range(N):
            p1 = shrunk_points[i]
            p2 = shrunk_points[(i + 1) % N]  # wrap around
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=colors[i % len(colors)], linewidth=0.2)
    
    def calc_shifted_limits(self):
        """Shifts the mesh such that the surface points form a unit chord from 0 to 1. This function is the inverse of the calc_unshifted_limits function."""
        self.plot_x_lower_lim, self.plot_x_upper_lim, self.plot_y_lower_lim, self.plot_y_upper_lim = -self.radial_distance+1, self.radial_distance+1, -self.radial_distance, self.radial_distance
        self.plot_x_lower_lim, self.plot_x_upper_lim = self.plot_x_lower_lim - self.zeta_center.real - self.leading_edge, self.plot_x_upper_lim - self.zeta_center.real - self.leading_edge
        self.plot_y_lower_lim, self.plot_y_upper_lim = self.plot_y_lower_lim - self.zeta_center.imag, self.plot_y_upper_lim - self.zeta_center.imag
        # now divide the x and y limits by the trailing edge
        self.plot_x_lower_lim, self.plot_x_upper_lim, self.plot_y_lower_lim, self.plot_y_upper_lim = self.plot_x_lower_lim / self.trailing_edge, self.plot_x_upper_lim / self.trailing_edge, self.plot_y_lower_lim / self.trailing_edge, self.plot_y_upper_lim / self.trailing_edge
        self.plot_x_start = self.plot_x_lower_lim
        # re-define radial_distance to be plot_y_upper_lim
        self.radial_distance = self.plot_y_upper_lim
        plt.xlim(-self.radial_distance*1.5, self.radial_distance*1.5)
        plt.ylim(-self.radial_distance*1.5, self.radial_distance*1.5)

    def calc_unshifted_limits(self):
        """"""
        self.plot_x_lower_lim, self.plot_x_upper_lim, self.plot_y_lower_lim, self.plot_y_upper_lim = -self.radial_distance+1, self.radial_distance+1, -self.radial_distance, self.radial_distance
        self.plot_x_lower_lim, self.plot_x_upper_lim = self.plot_x_lower_lim + self.zeta_center.real + self.leading_edge, self.plot_x_upper_lim + self.zeta_center.real + self.leading_edge
        self.plot_y_lower_lim, self.plot_y_upper_lim = self.plot_y_lower_lim + self.zeta_center.imag, self.plot_y_upper_lim + self.zeta_center.imag
        # # now divide the x and y limits by the trailing edge
        self.plot_x_lower_lim, self.plot_x_upper_lim, self.plot_y_lower_lim, self.plot_y_upper_lim = self.plot_x_lower_lim * self.trailing_edge, self.plot_x_upper_lim * self.trailing_edge, self.plot_y_lower_lim * self.trailing_edge, self.plot_y_upper_lim * self.trailing_edge
        self.plot_x_start = self.plot_x_lower_lim
        # re-define radial_distance to be plot_y_upper_lim 
        self.radial_distance = self.plot_y_upper_lim
        plt.xlim(-self.radial_distance*1.5, self.radial_distance*1.5)
        plt.ylim(-self.radial_distance*1.5, self.radial_distance*1.5)

    def convert_grid_into_SU2_format(self, xy_points: np.array, cells: np.array, inner_boundary_lines: np.array, outer_boundary_lines: np.array, mesh_filename: str = "complex_streamline_SU2_mesh_gen.su2"):
        """
        Converts grid data into SU2 mesh file format and writes it to a file.

        Parameters
        ----------
        xy_points : np.array
            Array of node coordinates with shape (N, 3), where each row contains the (x, y) coordinates of a point and the index of that coordinate pair.
        cells : np.array
            Array of quadrilateral cell connectivity with shape (M, 4), where each row contains the indices of the four corner points of a cell.
        inner_boundary_lines : np.array
            Array of inner boundary line segments with shape (K, 2), where each row contains the indices of the two endpoints of a line segment.
        outer_boundary_lines : np.array
            Array of outer boundary line segments with shape (L, 2), where each row contains the indices of the two endpoints of a line segment.
        mesh_filename : str, optional
            Name of the output SU2 mesh file. Default is "complex_streamline_SU2_mesh_gen.su2".

        Notes
        -----
        - The function assumes quadrilateral elements (shape type 9) for cells and line elements (shape type 3) for boundaries.
        - The mesh file is written in SU2 format, including node coordinates, element connectivity, and boundary markers for inner and outer boundaries.
        """
        num_dim = 2
        npoin = len(xy_points)
        nelem = len(cells)
        nmark = 2 #### change if we end up modeling multiple airfoils 
        #### create npoin array from the xy_points (each row contains 3 columns: x, y, index- these come directly from the xy_points array)
        npoin_array = xy_points 
        #### create a nelem array (each row contains 5 columns: 9, index1, index2, index3, index4) the 9 indicates the shape type which is a quadrilateral, the indices come directly from cells array
        nelem_array = np.zeros((nelem, 5), dtype=int)
        nelem_array[:, 0] = 9  # shape type for quadrilateral
        nelem_array[:, 1] = cells[:, 0].astype(int)  # first index
        nelem_array[:, 2] = cells[:, 1].astype(int)  # second index
        nelem_array[:, 3] = cells[:, 2].astype(int)  # third index
        nelem_array[:, 4] = cells[:, 3].astype(int)  # fourth index
        #### create a nmark array for the inner and outer boundary lines
        inner_marker_tag = "airfoil"
        outer_marker_tag = "farfield"
        inner_marker_elems = len(inner_boundary_lines)
        outer_marker_elems = len(outer_boundary_lines) 
        #### create a nmark_inner_array (each row contains 3 columns: 3, index1, index2) the 3 indicates the shape type which is a line, the indices come directly from inner_boundary_lines array
        nmark_inner_array = np.zeros((inner_marker_elems, 3), dtype=int)
        nmark_inner_array[:, 0] = 3  # shape type for line
        nmark_inner_array[:, 1] = inner_boundary_lines[:, 0].astype(int)  # first index
        nmark_inner_array[:, 2] = inner_boundary_lines[:, 1].astype(int)  # second index
        #### do the same for the outer boundary lines
        nmark_outer_array = np.zeros((outer_marker_elems, 3), dtype=int)
        nmark_outer_array[:, 0] = 3  # shape type for line
        nmark_outer_array[:, 1] = outer_boundary_lines[:, 0].astype(int)  # first index
        nmark_outer_array[:, 2] = outer_boundary_lines[:, 1].astype(int)  # second index
        #### write the mesh file in SU2 format
        with open(mesh_filename, "w", encoding="utf‑8") as f:
            #### Header: dimension of the problem
            f.write(f"NDIME= {num_dim}\n")
            #### Interior elements (quadrilaterals, VTK id = 9)
            f.write(f"NELEM= {nelem}\n")
            # include an explicit element index (optional in SU2, but matches
            # your reference file) ─ last integer on every line
            for idx, (etype, n1, n2, n3, n4) in enumerate(nelem_array):
                f.write(f"{etype}\t{n1}\t{n2}\t{n3}\t{n4}\t{idx}\n")
            #### Node coordinates
            f.write(f"NPOIN= {npoin}\n")
            # xy_points already stores (x, y, index)
            for x, y, idx in npoin_array:
                f.write(f"{x:.14f}\t{y:.14f}\t{int(idx)}\n")
            #### Boundary markers
            f.write(f"NMARK= {nmark}\n")
            #### inner boundary 
            f.write(f"MARKER_TAG= {inner_marker_tag}\n")
            f.write(f"MARKER_ELEMS= {inner_marker_elems}\n")
            for etype, n1, n2 in nmark_inner_array:
                f.write(f"{etype}\t{n1}\t{n2}\n")
            #### outer boundary 
            f.write(f"MARKER_TAG= {outer_marker_tag}\n")
            f.write(f"MARKER_ELEMS= {outer_marker_elems}\n")
            for etype, n1, n2 in nmark_outer_array:
                f.write(f"{etype}\t{n1}\t{n2}\n")

    # def create_grid(self, theta_aft: float = 2*np.pi):
    #     """Generates a polar grid in the chi plane."""
    #     num_theta_values = self.surface_nodes
    #     num_radial_values = self.radial_nodes
    #     self.theta_values = np.linspace(theta_aft, 2 * np.pi + theta_aft, num_theta_values)  # theta values from theta_aft to 2*pi + theta_aft
    #     r0 = self.cylinder_radius
    #     rN = self.radial_distance
    #     growth = (rN / r0) ** (1 / (num_radial_values - 1))
    #     self.r_values = r0 * growth ** np.arange(num_radial_values)
    #     index = 0
    #     points = []
    #     outer_boundary = []
    #     inner_boundary = []
    #     radial_line = []
    #     radial_lines = []
    #     # for i in range(num_theta_values - 1):
    #     for i in tqdm(range(num_theta_values-1), desc="Generating grid", unit="theta"):
    #         for j in range(num_radial_values):
    #             r = self.r_values[j]
    #             theta = self.theta_values[i]
    #             chi = r * np.exp(1j * theta)
    #             zeta = self.Chi_to_zeta(chi)
    #             z = self.zeta_to_z(zeta, self.epsilon)
    #             points.append([z.real, z.imag, index])
    #             if j == 0:
    #                 inner_boundary.append([z.real, z.imag, index])  # store the inner boundary point
    #             if j == num_radial_values - 1:
    #                 outer_boundary.append([z.real, z.imag, index])  # store the outer boundary point
    #             radial_line.append([z.real, z.imag, index])  # store the radial line point
    #             index += 1
    #         radial_lines.append(radial_line)  # store the radial line for this theta
    #         radial_line = []  # reset for the next theta
    #     self.radial_lines = np.array(radial_lines)  # store all radial lines
    #     self.outer_boundary_array = np.array(outer_boundary)
    #     self.inner_boundary_array = np.array(inner_boundary)
    #     self.all_points = np.array(points)
    #     return self.radial_lines, self.outer_boundary_array, self.inner_boundary_array, self.all_points

    def create_grid(self, theta_aft: float = 2*np.pi):
        """Generates a polar grid in the chi plane."""
        num_theta_values = self.surface_nodes
        num_radial_values = self.radial_nodes
        self.theta_values = np.linspace(theta_aft, 2 * np.pi + theta_aft, num_theta_values)
        # Compute radial spacing
        growth = self.radial_growth
        first_r_spacing = self.calc_deltaD_0(num_radial_values, growth, total_distance=self.radial_distance)
        # Compute r_values via cumulative sum of geometrically growing intervals
        intervals = first_r_spacing * growth ** np.arange(num_radial_values - 1)
        self.r_values = np.concatenate(([self.cylinder_radius],self.cylinder_radius + np.cumsum(intervals)))
        index = 0
        points = []
        outer_boundary = []
        inner_boundary = []
        radial_line = []
        radial_lines = []
        for i in tqdm(range(num_theta_values-1), desc="Generating grid", unit="theta"):
            for j in range(num_radial_values):
                r = self.r_values[j]
                theta = self.theta_values[i]
                chi = r * np.exp(1j * theta)
                zeta = self.Chi_to_zeta(chi)
                z = self.zeta_to_z(zeta, self.epsilon)
                points.append([z.real, z.imag, index])
                if j == 0:
                    inner_boundary.append([z.real, z.imag, index])  # store the inner boundary point
                if j == num_radial_values - 1:
                    outer_boundary.append([z.real, z.imag, index])  # store the outer boundary point
                radial_line.append([z.real, z.imag, index])  # store the radial line point
                index += 1
            radial_lines.append(radial_line)  # store the radial line for this theta
            radial_line = []  # reset for the next theta
        self.radial_lines = np.array(radial_lines)  # store all radial lines
        self.outer_boundary_array = np.array(outer_boundary)
        self.inner_boundary_array = np.array(inner_boundary)
        self.all_points = np.array(points)
        return self.radial_lines, self.outer_boundary_array, self.inner_boundary_array, self.all_points
    
    def chi_to_z_grid(self, chi_grid: np.array):
        """accepts a grid of points in the chi plane and transforms them to the z plane."""
        z_grid = np.zeros_like(chi_grid, dtype=float)
        # create an inner z_grid array which is the length of surface_nodes
        for i in range(len(chi_grid)):
            chi = chi_grid[i, 0] + 1j * chi_grid[i, 1]
            zeta = self.Chi_to_zeta(chi)
            z = self.zeta_to_z(zeta, self.epsilon)
            z_grid[i, 0] = z.real
            z_grid[i, 1] = z.imag
        return z_grid
    
    def create_boundary_lines(self, boundary_array: np.array):
        """"""
        length = len(boundary_array)
        boundary_lines = np.zeros((length, 2), dtype=int)
        # now loop through the boundary_array and create the boundary lines (connect each adjacent point)
        for i in range(length-1):
            boundary_lines[i] = [boundary_array[i, 2], boundary_array[i+1, 2]]  # use the index column to create the boundary lines
        # connect the last point to the first point
        boundary_lines[-1] = [boundary_array[-1, 2], boundary_array[0, 2]]  # connect the last point to the first point
        return boundary_lines
    
    def calc_radial_and_surface_nodes_based_on_doubling(self, stopper = 1000):
        """Takes in the surface and radial nodes and adjusts them based on doubling the cell count"""
        cell_count = (self.surface_nodes - 1) * (self.radial_nodes - 1) + (self.radial_nodes - 1)
        radial_and_surface_node_values = []
        radial_and_surface_node = self.surface_nodes
        while radial_and_surface_node <= stopper:
            radial_and_surface_node_values.append(radial_and_surface_node)
            root = (1 + 8*radial_and_surface_node*(radial_and_surface_node - 1))
            radial_and_surface_node = (1 + np.sqrt(root)) / 2
        return np.array(radial_and_surface_node_values, dtype=int)


if __name__ == "__main__":
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
    ## initialize the cylinder object
    mesh = mesh_generation_object()
    D_range_first = [0.0001,0.0009, 0.0001]
    D_values_first = hlp.list_to_range(D_range_first)
    D_range_second = [0.001, 0.009, 0.001]
    D_values_second = hlp.list_to_range(D_range_second)
    D_range_third = [0.01, 0.99, 0.01]
    D_values_third = hlp.list_to_range(D_range_third)
    D_values = np.concatenate((D_values_second, D_values_third), axis=0)
    # D_values = np.array([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    D_values = np.array([0.001])
    radial_and_surface_node_values = mesh.calc_radial_and_surface_nodes_based_on_doubling()
    print("radial_and_surface_node_values:", radial_and_surface_node_values)
    for rad_and_surf_val in radial_and_surface_node_values:
        # generate the grid for each D value
        for D in D_values:
            folder_name = "D_" + str(round(D, 4)) + "_grid_conv"
            mesh = mesh_generation_object()
            mesh.radial_nodes = rad_and_surf_val
            mesh.surface_nodes = rad_and_surf_val
            mesh.kutta_circulation = mesh.calc_circulation_J_airfoil()
            mesh.circulation = mesh.kutta_circulation
            # mesh.D is equal to D out to 4 decimal places
            mesh.D = np.round(D, 4)
            print("D value:", mesh.D)
            mesh.epsilon = mesh.calc_epsilon_from_D(mesh.D)
            #### the mesh filename includes the zeta_center, D value, angle of attack, radial nodes, radial distance, surface nodes, and radial growth
            theta_aft = 0.0#mesh.calculate_aft_stagnation_theta_in_Chi_from_Gamma(mesh.circulation)
            theta_forward = mesh.calculate_forward_stag_theta_in_Chi_from_aft_stag_theta(theta_aft)
            mesh.forward_stag, mesh.aft_stag = mesh.calculate_forward_and_aft_stag_locations_in_z(theta_aft)
            mesh.get_full_geometry_zeta(mesh.surface_nodes, theta_aft, theta_forward, 2*np.pi + theta_aft)
            mesh.get_full_geometry()
            mesh.shift_joukowski_cylinder()
            mesh.calc_unshifted_limits()
            mesh.radial_lines, mesh.outer_boundary_array, mesh.inner_boundary_array, mesh.all_points = mesh.create_grid(theta_aft)
            n_theta = len(mesh.radial_lines)
            n_r = len(mesh.radial_lines[0])
            num_cells = (n_theta - 1) * (n_r - 1) + (n_r - 1)
            mesh.calc_shifted_limits()
            mesh.mesh_filename = f"mesh_zeta0_{mesh.zeta_center.real:.3f}_{mesh.zeta_center.imag:.3f}_D_{mesh.D:.3f}_AoA_{np.rad2deg(mesh.angle_of_attack):.1f}deg_radial_distance_{mesh.radial_distance:.1f}_radial_growth_{mesh.radial_growth:.1f}_radial_nodes_{int(mesh.radial_nodes)}_surface_nodes_{int(mesh.surface_nodes)}_num_cells_{int(num_cells)}.su2"
            print("creating grid for D = ", mesh.D, ", zeta_center = ", mesh.zeta_center, ", surface_nodes = ", mesh.surface_nodes, ", radial_nodes = ", mesh.radial_nodes, ", radial_distance = ", mesh.radial_distance, ", radial_growth = ", mesh.radial_growth)
            cells = np.zeros((num_cells, 4, 3))  # 4 points per cell, 3 values per point (x, y, idx)
            cell_idx = 0
            for i in range(n_theta - 1):
                for j in range(n_r - 1):
                    cell_unsorted = np.array([
                        mesh.radial_lines[i][j],
                        mesh.radial_lines[i][j+1],
                        mesh.radial_lines[i+1][j+1],
                        mesh.radial_lines[i+1][j]
                    ])
                    cell_sorted = mesh.sort_points_ccw_by_centroid(cell_unsorted)
                    cells[cell_idx] = cell_sorted
                    cell_idx += 1

            for i in range(n_r - 1):
                cell_unsorted = np.array([
                    mesh.radial_lines[-1][i],
                    mesh.radial_lines[-1][i+1],
                    mesh.radial_lines[0][i+1],
                    mesh.radial_lines[0][i]
                ])
                cell_sorted = mesh.sort_points_ccw_by_centroid(cell_unsorted)
                cells[cell_idx] = cell_sorted
                cell_idx += 1

            # If you only need the indices:
            cells_array = np.array([
                (int(cell[0][2]), int(cell[1][2]), int(cell[2][2]), int(cell[3][2]))
                for cell in cells
            ])
            mesh.outer_boundary_array = mesh.sort_points_ccw_by_centroid(mesh.outer_boundary_array)
            mesh.inner_boundary_array = mesh.sort_points_ccw_by_centroid(mesh.inner_boundary_array)
            outer_boundary_lines = mesh.create_boundary_lines(mesh.outer_boundary_array) # these only include the indices
            inner_boundary_lines = mesh.create_boundary_lines(mesh.inner_boundary_array) # these only include the indices
            # convert
            points_of_entire_mesh = mesh.all_points
            shifted_xy_points_of_entire_mesh = mesh.shift_xy_points(mesh.all_points[:, :2])  # shift the x and y points
            shifted_xy_points_of_entire_mesh = np.column_stack((shifted_xy_points_of_entire_mesh, mesh.all_points[:, 2]))
            mesh.convert_grid_into_SU2_format(shifted_xy_points_of_entire_mesh, cells_array, inner_boundary_lines, outer_boundary_lines, mesh_filename=mesh.mesh_filename)
        # mesh.convert_grid_into_SU2_format(points_of_entire_mesh, cells_array, inner_boundary_lines, outer_boundary_lines, mesh_filename=mesh.mesh_filename)
            if mesh.is_visualize_mesh:
                # plt.show()
                import su2_viewer # type: ignore
                nodes, elements, boundaries = su2_viewer.read_su2(mesh.mesh_filename)
                su2_viewer.plot_su2_mesh(nodes, elements, boundaries)
