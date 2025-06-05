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
import matplotlib.patches as patches  # type: ignore

class mesh_generation_object(cylinder, vort_panel):
    """This class uses conformal mapping or NACA airfoil coordinates to generate a mesh for a given domain."""
    def __init__(self, mesh_json_file: str = "mesh_generation.json", vort_panel_json_file: str = "vortex_panel_input.json", joukowski_cylinder_json_file: str = "Joukowski_Cylinder.json"):
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
            self.is_conformal_mapping = hlp.parse_dictionary_or_return_default(input, ["mesh", "is_conformal_mapping"], True)
            self.surface_nodes = hlp.parse_dictionary_or_return_default(input, ["mesh", "general_settings", "surface_nodes"], 100)
            self.wake_nodes = hlp.parse_dictionary_or_return_default(input, ["mesh", "general_settings", "wake_nodes"], 100)
            self.radial_nodes = hlp.parse_dictionary_or_return_default(input, ["mesh", "general_settings", "radial_nodes"], 100)
            self.radial_distance = hlp.parse_dictionary_or_return_default(input, ["mesh", "general_settings", "radial_distance"], 10)
            self.radial_growth = hlp.parse_dictionary_or_return_default(input, ["mesh", "general_settings", "radial_growth"], 1.0)
            # set the conformal mapping parameters if is_conformal_mapping is True
            if self.is_conformal_mapping:
                cylinder.__init__(self, self.joukowski_cylinder_json_file) # this initializes the cylinder with the given JSON file
                self.text_file_attribute = "Joukowski_cylinder_zeta0_over_R_is_" + str(self.zeta_center.real) + "+" + str(self.zeta_center.imag) + "i"
                self.type = "cylinder" # default type is cylinder
                self.transformation_type = "joukowski" # default transformation type is Joukowski
                self.use_shape_parameter_D = True
                self.is_C_mesh = hlp.parse_dictionary_or_return_default(input, ["mesh", "conformal_mapping", "is_C_mesh"], True)
                self.cylinder_radius = hlp.parse_dictionary_or_return_default(input, ["mesh","conformal_mapping", "cylinder_radius"], 1.0)
                self.D = hlp.parse_dictionary_or_return_default(input, ["mesh","conformal_mapping", "shape_parameter_D"], 1.0)
                self.epsilon = self.calc_epsilon_from_D(self.D)
                self.kutta_circulation = self.calc_circulation_J_airfoil()
                zeta_center = hlp.parse_dictionary_or_return_default(input, ["mesh","conformal_mapping", "zeta_0"], [-0.09, 0.09])
                self.zeta_center = (zeta_center[0] + 1j*zeta_center[1])/self.cylinder_radius
                self.angle_of_attack = hlp.parse_dictionary_or_return_default(input, ["operating", "angle_of_attack[deg]"], 5.0)
                self.freestream_velocity = hlp.parse_dictionary_or_return_default(input, ["operating", "freestream_velocity"], 1.0)
                self.get_full_geometry_zeta(self.surface_nodes)
                self.get_full_geometry()
                self.shift_joukowski_cylinder()
                self.surface_mesh_points = self.shifted_combined_coords
                self.plot_geometry_settings()
                self.plot_shifted_joukowski_cylinder()
                if self.is_C_mesh:
                    self.C_mesh_wake = self.generate_C_mesh_wake()
                    self.C_mesh_above_wake, self.C_mesh_below_wake = self.generate_C_mesh_above_and_below_wake(self.C_mesh_wake)
                    self.C_mesh_body = self.generate_C_mesh_body()
                    self.combined_c_mesh = self.combine_C_mesh(self.C_mesh_wake, self.C_mesh_above_wake, self.C_mesh_below_wake, self.C_mesh_body)
                    self.plot_C_mesh(self.C_mesh_wake)
                plt.xlim(-self.radial_distance-1, self.radial_distance+1)
                plt.ylim(-self.radial_distance-1, self.radial_distance+1)
                plt.xlabel(r"$x/c$")
                plt.ylabel(r"$y/c$")
                plt.gca().set_aspect('equal', adjustable='box')
                # self.plot_geometry()
                plt.show()
                plt.close()
            else:
                vort_panel.__init__(self, self.vort_panel_json_file) # this initializes the vortex panel with the given JSON file
                self.type = "cylinder" # default type is cylinder
                self.is_text_file = False
                self.export_geometry = True
                self.Airfoil_type = hlp.parse_dictionary_or_return_default(input, ["mesh", "NACA_4_digit", "airfoil"], "0012")
                self.text_file_attribute = "NACA_" + self.Airfoil_type
                self.trailing_edge_condition = hlp.parse_dictionary_or_return_default(input, ["mesh", "NACA_4_digit", "trailing_edge"], "closed")
                self.n_geom_points = hlp.parse_dictionary_or_return_default(input, ["mesh", "general_settings", "surface_nodes"], 100)
                self.original_alpha = hlp.parse_dictionary_or_return_default(input, ["operating", "angle_of_attack[deg]"], 5.0)
                self.num_points_is_even()
                self.define_Airfoil()
                self.get_full_geometry_vortex()
                self.surface_mesh_points = self.surface_points
                self.plot_geometry()
                self.C_mesh_wake = self.generate_C_mesh_wake()
                self.C_mesh_above_wake, self.C_mesh_below_wake = self.generate_C_mesh_above_and_below_wake(self.C_mesh_wake)
                self.C_mesh_body = self.generate_C_mesh_body()
                self.combined_c_mesh = self.combine_C_mesh(self.C_mesh_wake, self.C_mesh_above_wake, self.C_mesh_below_wake, self.C_mesh_body)
                self.plot_C_mesh(self.C_mesh_wake)
                plt.xlim(-self.radial_distance-1, self.radial_distance+1)
                plt.ylim(-self.radial_distance-1, self.radial_distance+1)
                plt.xlabel(r"$x/c$")
                plt.ylabel(r"$y/c$")
                plt.gca().set_aspect('equal', adjustable='box')
                plt.show()

    def generate_C_mesh_wake(self):
        N = self.wake_nodes
        x_start, y_start = self.surface_mesh_points[0, 0], self.surface_mesh_points[0, 1]
        C_mesh_wake = np.zeros((N, 2))
        x_end, y_end = x_start + self.radial_distance, y_start  # Correct end point
        deltaD_0 = self.calc_deltaD_0(N, self.radial_growth, (x_start, y_start), (x_end, y_start))
        if self.radial_growth == 1.0:
            for i in range(N):
                C_mesh_wake[i, 0] = x_start + i * deltaD_0
                C_mesh_wake[i, 1] = y_start
        else:
            C_mesh_wake[0, 0] = x_start
            C_mesh_wake[0, 1] = y_start
            for i in range(1, N):
                C_mesh_wake[i, 0] = C_mesh_wake[i-1, 0] + deltaD_0 * self.radial_growth**(i-1)
                C_mesh_wake[i, 1] = y_start
        return C_mesh_wake # # Exclude the first point to avoid duplication with the surface mesh
    
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
    
    def generate_C_mesh_above_and_below_wake(self, C_mesh_wake: np.ndarray):
        """Generate a mesh above the wake mesh as a (N*M, 2) array of points, not including the wake itself."""
        N = len(C_mesh_wake)  # Number of points in the wake mesh
        M = self.radial_nodes
        x_coords = C_mesh_wake[:, 0]
        x_start, y_start = C_mesh_wake[0, 0], C_mesh_wake[0, 1]
        x_end, y_end = C_mesh_wake[0,0], C_mesh_wake[0,1] + self.radial_distance  # Correct end point for y-direction

        # Calculate deltaD_0 for the y-direction
        deltaD_0_upper= self.calc_deltaD_0(M + 1, self.radial_growth, (x_start, y_start), (x_end, y_end))  # For the upper mesh
        deltaD_0_lower = self.calc_deltaD_0(M + 1, self.radial_growth, (x_start, y_start), (x_end, -y_end))  # For the lower mesh
        # Compute y-coordinates for each row above the wake (skip the wake itself)
        y_upper_coords = np.zeros(M)
        y_lower_coords = np.zeros(M)
        y_prev_upper = y_start
        y_prev_lower = y_start
        for j in range(M):
            y_prev_upper +=  deltaD_0_upper* self.radial_growth**j
            y_prev_lower -= deltaD_0_lower * self.radial_growth**j
            y_upper_coords[j] = y_prev_upper
            y_lower_coords[j] = y_prev_lower
        # Allocate the (N*M, 2) array
        points_upper = np.zeros((N * M, 2))
        points_lower = np.zeros((N * M, 2))
        for j in range(M):
            for i in range(N):
                idx = j * N + i
                points_upper[idx, 0] = x_coords[i]
                points_lower[idx, 0] = x_coords[i]
                points_upper[idx, 1] = y_upper_coords[j]
                points_lower[idx, 1] = y_lower_coords[j]
        C_mesh_above_wake = points_upper  # shape (N*M, 2)
        C_mesh_below_wake = points_lower  # shape (N*M, 2)
        self.C_mesh_above_wake = C_mesh_above_wake
        self.C_mesh_below_wake = C_mesh_below_wake
        return C_mesh_above_wake, C_mesh_below_wake

    def generate_C_mesh_body(self):
        """Generate a C-type body mesh radiating outward from each surface point, starting with the first grown point (not on the surface)."""
        N = self.surface_nodes 
        M = self.radial_nodes
        radial_distance = self.radial_distance
        radial_growth = self.radial_growth
        trailing_edge_xy = self.surface_mesh_points[0, 0], self.surface_mesh_points[0, 1]
        # Shift surface points so trailing edge is at origin
        shifted_surface_points = self.surface_mesh_points - trailing_edge_xy
        # thetas = np.linspace(3*np.pi/2, np.pi/2, N)
        thetas = self.generate_clustered_thetas(N, radial_growth, theta_start=3*np.pi/2, theta_end=np.pi/2)
        # remove the first and last theta to avoid duplication with the surface mesh
        # thetas = thetas[1:-1] # thetas should be of size N-2
        deltaD_0 = self.calc_deltaD_0(M + 1, radial_growth, (0.0, 0.0), (0.0, radial_distance))
        # print("deltaD_0:", deltaD_0)
        # create outer mesh points that go out to radial_distance and are evenly spaced in theta
        outer_mesh_points = np.zeros((N, 2))
        for i in range(N):
            outer_mesh_points[i, 0] = shifted_surface_points[0, 0] + radial_distance * np.cos(thetas[i])
            outer_mesh_points[i, 1] = shifted_surface_points[0, 1] + radial_distance * np.sin(thetas[i])
        # Now connect each point in the outer mesh to a corresponding point on the surface mesh (they're the same size, so just match indices)
        C_mesh_body = np.zeros((N * M, 2))  # Allocate the mesh array
        # loop through each surface node first
        for i in range(N): # looping through each surface point
            for j in range(M): # looping through each radial point
                x_start, y_start = shifted_surface_points[i, 0], shifted_surface_points[i, 1]
                x_end, y_end = outer_mesh_points[i, 0], outer_mesh_points[i, 1]
                # Calculate deltaD_0 for the radial growth
                deltaD_0 = self.calc_deltaD_0(M+1, radial_growth, (x_start, y_start), (x_end, y_end))
                # Compute the points for each row in the radial direction
                idx = j * N + i
                # growth_distance = deltaD_0*radial_growth**j
                growth_distance = deltaD_0 * (1 - radial_growth ** (j + 1)) / (1 - radial_growth)
                C_mesh_body[idx, 0] = shifted_surface_points[i, 0] + growth_distance * np.cos(thetas[i])
                C_mesh_body[idx, 1] = shifted_surface_points[i, 1] + growth_distance * np.sin(thetas[i])
        # Ensure the last point in each row matches the outer mesh point
        # Shift the points back to the original coordinate system
        C_mesh_body[:, 0] += trailing_edge_xy[0]
        C_mesh_body[:, 1] += trailing_edge_xy[1]
        # C_mesh_body = outer_mesh_points.copy()
        return C_mesh_body
    
    def generate_clustered_thetas(self, N, radial_growth, theta_start=3*np.pi/2, theta_end=np.pi/2):
        """
        Generate `N` angles between theta_start and theta_end, clustered at both ends
        using a geometric progression controlled by `radial_growth`.
        """
        half_N = N // 2

        total_theta_span = abs(theta_start - theta_end)
        delta_theta_0 = total_theta_span / ((1 - radial_growth ** half_N) / (1 - radial_growth)) / 2

        # Build one half of the cumulative angular spacing
        offsets = np.array([delta_theta_0 * (1 - radial_growth**(i + 1)) / (1 - radial_growth) for i in range(half_N)])

        thetas = np.zeros(N)
        for i in range(half_N):
            thetas[i] = theta_start - offsets[i]  # from 3pi/2
            thetas[N - 1 - i] = theta_end + offsets[i]  # from pi/2
        # remove the middle point if N is odd
        if N % 2 != 0:
            thetas = np.delete(thetas, half_N)
            print("N is odd, removing the middle point from thetas.")

        return thetas
    
    def combine_C_mesh(self, C_mesh_wake: np.ndarray, C_mesh_above_wake: np.ndarray, C_mesh_below_wake: np.ndarray, C_mesh_body: np.ndarray):
        """Combine the C mesh arrays into a single array.""" 
        combined_C_mesh = np.vstack((C_mesh_wake, C_mesh_above_wake, C_mesh_below_wake, C_mesh_body)) # vstack combines the arrays vertically. So since each array is of shape (N, 2), the combined array will be of shape (4*N, 2)
        # save the combined mesh as a text file if is_export_mesh is True 
        if self.is_export_mesh:
            output_file = "text_files/mesh_files/full_mesh/C_mesh_" + self.text_file_attribute + "_surface_" + str(self.surface_nodes) + "_wake_" + str(self.wake_nodes) + "_radial_" + str(self.radial_nodes) + ".txt"
            np.savetxt(output_file, combined_C_mesh, header="C mesh points (x, y) in the format: x y", fmt='%.6f')
        return combined_C_mesh

    def plot_C_mesh(self, C_mesh_wake: np.ndarray):
        """Plot the C mesh wake."""
        plt.scatter(C_mesh_wake[:, 0], C_mesh_wake[:, 1], color='orange', s = 0.1)
        plt.scatter(self.C_mesh_above_wake[:, 0], self.C_mesh_above_wake[:, 1], color='blue', s = 0.1, label='C mesh above wake')
        plt.scatter(self.C_mesh_below_wake[:, 0], self.C_mesh_below_wake[:, 1], color='green', s = 0.1, label='C mesh below wake')
        plt.scatter(self.C_mesh_body[:, 0], self.C_mesh_body[:, 1], color='black', s = 0.1, label='C mesh body')

    def shift_from_trailing_edge_to_origin(self, origin_xy: tuple[float, float], point: tuple[float, float]):
        """Shift the points by the origin amount to get it back in terms of the actual coordinate system."""
        shifted_points = point[0] + origin_xy[0], point[1] + origin_xy[1]
        return shifted_points
    
    def shift_from_origin_to_trailing_edge(self, origin_xy: tuple[float, float], point: tuple[float, float]):
        """Shift the points by the origin amount to get it back in terms of the actual coordinate system."""
        shifted_points = point[0] - origin_xy[0], point[1] - origin_xy[1]
        return shifted_points
    

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
    mesh = mesh_generation_object()
    # print("Mesh generation object initialized.")
    # mesh_gen.shift_joukowski_cylinder()




    