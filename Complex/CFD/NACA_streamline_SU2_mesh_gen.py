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
                # self.plot_geometry_settings()
                self.plot_shifted_joukowski_cylinder()
                if self.is_C_mesh:
                    self.C_mesh_wake = self.generate_C_mesh_wake()
                    self.C_mesh_above_wake, self.C_mesh_below_wake = self.generate_C_mesh_above_and_below_wake(self.C_mesh_wake)
                    self.C_mesh_body = self.generate_C_mesh_body()
                    self.combined_c_mesh = self.combine_C_mesh()
                    # self.plot_C_mesh()
                plt.xlim(-self.radial_distance-1, self.radial_distance+1)
                plt.ylim(-self.radial_distance-1, self.radial_distance+1)
                plt.xlabel(r"$x/c$")
                plt.ylabel(r"$y/c$")
                plt.gca().set_aspect('equal', adjustable='box')
                # self.plot_geometry()
                # plt.show()
                # plt.close()
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
                # self.plot_geometry()
                self.C_mesh_wake = self.generate_C_mesh_wake()
                self.C_mesh_above_wake, self.C_mesh_below_wake = self.generate_C_mesh_above_and_below_wake(self.C_mesh_wake)
                self.C_mesh_body = self.generate_C_mesh_body()
                self.combined_c_mesh = self.combine_C_mesh()
                # self.plot_C_mesh()
                plt.xlim(-self.radial_distance-1, self.radial_distance+1)
                plt.ylim(-self.radial_distance-1, self.radial_distance+1)
                plt.xlabel(r"$x/c$")
                plt.ylabel(r"$y/c$")
                plt.gca().set_aspect('equal', adjustable='box')
                # plt.show()

    def generate_C_mesh_wake(self):
        N = self.wake_nodes
        x_start, y_start = self.surface_mesh_points[0, 0], self.surface_mesh_points[0, 1]
        C_mesh_wake = np.zeros((N, 3)) # the third column is for the node index 
        x_end, y_end = x_start + self.radial_distance, y_start  # Correct end point
        deltaD_0 = self.calc_deltaD_0(N, self.radial_growth, (x_start, y_start), (x_end, y_start))
        # start a list for the outer boundary of the entire C_mesh 
        self.Outer_boundary_mesh = [] # this will be a list of tuples (x, y, index) for the outer boundary of the C mesh
        if self.radial_growth == 1.0:
            for i in range(N):
                C_mesh_wake[i, 0] = x_start + i * deltaD_0
                C_mesh_wake[i, 1] = y_start
                C_mesh_wake[i, 2] = i  # Assign node index
        else:
            C_mesh_wake[0, 0] = x_start
            C_mesh_wake[0, 1] = y_start
            C_mesh_wake[0, 2] = 0  # Assign node index
            for i in range(1, N):
                C_mesh_wake[i, 0] = C_mesh_wake[i-1, 0] + deltaD_0 * self.radial_growth**(i-1)
                C_mesh_wake[i, 1] = y_start
                C_mesh_wake[i, 2] = i
        # get the last value of the 3 column and save it
        # C_mesh_wake = C_mesh_wake[1:] # shape (N-1, 3), excluding the first point which is the trailing edge
        # # subtract each index by 1 to make it start from 0
        # C_mesh_wake[:, 2] -= 1  # Adjust indices to start from 0
        self.last_index_C_mesh_wake = C_mesh_wake[-1, 2] # this is the last index of the C mesh wake
        self.Outer_boundary_mesh.append(C_mesh_wake[-1]) # save the last point (x, y, index) of the C mesh wake to the outer boundary mesh
        # return everything but the first point of the C mesh wake
        return C_mesh_wake  # shape (N-1, 3), excluding the first point which is the trailing edge

    def generate_C_mesh_above_and_below_wake(self, C_mesh_wake: np.ndarray):
        """Generate a mesh above and below the wake mesh as (N*M, 3) arrays of points, not including the wake itself.
        Indices increment radially outward for each x."""
        N = len(C_mesh_wake)  # Number of points in the wake mesh
        M = self.radial_nodes
        x_coords = C_mesh_wake[:, 0]
        x_start, y_start = C_mesh_wake[0, 0], C_mesh_wake[0, 1]
        x_end, y_end = C_mesh_wake[0, 0], C_mesh_wake[0, 1] + self.radial_distance
        # Calculate deltaD_0 for the y-direction
        deltaD_0_upper = self.calc_deltaD_0(M + 1, self.radial_growth, (x_start, y_start), (x_end, y_end))
        deltaD_0_lower = self.calc_deltaD_0(M + 1, self.radial_growth, (x_start, y_start), (x_end, -y_end))
        # Compute y-coordinates for each row above/below the wake (skip the wake itself)
        y_upper_coords = np.zeros(M)
        y_lower_coords = np.zeros(M)
        y_prev_upper = y_start
        y_prev_lower = y_start
        for j in range(M):
            y_prev_upper += deltaD_0_upper * self.radial_growth**j
            y_prev_lower -= deltaD_0_lower * self.radial_growth**j
            y_upper_coords[j] = y_prev_upper
            y_lower_coords[j] = y_prev_lower
        # Allocate the (N*M, 3) arrays
        points_upper = np.zeros((N * M, 3))
        points_lower = np.zeros((N * M, 3))
        starting_index = self.last_index_C_mesh_wake + 1
        for i in range(N):  # loop through each x (wake node)
            for j in range(M):  # loop through each radial node
                idx = i * M + j
                points_upper[idx, 0] = x_coords[i]
                points_lower[idx, 0] = x_coords[i]
                points_upper[idx, 1] = y_upper_coords[j]
                points_lower[idx, 1] = y_lower_coords[j]
                points_upper[idx, 2] = starting_index + idx
                points_lower[idx, 2] = starting_index + N * M + idx
                if i == N - 1 and j != M - 1:
                    self.Outer_boundary_mesh.append((points_upper[idx, 0], points_upper[idx, 1], points_upper[idx, 2]))
                    self.Outer_boundary_mesh.append((points_lower[idx, 0], points_lower[idx, 1], points_lower[idx, 2]))
                elif j == M - 1:
                    self.Outer_boundary_mesh.append((points_upper[idx, 0], points_upper[idx, 1], points_upper[idx, 2]))
                    self.Outer_boundary_mesh.append((points_lower[idx, 0], points_lower[idx, 1], points_lower[idx, 2]))
        C_mesh_above_wake = points_upper  # shape (N*M, 3)
        C_mesh_below_wake = points_lower  # shape (N*M, 3)
        self.C_mesh_above_wake = C_mesh_above_wake
        self.C_mesh_below_wake = C_mesh_below_wake
        self.last_index_C_mesh_below_wake = C_mesh_below_wake[-1, 2]
        return C_mesh_above_wake, C_mesh_below_wake

    def generate_C_mesh_body(self):
        """Generate a C-type body mesh radiating outward from each surface point, starting with the first grown point (not on the surface), EXCLUDING theta=3pi/2 and pi/2."""
        # remove the last point of the surface mesh points, which is the trailing edge point
        self.surface_mesh_points = self.surface_mesh_points#[:-1]  # Exclude the trailing edge point
        N = len(self.surface_mesh_points)  # Number of surface points
        M = self.radial_nodes + 1   # Go out radial_nodes + 1 points
        radial_distance = self.radial_distance
        radial_growth = self.radial_growth
        trailing_edge_xy = self.surface_mesh_points[0, 0], self.surface_mesh_points[0, 1]
        # Shift surface points so trailing edge is at origin
        shifted_surface_points = self.surface_mesh_points - trailing_edge_xy
        # Generate N+2 thetas, then exclude the endpoints
        # Corrected line
        thetas_full = np.linspace(3*np.pi/2, np.pi/2, N+2)
        print("length of thetas_full:", len(thetas_full))
        thetas = thetas_full[1:-1]  # Now this is length N
        print("length of thetas:", len(thetas))
        # if the length of thetas is ODD now, recreate the surface with even number of points
        # Create outer mesh points
        outer_mesh_points = np.zeros((N, 2))
        print("length of shifted_surface_points:", len(shifted_surface_points))
        for i in range(N):
            outer_mesh_points[i, 0] = shifted_surface_points[i, 0] + radial_distance * np.cos(thetas[i])
            outer_mesh_points[i, 1] = shifted_surface_points[i, 1] + radial_distance * np.sin(thetas[i])
        # Now connect each point in the outer mesh to a corresponding point on the surface mesh
        C_mesh_body = np.zeros((N * M, 3))  # the third column is for the node index
        starting_index = self.last_index_C_mesh_below_wake + 1
        self.Inner_boundary_mesh = []
        for i in range(N):      # surface points
            x_start, y_start = shifted_surface_points[i, 0], shifted_surface_points[i, 1]
            x_end, y_end = outer_mesh_points[i, 0], outer_mesh_points[i, 1]
            deltaD_0 = self.calc_deltaD_0(M, radial_growth, (x_start, y_start), (x_end, y_end))
            for j in range(M):  # radial points
                idx = i * M + j
                if radial_growth == 1.0:
                    growth_distance = j * deltaD_0
                else:
                    growth_distance = deltaD_0 * (1 - radial_growth ** j) / (1 - radial_growth)
                C_mesh_body[idx, 0] = x_start + growth_distance * np.cos(thetas[i])
                C_mesh_body[idx, 1] = y_start + growth_distance * np.sin(thetas[i])
                C_mesh_body[idx, 2] = starting_index + idx
                if j == 0:
                    self.Inner_boundary_mesh.append((x_start + trailing_edge_xy[0], y_start + trailing_edge_xy[1], C_mesh_body[idx, 2]))
                elif j == M - 1:
                    self.Outer_boundary_mesh.append((C_mesh_body[idx, 0] + trailing_edge_xy[0], C_mesh_body[idx, 1] + trailing_edge_xy[1], C_mesh_body[idx, 2]))
        # Shift the points back to the original coordinate system
        C_mesh_body[:, 0] += trailing_edge_xy[0]
        C_mesh_body[:, 1] += trailing_edge_xy[1]
        self.last_index_C_mesh_body = C_mesh_body[-1, 2]
        self.Outer_boundary_mesh = np.array(self.Outer_boundary_mesh)
        self.Inner_boundary_mesh = np.array(self.Inner_boundary_mesh)
        return C_mesh_body
    
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

    def combine_C_mesh(self):
        """Combine the C mesh arrays into a single array.""" 
        self.C_mesh_wake = self.C_mesh_wake[1:]  # Exclude the first point which is the trailing edge
        # now adjust the indices of each C_mesh array by subtracting 1 from the third column
        self.C_mesh_wake[:, 2] -= 1  # Adjust indices to start from 0
        self.C_mesh_above_wake[:, 2] -= 1  # Adjust indices to start from 0
        self.C_mesh_below_wake[:, 2] -= 1  # Adjust indices to start from 0
        self.C_mesh_body[:, 2] -= 1  # Adjust indices to start from 0
        # shift the C_mesh_wake to the origin
        combined_C_mesh = np.vstack((self.C_mesh_wake, self.C_mesh_above_wake, self.C_mesh_below_wake, self.C_mesh_body)) # vstack combines the arrays vertically. So since each array is of shape (N, 2), the combined array will be of shape (4*N, 2)
        # remove the very first point of combined_C_mesh, which is the trailing edge point
        # same with the outer and inner boundary meshes
        self.Outer_boundary_mesh[:, 2] -= 1
        self.Inner_boundary_mesh[:, 2] -= 1
        self.combined_C_mesh = combined_C_mesh
        if self.is_export_mesh:
            output_file = "text_files/mesh_files/full_mesh/C_mesh_" + self.text_file_attribute + "_surface_" + str(self.surface_nodes) + "_wake_" + str(self.wake_nodes) + "_radial_" + str(self.radial_nodes) + ".txt"
            np.savetxt(output_file, combined_C_mesh, header="C mesh points (x, y) in the format: x y", fmt='%.6f')
        return combined_C_mesh

    def plot_C_mesh(self):
        """Plot the C mesh wake."""
        plt.scatter(self.C_mesh_wake[:, 0], self.C_mesh_wake[:, 1], color='orange', s = 0.1)
        plt.scatter(self.C_mesh_above_wake[:, 0], self.C_mesh_above_wake[:, 1], color='blue', s = 0.1, label='C mesh above wake')
        plt.scatter(self.C_mesh_below_wake[:, 0], self.C_mesh_below_wake[:, 1], color='green', s = 0.1, label='C mesh below wake')
        plt.scatter(self.C_mesh_body[:, 0], self.C_mesh_body[:, 1], color='black', s = 0.1, label='C mesh body')
        ## text the indices of the C mesh points
        for i in range(len(self.combined_c_mesh)):
            plt.text(self.combined_c_mesh[i, 0] + 0.05, self.combined_c_mesh[i, 1] - 0.05, str(int(self.combined_c_mesh[i, 2])), fontsize=8, color='black') #C_mesh_wake[i, 0] + 0.05 means that the text will be of size 8 and will be placed down and to the right of the point
        ## text the outer and inner boundary meshes
        # for i in range(len(self.Outer_boundary_mesh)):
            # plt.text(self.Outer_boundary_mesh[i, 0] + 0.05, self.Outer_boundary_mesh[i, 1] - 0.05, str(int(self.Outer_boundary_mesh[i, 2])), fontsize=8, color='red')
        # for j in range(len(self.Inner_boundary_mesh)):
            # plt.text(self.Inner_boundary_mesh[j, 0], self.Inner_boundary_mesh[j, 1], str(int(self.Inner_boundary_mesh[j, 2])), fontsize=8, color='purple')

    def create_SU2_mesh_file(self, filename: str = "mesh.su2", text_file_mesh: str = "text_files/mesh_files/full_mesh/C_mesh_Joukowski_cylinder_zeta0_over_R_is_-0.09+0.09i_surface_100_wake_100_radial_100.txt", triangle_or_quad: str = "quad"):
        """Create a SU2 mesh file from the combined C mesh."""
        ## if the text file does not exist, raise an error
        if not os.path.exists(text_file_mesh):
            raise FileNotFoundError(f"The text file {text_file_mesh} does not exist. Please generate the mesh first.")
        ## read in the text file mesh
        combined_C_mesh = self.combined_C_mesh#np.loadtxt(text_file_mesh)
        ## ignore the first row which is the header
        ## the number of columns from the text file is the NDIME for an SU2 file, so calculate the number of columns
        NDIME = combined_C_mesh.shape[1] - 1  # the last column is the index, so subtract 1
        ## Node coordinates
        NPOIN = len(self.combined_c_mesh)  # the number of points in the combined C mesh
        POINT_ARRAY = self.calc_SU2_point_array(combined_C_mesh)
        ## inner element connectivity
        NELEM = len(self.combined_c_mesh) - 1  # 1 less than the number of points in the combined C mesh
        # ELEMENT_ARRAY = self.calc_SU2_element_connectivity_array(combined_C_mesh, triangle_or_quad)
        ELEMENT_ARRAY = self.calc_SU2_element_connectivity_array(combined_C_mesh, "quad")
        ## Boundary elements
        NMARK = 2  # we have two boundary markers: outer and inner
        MARKER_TAGS = ["outer", "inner"]
        MARKER_ELEMS = [len(self.Outer_boundary_mesh), len(self.Inner_boundary_mesh)]  # number of elements in each boundary marker
        # BOUNDARY_ARRAY = self.calc_SU2_boundary_array(combined_C_mesh, triangle_or_quad) # 3 means that the boundaries are lines
        
    def calc_SU2_point_array(self, combined_C_mesh: np.ndarray):
        """Takes the point array and returns the SU2 Point array, which is a list of tuples (x, y, index) for each point. The columns are delimited by tabs."""
        point_array = combined_C_mesh[:, :2]  # take only the first two columns (x, y)
        points= "\n".join([f"{x:.12f}\t{y:.12f}" for x, y in point_array])  # format each point as "x\ty"
        return points
    
    def calc_SU2_element_connectivity_array(self, combined_C_mesh: np.ndarray, triangle_or_quad: str):
        """Calculate the SU2 element connectivity array based on the combined C mesh and the type of elements (triangle or quad)."""
        if triangle_or_quad == "quad":
            # For quadrilateral elements, we need to connect points in groups of 4
            first_column_is_always = 9 # 9 is the SU2 code for quadrilateral elements
        elements = []
        # loop through the wake mesh points, make quads above and below the wake based on how we generated the C mesh. ALL of the C mesh quads should be connected counterclockwise, so we can just connect the points in pairs.
        i = 0
        first = self.C_mesh_body[0]
        second = self.C_mesh_wake[i]
        third_above = self.C_mesh_above_wake[(i+1)*(i + self.radial_nodes)]
        fourth_above = self.C_mesh_above_wake[i]
        third_below = self.C_mesh_below_wake[(i+1)*(i + self.radial_nodes)]
        fourth_below = self.C_mesh_below_wake[i]
        above_arrays = np.array([first, second, third_above, fourth_above])
        below_arrays = np.array([first, second, third_below, fourth_below])
        above_ordered = self.order_points_counterclockwise(above_arrays)
        below_ordered = self.order_points_counterclockwise(below_arrays)
        quad_above = f"{first_column_is_always}\t{above_ordered[0]}\t{above_ordered[1]}\t{above_ordered[2]}\t{above_ordered[3]}"
        quad_below = f"{first_column_is_always}\t{below_ordered[0]}\t{below_ordered[1]}\t{below_ordered[2]}\t{below_ordered[3]}"
        elements.append(quad_above)
        elements.append(quad_below)
        # left_upper is the first point in the last radial row of the C mesh body which is len(self.C_mesh_body) - 1
        left_upper = self.C_mesh_body[len(self.C_mesh_body) - self.radial_nodes] 
        left_center_upper = self.C_mesh_body[0 + len(self.C_mesh_body) - self.radial_nodes - 1] 
        left_center_lower = self.C_mesh_body[0 + self.radial_nodes + 1] 
        # left_lower is the first point in the last radial row of the C mesh body 
        left_lower = self.C_mesh_body[0 + 1] 
        left_above_ordered = self.order_points_counterclockwise(np.array([first, fourth_above, left_upper, left_center_upper]))
        left_below_ordered = self.order_points_counterclockwise(np.array([first, fourth_below, left_center_lower, left_lower]))
        quad_left_above = f"{first_column_is_always}\t{left_above_ordered[0]}\t{left_above_ordered[1]}\t{left_above_ordered[2]}\t{left_above_ordered[3]}"
        quad_left_below = f"{first_column_is_always}\t{left_below_ordered[0]}\t{left_below_ordered[1]}\t{left_below_ordered[2]}\t{left_below_ordered[3]}"
        print(f"{quad_left_below}")
        elements.append(quad_left_above)
        elements.append(quad_left_below)
        # Get coordinates from indices for plotting
        coords_above = combined_C_mesh[above_ordered, :2]
        coords_below = combined_C_mesh[below_ordered, :2]
        plt.plot(
            np.append(coords_above[:, 0], coords_above[0, 0]),
            np.append(coords_above[:, 1], coords_above[0, 1]),
            color='red', linewidth=0.5
        )
        plt.plot(
            np.append(coords_below[:, 0], coords_below[0, 0]),
            np.append(coords_below[:, 1], coords_below[0, 1]),
            color='red', linewidth=0.5
        )
        coords_left_above = combined_C_mesh[left_above_ordered, :2]
        coords_left_below = combined_C_mesh[left_below_ordered, :2]
        plt.plot(
            np.append(coords_left_above[:, 0], coords_left_above[0, 0]),
            np.append(coords_left_above[:, 1], coords_left_above[0, 1]),
            color='red', linewidth=0.5
        )
        plt.plot(
            np.append(coords_left_below[:, 0], coords_left_below[0, 0]),
            np.append(coords_left_below[:, 1], coords_left_below[0, 1]),
            color='red', linewidth=0.5
        )
        for i in range(1, len(self.C_mesh_wake)):
                first = self.C_mesh_wake[i-1]
                second = self.C_mesh_wake[i]
                third_above = self.C_mesh_above_wake[(i+1)*self.radial_nodes]
                fourth_above = self.C_mesh_above_wake[(i)*self.radial_nodes]
                third_below = self.C_mesh_below_wake[((i+1)*self.radial_nodes)]
                fourth_below = self.C_mesh_below_wake[(i)*self.radial_nodes]
                above_arrays = np.array([first, second, third_above, fourth_above])
                below_arrays = np.array([first, second, third_below, fourth_below])
                above_ordered = self.order_points_counterclockwise(above_arrays)
                below_ordered = self.order_points_counterclockwise(below_arrays)
                quad_above = f"{first_column_is_always}\t{above_ordered[0]}\t{above_ordered[1]}\t{above_ordered[2]}\t{above_ordered[3]}"
                quad_below = f"{first_column_is_always}\t{below_ordered[0]}\t{below_ordered[1]}\t{below_ordered[2]}\t{below_ordered[3]}"
                elements.append(quad_above)
                elements.append(quad_below)
                # Get coordinates from indices for plotting
                coords_above = combined_C_mesh[above_ordered, :2]
                coords_below = combined_C_mesh[below_ordered, :2]
                plt.plot(
                    np.append(coords_above[:, 0], coords_above[0, 0]),
                    np.append(coords_above[:, 1], coords_above[0, 1]),
                    color='red', linewidth=0.5
                )
                plt.plot(
                    np.append(coords_below[:, 0], coords_below[0, 0]),
                    np.append(coords_below[:, 1], coords_below[0, 1]),
                    color='red', linewidth=0.5
                )
        # now loop through the upper_wake and lower_wake meshes, and connect the points in pairs to make quads connecting the C mesh body to the C mesh above and below wake
        for i in range(self.radial_nodes-1):
            ## if i <= radial nodes, then each point will have two quads, one to the left which connects to the first radial row in the C mesh body, and one to the right which connects to the next radial row in the C_mesh_wake_above or C_mesh_wake_below
            left_first_above = self.C_mesh_above_wake[i]
            # print("left_first_above index:", left_first_above)
            left_first_below = self.C_mesh_below_wake[i]
            left_second_above = self.C_mesh_above_wake[i + 1]
            # print("left_second_above index:", left_second_above)
            left_second_below = self.C_mesh_below_wake[i + 1]
            ## third above is the first point in the last radial row of the C mesh body which is len(self.C_mesh_body) - 1
            left_third_above = self.C_mesh_body[len(self.C_mesh_body) - self.radial_nodes + i] 
            # print("left_third_above index:", left_third_above)
            left_third_below = self.C_mesh_body[i + 1]
            left_fourth_above = self.C_mesh_body[len(self.C_mesh_body) - self.radial_nodes + i + 1]
            # print("left_fourth_above index:", left_fourth_above)
            left_fourth_below = self.C_mesh_body[i + 2]
            left_above_arrays = np.array([left_first_above, left_second_above, left_third_above, left_fourth_above])
            left_below_arrays = np.array([left_first_below, left_second_below, left_third_below, left_fourth_below])
            left_above_ordered = self.order_points_counterclockwise(left_above_arrays)
            left_below_ordered = self.order_points_counterclockwise(left_below_arrays)
            quad_left_above = f"{first_column_is_always}\t{left_above_ordered[0]}\t{left_above_ordered[1]}\t{left_above_ordered[2]}\t{left_above_ordered[3]}"
            quad_left_below = f"{first_column_is_always}\t{left_below_ordered[0]}\t{left_below_ordered[1]}\t{left_below_ordered[2]}\t{left_below_ordered[3]}"
            elements.append(quad_left_above)
            elements.append(quad_left_below)
            # Get coordinates from indices for plotting
            coords_left_above = combined_C_mesh[left_above_ordered, :2]
            coords_left_below = combined_C_mesh[left_below_ordered, :2]
            plt.plot(
                np.append(coords_left_above[:, 0], coords_left_above[0, 0]),
                np.append(coords_left_above[:, 1], coords_left_above[0, 1]),
                color='red', linewidth=0.5
            )
            plt.plot(
                np.append(coords_left_below[:, 0], coords_left_below[0, 0]),
                np.append(coords_left_below[:, 1], coords_left_below[0, 1]),
                color='red', linewidth=0.5
            )
        # now, loop through the C_mesh_wake to make quads through the C_mesh_wake_above and C_mesh_wake_below
        for i in range(self.radial_nodes - 1):
            for j in range(self.wake_nodes - 1):
                idx = i * (self.wake_nodes) + j 
                first_above = self.C_mesh_above_wake[idx]
                second_above = self.C_mesh_above_wake[idx + self.radial_nodes]
                third_above = self.C_mesh_above_wake[idx + self.radial_nodes + 1]
                fourth_above = self.C_mesh_above_wake[idx + 1]
                first_below = self.C_mesh_below_wake[idx]
                second_below = self.C_mesh_below_wake[idx + self.radial_nodes]
                third_below = self.C_mesh_below_wake[idx + self.radial_nodes + 1]
                fourth_below = self.C_mesh_below_wake[idx + 1]
                above_arrays = np.array([first_above, second_above, third_above, fourth_above])
                below_arrays = np.array([first_below, second_below, third_below, fourth_below])
                above_ordered = self.order_points_counterclockwise(above_arrays)
                below_ordered = self.order_points_counterclockwise(below_arrays)
                quad_above = f"{first_column_is_always}\t{above_ordered[0]}\t{above_ordered[1]}\t{above_ordered[2]}\t{above_ordered[3]}"
                quad_below = f"{first_column_is_always}\t{below_ordered[0]}\t{below_ordered[1]}\t{below_ordered[2]}\t{below_ordered[3]}"
                elements.append(quad_above)
                elements.append(quad_below)
                # Get coordinates from indices for plotting
                coords_above = combined_C_mesh[above_ordered, :2]
                coords_below = combined_C_mesh[below_ordered, :2]
                plt.plot(
                    np.append(coords_above[:, 0], coords_above[0, 0]),
                    np.append(coords_above[:, 1], coords_above[0, 1]),
                    color='red', linewidth=0.5
                )
                plt.plot(
                    np.append(coords_below[:, 0], coords_below[0, 0]),
                    np.append(coords_below[:, 1], coords_below[0, 1]),
                    color='red', linewidth=0.5
                )
        # now loop through the C_mesh_body to make quads, and one tri
        num_radial_body = self.radial_nodes + 1  # Number of nodes along the "radial" direction of the body
        num_surface_body = self.surface_nodes - 1  # Number of nodes along the "surface" of the body (after removing trailing edge)

        for i in range(num_radial_body - 1):  # loop through rows
            for j in range(num_surface_body - 1):  # loop through columns
                print(f"DEBUG: i = {i}, j = {j}")
                idx = i * num_surface_body + j
                first  = self.C_mesh_body[idx]  # bottom-left
                second = self.C_mesh_body[idx + 1]  # bottom-right
                third  = self.C_mesh_body[idx + num_surface_body + 1]  # top-right
                fourth = self.C_mesh_body[idx + num_surface_body]  # top-left

                arrays = np.array([first, second, third, fourth])
                ordered = self.order_points_counterclockwise(arrays)
                quad = f"{first_column_is_always}\t{ordered[0]}\t{ordered[1]}\t{ordered[2]}\t{ordered[3]}"
                elements.append(quad)

                coords = combined_C_mesh[ordered, :2]
                plt.plot(
                    np.append(coords[:, 0], coords[0, 0]),
                    np.append(coords[:, 1], coords[0, 1]),
                    color='red', linewidth=0.5
                )

        # Now, add the single triangle at the very end (last column, top-most row)
        tri_p1_idx = (num_radial_body - 2) * num_surface_body + (num_surface_body - 1)
        tri_p2_idx = (num_radial_body - 1) * num_surface_body + (num_surface_body - 1)
        tri_p3_idx = (num_radial_body - 1) * num_surface_body + (num_surface_body - 2)
        tri_p1 = self.C_mesh_body[tri_p1_idx]
        tri_p2 = self.C_mesh_body[tri_p2_idx]
        tri_p3 = self.C_mesh_body[tri_p3_idx]

        # For the triangle at the end
        tri_arrays = np.array([tri_p1, tri_p2, tri_p3])
        if tri_arrays.shape[0] == 4:
            tri_ordered = self.order_points_counterclockwise(tri_arrays)
        elif tri_arrays.shape[0] == 3:
            # For triangles, just use the indices as-is (or implement a CCW order if needed)
            tri_ordered = tri_arrays[:, 2].astype(int)
        else:
            raise ValueError("Triangle/quad array must have 3 or 4 points.")

        first_column_is_always_tri = '3'
        triangle = f"{first_column_is_always_tri}\t{tri_ordered[0]}\t{tri_ordered[1]}\t{tri_ordered[2]}"
        elements.append(triangle)

            
    def order_points_counterclockwise(self, arr: np.ndarray) -> np.ndarray:
        """
        Order four unique points counterclockwise starting from the bottom-left.
        Each row in `arr` is [x, y, index].
        Returns:
            A 1D numpy array of 4 unique indices in CCW order: [BL, BR, TR, TL]
        """
        if arr.shape != (4, 3):
            raise ValueError("Input must be exactly four [x, y, index] points.")
        coords = arr[:, :2]
        indices = arr[:, 2].astype(int)

        # Sort by y, then x
        sorted_idx = np.lexsort((coords[:, 0], coords[:, 1]))
        sorted_coords = coords[sorted_idx]
        sorted_indices = indices[sorted_idx]

        # First two are bottom (lowest y), last two are top (highest y)
        bottom = sorted_coords[:2]
        bottom_indices = sorted_indices[:2]
        top = sorted_coords[2:]
        top_indices = sorted_indices[2:]

        # For bottom, left is min x, right is max x
        if bottom[0, 0] < bottom[1, 0]:
            bl, br = bottom_indices[0], bottom_indices[1]
        else:
            bl, br = bottom_indices[1], bottom_indices[0]

        # For top, left is min x, right is max x
        if top[0, 0] < top[1, 0]:
            tl, tr = top_indices[0], top_indices[1]
        else:
            tl, tr = top_indices[1], top_indices[0]

        return np.array([bl, br, tr, tl])
        

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
    # mesh.create_SU2_mesh_file(filename="mesh.su2", text_file_mesh="text_files/mesh_files/full_mesh/C_mesh_Joukowski_cylinder_zeta0_over_R_is_-0.09+0.0i_surface_6_wake_6_radial_6.txt", triangle_or_quad="quad")
    # mesh.plot_C_mesh()
    mesh.plot()
    # plt.show()
    # plt.close()
    # print("Mesh generation object initialized.")




    