import json
import numpy as np # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from tqdm import tqdm # type: ignore
from multiprocessing import Pool # type: ignore
import helper as hlp

class potential_flow_object:
    """This is a class that has functions that can be used for a  variety of potential flow objects (ie. cylinder, airfoil, etc.)"""
    def __init__(self, json_file):
        self.json_file = json_file
        self.parse_json()
        
    def parse_json(self):
        """This function reads the json file and stores the data in a dictionary"""
        with open(self.json_file, 'r') as json_handle:
            input = json.load(json_handle)
            self.plot_x_lower_lim = input["plot"]["x_lower_limit"]/self.original_cylinder_radius#self.cylinder_radius # this is the lower limit of the plot in x direction.
            self.plot_x_upper_lim = input["plot"]["x_upper_limit"]/self.original_cylinder_radius#self.cylinder_radius # this is the lower limit of the plot in x direction.
            self.plot_x_start = input["plot"]["x_start"]/self.original_cylinder_radius#self.cylinder_radius # this is the lower limit of the plot in x direction.
            self.plot_delta_s = input["plot"]["delta_s"]
            self.plot_n_lines = input["plot"]["n_lines"]
            self.plot_delta_y = input["plot"]["delta_y"]/self.original_cylinder_radius#self.cylinder_radius # this is the spacing between streamlines in the y direction.
            if self.type == "cylinder":
                self.x_leading_edge = (self.zeta_center.real - self.cylinder_radius)/self.cylinder_radius
                self.x_trailing_edge = (self.zeta_center.real + self.cylinder_radius)/self.cylinder_radius
                zeta_leading_edge = (self.x_leading_edge + 1j*self.zeta_center.imag)/self.cylinder_radius
                zeta_trailing_edge = (self.x_trailing_edge + 1j*self.zeta_center.imag)/self.cylinder_radius
                self.z_leading_edge = self.zeta_to_z(zeta_leading_edge, self.epsilon)
                self.z_trailing_edge = self.zeta_to_z(zeta_trailing_edge, self.epsilon)
            else:
                self.x_leading_edge = self.zeta_center.real - self.cylinder_radius
                self.x_trailing_edge = self.zeta_center.real + self.cylinder_radius
                zeta_leading_edge = self.x_leading_edge + 1j*self.zeta_center.imag
                zeta_trailing_edge = self.x_trailing_edge + 1j*self.zeta_center.imag
                self.zeta_leading_intercept, self.zeta_trailing_intercept = self.calc_zeta_real_intercepts_J_airfoil()
                self.z_leading_edge = self.zeta_to_z(self.zeta_leading_intercept)
                self.z_trailing_edge = self.zeta_to_z(self.zeta_trailing_intercept)

    def surface_tangent(self, x_coord: float): # A3
        """
        Calculate the unit surface tangent vectors (upper and lower) at a given x-coordinate.
        Parameters:
        - x_coord (float): The x-coordinate at which to calculate the unit surface tangent vector.
        Returns:
        - unit_tangent (tuple): The unit surface tangent vectors (upper and lower) at the given x-coordinate.
        """
        delta_x = 1e-12  # Use a small delta for numerical differentiation
        if not isinstance(x_coord, float):
            raise TypeError("The x-coordinate must be a float. Please provide a float value.")
        if x_coord - self.x_leading_edge <= delta_x:
            x_coord += delta_x
            x_upper = self.geometry(x_coord)[0][0] # get the x value of the upper surface
            y_upper = self.geometry(x_coord)[0][1] # get the y value of the upper surface
            x_lower = self.geometry(x_coord)[1][0] # get the x value of the lower surface
            y_lower = self.geometry(x_coord)[1][1] # get the y value of the lower surface
            length = np.sqrt((x_upper - x_lower)**2 + (y_upper - y_lower)**2)
            tangent_upper = [(x_upper - x_lower)/length, (y_upper - y_lower)/length]
            tangent_lower = tangent_upper
        elif self.x_trailing_edge - x_coord <= delta_x:
            x_coord -= delta_x
            x_upper = self.geometry(x_coord)[0][0]
            y_upper = self.geometry(x_coord)[0][1]
            x_lower = self.geometry(x_coord)[1][0]
            y_lower = self.geometry(x_coord)[1][1]
            length = np.sqrt((x_upper - x_lower)**2 + (y_upper - y_lower)**2)
            tangent_upper = [-1*(x_upper - x_lower)/length, -1*(y_upper - y_lower)/length]
            tangent_lower = tangent_upper
        else: # for when you're not on the leading or trailing edge. 
            x_plus_delta = x_coord + 0.5*delta_x
            x_minus_delta = x_coord - 0.5*delta_x
            # Calculate the y-coordinates at x + delta and x - delta
            x_upper_plus_delta = self.geometry(x_plus_delta)[0][0]
            y_upper_plus_delta = self.geometry(x_plus_delta)[0][1]
            x_upper_minus_delta = self.geometry(x_minus_delta)[0][0]
            y_upper_minus_delta = self.geometry(x_minus_delta)[0][1]
            x_lower_plus_delta = self.geometry(x_plus_delta)[1][0]
            y_lower_plus_delta = self.geometry(x_plus_delta)[1][1]
            x_lower_minus_delta = self.geometry(x_minus_delta)[1][0]
            y_lower_minus_delta = self.geometry(x_minus_delta)[1][1]
            length_upper = np.sqrt((x_upper_plus_delta - x_upper_minus_delta)**2 + (y_upper_plus_delta - y_upper_minus_delta)**2)
            length_lower = np.sqrt((x_lower_plus_delta - x_lower_minus_delta)**2 + (y_lower_plus_delta - y_lower_minus_delta)**2)
            tangent_upper = [(x_upper_plus_delta - x_upper_minus_delta)/length_upper, (y_upper_plus_delta - y_upper_minus_delta)/length_upper]
            tangent_lower = [-1*(x_lower_plus_delta - x_lower_minus_delta)/length_lower, -1*(y_lower_plus_delta - y_lower_minus_delta)/length_lower]
        upper_surface_tangent = tangent_upper
        lower_surface_tangent = tangent_lower
        return upper_surface_tangent, lower_surface_tangent

    def surface_normal(self, x_coord: float): # A3 on project
        """Uses surface tangent function to calculate the unit normal vector
            Parameters:
            - x_coord (float): The x-coordinate at which to calculate the unit surface tangent vector.

            Returns:
            - unit_tangent (tuple): The unit surface tangent vectors (upper and lower) at the given x-coordinate.
        """
        upper_surface_normal = [-self.surface_tangent(x_coord)[0][1], self.surface_tangent(x_coord)[0][0]]
        lower_surface_normal = [-self.surface_tangent(x_coord)[1][1], self.surface_tangent(x_coord)[1][0]]
        return upper_surface_normal, lower_surface_normal

    def calc_single_streamline(self, start_point: np.array, direction: int):
        """This function calculates the streamline of a given flow field, updating the progress bar based only on x-distance moved."""
        # Ensure start_point is a NumPy array with two float elements
        if not isinstance(start_point, np.ndarray) or start_point.shape != (2,) or not np.issubdtype(start_point.dtype, np.floating):
            raise ValueError("The start_point must be a NumPy array of two float values representing x and y coordinates.")
        point = np.array(start_point)
        streamline_points = [point]
        i = 0
        max_iter = 100000  # Maximum number of iterations to prevent infinite loops
        # Calculate total x-distance to cover based on the direction
        if direction == -1:
            # Leftward integration: distance to the lower x-limit
            x_total_distance = abs(point[0] - self.plot_x_lower_lim)
            x_limit = self.plot_x_lower_lim
        elif direction == 1:
            # Rightward integration: distance to the upper x-limit
            x_total_distance = abs(self.plot_x_upper_lim - point[0])
            x_limit = self.plot_x_upper_lim
        else:
            raise ValueError("Invalid direction: must be 1 (right) or -1 (left).")
        # print(f"Lower x limit: {self.plot_x_lower_lim}, Upper x limit: {self.plot_x_upper_lim}")
        # Initialize tqdm progress bar based on x-distance only
        with tqdm(total=x_total_distance, desc="Calculating Streamline", unit="x-distance", dynamic_ncols=True) as pbar:
            while self.plot_x_lower_lim <= point[0] <= self.plot_x_upper_lim:
                # Perform the RK4 integration step to get the next point on the streamline
                point_new = hlp.rk4(point, direction, self.plot_delta_s, self.surface_normal, self.unit_velocity)
                # Check if the new point will exceed the x_limits
                if point_new[0] < self.plot_x_lower_lim or point_new[0] > self.plot_x_upper_lim:
                    break  # Stop the integration if the new point goes beyond the x_limits
                streamline_points.append(point_new)
                # Calculate the x-distance covered between the current and new point (ignoring y-distance)
                x_distance_covered = abs(point_new[0] - point[0])
                # Prevent progress bar from going over 100% by calculating the remaining distance
                remaining_distance = pbar.total - pbar.n # pbar.n is the current progress
                update_distance = min(x_distance_covered, remaining_distance)  # Only update with the smaller value
                pbar.update(update_distance)  # Update progress bar with the bounded x-distance
                point = point_new
                if  i > max_iter:
                    break
                i += 1
        streamline_points = np.array(streamline_points)
        return streamline_points

    def shift_xy_points(self, point_xy: np.array):
        """
        Shift a point or array of points the same way as the J-Cylinder transformation.
        
        Parameters: 
        - point_xy (np.array): shape (2,) for a single point or (n, 2) for multiple points.

        Returns:
        - shifted_array (np.array): shifted point(s), same shape as input.
        """
        point_xy = np.asarray(point_xy)
        if point_xy.ndim == 1 and point_xy.shape[0] == 2: # Single point
            shifted = np.empty(2)
            shifted[0], shifted[1] = point_xy[0] - self.zeta_center.real - self.leading_edge, point_xy[1] - self.zeta_center.imag
            shifted /= self.trailing_edge
            return shifted
        elif point_xy.ndim == 2 and point_xy.shape[1] == 2: # Array of points
            shifted = np.empty_like(point_xy)
            shifted[:, 0], shifted[:, 1] = point_xy[:, 0] - self.zeta_center.real - self.leading_edge, point_xy[:, 1] - self.zeta_center.imag
            shifted /= self.trailing_edge
            return shifted
        else:
            raise ValueError("point_xy must be shape (2,) or (n, 2)")
    
    def calc_single_shifted_streamline(self, streamline_array: np.array):
        """This function calculates the streamline using the streamline function, but then shifts the points in the same way a unit airfoil was created for the J-Cylinder"""
        shifted_streamline_points = self.shift_xy_points(streamline_array)
        return shifted_streamline_points
    
    def unit_velocity(self, point_xy: np.array): # A4 on project
        """This function calculates the unit velocity at a given point in the flow field in cartesian coordinates."""
        if self.velocity(point_xy, self.circulation)[0] == 0.0 and self.velocity(point_xy, self.circulation)[1] == 0.0:
            velocity = np.array([0.0, 0.0])
        else:
            velocity = hlp.unit_vector(self.velocity(point_xy, self.circulation))
        return velocity
    
    def stagnation(self):
        """This function returns the stagnation point of a cylinder given the velocity field properties."""
        aft_stag_angle = self.calculate_aft_stagnation_theta_in_Chi_from_Gamma(self.circulation)
        forward_stag, aft_stag = self.calculate_forward_and_aft_stag_locations_in_z(aft_stag_angle)
        forward_stag[0] -= 1e-12 # small adjustment to avoid being exactly where velocity is zero
        aft_stag[0] += 1e-12 # small adjustment to avoid being exactly where velocity is zero
        self.forward_stag = forward_stag
        self.aft_stag = aft_stag
        return self.forward_stag, self.aft_stag
    
    def plot_geometry(self): # A2 on project
        """This function plots the geometry in question"""
        plt.scatter(self.full_z_surface[:,0], self.full_z_surface[:,1], color = "black", label = "J-Cyl", s=0.5) 
        plt.xlim(self.plot_x_lower_lim, self.plot_x_upper_lim)
        plt.ylim(self.plot_x_lower_lim, self.plot_x_upper_lim)

    def calc_stagnation_streamlines(self):
        """This function calculates the stagnation streamlines for the forward and aft stagnation points."""
        self.forward_stag, self.aft_stag = self.stagnation()
        self.forward_stag_streamline = self.calc_single_streamline(self.forward_stag, -1)
        self.aft_stag_streamline = self.calc_single_streamline(self.aft_stag, 1)
        return self.forward_stag_streamline, self.aft_stag_streamline

    def calc_shifted_streamlines(self, streamlines):
        """This function calculates the streamlines for a given starting y-coordinate, number of lines, and delta y."""
        for i in range(len(streamlines)):
            streamlines[i] = self.calc_single_shifted_streamline(streamlines[i])
        return streamlines  
        
    def calc_streamlines(self):
        """This function calculates the streamlines for the forward and aft stagnation points."""
        streamlines = []  # Initialize a list to store streamlines
        self.forward_stag, self.aft_stag = self.stagnation()
        self.forward_stag_streamline = self.calc_single_streamline(self.forward_stag, -1)
        streamlines.append(self.forward_stag_streamline)  # Add the forward stagnation streamline to the list
        self.aft_stag_streamline = self.calc_single_streamline(self.aft_stag, 1)
        streamlines.append(self.aft_stag_streamline)  # Add the aft stagnation streamline to the list
        y_coord = self.forward_stag_streamline[-1][1]  # Get the second to last y coordinate of the forward stagnation streamline
        for i in range(int(self.plot_n_lines)):
            y_coord += self.plot_delta_y
            # Plot the streamlines above the forward stagnation streamline
            streamline = self.calc_single_streamline(np.array([self.plot_x_start, y_coord]), 1)
            streamlines.append(streamline)
        y_coord = self.forward_stag_streamline[-1][1]  # Get the last y coordinate of the forward stagnation streamline
        for j in range(int(self.plot_n_lines)):
            y_coord -= self.plot_delta_y
            # Plot the streamlines below the forward stagnation streamline
            streamline = self.calc_single_streamline(np.array([self.plot_x_start, y_coord]), 1)
            streamlines.append(streamline)
        return streamlines

    def plot(self, streamlines):
        """This function plots the geometry, streamlines, and stagnation lines."""
        for i in range(len(streamlines)):
            hlp.plot_xy_array(streamlines[i])
