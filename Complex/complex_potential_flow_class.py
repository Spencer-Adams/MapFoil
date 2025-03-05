import json
import numpy as np # type: ignore
import math 
import matplotlib.pyplot as plt  # type: ignore
from tqdm import tqdm
from multiprocessing import Pool # this is for parallel processing
import scipy.integrate as spi # this is for numerical integration
# import bisection from scipy.optimize
from scipy import optimize
import helper as hlp

# set print precision to 4 digits
# np.set_printoptions(precision=4)

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
            # print("x lower limit", self.plot_x_lower_lim)
            self.plot_x_upper_lim = input["plot"]["x_upper_limit"]/self.original_cylinder_radius#self.cylinder_radius # this is the lower limit of the plot in x direction.
            # print("x upper limit", self.plot_x_upper_lim)
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
                # print("zeta leading edge", zeta_leading_edge)
                # print("zeta trailing edge", zeta_trailing_edge)
            else:
                # self.x_leading_edge, self.x_trailing_edge = self.calc_zeta_real_intercepts()
                self.x_leading_edge = self.zeta_center.real - self.cylinder_radius
                self.x_trailing_edge = self.zeta_center.real + self.cylinder_radius
                zeta_leading_edge = self.x_leading_edge + 1j*self.zeta_center.imag
                zeta_trailing_edge = self.x_trailing_edge + 1j*self.zeta_center.imag
                self.zeta_leading_intercept, self.zeta_trailing_intercept = self.calc_zeta_real_intercepts_J_airfoil()
                self.z_leading_edge = self.zeta_to_z(self.zeta_leading_intercept)
                self.z_trailing_edge = self.zeta_to_z(self.zeta_trailing_intercept)
                # print("zeta leading edge", zeta_leading_edge)
                # print("zeta trailing edge", zeta_trailing_edge)
                # print("zeta leading intercept", self.zeta_leading_intercept)
                # print("zeta trailing intercept", self.zeta_trailing_intercept)

            # print("\n")
            # print("z leading edge", self.z_leading_edge)
            # print("z trailing edge", self.z_trailing_edge)
            # print("\n")

            # shift it more according to the conformal mapping 
            # now read in potential flow elements as well. 

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

        # Adjust x_coord if it's too close to the edges
        if x_coord - self.x_leading_edge <= delta_x:
            # print("we're in the leading edge tangent function")
            # print("x_coord", x_coord)
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

    
    def surface_tangential_velocity(self, x_coord: float): # A5 on project  # make this callable so we can use it for other geoms
        """
        Calculates the velocity at a given point on the surface of the shape in cartesian coordinates.

        Parameters:
        - x_coord (float): The x-coordinate at which to calculate the velocity.

        Returns:
        - velocity (list): The velocity at the given point.
        """
        delta_x = 1e-12  # Use a small delta for numerical differentiation("The x-coordinate of the surface tangent vector cannot be outside the domain of the geometry.")
        
        x_upper = self.geometry(x_coord)[0][0]
        y_upper = self.geometry(x_coord)[0][1]
        x_lower = self.geometry(x_coord)[1][0]
        y_lower = self.geometry(x_coord)[1][1]
        # calculate Vx and Vy for the upper and lower surfaces using the velocity function and the surface tangent function
        Velocity_upper = self.velocity([x_upper, y_upper], self.circulation)
        upper_tangent = self.surface_tangent(x_coord)[0]
        upper_tangential_velocity = np.dot(Velocity_upper, upper_tangent)
        Velocity_lower = self.velocity([x_lower, y_lower], self.circulation)
        lower_tangent = self.surface_tangent(x_coord)[1]
        lower_tangential_velocity = np.dot(Velocity_lower, lower_tangent)
        return upper_tangential_velocity, lower_tangential_velocity

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

    def streamline(self, start_point: np.array, direction: int):
        """This function calculates the streamline of a given flow field, updating the progress bar based only on x-distance moved."""
        # Ensure start_point is a NumPy array with two float elements
        if not isinstance(start_point, np.ndarray) or start_point.shape != (2,) or not np.issubdtype(start_point.dtype, np.floating):
            raise ValueError("The start_point must be a NumPy array of two float values representing x and y coordinates.")
        
        point = np.array(start_point)
        streamline_points = [point]
        i = 0
        # Convergence and iteration parameters
        max_iter = 10000  # Maximum number of iterations to prevent infinite loops

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
        # Initialize tqdm progress bar based on x-distance only
        if not self.is_element:
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

                    # Check for convergence: if the iteration limit is reached
                    if  i > max_iter:
                        break

                    # Increment iteration counter
                    i += 1
        else:
            while self.plot_x_lower_lim <= point[0] <= self.plot_x_upper_lim:
                # Perform the RK4 integration step to get the next point on the streamline
                point_new = hlp.rk4(point, direction, self.plot_delta_s, self.surface_normal, self.unit_velocity)
                streamline_points.append(point_new)
                # Check if the new point will exceed the x_limits
                if point_new[0] < self.plot_x_lower_lim or point_new[0] > self.plot_x_upper_lim:
                    break
                i+=1
                point = point_new
                if i > max_iter:
                    break
            

        streamline_points = np.array(streamline_points)
        return streamline_points
    
    def unit_velocity(self, point_xy: np.array): # A4 on project
        """This function calculates the unit velocity at a given point in the flow field in cartesian coordinates."""
        if self.velocity(point_xy, self.circulation)[0] == 0.0 and self.velocity(point_xy, self.circulation)[1] == 0.0:
            velocity = np.array([0.0, 0.0])
        else:
            velocity = hlp.unit_vector(self.velocity(point_xy, self.circulation))
        return velocity
    
    def stagnation(self):
        """This function returns the stagnation point of a cylinder given the velocity field properties."""
        if self.type == "cylinder":
            leading = self.x_leading_edge
            trailing = self.x_trailing_edge
            middle = leading + (trailing - leading) / 2
            # print("\n")
            # print("Leading edge of the cylinder", leading)
            # print("Middle of the cylinder", middle)
            # print("Trailing edge of the cylinder", trailing)
        else:
            leading = self.zeta_leading_intercept
            trailing = self.zeta_trailing_intercept
            middle = leading + (trailing - leading) / 2
            # print("Leading edge of the airfoil", leading)
            # print("Middle of the airfoil", middle)
            # print("Trailing edge of the airfoil", trailing)
            # print("\n")
        
        forward_stag = []
        aft_stag = []
        # add dots to show progress
        print("Finding Stagnation Points...")
        if self.surface_tangential_velocity(leading)[0]*self.surface_tangential_velocity(middle)[0] < 0:
            forward_stag.append(optimize.bisect(lambda x: self.surface_tangential_velocity(x)[0], leading, middle, xtol=1e-12))
            self.forward_stag = [self.geometry(forward_stag[0])[0][0], self.geometry(forward_stag[0])[0][1]]
        # now check the lower suface within x_values
        elif self.surface_tangential_velocity(leading)[1]*self.surface_tangential_velocity(middle)[1] <0:
            forward_stag.append(optimize.bisect(lambda x: self.surface_tangential_velocity(x)[1], leading, middle, xtol=1e-12))
            self.forward_stag = [self.geometry(forward_stag[0])[1][0], self.geometry(forward_stag[0])[1][1]]
        # now check the upper surface within x_values_2
        if self.surface_tangential_velocity(middle)[0] * self.surface_tangential_velocity(trailing)[0] < 0:
            aft_stag.append(optimize.bisect(lambda x: self.surface_tangential_velocity(x)[0], middle, trailing, xtol=1e-12))
            self.aft_stag = [self.geometry(aft_stag[0])[0][0], self.geometry(aft_stag[0])[0][1]]
        # now check the lower surface within x_values_2
        elif self.surface_tangential_velocity(middle)[1] * self.surface_tangential_velocity(trailing)[1] < 0:
            aft_stag.append(optimize.bisect(lambda x: self.surface_tangential_velocity(x)[1], middle, trailing, xtol=1e-12))
            self.aft_stag = [self.geometry(aft_stag[0])[1][0], self.geometry(aft_stag[0])[1][1]]
        
        if len(forward_stag) == 0:
            print("No forward stagnation point found within the given range. Using the leading edge.")
            self.forward_stag = [self.z_leading_edge.real, self.z_leading_edge.imag]
        if len(aft_stag) == 0:
            print("No aft stagnation point found within the given range. Using the trailing edge.")
            self.aft_stag = [self.z_trailing_edge.real, self.z_trailing_edge.imag]

        # check if the stagnation points are incidental with any of the singularities.
        if np.isclose(self.forward_stag[0], self.z_leading_edge_focus.real, atol=1e-12):
            print("Forward Stagnation Point is at Leading Edge Focus... Moving the forward stagnation point in the negative free stream direction by 1e-12")
            self.forward_stag = self.forward_stag - np.array([1e-12*np.cos(self.angle_of_attack), 1e-12*np.sin(self.angle_of_attack)])
        elif np.isclose(self.forward_stag[0], self.z_trailing_edge_focus.real, atol=1e-12).all():
            print("Forward Stagnation Point is at Trailing Edge Focus... Moving the forward stagnation point in the positive free stream direction by 1e-12")
            self.forward_stag = self.forward_stag + np.array([1e-12*np.cos(self.angle_of_attack), 1e-12*np.sin(self.angle_of_attack)])

        if np.isclose(self.aft_stag[0], self.z_leading_edge_focus.real, atol=1e-12):
            print("Aft Stagnation Point is at Leading Edge Focus... Moving the aft stagnation point in the negative free stream direction by 1e-12")
            self.aft_stag = self.aft_stag - np.array([1e-12*np.cos(self.angle_of_attack), 1e-12*np.sin(self.angle_of_attack)])
        elif np.isclose(self.aft_stag[0], self.z_trailing_edge_focus.real, atol=1e-12):
            print("Aft Stagnation Point is at Trailing Edge Focus... Moving the aft stagnation point in the positive free stream direction by 1e-12")
            self.aft_stag = self.aft_stag + np.array([1e-12*np.cos(self.angle_of_attack), 1e-12*np.sin(self.angle_of_attack)])
        
        self.forward_stag = np.array(self.forward_stag)
        self.aft_stag = np.array(self.aft_stag)
        x_for_stag_forward = self.forward_stag[0]
        # print("chordwise x/c value for the forward stagnation", x_for_stag_forward)
        print("FORWARD STAGNATION",self.forward_stag)
        x_for_stag_aft = self.aft_stag[0]
        # print("chordwise x/c value for the aft stagnation", x_for_stag_aft)
        print("AFT STAGNATION",self.aft_stag)
        print("Velocity at Forward Stag", self.velocity(self.forward_stag, self.circulation))
        pressure_forward = 1-(hlp.vector_magnitude(self.velocity(self.forward_stag, self.circulation))/self.freestream_velocity)**2
        print("Pressure at Forward Stag", pressure_forward)
        print("Velocity at Aft Stag", self.velocity(self.aft_stag, self.circulation))
        pressure_aft = 1-(hlp.vector_magnitude(self.velocity(self.aft_stag, self.circulation))/self.freestream_velocity)**2
        print("Pressure at Aft Stag", pressure_aft, "\n")
        return self.forward_stag, self.aft_stag
    
    def plot_geometry(self): # A2 on project
        """This function plots the geometry in question"""
        # define the angle
        # self.get_full_geometry()
        plt.plot(self.upper_coords[:,0], self.upper_coords[:,1], color = "black", label = "J-Cyl") # upper plot
        plt.plot(self.lower_coords[:,0], self.lower_coords[:,1] , color = "black") # lower plot
        plt.xlim(self.plot_x_lower_lim, self.plot_x_upper_lim)
        # print("x lower limit", self.plot_x_lower_lim)
        # print("x upper limit", self.plot_x_upper_lim)
        plt.ylim(self.plot_x_lower_lim, self.plot_x_upper_lim)
        # print("y lower limit", self.plot_x_lower_lim)
        # print("y upper limit", self.plot_x_upper_lim)
        # # use plot_surface_tangents and plot_surface_normals to plot the surface tangents on the geometry (skip 5 out of 6 geom points for less dense tangent vector plot, but make sure to include the leading and trailing edges)
        # for i in range(len(self.upper_coords)):
        #     if i % 6 == 0 or i == 0 or i == len(self.upper_coords) - 1: #
        #         self.plot_surface_tangents(self.upper_zeta_coords[i][0])
        #         self.plot_surface_normals(self.upper_zeta_coords[i][0])
        
    def plot_surface_tangents(self, x_coord: float): 
        """This function plots the surface tangents on the geometry"""
        upper_tangent, lower_tangent = self.surface_tangent(x_coord)
        # get the y coordinates of the upper and lower surfaces
        x_upper = self.geometry(x_coord)[0][0]
        x_lower = self.geometry(x_coord)[1][0]
        y_upper = self.geometry(x_coord)[0][1]
        y_lower = self.geometry(x_coord)[1][1]
        # plot the surface tangents vectors with arrows
        plt.quiver(x_upper, y_upper, upper_tangent[0], upper_tangent[1], color="black", scale=10, scale_units='xy')
        plt.quiver(x_lower, y_lower, lower_tangent[0], lower_tangent[1], color="black", scale=10, scale_units='xy')

    def plot_surface_normals(self, x_coord: float):
        """This function plots the surface normals on the geometry"""
        upper_normal, lower_normal = self.surface_normal(x_coord)
        # get the y coordinates of the upper and lower surfaces
        x_upper = self.geometry(x_coord)[0][0]
        x_lower = self.geometry(x_coord)[1][0]
        y_upper = self.geometry(x_coord)[0][1]
        y_lower = self.geometry(x_coord)[1][1]
        # plot the surface normal vectors with arrows
        plt.quiver(x_upper, y_upper, upper_normal[0], upper_normal[1], color="black", scale=10, scale_units='xy')
        plt.quiver(x_lower, y_lower, lower_normal[0], lower_normal[1], color="black", scale=10, scale_units='xy')

    def plot_streamline(self, start_point: np.array, direction): # A6 on project
        """ Plots a streamline starting from the given start_point using the provided velocity_function.
            Parameters:
            - start_point (array): A numpy array of two float values representing the x and y coordinates of the starting point.

            Raises:
            - ValueError: If the start_point is not a list of two float values.

            Returns:
            - None
        """
        # make sure that start_point is a list of two float elements
        if len(start_point) < 2 or not all(isinstance(coord, float) for coord in start_point):
            raise ValueError("The start_point must be a list of two float values representing x and y coordinates.")
        # calculate the streamline points
        start_point = np.array(start_point)
        streamline_points = self.streamline(start_point, direction)
        # plot the streamline
        plt.plot([point[0] for point in streamline_points], [point[1] for point in streamline_points], color = "black", linewidth=0.5)
        return streamline_points

    def plot_stagnation(self): 
        """This function plots the stagnation points on the geometry"""
        # self.get_stagnation_streamlines()
        self.forward_stag_streamline = self.plot_streamline(self.forward_stag, -1)
        self.aft_stag_streamline = self.plot_streamline(self.aft_stag, 1)
    
    def plot(self):
        """This function plots the geometry, streamlines, and stagnation lines."""
        # get stagnation points Here. 
        self.stagnation()
        # plot the stagnation lines extending from the stagnation points
        self.plot_stagnation()
        # get the second to last y coordinate of the forward stagnation streamline
        y_coord = self.forward_stag_streamline[-1][1]
        
        # for i in tqdm(range(int(self.plot_n_lines)), desc="Plotting Upper Streamlines (Total number of streamlines = {})".format(int(self.plot_n_lines))):
        for i in range(int(self.plot_n_lines)):
            y_coord += self.plot_delta_y
            # plot the streamlines above the forward stagnation streamline
            streamline = self.plot_streamline(np.array([self.plot_x_start, y_coord]), 1)

        # get the last y coordinate of the forward stagnation streamline
        y_coord = self.forward_stag_streamline[-1][1]
        # for i in tqdm(range(int(self.plot_n_lines)), desc="Plotting Lower Streamlines (Total number of streamlines = {})".format(int(self.plot_n_lines))):
        for i in range(int(self.plot_n_lines)):
            y_coord -= self.plot_delta_y
            # plot the streamlines below the forward stagnation streamline
            self.plot_streamline(np.array([self.plot_x_start, y_coord]), 1)
