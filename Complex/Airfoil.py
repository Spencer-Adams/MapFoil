import json
import numpy as np # type: ignore
import math 
import matplotlib.pyplot as plt  # type: ignore
from tqdm import tqdm
from multiprocessing import Pool # this is for parallel processing
import scipy.integrate as spi # this is for numerical integration
# import bisection from scipy.optimize
from scipy import optimize

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
            self.plot_x_lower_lim = input["plot"]["x_lower_limit"]
            self.plot_x_upper_lim = input["plot"]["x_upper_limit"]
            self.plot_x_start = input["plot"]["x_start"]
            self.plot_y_start = input["plot"]["y_start"]
            self.plot_delta_s = input["plot"]["delta_s"]
            self.plot_n_lines = input["plot"]["n_lines"]
            self.plot_delta_y = input["plot"]["delta_y"]
            self.plot_y_upper_lim = input["plot"]["y_upper_limit"]
            self.plot_y_lower_lim = input["plot"]["y_lower_limit"]
            self.freestream_velocity = input["operating"]["freestream_velocity"]
            angle_of_attack = input["operating"]["angle_of_attack[deg]"]
            self.angle_of_attack = np.radians(angle_of_attack)
            self.x_leading_edge = input["geometry"]["general"]["x_leading_edge"]
            self.x_trailing_edge = input["geometry"]["general"]["x_trailing_edge"]
            self.x_center = input["geometry"]["general"]["x_center"]
            self.y_center = input["geometry"]["general"]["y_center"]
            self.point_density = input["geometry"]["general"]["point_density"]
            self.circulation = input["operating"]["vortex_strength"]
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
        
        if x_coord < self.x_leading_edge - 1 or x_coord > self.x_trailing_edge + 1:
            print("\n")
            print("\n")
            print("x_coord", x_coord)
            print("leading edge", self.x_leading_edge)
            print("trailing edge", self.x_trailing_edge)
            print("\n")
            print("\n")
            raise ValueError("The x-coordinate of the surface tangent vector cannot be outside the domain of the geometry.")
        
        # Adjust x_coord if it's too close to the edges
        if x_coord - self.x_leading_edge <= delta_x:
            x_coord += delta_x
            x_upper = self.geometry(x_coord)[0][0] # get the x value of the upper surface
            y_upper = self.geometry(x_coord)[0][1] # get the y value of the upper surface
            x_lower = self.geometry(x_coord)[1][0] # get the x value of the lower surface
            y_lower = self.geometry(x_coord)[1][1] # get the y value of the lower surface
            length = np.sqrt((x_upper - x_lower)**2 + (y_upper - y_lower)**2)
            tangent_upper = [(x_upper - x_lower)/length, (y_upper - y_lower)/length]
            tangent_lower = [-1*(x_upper - x_lower)/length, -1*(y_upper - y_lower)/length]
        elif self.x_trailing_edge - x_coord <= delta_x:
            x_coord -= delta_x
            x_upper = self.geometry(x_coord)[0][0]
            y_upper = self.geometry(x_coord)[0][1]
            x_lower = self.geometry(x_coord)[1][0]
            y_lower = self.geometry(x_coord)[1][1]
            length = np.sqrt((x_upper - x_lower)**2 + (y_upper - y_lower)**2)
            tangent_upper = [(x_upper - x_lower)/length, (y_upper - y_lower)/length]
            tangent_lower = [-1*(x_upper - x_lower)/length, -1*(y_upper - y_lower)/length]
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
        Velocity_upper = self.velocity([x_upper, y_upper])
        upper_tangent = self.surface_tangent(x_coord)[0]
        upper_tangential_velocity = np.dot(Velocity_upper, upper_tangent)
        Velocity_lower = self.velocity([x_lower, y_lower])
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
        # Initialize tqdm progress bar based on x-distance only
        if not self.is_element:
            with tqdm(total=x_total_distance, desc="Calculating Streamline", unit="x-distance", dynamic_ncols=True) as pbar:
                while self.plot_x_lower_lim <= point[0] <= self.plot_x_upper_lim:
                    # Perform the RK4 integration step to get the next point on the streamline
                    point_new = rk4(point, direction, self.plot_delta_s, self.surface_normal, self.unit_velocity)

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
                point_new = rk4(point, direction, self.plot_delta_s, self.surface_normal, self.unit_velocity)
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
        if self.velocity(point_xy)[0] == 0.0 and self.velocity(point_xy)[1] == 0.0:
            velocity = np.array([0.0, 0.0])
        else:
            velocity = unit_vector(self.velocity(point_xy))
        return velocity

    def stagnation(self):
        # check to see if the geometry is a text file or not 
        # check for a stagnation point on the lower surface using the leading and trailing edge first
        self.forward_stag = np.array([self.x_leading_edge, 0.0])
        self.aft_stag = np.array([self.x_trailing_edge, 0.0])
        delta_x = 1e-6
        stagnation_points = []
        lower_bound = self.x_leading_edge
        if self.is_text_file:
            print("Text file stagnation points are", self.forward_stag, self.aft_stag)
        else:
            if self.is_airfoil:
                lower_bound = self.x_leading_edge + delta_x
                upper_bound = self.x_trailing_edge - delta_x
            else:
                upper_bound = self.x_trailing_edge
            lower_bound_surface_tangential_velocity = self.surface_tangential_velocity(lower_bound)[1]
            upper_bound_surface_tangential_velocity = self.surface_tangential_velocity(upper_bound)[1]
            # check to see if the lower surface tangential velocity changes sign from the lower bound to the upper bound
            if lower_bound_surface_tangential_velocity * upper_bound_surface_tangential_velocity < 0:
                # find the stagnation point using bisection
                stagnation_points.append(optimize.bisect(lambda x: self.surface_tangential_velocity(x)[1], lower_bound, upper_bound))
                stagnation_points.append("lower")
            # if not, there may have been 2 stagnation points on the lower surface, and the sign change flipped back to positive, so check from the middle to the left, and then the middle to the right
            else:
                middle = (lower_bound + upper_bound) / 2
                middle_surface_tangential_velocity = self.surface_tangential_velocity(middle)[1]
                if lower_bound_surface_tangential_velocity * middle_surface_tangential_velocity < 0:
                    stagnation_points.append(optimize.bisect(lambda x: self.surface_tangential_velocity(x)[1], lower_bound, middle))
                    stagnation_points.append("lower")
                if middle_surface_tangential_velocity * upper_bound_surface_tangential_velocity < 0:
                    stagnation_points.append(optimize.bisect(lambda x: self.surface_tangential_velocity(x)[1], middle, upper_bound))
                    stagnation_points.append("lower")
            # check for a stagnation point on the upper surface using the leading and trailing edge first
            lower_bound = self.x_leading_edge
            upper_bound = self.x_trailing_edge
            lower_bound_surface_tangential_velocity = self.surface_tangential_velocity(lower_bound)[0] # returning the upper surface tangential velocity
            upper_bound_surface_tangential_velocity = self.surface_tangential_velocity(upper_bound)[0] # returning the upper surface tangential velocity
            # check to see if the upper surface tangential velocity changes sign from the lower bound to the upper bound
            if lower_bound_surface_tangential_velocity * upper_bound_surface_tangential_velocity < 0:
                # find the stagnation point using bisection
                stagnation_points.append(optimize.bisect(lambda x: self.surface_tangential_velocity(x)[0], lower_bound, upper_bound))
                stagnation_points.append("upper")
            # if not, there may have been 2 stagnation points on the upper surface, and the sign change flipped back to positive, so check from the middle to the left, and then the middle to the right
            else:
                middle = (lower_bound + upper_bound) / 2
                middle_surface_tangential_velocity = self.surface_tangential_velocity(middle)[0]
                if lower_bound_surface_tangential_velocity * middle_surface_tangential_velocity < 0:
                    stagnation_points.append(optimize.bisect(lambda x: self.surface_tangential_velocity(x)[0], lower_bound, middle))
                    stagnation_points.append("upper")
                if middle_surface_tangential_velocity * upper_bound_surface_tangential_velocity < 0:
                    stagnation_points.append(optimize.bisect(lambda x: self.surface_tangential_velocity(x)[0], middle, upper_bound))
                    stagnation_points.append("upper")

            if len(stagnation_points) == 0:
                print("No stagnation points found inside shape, assuming leading and trailing edges are the stagnation points")
                stagnation_points.append(self.x_leading_edge)
                stagnation_points.append("lower")
                stagnation_points.append(self.x_trailing_edge)
                stagnation_points.append("upper")
                print("\n")
            else:
                print("")

            # check to see which stagnation in stagnation_points is farther to the left, see if those points are upper or lower and assign those values to forward_stag, and the other to aft_stag
            if not self.is_airfoil and len(stagnation_points) == 4:
                # starting with the leading edge, check to see if the stagnation point is upper or lower, and assign the appropriate y value to the forward_stag
                if stagnation_points[0] < stagnation_points[2]: # stag_point[0] is the x value of the leading edge stagnation point
                    if stagnation_points[1] == "lower":
                        x_for_stag = stagnation_points[0]
                        self.forward_stag = np.array([self.geometry(stagnation_points[0])[1][0], self.geometry(stagnation_points[0])[1][1]]) # [1][1] on the geometry function is the y value of the lower surface
                        if stagnation_points[3] == "lower":
                            x_for_stag = stagnation_points[2]
                            self.aft_stag = np.array([self.geometry(stagnation_points[2])[1][0], self.geometry(stagnation_points[2])[1][1]]) # [1][1] on the geometry function is the y value of the lower surface
                        elif stagnation_points[3] == "upper":
                            x_for_stag = stagnation_points[2]
                            self.aft_stag = np.array([self.geometry(stagnation_points[2])[0][0], self.geometry(stagnation_points[2])[0][1]]) # [0][1] on the geometry function is the y value of the upper surface
                    elif stagnation_points[1] == "upper":
                        x_for_stag = stagnation_points[0]
                        self.forward_stag = np.array([self.geometry(stagnation_points[0])[0][0], self.geometry(stagnation_points[0])[0][1]]) # [0][1] on the geometry function is the y value of the upper surface
                        if stagnation_points[3] == "lower":
                            x_for_stag = stagnation_points[2]
                            self.aft_stag = np.array([self.geometry(stagnation_points[2])[1][0], self.geometry(stagnation_points[2])[1][1]]) # [1][1] on the geometry function is the y value of the lower surface
                        elif stagnation_points[3] == "upper":
                            x_for_stag = stagnation_points[2]
                            self.aft_stag = np.array([self.geometry(stagnation_points[2])[0][0], self.geometry(stagnation_points[2])[0][1]]) # [0][1] on the geometry function is the y value of the upper surface
                elif stagnation_points[0] > stagnation_points[2]: # stag_point[2] is the x value of the trailing edge stagnation point
                    if stagnation_points[3] == "lower":
                        x_for_stag = stagnation_points[2]
                        self.forward_stag = np.array([self.geometry(stagnation_points[2])[1][0], self.geometry(stagnation_points[2])[1][1]])
                        if stagnation_points[1] == "lower":
                            x_for_stag = stagnation_points[0]
                            self.aft_stag = np.array([self.geometry(stagnation_points[0])[1][0], self.geometry(stagnation_points[0])[1][1]])
                        elif stagnation_points[1] == "upper":
                            x_for_stag = stagnation_points[0]
                            self.aft_stag = np.array([self.geometry(stagnation_points[0])[0][0], self.geometry(stagnation_points[0])[0][1]])
                    elif stagnation_points[3] == "upper":
                        x_for_stag = stagnation_points[2]
                        self.forward_stag = np.array([self.geometry(stagnation_points[2])[0][0], self.geometry(stagnation_points[2])[0][1]])
                        if stagnation_points[1] == "lower":
                            x_for_stag = stagnation_points[0]
                            self.aft_stag = np.array([self.geometry(stagnation_points[0])[1][0], self.geometry(stagnation_points[0])[1][1]])
                        elif stagnation_points[1] == "upper":
                            x_for_stag = stagnation_points[0]
                            self.aft_stag = np.array([self.geometry(stagnation_points[0])[0][0], self.geometry(stagnation_points[0])[0][1]])
                                                    
            elif self.is_airfoil and len(stagnation_points) == 4 or len(stagnation_points) == 2:
                if stagnation_points[1] == "lower":
                    # if the geometry at this point returns a nan, then move the point to the right by delta_x
                    # while math.isnan(self.geometry(stagnation_points[0])[1][1]):
                        # stagnation_points[0] += delta_x
                    x_for_stag = stagnation_points[0]
                    self.forward_stag = np.array([self.geometry(stagnation_points[0])[1][0], self.geometry(stagnation_points[0])[1][1]]) # [1][1] on the geometry function is the y value of the lower surface
                elif stagnation_points[1] == "upper":
                    # if the geometry at this point returns a nan, then move the point to the right by delta_x
                    # while math.isnan(self.geometry(stagnation_points[0])[0][1]):
                        # stagnation_points[0] += delta_x
                    x_for_stag = stagnation_points[0]
                    self.forward_stag = np.array([self.geometry(stagnation_points[0])[0][0], self.geometry(stagnation_points[0])[0][1]]) # [0][1] on the geometry function is the y value of the upper surface
                self.aft_stag = np.array([self.x_trailing_edge, self.geometry(self.x_trailing_edge)[2][1]]) # [2][1] on the geometry function is the y value of the camber line
            
            if np.isclose(self.forward_stag[0], self.x_leading_edge, rtol=1e-6):
                self.forward_stag = np.array([self.geometry(self.x_leading_edge)[2][0], self.geometry(self.x_leading_edge)[2][1]]) # [0][1] on the geometry function is the y value of the upper surface
            if np.isclose(self.aft_stag[0], self.x_trailing_edge, rtol=1e-6):
                self.aft_stag = np.array([self.geometry(self.x_trailing_edge)[2][0], self.geometry(self.x_trailing_edge)[2][1]])
            if self.is_airfoil and np.isclose(self.forward_stag[0], self.x_trailing_edge, rtol=1e-6):
                # set the forward stagnation point to the leading edge if the trailing edge is the stagnation point
                self.forward_stag = np.array([self.geometry(self.x_leading_edge)[2][0], self.geometry(self.x_leading_edge)[2][1]])
            
            if self.is_airfoil:
                print("\n")
                print("chordwise x/c value", x_for_stag)
                print("forward_stag",self.forward_stag)
                print("Velocity at Forward Stag", self.velocity(self.forward_stag))
                pressure_forward = 1-(vector_magnitude(self.velocity(self.forward_stag))/self.freestream_velocity)**2
                print("Pressure at Forward Stag", pressure_forward)
                if self.forward_stag[1] < 0:
                    print("Forward Stag is on the lower surface")
                    print("Surface_tangent velocity at Forward Stag", self.surface_tangential_velocity(x_for_stag)[1])
                elif self.forward_stag[1] > 0:
                    print("Forward Stag is on the upper surface")
                    print("Surface_tangent velocity at Forward Stag", self.surface_tangential_velocity(x_for_stag)[0])
                print("\n")
            elif not self.is_airfoil:
                print("\n")
                print("forward_stag",self.forward_stag, "\naft_stag",self.aft_stag)
                print("Velocity at Forward Stag", self.velocity(self.forward_stag))
                print("Velocity at Aft Stag", self.velocity(self.aft_stag))
                pressure_forward = 1-(vector_magnitude(self.velocity(self.forward_stag))/self.freestream_velocity)**2
                print("Pressure at Forward Stag", pressure_forward)
                pressure_aft = 1-(vector_magnitude(self.velocity(self.aft_stag))/self.freestream_velocity)**2
                print("Pressure at Aft Stag", pressure_aft)
                print("Surface_tangent velocity at Forward Stag", self.surface_tangential_velocity(self.forward_stag[0]))
                print("Surface_tangent velocity at Aft Stag", self.surface_tangential_velocity(self.aft_stag[0]))
                print("\n")
        return self.forward_stag, self.aft_stag

    
    def plot_geometry(self): # A2 on project
        """This function plots the geometry in question"""
        # define the angle
        # self.get_full_geometry()
        plt.plot(self.upper_coords[:,0], self.upper_coords[:,1], color = "black") # upper plot
        plt.plot(self.lower_coords[:,0], self.lower_coords[:,1] , color = "black") # lower plot
        plt.plot(self.camber_coords[:,0], self.camber_coords[:,1] , color = "red") # camber plot
        # print all the camber points
        plt.xlim(self.plot_x_lower_lim, self.plot_x_upper_lim)
        plt.ylim(self.plot_x_lower_lim, self.plot_x_upper_lim)
        # use plot_surface_tangents and plot_surface_normals to plot the surface tangents on the geometry (skip 5 out of 6 geom points for less dense tangent vector plot, but make sure to include the leading and trailing edges)
        # for i in range(len(self.upper_coords)):
        #     if i % 6 == 0 or i == 0 or i == len(self.upper_coords) - 1:
        #         self.plot_surface_tangents(self.upper_coords[i][0])
        #         self.plot_surface_normals(self.upper_coords[i][0])
        
    def plot_surface_tangents(self, x_coord: float): 
        """This function plots the surface tangents on the geometry"""
        upper_tangent, lower_tangent = self.surface_tangent(x_coord)
        # get the y coordinates of the upper and lower surfaces
        y_upper = self.geometry(x_coord)[0][1]
        y_lower = self.geometry(x_coord)[1][1]
        # plot the surface tangents vectors with arrows
        plt.quiver(x_coord, y_upper, upper_tangent[0], upper_tangent[1], color="black", scale=10, scale_units='xy')
        plt.quiver(x_coord, y_lower, lower_tangent[0], lower_tangent[1], color="black", scale=10, scale_units='xy')

    def plot_surface_normals(self, x_coord: float):
        """This function plots the surface normals on the geometry"""
        upper_normal, lower_normal = self.surface_normal(x_coord)
        # get the y coordinates of the upper and lower surfaces
        y_upper = self.geometry(x_coord)[0][1]
        y_lower = self.geometry(x_coord)[1][1]
        # plot the surface normal vectors with arrows
        plt.quiver(x_coord, y_upper, upper_normal[0], upper_normal[1], color="black", scale=10, scale_units='xy')
        plt.quiver(x_coord, y_lower, lower_normal[0], lower_normal[1], color="black", scale=10, scale_units='xy')

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
        plt.plot([point[0] for point in streamline_points], [point[1] for point in streamline_points], color = "black")
        return streamline_points

    def plot_stagnation(self): 
        """This function plots the stagnation points on the geometry"""
        # self.get_stagnation_streamlines()
        self.forward_stag_streamline = self.plot_streamline(self.forward_stag, -1)
        self.aft_stag_streamline = self.plot_streamline(self.aft_stag, 1)
    
    def plot(self):
        """This function plots the geometry, streamlines, and stagnation lines."""
        # plot the geometry
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
            streamline = self.plot_streamline(np.array([self.plot_x_start, y_coord]), 1)

class cylinder(potential_flow_object):
    """This is a class that creates a cylinder object and performs calculations specific to a cylinder"""
    def __init__(self, json_file):
        self.cyl_json_file = json_file
        self.parse_cylinder_json()
        super().__init__(json_file) # this is doing self.() of all the variables we want to in the super class

    def parse_cylinder_json(self):
        """This function reads the json file and stores the cylinder specific data in a dictionary"""
        with open(self.cyl_json_file, 'r') as json_handle:
            input = json.load(json_handle)
            self.cylinder_radius = input["geometry"]["cylinder"]["radius"]
            self.is_airfoil = False
    
    def geometry(self, x_coord: float): # A1 on project
        """
        Calculate the geometry of an airfoil at a given x-coordinate.

        Parameters:
        - x_coord (float): The x-coordinate at which to calculate the airfoil geometry.

        Returns:
        - upper (list): The upper surface coordinates of the airfoil at the given x-coordinate.
        - lower (list): The lower surface coordinates of the airfoil at the given x-coordinate.
        - camber (list): The camber line coordinates of the airfoil at the given x-coordinate.
        """
        # make sure that x_coord is a float
        if not isinstance(x_coord, float):
            raise TypeError("The x-coordinate must be a float. Please provide a float value.")
        # check if the x-coordinate is outside the cylinder
        if abs(x_coord) > self.cylinder_radius:
            print("The x-coordinate is outside the cylinder. Try finding geom for an x coordinate greater than zero and less than", self.cylinder_radius)
        else:
            theta = np.arccos(x_coord/self.cylinder_radius)
            upper = self.cylinder_radius*np.sin(theta)
            lower = -upper
            camber = 0
        return [x_coord, upper], [x_coord, lower], [x_coord, camber]

    def get_full_geometry(self):
        """This function calls the geometry function across every point on the cylinder"""
        # create an empty array for the upper and lower surfaces and camber 
        upper = np.zeros((self.point_density,2))
        lower = np.zeros((self.point_density,2))
        camber = np.zeros((self.point_density,2))
        x_coords = np.linspace(self.x_leading_edge, self.x_trailing_edge, self.point_density) # this is creating an array of x coordinates
        for i in range(self.point_density):
            upper[i], lower[i], camber[i] = self.geometry(x_coords[i])
        self.upper_coords = upper
        self.lower_coords = lower
        self.camber_coords = camber
    
    def velocity(self, point_xy: np.array): # A4 on project
        """
        Calculates the velocity at a given point in the flow field in cartesian coordinates using cylindrical velocity equations.

        Parameters:
        - point_xy (list): A list containing the x and y coordinates of the point at which to calculate the velocity.

        Returns:
        - velocity (list): The velocity at the given point.
        """
        # make sure that point_xy contains exactly two float elements
        if len(point_xy) != 2 or not all(isinstance(coord, float) for coord in point_xy):
            raise ValueError("The point_xy must be a list of two float values representing x and y coordinates.")
        
        r = np.sqrt(point_xy[0]**2 + point_xy[1]**2) # calculate the radius of the point from the origin
        theta = math.atan2(point_xy[1], point_xy[0])
        Vr = self.freestream_velocity*(1 - self.cylinder_radius**2/(r**2))*np.cos(theta - self.angle_of_attack)
        Vtheta = -self.freestream_velocity*(1 + self.cylinder_radius**2/(r**2))*np.sin(theta - self.angle_of_attack) - self.circulation/(2*np.pi*r)
        Vx = Vr*np.cos(theta) - Vtheta*np.sin(theta)
        Vy = Vr*np.sin(theta) + Vtheta*np.cos(theta)
        velocity_cartesian = [Vx, Vy]
        velocity_cartesian = np.array(velocity_cartesian)
        return velocity_cartesian


# Helper functions below
def vector_magnitude(vector: np.array): # Used to help A6 in the project. The streamlines need to be integrated forward using the unit velocity vector
    """
    Calculate the magnitude of a vector.

    Parameters:
    vector (list): A list representing the vector.

    Returns:
    float: The magnitude of the vector.
    """
    magnitude = math.sqrt(sum([element**2 for element in vector]))
    return magnitude

def unit_vector(vector: np.array): # Used to help A6 in the project. The streamlines need to be integrated forward using the unit velocity vector
    """Calculate the unit vector of a given vector."""
    # make sure that vector is a list of two or three float elements
    if len(vector) < 2 or not all(isinstance(coord, float) for coord in vector):
        raise ValueError("The vector must be a list of two or three float elements.")
    # calculate the magnitude of the vector
    magnitude = vector_magnitude(vector)
    # calculate the unit vector
    unit_vector = [element/magnitude for element in vector]
    return unit_vector

def rk4(start: np.array, direction: int, step_size: float,  move_off_direction_func: callable, function: callable): # Helps with A6 on the project
    """This function performs a Runge-Kutta 4th order integration

    Args:
        - start (list): A list of two float values representing x and y coordinates.
        - direction (int): An integer value representing the direction of integration.
        - step_size (float): The step size for the integration.
        - function (callable): A function that calculates the derivative at a given point.

    Returns:
        list: A numpy array of float values representing the new coordinates after integration.
    """
    # make sure that start is a list of two or three float elements
    if len(start) < 2 or not all(isinstance(coord, float) for coord in start):
        raise ValueError("The start must be a list of two float values representing x and y coordinates.")
    # make sure that direction is an integer
    if not isinstance(direction, int):
        raise TypeError("The direction must be an integer. Please provide an integer value.")
    # set the step size
    # print(direction)
    point = start
    h = direction*step_size
    # if all of the function values in the np.array are really close to zero, step the point off in the move off direction. 
    while np.all(np.abs(function(point)) < 1e-12):
        for i in range(len(point)):
            print("point", point)
            print("velocity", function(point))
            move_off_direction = move_off_direction_func(point[0])[i]
            point[i] = point[i] + move_off_direction[i]*1e-12
            break
    # set the initial values of k1, k2, k3, and k4
    k1 = np.array(function(point))
    k2 = np.array(function(point + 0.5 * h * k1))
    k3 = np.array(function(point + 0.5 * h * k2))
    k4 = np.array(function(point + h * k3))
    point_new = point + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    point_new = np.array(point_new)
    return point_new


if __name__ == "__main__":
    # print("here")
    # initialize the cylinder object
    cyl = cylinder("Airfoil_Input.json")
    cyl.get_full_geometry()
    cyl.plot()
    plt.show()
    # initialize the potential flow elements object
    

