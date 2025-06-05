import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from tabulate import tabulate
from complex_potential_flow_class import potential_flow_object
from tqdm import tqdm
import time
np.set_printoptions(precision=15)

class vort_panel(potential_flow_object):
    """This class contains functions that calculates position of nodes, control points, li, xi, eta, phi, psi, P_matrix, A_matrix, gamma_vector, cartesian_velocity, C_p, C_L, C_mle, C_mc/4"""

    def __init__(self, json_file):
        self.json_file = json_file
        self.load_json()

    def load_json(self):
        """This function pulls in all the input values from the json"""
        with open(self.json_file, 'r') as json_handle:
            input_vals = json.load(json_handle)
            self.is_airfoil = True
            self.is_element = False
            # geometry stuff
            self.Airfoil_type = input_vals["geometry"]["airfoil"]
            if self.Airfoil_type == "file":
                self.is_text_file = True
            else:
                self.is_text_file = False
            self.text_file_name = input_vals["geometry"]["filename"]
            self.n_geom_points = input_vals["geometry"]["n_points"]
            self.filename = input_vals["geometry"]["filename"]
            self.CL_design = input_vals["geometry"]["CL_design"]
            self.trailing_edge_condition = input_vals["geometry"]["trailing_edge"]
            # operating stuff
            self.freestream_velocity = input_vals["operating"]["freestream_velocity"]
            self.original_alpha = input_vals["operating"]["alpha[deg]"]
            # alpha sweep stuff
            self.alpha_start = (input_vals["alpha_sweep"]["start[deg]"])
            self.alpha_end = (input_vals["alpha_sweep"]["end[deg]"])
            self.alpha_increment = (input_vals["alpha_sweep"]["increment[deg]"])
            # plotting options 
            self.plot_x_start = input_vals["plot_options"]["x_start"]
            self.plot_x_lower_lim = input_vals["plot_options"]["x_lower_limit"]
            self.plot_x_upper_lim = input_vals["plot_options"]["x_upper_limit"]
            self.plot_delta_s = input_vals["plot_options"]["delta_s"]
            self.plot_n_lines = input_vals["plot_options"]["n_lines"]
            self.plot_delta_y = input_vals["plot_options"]["delta_y"]
            # run commands
            self.plot_streamlines = input_vals["run_commands"]["plot_streamlines"] # this is a boolean value
            self.plot_pressure = input_vals["run_commands"]["plot_pressure"] # this is a boolean value
            self.alpha_sweep = input_vals["run_commands"]["alpha_sweep"] # this is a boolean value
            self.export_geometry = input_vals["run_commands"]["export_geometry"] # this is a boolean value
            # other stuff
            self.full_chord = 1.0 # this is the full non-dimensional chord length of the airfoil
            self.x_leading_edge = 0.0
            self.x_trailing_edge = 1.0

    def define_Airfoil(self):
        """This function defines the NACA 4-digit airfoil"""
        # convert the NACA type string into 4 integers
        # if first two digits are not U and L then do the NACA 4-digit airfoil
        # see if the airfoil is a text file or not
        if self.Airfoil_type == "file":
            self.is_NACA_4_digit = False
            self.is_UL_NACA_1 = False
            pass
        else:
            c = int(self.Airfoil_type[2])
            a2 = int(self.Airfoil_type[3])
            if self.Airfoil_type[0] != "U" and self.Airfoil_type[1] != "L":
                self.is_NACA_4_digit = True
                self.is_UL_NACA_1 = False
                n = int(self.Airfoil_type[0])
                a1 = int(self.Airfoil_type[1])

                # calculate max camber in percent chord. This is the first digit of the NACA 4-digit code. (use n from above)
                self.max_camber_percent_chord = n / 100

                # the second digit (a1 here) is the position of the max camber in tenths of the chord away from the leading edge
                self.max_camber_position_tenths_chord = a1 / 10

                # the third and fourth digits (c and a2 here) together represent the max thickness of the airfoil in percent of chord 
                self.max_thickness_percent_chord = (c * 10 + a2) / 100

            elif self.Airfoil_type[0] == "U" and self.Airfoil_type[1] == "L":
                print("")
                self.is_UL_NACA_1 = True
                self.is_NACA_4_digit = False
                self.max_thickness_percent_chord = (c * 10 + a2) / 100
            

    def num_points_is_even(self):
        """This function checks if the number of nodes is even by using the modulo operator"""
        if self.n_geom_points % 2 == 0: # the % operator works by dividing the number of nodes by 2 and checking if there is a remainder
            self.even_num_points = True # if there is no remainder, then the number of nodes is even
        else:
            self.even_num_points = False # if there is a remainder, then the number of nodes is odd

    def calculate_point_camb(self, x: float):
        """
        Calculates the coordinates of a point on the camber line of an airfoil provided an x-coordinate in percent chord.

        Parameters:
        -x (float): The x-coordinate of the point on the camber line.

        Returns:
            Tuple[float, float]: The x and y coordinates of the point on the camber line.

        """
        if self.is_NACA_4_digit:
            xmc = self.max_camber_position_tenths_chord # x-coordinate of the max camber position
            ymc = self.max_camber_percent_chord # y-coordinate of the camber line at the max camber position
            c = self.full_chord # full chord length
            if -1 <= x <= xmc and xmc != 0:
                xc = x  # x-coordinate of the point on the camber line
                yc = ymc*(2*(x/xmc) - (x/xmc)**2) # y-coordinate of the point on the camber line
            elif xmc < x <= c and xmc != 0:
                xc = x # x-coordinate of the point on the camber line
                yc = ymc*(2*((c - x)/(c - xmc)) - ((c - x)/(1 - xmc))**2) # y-coordinate of the point on the camber line
            else:
                xc = x
                yc = 0.0
                # print("Invalid x-coordinate value. Please enter a value between 0 and 1 for an x position in percent chord.")
        elif self.is_UL_NACA_1:
            c = self.full_chord
            if x/c == 0 or x/c == 1:
                xc = x
                yc = 0.0
            else:
                outside_design_lift = self.CL_design/(4*np.pi)
                first_term = ((x/c)-1)*np.log(1-(x/c))
                second_term = -1*(x/c)*np.log(x/c)
                xc = x # x-coordinate of the point on the camber line
                yc = outside_design_lift*(first_term + second_term)
        return xc, yc
    
    def calculate_thickness(self, x: float):
        """
        Calculates the thickness of an airfoil at a given x-coordinate as a function of the maximum thickness.

        Parameters:
        -x (float): The x-coordinate of the point on the airfoil in percent chord.

        Returns:
            float: The thickness of the airfoil at the given x-coordinate.

        """
        # check trailing edge type
        if self.trailing_edge_condition == "closed": # if the trailing edge is closed
            c = self.full_chord
            tmax = self.max_thickness_percent_chord
            first_term = 2.980 * np.sqrt(x/c)
            second_term = -1.320 * (x/c)
            third_term = -3.286 * (x/c)**2
            fourth_term = 2.441 * (x/c)**3
            fifth_term = -0.815 * (x/c)**4
            t_at_x = tmax*(first_term + second_term + third_term + fourth_term + fifth_term)
        else: # if the trailing edge is open
            c = self.full_chord # full chord length
            tmax = self.max_thickness_percent_chord # max thickness of the airfoil
            first_term = 2.969 * np.sqrt(x/c)  
            second_term = -1.260 * (x/c)
            third_term = -3.516 * (x/c)**2
            fourth_term = 2.843 * (x/c)**3
            fifth_term = -1.015 * (x/c)**4
            t_at_x = tmax*(first_term + second_term + third_term + fourth_term + fifth_term) # thickness of the airfoil at the given x-coordinate
        return t_at_x
    
    def calculate_dy_dx(self, x: float): # this is used to find the upper and lower surface coordinates at a given x-coordinate in percent chord
        """
        Calculates the derivative of the camber line at a given x-coordinate.

        Parameters:
        -x (float): The x-coordinate of the point on the camber line.

        Returns:
            float: The derivative of the camber line at the given x-coordinate.

        """
        # check if it's uniform load or not
        if self.is_NACA_4_digit:
        # check if the x value is closer to the leading edge or the trailing edge
            diff_leading = np.abs(x - self.x_leading_edge)
            diff_trailing = np.abs(x - self.x_trailing_edge)
            if diff_leading < diff_trailing:
                closer_leading = True
            else:
                closer_leading = False
            
            xmc = self.max_camber_position_tenths_chord
            ymc = self.max_camber_percent_chord
            # mmake dy_dx zero if the airfoil is symmetric
            if self.max_camber_percent_chord == 0:
                return 0.0
            c = self.full_chord
            if x <= xmc:
                dy_dx = 2*ymc/xmc*(1 - x/xmc) # derivative of the camber line at the given x-coordinate before and up to the max camber position
            elif xmc < x:
                dy_dx = (2*ymc/(c-xmc))*(((c-x)/(c-xmc)) - 1) # derivative of the camber line at the given x-coordinate after the max camber position
                # dy_dx = ymc * (2 * (c - x) / (c - xmc) - (c - x) / (c - xmc)**2) # derivative of the camber line at the given x-coordinate after the max camber position
            else:
                # move the x value forward if it's less than zero and if it's closer to the leading edge
                if closer_leading:
                    x = x + diff_leading
                    dy_dx = 2*ymc/xmc*(1 - x/xmc)
                else:
                    x = x - diff_trailing
                    dy_dx = (2*ymc/(c-xmc))*(((c-x)/(c-xmc)) - 1)
                # print("Invalid x-coordinate value. Please enter a value between 0 and 1 for an x position in percent chord.")
        elif self.is_UL_NACA_1:
            if x == 0:
                x = x + 1e-6
            elif x == 1:
                x = x - 1e-6
            c = self.full_chord
            outside_design_lift = self.CL_design/(4*np.pi)
            first_term = np.log(1-(x/c))
            second_term = -1*(np.log(x/c))
            dy_dx = outside_design_lift*(first_term + second_term)
        return dy_dx
    
    def calculate_point_upper(self, x: float):
        """
        Calculates the coordinates of the upper surface of the airfoil at a given x-coordinate.

        Parameters:
        - x (float): The x-coordinate at which to calculate the upper surface coordinates.

        Returns:
        - xu (float): The x-coordinate of the upper surface point.
        - yu (float): The y-coordinate of the upper surface point.
        """
        yc = self.calculate_point_camb(x)[1]
        dy_dx = self.calculate_dy_dx(x)
        t = self.calculate_thickness(x)
        xu = x - (t/(2*np.sqrt(1 + dy_dx**2)))*dy_dx
        yu = yc + (t/(2*np.sqrt(1 + dy_dx**2)))
        return xu, yu

    def calculate_point_lower(self, x: float):
        """
        Calculates the coordinates of the lower point on the airfoil at a given x-coordinate.

        Parameters:
        - x (float): The x-coordinate at which to calculate the lower point.

        Returns:
        - xl (float): The x-coordinate of the lower point.
        - yl (float): The y-coordinate of the lower point.
        """
        yc = self.calculate_point_camb(x)[1]
        dy_dx = self.calculate_dy_dx(x)
        t = self.calculate_thickness(x)
        xl = x + (t/(2*np.sqrt(1 + dy_dx**2)))*dy_dx
        yl = yc - (t/(2*np.sqrt(1 + dy_dx**2)))
        return xl, yl
    
    def geometry_vortex(self, x: float):
        """This function takes in a chord-wise x coordinate and returns the x_camber, x_upper, x_lower,  and  coordinate of the NACA airfoil (remember that the upper and lower surface positions are perpendicular to the camber line)"""
        if not isinstance(x, float):
            raise TypeError("The input value must be a float")
        x_camber, y_camber = self.calculate_point_camb(x)
        x_upper, y_upper = self.calculate_point_upper(x)
        x_lower, y_lower = self.calculate_point_lower(x)
        return [x_upper, y_upper], [x_lower, y_lower], [x_camber, y_camber]

    def get_full_geometry_vortex(self):
        
        # check if the airfoil is a text file read in or anything else
        # if it's a text file, then read in the text file
        # if it's not a text file, then calculate the geometry
        if self.Airfoil_type == "file":
            self.read_in_txt_file()
            # change the n_geom_points to be the length of the surface points
            self.n_geom_points = len(self.surface_points)
        else:
            # Initialize arrays
            upper_coords = np.zeros((self.n_geom_points+1, 2))
            lower_coords = np.zeros((self.n_geom_points+1, 2))
            camber_coords = np.zeros((self.n_geom_points+1, 2))

            # Odd number of points case
            if not self.even_num_points:
                theta = 0.0  # Initialize theta
                x = 0.5 * (1 - np.cos(theta))  # Cosine clustering for x-coordinate
                upper_last, lower_first, camber_first = self.geometry_vortex(x)
                delta_theta = np.pi / np.real(self.n_geom_points / 2)  # Cosine clustering increment
                # Loop through and fill the coordinates
                for i in range(1, self.n_geom_points // 2 + 1):
                    itheta = np.real(i) * delta_theta
                    x = 0.5 * (1 - np.cos(itheta))  # Cosine clustering for x-coordinate

                    # Call geometry function 
                    upper, lower, camber = self.geometry_vortex(x)
                    upper_coords[self.n_geom_points // 2 + 1 + i][0] = upper[0]  # upper surface x-coordinate
                    upper_coords[self.n_geom_points // 2 + 1 + i][1] = upper[1]  # upper surface y-coordinate
                    lower_coords[self.n_geom_points // 2 + 1 - i][0] = lower[0]  # lower surface x-coordinate
                    lower_coords[self.n_geom_points // 2 + 1 - i][1] = lower[1]  # lower surface y-coordinate
                    camber_coords[i][0] = camber[0]  # camber x-coordinate
                    camber_coords[i][1] = camber[1]

                # Delete all rows that are completely zero
                upper_coords = upper_coords[~np.all(upper_coords == 0, axis=1)]
                lower_coords = lower_coords[~np.all(lower_coords == 0, axis=1)]
                camber_coords = camber_coords[~np.all(camber_coords == 0, axis=1)]

                # find the index where magnitude of x and y is closest to zero when evaluating the index of the minimum x and y coordinates in the upper chord
                min_index_upper = np.argmin(np.linalg.norm(upper_coords, axis=1))
                # find the index where magnitude of x and y is closest to zero when evaluating the index of the minimum x and y coordinates in the camber chord
                min_index_camber = np.argmin(np.linalg.norm(camber_coords, axis=1))

                # add a new row of zeros next to the minimum x and y coordinates in the upper chord
                # if the min_index_upper is the last index, then add the new row of zeros after min_index_upper
                if min_index_upper == len(upper_coords) - 1:
                    upper_coords = np.insert(upper_coords, min_index_upper+1, [0, 0], axis=0)
                elif min_index_upper == 0:
                    upper_coords = np.insert(upper_coords, min_index_upper, [0, 0], axis=0)
                if min_index_camber == len(camber_coords) - 1:
                    camber_coords = np.insert(camber_coords, min_index_camber+1, [0, 0], axis=0)
                elif min_index_camber == 0:
                    camber_coords = np.insert(camber_coords, min_index_camber, [0, 0], axis=0)

            # Even number of points case
            else:
                theta = 0.0  # Initialize theta
                x = 0.5 * (1 - np.cos(theta))  # Cosine clustering for x-coordinate
                upper, lower, camber = self.geometry_vortex(x)
                upper_coords[self.n_geom_points // 2][0] = upper[0]
                upper_coords[self.n_geom_points // 2][1] = upper[1]
                lower_coords[self.n_geom_points // 2][0] = lower[0]
                lower_coords[self.n_geom_points // 2][1] = lower[1]
                camber_coords[0][0] = camber[0]
                camber_coords[0][1] = camber[1]
                delta_theta = np.pi / ((self.n_geom_points / 2) - 0.5)  # Adjusted for even case
                # Loop through and fill the coordinates
                for i in range(1, self.n_geom_points // 2 + 1):
                    itheta = np.real(i) * delta_theta
                    x = 0.5 * (1 - np.cos(itheta - 0.5 * delta_theta))  # Cosine clustering for x-coordinate

                    # Call geometry function
                    upper, lower, camber = self.geometry_vortex(x)
                    upper_coords[self.n_geom_points // 2 + i][0] = upper[0]  # upper surface x-coordinate
                    upper_coords[self.n_geom_points // 2 + i][1] = upper[1]  # upper surface y-coordinate
                    lower_coords[self.n_geom_points // 2 + 1 - i][0] = lower[0]  # lower surface x-coordinate
                    lower_coords[self.n_geom_points // 2 + 1 - i][1] = lower[1]  # lower surface y-coordinate
                    camber_coords[i][0] = camber[0]  # camber x-coordinate
                    camber_coords[i][1] = camber[1]

                # Delete all rows that are completely zero
                upper_coords = upper_coords[~np.all(upper_coords == 0, axis=1)] # the ~ operator is a NOT operator
                lower_coords = lower_coords[~np.all(lower_coords == 0, axis=1)]
                camber_coords = camber_coords[~np.all(camber_coords == 0, axis=1)]
            
            # delete all rows that have a nan value in any column
            upper_coords = upper_coords[~np.isnan(upper_coords).any(axis=1)]
            lower_coords = lower_coords[~np.isnan(lower_coords).any(axis=1)]
            # now if there are any nan values in the camber_coords, delete the row and print out the x and y values of the camber coords
            # camber_coords = camber_coords[~np.isnan(camber_coords).any(axis=1), print("Camber coords are", camber_coords)]
            camber_coords = camber_coords[~np.isnan(camber_coords).any(axis=1)]
            
            # Convert to numpy arrays and store
            self.upper_coords = np.array(upper_coords)
            self.lower_coords = np.array(lower_coords)
            self.camber_coords = np.array(camber_coords)
            
            # make total surface_points array that starts with lower, then does upper
            self.surface_points = np.concatenate((self.lower_coords, self.upper_coords), axis=0)

            # if self.export_geometry is True, then export the geometry to a txt file
            if self.export_geometry:
                print("Exporting geometry to csv file...")
                file_name = "text_files/mesh_files/surface_mesh/naca/" + self.Airfoil_type + "_" + str(self.n_geom_points) + ".txt"
                # save text file
                np.savetxt(file_name, self.surface_points, delimiter = ",", header = "x, y", comments = "")
                print("Geometry exported to", file_name)

        return self.upper_coords, self.lower_coords, self.camber_coords
    
    def read_in_txt_file(self):
        """This function reads in a txt file and returns the x and y coordinates"""
        # skip the first row because it's the header
        if self.Airfoil_type == "file":
            x, y = np.loadtxt(self.filename, delimiter = ",", skiprows = 1, unpack = True)
            # set self.surface_points to be an empty numpy array equal to the length of self.surface points (we're redefining self.surface_points)
            self.surface_points = np.zeros((len(x), 2)) # the 2 is for the x and y coordinates
            # loop through the x and y coordinates and set the x and y coordinates of the surface points to be equal to the x and y coordinates of the txt file
            for i in tqdm(range(len(self.surface_points)), desc = "Reading in txt file..."):
                self.surface_points[i] = [x[i], y[i]]
            # self.surface_points = np.array([x, y])
            self.upper_coords = self.surface_points
            self.lower_coords = self.surface_points
            self.camber_coords = self.surface_points
        else: # raise an error if this function is called and Airfoil_type is not a txt file
            raise ValueError("The airfoil type must be a txt file")
        return self.surface_points
    
    def calc_control_points(self):
        """Given a list of nodes, this function calculates the control points by taking the average position of each pair adjacent nodes and returns them in a Nx2 list"""
        control_points = []
        for i in range(0, len(self.surface_points)-1):
            x1, y1 = self.surface_points[i]
            x2, y2 = self.surface_points[i+1]
            point = [(x1 + x2)/2, (y1 + y2)/2]
            control_points.append(point)
        control_points_array = np.array(control_points)
        self.control_points = control_points_array
        return self.control_points
    
    def calc_L(self):
        x, y = self.surface_points[:, 0], self.surface_points[:, 1]
        diff_x = np.zeros(len(x)-1)
        diff_y = np.zeros(len(y)-1)
        # loop through the x and y coordinates and calculate the difference between each point
        for i in range(len(x)):
            if i == len(x)-1: # if the index is the last index, break the loop because there is no next point
                break
            diff_x[i] = x[i+1] - x[i]
            diff_y[i] = y[i+1] - y[i]
    
        L_vals = np.sqrt(diff_x**2 + diff_y**2)
        self.L_vals = L_vals
        return self.L_vals
    
    def transform_xi_eta(self, control_x, control_y, point_x_2, point_x_1, point_y_2, point_y_1, L):
        """
        Transforms the coordinates (control_x, control_y) to the local coordinate system (xi, eta)
        defined by the points (point_x_1, point_y_1) and (point_x_2, point_y_2) with length L.

        Parameters:
        control_x (float): x-coordinate of the control point.
        control_y (float): y-coordinate of the control point.
        point_x_2 (float): x-coordinate of the second point defining the local coordinate system.
        point_x_1 (float): x-coordinate of the first point defining the local coordinate system.
        point_y_2 (float): y-coordinate of the second point defining the local coordinate system.
        point_y_1 (float): y-coordinate of the first point defining the local coordinate system.
        L (float): Length between the points (point_x_1, point_y_1) and (point_x_2, point_y_2).

        Returns:
        np.ndarray: A 2x2 array containing the transformed coordinates in the local coordinate system.
        """
        

        xi = (1/L)*((point_x_2-point_x_1)*(control_x-point_x_1)+(point_y_2-point_y_1)*(control_y-point_y_1))
        eta = (1/L)*(-(point_y_2-point_y_1)*(control_x-point_x_1)+(point_x_2-point_x_1)*(control_y-point_y_1))
        phi = np.arctan2(eta*L, eta**2+xi**2-xi*L)
        # print("phi", phi)
        if (xi-L)**2+eta**2 < 1e-10:
            eta += 1e-6
        else:
            pass
        if (xi**2+eta**2)/((xi-L)**2+eta**2) <= 1e-10:
            eta += 1e-6
        psi = 0.5*np.log((xi**2+eta**2)/((xi-L)**2+eta**2))

        # print("psi", psi)
        transform_xi_eta = np.array([[(L-xi)*phi+(eta*psi), (xi*phi)-(eta*psi)], [(eta*phi)-((L-xi)*psi)-L, (-eta*phi)-(xi*psi)+L]])
        return transform_xi_eta
    
    def calc_untransformed_p(self, x_j_plus_one, x_j, y_j_plus_one, y_j):
        """
        Calculates the untransformed matrix of the p matrix. This is in the global coordinate system.

        Parameters:
        x_j_plus_one (float): x-coordinate of the next node.
        x_j (float): x-coordinate of the current node.
        y_j_plus_one (float): y-coordinate of the next node.
        y_j (float): y-coordinate of the current node.

        Returns:
        np.ndarray: A 2x2 array containing the untransformed matrix of the p matrix.
        """
        untransformed_p = np.array([[x_j_plus_one-x_j, -(y_j_plus_one-y_j)], [y_j_plus_one-y_j, x_j_plus_one-x_j]])
        return untransformed_p
    
    def calc_p_matrix(self, mat_1, mat_2, L):
        """
        Calculates the p matrix. This is in the local coordinate system.

        Parameters:
        mat_1 (np.ndarray): The first matrix p.
        mat_2 (np.ndarray): The second matrix which is the transformation matrix xi,eta.
        L (float): The length between the points.

        Returns:
        np.ndarray: A 2x2 array containing the p matrix.
        """
        p_matrix = (1/(2*np.pi*(L**2)))*np.matmul(mat_1,mat_2)
        return p_matrix
    
    def calc_a_matrix(self):
        """This function finds the nxn a matrix given a list of nodes, control points, and correct functions that calculate xi,eta,phi,psi,li, and lj"""
        # self.x_leading_edge = 0.0 # this is the x-coordinate of the leading edge of the airfoil
        # self.x_trailing_edge = 1.0 # this is the x-coordinate of the trailing edge of the airfoil
        x, y = self.surface_points[:,0], self.surface_points[:,1]
        x_control = self.control_points[:,0]
        y_control = self.control_points[:,1]
        a_vals = np.zeros((len(self.surface_points), len(self.surface_points)))  # Initialize an empty array
        for i in range(0,len(self.surface_points)-1):
            for j in range(0,len(self.surface_points)-1):              
                l_j = self.L_vals[j]                
                l_i = self.L_vals[i]

                # define rotation matrix for x and y in the p_matrix calculation
                p_first = self.calc_untransformed_p(x[j+1], x[j], y[j+1], y[j])

                # define rotation matrix for xi and eta
                xi_eta = self.transform_xi_eta(x_control[i], y_control[i], x[j+1], x[j], y[j+1], y[j], l_j)

                # Calculate the P matrix at i, j by multiplying the untransformed p matrix by the transformation matrix, this gets the p matrix in the local coordinate system
                p_matrix = self.calc_p_matrix(p_first, xi_eta, l_j) # eq 1.6.23 Mechanics of Flight 

                a_vals[i,j] = a_vals[i,j] + ((x[i+1]-x[i])*p_matrix[1,0]-(y[i+1]-y[i])*p_matrix[0,0])/l_i # eq 1.6.25 Mechanics of Flight
                a_vals[i,j+1] = a_vals[i,j+1] + ((x[i+1]-x[i])*p_matrix[1,1]-(y[i+1]-y[i])*p_matrix[0,1])/l_i # eq 1.6.25 Mechanics of Flight   
        a_vals[len(self.surface_points)-1,0] = 1.0 # eq 1.6.26 Mechanics of Flight
        a_vals[len(self.surface_points)-1,len(self.surface_points)-1] = 1.0 # eq 1.6.27 Mechanics of Flight
        self.A_matrix = a_vals
        return self.A_matrix
    
    def calc_B_matrix(self): # this matrix is the Nx1 matrix that's in equation 4.32 in the Aeronautics engineering handbook
        """This function finds the d_matrix given the a_matrix and vel_inf"""
        x, y = self.surface_points[:,0], self.surface_points[:,1]
        B_matrix = np.zeros(len(self.surface_points))
        for i in range(0,len(self.surface_points)-1):
            diff_x = x[i+1]-x[i]
            diff_y = y[i+1]-y[i]
            l_val = self.L_vals[i]
            B_val = ((diff_y*np.cos(self.alpha))-(diff_x*np.sin(self.alpha)))/l_val
            B_matrix[i] = B_val
        self.B_matrix = B_matrix

    def calc_gamma_vector(self):
        """This function finds the gamma values given matrix_a, matrix_d and vel_inf"""
        self.gammas = np.linalg.solve(self.A_matrix, self.freestream_velocity*self.B_matrix)
        return self.gammas

    def calc_CL(self):
        """This function finds CL given gammas, vel_inf, a geometry, and l_i"""
        l_i = self.L_vals
        Coeff_L = 0.0
        for i in range(0, len(self.surface_points)-1):
            Co_L = l_i[i]*((self.gammas[i]+self.gammas[i+1])/(self.freestream_velocity)) # eq 1.6.32 Mechanics of Flight
            Coeff_L += Co_L
        self.Coeff_L = Coeff_L
        return self.Coeff_L

    def calc_Cm_le(self):
        """This function finds the moment coefficient calculated at the leading edge"""
        n = int(len(self.surface_points))
        x, y = self.surface_points[:,0], self.surface_points[:,1]
        Coeff_mle = 0
        for i in range(0,n-1):
            cos_coeff = (2*x[i]*self.gammas[i]+x[i]*self.gammas[i+1]+x[i+1]*self.gammas[i]+2*x[i+1]*self.gammas[i+1])
            sin_coeff = (2*y[i]*self.gammas[i]+y[i]*self.gammas[i+1]+y[i+1]*self.gammas[i]+2*y[i+1]*self.gammas[i+1])
            Cmle = self.L_vals[i]*(cos_coeff*np.cos(self.alpha)+sin_coeff*np.sin(self.alpha))
            Coeff_mle = Coeff_mle + Cmle
        Coeff_mle = (-1/3)*Coeff_mle/self.freestream_velocity # eq 1.6.33 Mechanics of Flight
        self.Coeff_mle = Coeff_mle
        return self.Coeff_mle
    
    def calc_Cm_c4(self):
        """This function finds the moment coefficient calculated at the quarter chord"""
        self.Cm_4 = self.Coeff_mle + 0.25*self.Coeff_L*np.cos(self.alpha)
        return self.Cm_4
    
    def force_moment_coefficients(self):
        """This is a runner function that calculates the force and moment coefficients"""
        results = []
        # create an alpha range using alpha_start, alpha_end, and alpha_increment and linspace
        if self.alpha_end - self.alpha_start != 0:
            num_steps = int((self.alpha_end - self.alpha_start) / self.alpha_increment)
            if self.alpha_sweep:
                print("\nAlpha Sweep Parameters: ")
                print("Alpha start: ", self.alpha_start)
                print("Alpha end: ", self.alpha_end)
                print("Alpha increment: ", self.alpha_increment)
            else:
                print("\nAlpha: ", self.original_alpha)
            alpha_range = []
            alpha_rad_range = []
            for i in range(num_steps+1):
                alpha_range.append(self.alpha_start + i*self.alpha_increment)
                alpha_rad_range.append(np.deg2rad(self.alpha_start + i*self.alpha_increment))
            alpha_range = np.array(alpha_range)
            alpha_rad_range = np.array(alpha_rad_range)
        else:
            alpha_range = np.array([self.alpha_start])
            alpha_rad_range = np.array([np.deg2rad(self.alpha_start)])
        if self.alpha_sweep:
            for j in range(len(alpha_range)):
                self.alpha = alpha_rad_range[j]
                alpha_deg = alpha_range[j]
                self.calc_B_matrix()
                self.calc_gamma_vector()
                self.calc_CL()
                self.calc_Cm_le()
                self.calc_Cm_c4()
                data = [alpha_deg, self.Coeff_L, self.Coeff_mle, self.Cm_4]
                results.append(data)
        else:
            self.alpha = np.deg2rad(self.original_alpha)
            self.calc_B_matrix()
            self.calc_gamma_vector()
            self.calc_CL()
            self.calc_Cm_le()
            self.calc_Cm_c4()
            data = [self.original_alpha, self.Coeff_L, self.Coeff_mle, self.Cm_4]
            results.append(data)
        if self.trailing_edge_condition == "closed":
            label = str(self.Airfoil_type) + " with " + str(self.n_geom_points) + " points and a " + str(self.trailing_edge_condition) + " trailing edge"
        else:
            label = str(self.Airfoil_type) + " with " + str(self.n_geom_points) + " points and an " + str(self.trailing_edge_condition) + " trailing edge"
        headers = ["Alpha[deg]", "C_L", "C_mle", "C_m/4"]
        print("\n", label)
        # print a horizontal line
        print("-"*len(label))
        # print the tabulated results out to 5 decimal places
        # out to 5 decimal places
        print(tabulate(results, headers, floatfmt=".5f"))
        # self.plot_geometry()
        # send the results to a csv file
        csv_title = "NACA_" + str(self.Airfoil_type) + "_" + str(self.n_geom_points) + "points.csv"
        np.savetxt(csv_title, results, delimiter = ",", header = "Alpha[deg],C_L,C_mle,C_m/4", comments = "")
        return results
    
    def initialize_velocity_calcs(self):
        """This function initializes the velocity calculations"""
        self.alpha = np.deg2rad(self.original_alpha)
        self.calc_B_matrix()
        self.calc_gamma_vector()
    
    def velocity(self, point_xy: np.array):
        """This function calculates the velocity at a point using the p_matrix and gamma values"""
        # call the init function again to get the desired angle of attack and freestream velocity from airfoil.py
        x, y = self.surface_points[:,0], self.surface_points[:,1]
        # print("angle of attack:", np.rad2deg(self.alpha))
        Vx = self.freestream_velocity*np.cos(self.alpha)
        Vy = self.freestream_velocity*np.sin(self.alpha)
        # the x velocity is the freestream vel * cos(alpha) + sum of p_matrices * gamma_i
        # the y velocity is the freestream vel * sin(alpha) + sum of p_matrices * gamma_i+1
        for i in range(0, len(self.surface_points)-1):
            l_val = self.L_vals[i]
            p_matrix = self.calc_untransformed_p(x[i+1], x[i], y[i+1], y[i])
            xi_eta = self.transform_xi_eta(point_xy[0], point_xy[1], x[i+1], x[i], y[i+1], y[i], l_val)
            p_matrix = self.calc_p_matrix(p_matrix, xi_eta, l_val)
            Vx += p_matrix[0,0]*self.gammas[i] + p_matrix[0,1]*self.gammas[i+1]
            Vy += p_matrix[1,0]*self.gammas[i] + p_matrix[1,1]*self.gammas[i+1]
        velocity_cartesian = np.array([Vx, Vy])
        return velocity_cartesian
    
    def calculate_surface_pressure(self, point_xy: np.array):
        """This function calculates the surface pressure at a point using the velocity at that point"""
        velocity = self.velocity(point_xy)
        V_xy = np.dot(velocity, velocity)
        # V_xy = np.linalg.norm(velocity)
        pressure = 1-(V_xy/self.freestream_velocity**2)
        return pressure

    def calculate_surface_pressures(self):
        """This function calculates the surface pressures at all the points on the airfoil"""
        pressures = []
        # loop through the control points and calculate the surface pressures
        for i in range(len(self.control_points)):
            # step off the point in the normal direction
            x, y = self.control_points[i]
            node_i_plus_one = self.surface_points[i+1]
            node_i = self.surface_points[i] 
            # calc tangent vector by subtracting the two nodes
            tangent_vector = node_i_plus_one - node_i
            # calc normal vector by rotating the tangent vector 90 degrees
            normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])
            # normalize the normal vector
            normal_vector = normal_vector/np.linalg.norm(normal_vector)
            # step off the point in the normal direction
            point_xy = np.array([x, y]) + 1e-6*normal_vector
            pressure = self.calculate_surface_pressure(point_xy)
            pressures.append([point_xy[0], pressure])
        pressures = np.array(pressures)
        return pressures

    def plot_surface_pressures(self):
        """This function plots the surface pressures"""
        pressures = self.calculate_surface_pressures()
        plt.plot(pressures[:, 0], pressures[:, 1], color="black")
        plt.xlabel("x/c")
        plt.ylabel("Pressure Coefficient")
        plt.title("Surface Pressure Coefficient")
        # plt.xlim(0, 1.0)
        # plt.ylim(-0.7, 1)

        # Create the annotation text
        if self.is_NACA_4_digit:
            airfoil_type_annotation = "NACA " + str(self.Airfoil_type) + " " + str(self.n_geom_points) + " nodes"
        else:
            if self.Airfoil_type == "file":
                airfoil_type_annotation = str(self.filename) + " " + str(self.n_geom_points) + " nodes"
            else:
                airfoil_type_annotation = str(self.Airfoil_type) + " " + str(self.n_geom_points) + " nodes and CLd = " + str(self.CL_design)
        closed_open_edge_annotation = "Closed Trailing Edge" if self.trailing_edge_condition == "closed" else "Open Trailing Edge"
        alpha_annotation = "$\\alpha$ = " + str(np.round(np.rad2deg(self.alpha), 7)) + "$\\degree$"
        CL_annotation = "$C_L$ = " + str(np.round(self.Coeff_L, 8))
        Cmc4_annotation = "$C_{m/4}$ = " + str(np.round(self.Cm_4, 8))
        CmLE_annotation = "$C_{m_{LE}}$ = " + str(np.round(self.Coeff_mle, 8))
        annotation_text = f"{airfoil_type_annotation}\n{closed_open_edge_annotation}\n{alpha_annotation}\n{CL_annotation}\n{Cmc4_annotation}\n{CmLE_annotation}"

        # Create an AnchoredText object
        at = AnchoredText(annotation_text, prop={"ma": 'right'}, frameon=True, loc='upper right')
        at.patch.set_boxstyle("square,pad=0.")
        at.patch.set_linewidth(0.5)
        at.patch.set_edgecolor('black')
        at.patch.set_facecolor('white')
        at.patch.set_alpha(0.5)
        # add shadow to box

        plt.xlim(0, 1.0)

        # Add the AnchoredText object to the plot
        plt.gca().add_artist(at)

        # Make the plot decrease in the positive y direction
        plt.gca().invert_yaxis()

        # Save the figure
        if self.Airfoil_type == "file":
            plt.savefig(str(self.filename) + "_surface_pressure_at_" + str(self.original_alpha) + "_degrees.png")
        else:
            plt.savefig(str(self.Airfoil_type) + "_surface_pressure_at_" + str(self.original_alpha) + "_degrees.png")
        plt.show()

if __name__ == "__main__":
    json_file = "vortex_panel_input.json"
    vort = vort_panel(json_file)
    time_1 = time.time()    
    vort.num_points_is_even()
    vort.define_Airfoil()
    vort.get_full_geometry_vortex()
    vort.calc_L()
    vort.calc_control_points()
    vort.calc_a_matrix()
    vort.force_moment_coefficients()
    vort.initialize_velocity_calcs()

    if vort.plot_streamlines:
        vort.plot_geometry()
        # vort.plot()
        plt.xlim(vort.plot_x_lower_lim, vort.plot_x_upper_lim)
        plt.ylim(-0.5, 0.5)
        plt.gca().set_aspect("equal")
        plt.show()
    # now if the user wants to plot the surface pressures, then plot the surface pressures
    if vort.plot_pressure:
        plt.figure()
        vort.plot_surface_pressures()
        plt.show()

    time_2 = time.time()
    print("Time taken:", time_2-time_1)
    