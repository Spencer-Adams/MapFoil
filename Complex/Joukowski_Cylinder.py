import json
import numpy as np # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import scipy.integrate as spi # this is for numerical integration
# import bisection from scipy.optimize
import helper as hlp
from complex_potential_flow_class import potential_flow_object


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
            self.is_test_appellian = input["operating"]["is_test_appellian"]
            self.save_fig = input["plot"]["save_fig"]
            self.show_fig = input["plot"]["show_fig"]
            self.plot_text = input["plot"]["plot_text"]
            self.do_plot_streamlines = input["plot"]["do_plot_streamlines"]
            self.cylinder_radius = input["geometry"]["cylinder_radius"] # this is the radius of the cylinder in meters. We are normalizing everything by this value. This is used in the geometry function to calculate the geometry of the cylinder.
            self.original_cylinder_radius = self.cylinder_radius # this is the original cylinder radius in meters. We are normalizing everything by this value. This is used in the geometry function to calculate the geometry of the cylinder.
            self.cylinder_radius/=self.original_cylinder_radius # normalize the cylinder radius to 1. This makes the math easier. We are normalizing everything by the cylinder radius. This is used in the geometry function to calculate the geometry of the cylinder.
            self.is_airfoil = False
            self.is_text_file = False
            self.is_element = False
            self.is_cylinder = True
            self.type = input["geometry"]["type"]
            angle_of_attack = input["operating"]["angle_of_attack[deg]"]
            self.angle_of_attack = np.radians(angle_of_attack)
            self.freestream_velocity = input["operating"]["freestream_velocity"]
            self.design_CL = input["geometry"]["design_CL"]
            print("\ndesign CL", self.design_CL)
            self.design_thickness = input["geometry"]["design_thickness"]
            print("design thickness", self.design_thickness)
            self.output_points = input["geometry"]["output_points"]
            if self.output_points % 2 == 0:
                self.even_num_points = True
            else:
                self.even_num_points = False
            self.n_geom_points = self.output_points

            if self.type == "cylinder":
                self.circulation = input["operating"]["vortex_strength"]
                zeta_center = input["geometry"]["zeta_0"] # this is a list of two elements, real and imaginary parts of zeta_0. We need to convert it to a complex number
                self.zeta_center = (zeta_center[0] + 1j*zeta_center[1])/self.cylinder_radius
                self.epsilon = input["geometry"]["epsilon"]
            else:
                self.zeta_center = self.calc_J_airfoil_zeta_center() # uses design CL and thickness to calculate zeta center
                print("J airfoil zeta center", self.zeta_center)
                self.epsilon = self.calc_J_airfoil_epsilon() # uses the cylinder radius, and zeta center to calculate epsilon
                print("J airfoil epsilon", self.epsilon)
                self.circulation = self.calc_circulation_J_airfoil() # uses the Kutta condition to calculate circulation
                print("J airfoil circulation", self.circulation)
                # calculate gamma so that it satisfies the Kutta condition for the airfoil
    
    def geometry_zeta(self, x_coord: float): # A1 on project
        """
        Calculate the geometry of an airfoil at a given x-coordinate.

        Parameters:
        - x_coord (float): The x-coordinate at which to calculate the airfoil geometry.

        Returns:
        - upper (list): The upper surface coordinates of the airfoil at the given x-coordinate.
        - lower (list): The lower surface coordinates of the airfoil at the given x-coordinate.
        - camber (list): The camber line coordinates of the airfoil at the given x-coordinate.
        """
        x_shifted = (x_coord - np.real(self.zeta_center))/self.cylinder_radius
        radius = self.cylinder_radius
        theta = np.arccos(x_shifted/radius)
        upper = (x_coord + 1j*radius*np.sin(theta) + 1j*self.zeta_center.imag)/self.cylinder_radius
        lower = (x_coord - 1j*radius*np.sin(theta) + 1j*self.zeta_center.imag)/self.cylinder_radius
        camber = (x_coord + 1j*self.zeta_center.imag)/self.cylinder_radius
        # shift the coordinates to the center of the cylinder based on zeta_center
        return [upper.real, upper.imag], [lower.real, lower.imag], [camber.real, camber.imag]

    def get_full_geometry_zeta(self):
        """This function calls the geometry function across every point on the cylinder"""
        # create an empty array for the upper and lower surfaces and camber 
        n = 1000
        upper = np.zeros((n,2))
        lower = np.zeros((n,2))
        camber = np.zeros((n,2))
        x_coords = np.linspace(self.x_leading_edge, self.x_trailing_edge, n) # this is creating an array of x coordinates
        for i in range(n):
            upper[i], lower[i], camber[i] = self.geometry_zeta(np.real(x_coords[i]))
            # print("upper", upper[i], "lower", lower[i])
        self.upper_zeta_coords = upper
        self.lower_zeta_coords = lower
        self.camber_zeta_coords = camber
    
    def plot_geometry_zeta(self):
        """"""
        # make the line type dashed
        linetype = '--'
        color = 'black'
        plt.plot(self.upper_zeta_coords[:,0], self.upper_zeta_coords[:,1], label = "Cyl", linestyle=linetype, color=color)
        plt.plot(self.lower_zeta_coords[:,0], self.lower_zeta_coords[:,1], linestyle=linetype, color=color)
        # plt.plot(self.camber_zeta_coords[:,0], self.camber_zeta_coords[:,1], label="Camber Line", linestyle=linetype, color=color)
        # plot the zeta center with a hallow circle marker
        size = 40
        plt.scatter(self.zeta_center.real, self.zeta_center.imag, color=color, marker='o', s = size, facecolors='none', label="$\\zeta_0$")

    def geometry(self, x_from_zeta: float): # A1 on project
        """This function calculates the geometry of the cylinder at a given x-coordinate"""
        zeta_upper = self.geometry_zeta(x_from_zeta)[0][0] + 1j*self.geometry_zeta(x_from_zeta)[0][1]
        zeta_lower = self.geometry_zeta(x_from_zeta)[1][0] + 1j*self.geometry_zeta(x_from_zeta)[1][1]
        zeta_camber = self.geometry_zeta(x_from_zeta)[2][0] + 1j*self.geometry_zeta(x_from_zeta)[2][1]

        z_upper = self.zeta_to_z(zeta_upper)
        z_lower = self.zeta_to_z(zeta_lower)
        z_camber = self.zeta_to_z(zeta_camber)
        return [z_upper.real, z_upper.imag], [z_lower.real, z_lower.imag], [z_camber.real, z_camber.imag]  

    def get_full_geometry(self):
        """This function calculates the geometry of the cylinder at every point"""
        if self.type == "airfoil" and self.output_points <= 3:
            # raise value error if the number of points is less than or equal to  3
            print("\n")
            raise ValueError("The number of points must be greater than 3")
        elif self.type == "airfoil" and self.output_points > 3:
            num_points = self.output_points//2 
        else:
            num_points = 500
        # num_points = 50000
        upper = np.zeros((num_points, 2))  # Ensure upper has the correct size
        lower = np.zeros((num_points, 2))
        camber = np.zeros((num_points, 2))
        
        # Make sure that x_coords has points at the leading and trailing edges
        x_coords = np.linspace(self.x_leading_edge, self.x_trailing_edge, num_points)
        
        for i in range(num_points):
            upper[i], lower[i], camber[i] = self.geometry(x_coords[i])
        
        self.upper_coords = upper
        self.lower_coords = lower
        self.camber_coords = camber
        return self.upper_coords, self.lower_coords, self.camber_coords
    
    def get_and_plot_foci(self):
        """This function calculates and plots the foci of the ellipse using epsilon"""
        self.zeta_trailing_edge_focus = 1*(self.cylinder_radius- self.epsilon)/self.cylinder_radius
        self.zeta_leading_edge_focus = -1*(self.cylinder_radius - self.epsilon)/self.cylinder_radius
        self.z_trailing_edge_focus = self.zeta_to_z(self.zeta_trailing_edge_focus)#2*(self.cylinder_radius - self.epsilon)
        self.z_leading_edge_focus = self.zeta_to_z(self.zeta_leading_edge_focus)#-2*(self.cylinder_radius - self.epsilon)
        print("Real component of z_leading_edge singularity", self.z_leading_edge_focus)
        print("Real component of z_trailing_edge singularity", self.z_trailing_edge_focus)
        # list_of_all_possible_markers = ['o', 's', 'D', 'v', '^', '<', '>', 'p', 'P', '*', 'X', 'd', 'H', 'h', '+', '|', '_', '1', '2', '3', '4', '8']
        # D is a diamond, s is a square, o is a circle, v is a triangle pointing down, ^ is a triangle pointing up
        # reduce size of markers 
        size = 30
        size_tri = 40
        plt.scatter(self.zeta_trailing_edge_focus, 0.0, color='black', marker='^', s = size_tri, label="sing")
        plt.scatter(self.zeta_leading_edge_focus, 0.0, color='black', marker='^', s = size_tri)
        plt.scatter(self.z_trailing_edge_focus, 0.0, color='black', marker='s', s = size, label = "J-np.sing")
        plt.scatter(self.z_leading_edge_focus, 0.0, color='black', marker='s', s = size)

    def velocity_zeta(self, point_xy, Gamma): # A4 on project
        """
        Calculates the velocity at a given point in the flow field in cartesian coordinates using cylindrical velocity equations.
        Parameters:
        - point_xy (list): An xy coordinate.
        Returns:
        - velocity (list): The velocity at the given point.
        """
        z = point_xy[0] + 1j*point_xy[1]
        w_z = self.freestream_velocity*((np.exp(-1j*self.angle_of_attack)) + 1j*Gamma/(2*np.pi*self.freestream_velocity*(z-self.zeta_center)) + -1*(np.exp(1j*self.angle_of_attack))*self.cylinder_radius**2/(z-self.zeta_center)**2)  
        velocity_complex = np.array([w_z.real, -w_z.imag])     
        return velocity_complex
    
    def velocity(self, point_xy_in_z_plane, Gamma):
        """This function calculates the velocity at a given point in the flow field in the z plane"""
        z = point_xy_in_z_plane[0] + 1j*point_xy_in_z_plane[1]
        zeta = self.z_to_zeta(z)
        xi, eta = zeta.real, zeta.imag
        velocity = self.freestream_velocity*(np.exp(-1j*self.angle_of_attack) + 1j*Gamma/(2*np.pi*self.freestream_velocity*(zeta-self.zeta_center)) - np.exp(1j*self.angle_of_attack)*self.cylinder_radius**2/(zeta-self.zeta_center)**2) / (1 - (self.cylinder_radius-self.epsilon)**2/(zeta)**2) # eq 107
        velocity_complex = np.array([velocity.real, -velocity.imag])
        return velocity_complex

    def split_velocity_z(self, point_xy_in_z_plane, Gamma):
        """This function calculates the velocity at a given point in the flow field in the z plane"""
        z = point_xy_in_z_plane[0] + 1j*point_xy_in_z_plane[1]
        zeta = self.z_to_zeta(z)
        xi, eta = zeta.real, zeta.imag
        G1, G2, G3, G4, G5, G6 = self.calc_z_G_values(xi, eta, self.angle_of_attack, self.epsilon, self.cylinder_radius, self.zeta_center.real, self.zeta_center.imag)
        omega_real = (Gamma/(2*np.pi))*((G1*G5+G2*G6)/(G5**2 + G6**2)) + self.freestream_velocity*((G5*np.cos(self.angle_of_attack)-G6*np.sin(self.angle_of_attack)-G3*G5-G4*G6)/(G5**2 + G6**2))
        omega_imag = -(Gamma/(2*np.pi))*((G2*G5-G1*G6)/(G5**2 + G6**2)) + self.freestream_velocity*((G6*np.cos(self.angle_of_attack)+G5*np.sin(self.angle_of_attack)+G4*G5-G3*G6)/(G5**2 + G6**2))
        return omega_real, omega_imag

    def velocity_chi(self, point_xy_in_Chi_plane, Gamma):
        """Start with a Chi value that is shifted from zeta_center"""
        xi, eta = point_xy_in_Chi_plane[0], point_xy_in_Chi_plane[1]
        r, theta = hlp.xy_to_r_theta(xi, eta)
        zeta_center = self.zeta_center
        xio, etao = zeta_center.real, zeta_center.imag
        r0, theta0 = hlp.xy_to_r_theta(xio, etao)
        V_inf, R, alpha, epsilon = self.freestream_velocity, self.cylinder_radius, self.angle_of_attack, self.epsilon
        G1, G2, G3, G4, G5, G6 = self.calc_Chi_G_values(r, theta, alpha, epsilon, R, r0, theta0)
        V_real = (Gamma/(2*np.pi))*((G1*G5+G2*G6)/(G5**2 + G6**2)) + V_inf*((G5*np.cos(alpha)-G6*np.sin(alpha)-G3*G5-G4*G6)/(G5**2 + G6**2))
        V_imag = (-1*Gamma/(2*np.pi))*((G2*G5-G1*G6)/(G5**2 + G6**2)) + V_inf*((G6*np.cos(alpha)+G5*np.sin(alpha)+G4*G5-G3*G6)/(G5**2 + G6**2))
        velocity_complex = np.array([V_real, V_imag])
        return velocity_complex
    
    def chi_acceleration(self, point_xy_in_Chi_plane, Gamma):
        """This function calculates the acceleration at a given point in the flow field in the Chi plane"""
        chi = point_xy_in_Chi_plane[0] + 1j*point_xy_in_Chi_plane[1]
        xi, eta = point_xy_in_Chi_plane[0], point_xy_in_Chi_plane[1]
        r, theta = hlp.xy_to_r_theta(xi, eta)
        xio, etao = self.zeta_center.real, self.zeta_center.imag
        r0, theta0 = hlp.xy_to_r_theta(xio, etao)
        V_inf, R, alpha, epsilon = self.freestream_velocity,  self.cylinder_radius, self.angle_of_attack, self.epsilon
        acceleration = V_inf*(-1j*Gamma/(2*np.pi*V_inf*chi**2) + 2*np.exp(1j*alpha)*R**2/(chi**3)) / (1 - (R-epsilon)**2/(chi+self.zeta_center)**2)**2
        acceleration_complex = np.array([acceleration.real, -acceleration.imag])
        acceleration_r = np.cos(theta)*acceleration_complex[0] + np.sin(theta)*acceleration_complex[1]
        acceleration_theta = np.cos(theta)*acceleration_complex[1] - np.sin(theta)*acceleration_complex[0]
        acceleration_polar = np.array([acceleration_r, acceleration_theta])
        # print("\n\nradial acceleration!", acceleration_r)
        # print("theta acceleration!", acceleration_theta)
        # print("acceleration squared!", polar_acceleration_squared)
        return acceleration_polar
    
    def chi_acceleration_squared(self, point_xy_in_Chi_plane, Gamma):
        """This function calculates the acceleration squared at a given point in the flow field in the Chi plane"""
        acceleration = self.chi_acceleration(point_xy_in_Chi_plane, Gamma)
        acceleration_squared = np.dot(acceleration, acceleration)
        return acceleration_squared

    def polar_velocity(self, theta, cartesian_velocity):
        """This function converts the cartesian velocity to polar velocity (can go from z, zeta, or chi plane to polar velocity)"""
        velocity_r = cartesian_velocity[0]*np.cos(theta) + cartesian_velocity[1]*np.sin(theta)
        # print("\nradial velocity", velocity_r)
        velocity_theta = cartesian_velocity[1]*np.cos(theta) - cartesian_velocity[0]*np.sin(theta)
        # print("theta velocity", velocity_theta)
        polar_velocity = np.array([velocity_r, velocity_theta])
        return polar_velocity
    
    def calc_z_G_values(self, xi, eta, alpha, epsilon, R, xio, etao):
        """Cartesian G values for the z plane"""
        G1 = (eta-etao)/((xi-xio)**2 + (eta-etao)**2)
        G2 = (xi-xio)/((xi-xio)**2 + (eta-etao)**2)
        G3 = R**2*(np.cos(alpha)*((xi-xio)**2-(eta-etao)**2)+2*np.sin(alpha)*((xi-xio)*(eta-etao)))/(((xi-xio)**2 -(eta-etao)**2)**2 + (2*(xi-xio)*(eta-etao))**2)
        G4 = R**2*(np.sin(alpha)*((xi-xio)**2-(eta-etao)**2)-2*np.cos(alpha)*((xi-xio)*(eta-etao)))/(((xi-xio)**2 -(eta-etao)**2)**2 + (2*(xi-xio)*(eta-etao))**2)
        G5 = 1 - ((xi**2 - eta**2)*(R-epsilon)**2)/(((xi**2 - eta**2)**2 + (2*xi*eta)**2))
        G6 = ((2*xi*eta)*(R-epsilon)**2)/(((xi**2 - eta**2)**2 + (2*xi*eta)**2))
        return G1, G2, G3, G4, G5, G6
    
    def calc_Chi_G_values(self, r, theta, alpha, epsilon, R, r0, theta0):
        """takes in r, theta, alpha, epsilon, R, r0, theta0 and calculates the G values"""
        G1 = np.sin(theta)/r
        G2 = np.cos(theta)/r
        G3 = R**2*np.cos(alpha-2*theta)/r**2
        G4 = R**2*np.sin(alpha-2*theta)/r**2
        G5 = 1 - ((R-epsilon)**2*(r**2*np.cos(2*theta)+r0**2*np.cos(2*theta0)+2*r*r0*np.cos(theta+theta0)))/((r**2*np.cos(2*theta)+r0**2*np.cos(2*theta0)+2*r*r0*np.cos(theta+theta0))**2+(r**2*np.sin(2*theta)+r0**2*np.sin(2*theta0)+2*r*r0*np.sin(theta+theta0))**2)
        G6 = ((R-epsilon)**2*(r**2*np.sin(2*theta)+r0**2*np.sin(2*theta0)+2*r*r0*np.sin(theta+theta0)))/((r**2*np.cos(2*theta)+r0**2*np.cos(2*theta0)+2*r*r0*np.cos(theta+theta0))**2+(r**2*np.sin(2*theta)+r0**2*np.sin(2*theta0)+2*r*r0*np.sin(theta+theta0))**2)
        return G1, G2, G3, G4, G5, G6
    
    def all_partials(self, r, theta, alpha, epsilon, R, r0, theta0, Gamma):
        """This function calculates all the partial derivatives of omega r and omega theta"""
        G1, G2, G3, G4, G5, G6 = self.calc_Chi_G_values(r, theta, alpha, epsilon, R, r0, theta0)
        A_omega_rG, B_omega_rG, C_omega_rG, D_omega_rG, E_omega_rG, F_omega_rG, G_omega_rG, H_omega_rG = self.partial_coefficients_of_omega_r_wrt_G_values(self.freestream_velocity, theta, self.angle_of_attack, G1, G2, G3, G4, G5, G6)
        A_omega_thetaG, B_omega_thetaG, C_omega_thetaG, D_omega_thetaG, E_omega_thetaG, F_omega_thetaG, G_omega_thetaG, H_omega_thetaG = self.partial_coefficients_of_omega_theta_wrt_G_values(self.freestream_velocity, theta, self.angle_of_attack, G1, G2, G3, G4, G5, G6)
        AGr, BGr, CGr, DGr, EGr, FGr = self.partial_coefficients_of_G_wrt_r(r, theta, alpha, epsilon, R, r0, theta0)
        AGtheta, BGtheta, CGtheta, DGtheta, EGtheta, FGtheta = self.partial_coefficients_of_G_wrt_theta(r, theta, alpha, epsilon, R, r0, theta0)
        # omega r partials
        d_omega_r_dr = Gamma*A_omega_rG*AGr + Gamma*B_omega_rG*BGr + C_omega_rG*CGr + D_omega_rG*DGr + EGr*(Gamma*E_omega_rG + F_omega_rG) + FGr*(Gamma*G_omega_rG + H_omega_rG)
        d_omega_r_dtheta = Gamma*A_omega_rG*AGtheta + Gamma*B_omega_rG*BGtheta + C_omega_rG*CGtheta + D_omega_rG*DGtheta + EGtheta*(Gamma*E_omega_rG + F_omega_rG) + FGtheta*(Gamma*G_omega_rG + H_omega_rG)
        # omega theta partials
        d_omega_theta_dr = Gamma*A_omega_thetaG*AGr + Gamma*B_omega_thetaG*BGr + C_omega_thetaG*CGr + D_omega_thetaG*DGr + EGr*(Gamma*E_omega_thetaG + F_omega_thetaG) + FGr*(Gamma*G_omega_thetaG + H_omega_thetaG)
        d_omega_theta_dtheta = Gamma*A_omega_thetaG*AGtheta + Gamma*B_omega_thetaG*BGtheta + C_omega_thetaG*CGtheta + D_omega_thetaG*DGtheta + EGtheta*(Gamma*E_omega_thetaG + F_omega_thetaG) + FGtheta*(Gamma*G_omega_thetaG + H_omega_thetaG)
        print("d_omega_r_dr:                    ", d_omega_r_dr)
        print("d_omega_r_dtheta:                    ", d_omega_r_dtheta)
        print("d_omega_theta_dr                     ", d_omega_theta_dr)
        print("d_omega_theta_dtheta                     ", d_omega_theta_dtheta)
        return d_omega_r_dr, d_omega_r_dtheta, d_omega_theta_dr, d_omega_theta_dtheta
    
    def partial_coefficients_of_omega_r_wrt_G_values(self, V_inf, theta, alpha, G1, G2, G3, G4, G5, G6):
        """This function retrieves the partial derivatives of omega r with respect to G values"""
        # G1, G2, G3, G4, G5, G6 = self.calc_Chi_G_values(r, theta, alpha, epsilon, R, r0, theta0)
        # omega r partials with respect to G1, G2, G3, G4, G5, G6
        A_omega_rG = 1/(2*np.pi)*((G5*np.cos(theta))/(G5**2 + G6**2)+ (G6*np.sin(theta))/(G5**2 + G6**2))
        B_omega_rG = 1/(2*np.pi)*((G6*np.cos(theta))/(G5**2 + G6**2) - (G5*np.sin(theta))/(G5**2 + G6**2))
        C_omega_rG = - V_inf*(G5*np.cos(theta)/(G5**2 + G6**2) + G6*np.sin(theta)/(G5**2 + G6**2))
        D_omega_rG = V_inf*(G5*np.sin(theta)/(G5**2 + G6**2) - G6*np.cos(theta)/(G5**2 + G6**2))
        E_omega_rG = (G1*np.cos(theta))/(2*np.pi*(G5**2 + G6**2)) - (G2*np.sin(theta))/(2*np.pi*(G5**2 + G6**2)) - (G5*np.cos(theta)*(G1*G5 + G2*G6))/(np.pi*(G5**2 + G6**2)**2) - (G5*np.sin(theta)*(G1*G6 - G2*G5))/(np.pi*(G5**2 + G6**2)**2)
        F_omega_rG = V_inf*((np.sin(theta)*(G4+np.sin(alpha)))/(G5**2+G6**2)-(np.cos(theta)*(G3-np.cos(alpha)))/(G5**2+G6**2)+(2*G5*np.cos(theta)*(G6*np.sin(alpha)-G5*np.cos(alpha)+G3*G5+G4*G6))/(G5**2+G6**2)**2-(2*G5*np.sin(theta)*(G6*np.cos(alpha)+G5*np.sin(alpha)-G3*G6+G4*G5))/(G5**2+G6**2)**2)
        G_omega_rG = (G2*np.cos(theta))/(2*np.pi*(G5**2 + G6**2)) + (G1*np.sin(theta)/(2*np.pi*(G5**2+G6**2))) - (G6*np.cos(theta)*(G1*G5+G2*G6))/(np.pi*(G5**2+G6**2)**2) - (G6*np.sin(theta)*(G1*G6-G2*G5))/(np.pi*(G5**2+G6**2)**2)
        H_omega_rG = V_inf*((2*G6*np.cos(theta)*(G6*np.sin(alpha)-G5*np.cos(alpha)+G3*G5+G4*G6))/(G5**2+G6**2)**2-(np.sin(theta)*(G3-np.cos(alpha)))/(G5**2+G6**2)-(np.cos(theta)*(G4+np.sin(alpha)))/(G5**2+G6**2)-(2*G6*np.sin(theta)*(G6*np.cos(alpha)+G5*np.sin(alpha)-G3*G6+G4*G5))/(G5**2+G6**2)**2)
        return A_omega_rG, B_omega_rG, C_omega_rG, D_omega_rG, E_omega_rG, F_omega_rG, G_omega_rG, H_omega_rG

    def partial_coefficients_of_omega_theta_wrt_G_values(self, V_inf, theta, alpha, G1, G2, G3, G4, G5, G6):
        """This function retrieves the partial derivatives of omega theta with respect to G values"""
        A_omega_thetaG = 1/(2*np.pi)*((G6*np.cos(theta))/(G5**2 + G6**2) - (G5*np.sin(theta))/(G5**2 + G6**2))
        B_omega_thetaG = 1/(2*np.pi)*(-(G5*np.cos(theta))/(G5**2 + G6**2) - (G6*np.sin(theta))/(G5**2 + G6**2))
        C_omega_thetaG = V_inf*(G5*np.sin(theta)/(G5**2 + G6**2) - G6*np.cos(theta)/(G5**2 + G6**2))
        D_omega_thetaG = V_inf*(G5*np.cos(theta)/(G5**2 + G6**2) + G6*np.sin(theta)/(G5**2 + G6**2))
        E_omega_thetaG = -((G2*np.cos(theta))/(2*np.pi*(G5**2 + G6**2)) + (G1*np.sin(theta)/(2*np.pi*(G5**2+G6**2))) + (G5*np.cos(theta)*(G1*G6-G2*G5))/(np.pi*(G5**2+G6**2)**2) - (G5*np.sin(theta)*(G1*G5+G2*G6))/(np.pi*(G5**2+G6**2)**2))
        F_omega_thetaG = V_inf*((np.cos(theta)*(G4+np.sin(alpha)))/(G5**2+G6**2)+(np.sin(theta)*(G3-np.cos(alpha)))/(G5**2+G6**2)-(2*G5*np.cos(theta)*(G6*np.cos(alpha)+G5*np.sin(alpha)-G3*G6+G4*G5))/(G5**2+G6**2)**2-(2*G5*np.sin(theta)*(G6*np.sin(alpha)-G5*np.cos(alpha)+G3*G5+G4*G6))/(G5**2+G6**2)**2)
        G_omega_thetaG = (G1*np.cos(theta))/(2*np.pi*(G5**2 + G6**2)) - (G2*np.sin(theta)/(2*np.pi*(G5**2+G6**2))) - (G6*np.cos(theta)*(G1*G6-G2*G5))/(np.pi*(G5**2+G6**2)**2) + (G6*np.sin(theta)*(G1*G5+G2*G6))/(np.pi*(G5**2+G6**2)**2)
        H_omega_thetaG = V_inf*((np.sin(theta)*(G4+np.sin(alpha)))/(G5**2+G6**2)-(np.cos(theta)*(G3-np.cos(alpha)))/(G5**2+G6**2)-(2*G6*np.cos(theta)*(G6*np.cos(alpha)+G5*np.sin(alpha)-G3*G6+G4*G5))/(G5**2+G6**2)**2-(2*G6*np.sin(theta)*(G6*np.sin(alpha)-G5*np.cos(alpha)+G3*G5+G4*G6))/(G5**2+G6**2)**2)
        return A_omega_thetaG, B_omega_thetaG, C_omega_thetaG, D_omega_thetaG, E_omega_thetaG, F_omega_thetaG, G_omega_thetaG, H_omega_thetaG

    def partial_coefficients_of_G_wrt_r(self, r, theta, alpha, epsilon, R, r0, theta0):
        """This function retrieves the partial derivatives of G values with respect to r"""
        AGr = -np.sin(theta)/r**2
        BGr = -np.cos(theta)/r**2
        CGr = (R**2*(r**2*np.cos(2*theta)*np.cos(alpha) + r**2*np.sin(2*theta)*np.sin(alpha))*(2*r**3*(np.cos(4*theta) - 1) - 4*r**3*np.cos(2*theta)**2))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2)**2 - (R**2*(2*r*np.cos(2*theta)*np.cos(alpha) + 2*r*np.sin(2*theta)*np.sin(alpha)))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2)
        DGr = (R**2*(r**2*np.cos(2*theta)*np.sin(alpha) - r**2*np.sin(2*theta)*np.cos(alpha))*(2*r**3*(np.cos(4*theta) - 1) - 4*r**3*np.cos(2*theta)**2))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2)**2 - (R**2*(2*r*np.cos(2*theta)*np.sin(alpha) - 2*r*np.sin(2*theta)*np.cos(alpha)))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2)
        EGr = ((R - epsilon)**2*(2*(2*r*np.cos(2*theta) + 2*r0*np.cos(theta + theta0))*(np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2) + 2*(2*r*np.sin(2*theta) + 2*r0*np.sin(theta + theta0))*(np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))*(np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2)**2 - ((R - epsilon)**2*(2*r*np.cos(2*theta) + 2*r0*np.cos(theta + theta0)))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2)
        FGr= ((2*r*np.sin(2*theta) + 2*r0*np.sin(theta + theta0))*(R - epsilon)**2)/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2) - ((R - epsilon)**2*(2*(2*r*np.cos(2*theta) + 2*r0*np.cos(theta + theta0))*(np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2) + 2*(2*r*np.sin(2*theta) + 2*r0*np.sin(theta + theta0))*(np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))*(np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2)**2
        return AGr, BGr, CGr, DGr, EGr, FGr

    def partial_coefficients_of_G_wrt_theta(self, r, theta, alpha, epsilon, R, r0, theta0):
        AGtheta = np.cos(theta)/r
        BGtheta  = -np.sin(theta)/r
        CGtheta = -(R**2*(2*r**2*np.cos(2*theta)*np.sin(alpha) - 2*r**2*np.sin(2*theta)*np.cos(alpha)))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2) - (R**2*(2*r**4*np.sin(4*theta) - 4*r**4*np.cos(2*theta)*np.sin(2*theta))*(r**2*np.cos(2*theta)*np.cos(alpha) + r**2*np.sin(2*theta)*np.sin(alpha)))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2)**2
        DGtheta = (R**2*(2*r**2*np.cos(2*theta)*np.cos(alpha) + 2*r**2*np.sin(2*theta)*np.sin(alpha)))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2) - (R**2*(2*r**4*np.sin(4*theta) - 4*r**4*np.cos(2*theta)*np.sin(2*theta))*(r**2*np.cos(2*theta)*np.sin(alpha) - r**2*np.sin(2*theta)*np.cos(alpha)))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2)**2
        EGtheta = ((R - epsilon)**2*(2*np.sin(2*theta)*r**2 + 2*r0*np.sin(theta + theta0)*r))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2) - ((2*(2*np.sin(2*theta)*r**2 + 2*r0*np.sin(theta + theta0)*r)*(np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2) - 2*(2*np.cos(2*theta)*r**2 + 2*r0*np.cos(theta + theta0)*r)*(np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))*(R - epsilon)**2*(np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2)**2
        FGtheta = ((R - epsilon)**2*(2*np.cos(2*theta)*r**2 + 2*r0*np.cos(theta + theta0)*r))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2) + ((2*(2*np.sin(2*theta)*r**2 + 2*r0*np.sin(theta + theta0)*r)*(np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2) - 2*(2*np.cos(2*theta)*r**2 + 2*r0*np.cos(theta + theta0)*r)*(np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))*(R - epsilon)**2*(np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2)**2
        return AGtheta, BGtheta, CGtheta, DGtheta, EGtheta, FGtheta
    
    def calc_A1_through_A12(self,  V_inf, theta, alpha, G1, G2, G3, G4, G5, G6, A_omega_rG, B_omega_rG, C_omega_rG, D_omega_rG, E_omega_rG, F_omega_rG, G_omega_rG, H_omega_rG, A_omega_thetaG, B_omega_thetaG, C_omega_thetaG, D_omega_thetaG, E_omega_thetaG, F_omega_thetaG, G_omega_thetaG, H_omega_thetaG, AGr, BGr, CGr, DGr, EGr, FGr, AGtheta, BGtheta, CGtheta, DGtheta, EGtheta, FGtheta):
        """This function calculates the A1 through A12 values"""
        A1 = 1/(2*np.pi)*(np.cos(theta)*(G1*G5+G2*G6)/(G5**2 + G6**2) - np.sin(theta)*(G2*G5-G1*G6)/(G5**2 + G6**2))
        A2 = V_inf*(np.cos(theta)*(G5*np.cos(alpha)-G6*np.sin(alpha)-G3*G5-G4*G6)/(G5**2 + G6**2) + np.sin(theta)*(G6*np.cos(alpha)+G5*np.sin(alpha)+G4*G5-G3*G6)/(G5**2 + G6**2))
        A3 = 1/(2*np.pi)*(-np.cos(theta)*(G2*G5-G1*G6)/(G5**2 + G6**2) - np.sin(theta)*(G1*G5+G2*G6)/(G5**2 + G6**2))
        A4 = V_inf*(np.cos(theta)*(G6*np.cos(alpha)+G5*np.sin(alpha)+G4*G5-G3*G6)/(G5**2 + G6**2) - np.sin(theta)*(G5*np.cos(alpha)-G6*np.sin(alpha)-G3*G5-G4*G6)/(G5**2 + G6**2))
        A5 = A_omega_rG*AGr + B_omega_rG*BGr + E_omega_rG*EGr + G_omega_rG*FGr
        A6 = C_omega_rG*CGr + D_omega_rG*DGr + F_omega_rG*EGr + H_omega_rG*FGr
        A7 = A_omega_rG*AGtheta + B_omega_rG*BGtheta + E_omega_rG*EGtheta + G_omega_rG*FGtheta
        A8 = C_omega_rG*CGtheta + D_omega_rG*DGtheta + F_omega_rG*EGtheta + H_omega_rG*FGtheta
        A9 = A_omega_thetaG*AGtheta + B_omega_thetaG*BGtheta + E_omega_thetaG*EGtheta + G_omega_thetaG*FGtheta
        A10 = C_omega_thetaG*CGtheta + D_omega_thetaG*DGtheta + F_omega_thetaG*EGtheta + H_omega_thetaG*FGtheta
        A11 = A_omega_thetaG*AGr + B_omega_thetaG*BGr + E_omega_thetaG*EGr + G_omega_thetaG*FGr
        A12 = C_omega_thetaG*CGr + D_omega_thetaG*DGr + F_omega_thetaG*EGr + H_omega_thetaG*FGr
        return A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12

    def chi_partial_derivatives(self, Gamma, A5, A6, A7, A8, A9, A10, A11, A12):
        """This function retrieves the values of the partial derivatives of omega r and omega theta with respect to r and theta"""
        partial_omega_r_wrt_r = Gamma*A5 + A6
        partial_omega_r_wrt_theta = Gamma*A7 + A8
        partial_omega_theta_wrt_theta = Gamma*A9 + A10
        partial_omega_theta_wrt_r = Gamma*A11 + A12
        return partial_omega_r_wrt_r, partial_omega_r_wrt_theta, partial_omega_theta_wrt_theta, partial_omega_theta_wrt_r
    
    def calc_B1_through_B6(self, r, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12):
        """This function calculates the B1 through B6 values"""
        B1 = A1*A5 + (1/r)*(A3*(A7-A3))
        B2 = A1*A6 + A2*A5 + (1/r)*(A3*A8 + A4*(A7-2*A3))
        B3 = A2*A6 + (1/r)*(A4*(A8-A4))
        B4 = A1*A11 + (1/r)*(A3*(A9+A1))
        B5 = A1*A12 + A2*A11 + (1/r)*(A3*(A10+A2) + A4*(A9+A1))
        B6 = A2*A12 + (1/r)*(A4*(A10+A2))
        return B1, B2, B3, B4, B5, B6
    
    def calc_C1_through_C5(self, B1, B2, B3, B4, B5, B6):
        """This function calculates the C1 through C5 values"""
        C1 = B1**2 + B4**2
        C2 = 2*(B1*B2 + B4*B5)
        C3 = 2*(B1*B3 + B4*B6) + B2**2 + B5**2
        C4 = 2*(B2*B3 + B5*B6)
        C5 = B3**2 + B6**2
        return C1, C2, C3, C4, C5
    
    def chi_convective_acceleration(self, point_xy_in_Chi_plane, Gamma):
        """Start with a Chi value that is shifted from zeta_center"""
        xi, eta = point_xy_in_Chi_plane[0], point_xy_in_Chi_plane[1]
        r, theta = hlp.xy_to_r_theta(xi, eta)
        xio, etao = self.zeta_center.real, self.zeta_center.imag
        r0, theta0 = hlp.xy_to_r_theta(xio, etao)
        V_inf, R, alpha, epsilon = self.freestream_velocity, self.cylinder_radius, self.angle_of_attack, self.epsilon
        # omega_chi_unsplit = V_inf*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf*(xi+1j*eta)) - np.exp(1j*alpha)*R**2/(xi+1j*eta)**2) / (1 - (R-epsilon)**2/(xi+xio+1j*(eta+etao))**2)
        G1, G2, G3, G4, G5, G6 = self.calc_Chi_G_values(r, theta, alpha, epsilon, R, r0, theta0)
        # velocity_r = np.cos(theta)*Gamma/(2*np.pi)*(G1*G5+G2*G6)/(G5**2 + G6**2)+np.cos(theta)*V_inf*(G5*np.cos(alpha)-G6*np.sin(alpha)-G3*G5-G4*G6)/(G5**2 + G6**2) - np.sin(theta)*Gamma/(2*np.pi)*(G2*G5-G1*G6)/(G5**2 + G6**2) + np.sin(theta)*V_inf*(G6*np.cos(alpha)+G5*np.sin(alpha)+G4*G5-G3*G6)/(G5**2 + G6**2)
        # velocity_theta = np.cos(theta)*V_inf*(G6*np.cos(alpha)+G5*np.sin(alpha)+G4*G5-G3*G6)/(G5**2 + G6**2) - np.cos(theta)*Gamma/(2*np.pi)*(G2*G5-G1*G6)/(G5**2 + G6**2) - np.sin(theta)*Gamma/(2*np.pi)*(G1*G5+G2*G6)/(G5**2 + G6**2) - np.sin(theta)*V_inf*(G5*np.cos(alpha)-G6*np.sin(alpha)-G3*G5-G4*G6)/(G5**2 + G6**2)
        A_omega_rG, B_omega_rG, C_omega_rG, D_omega_rG, E_omega_rG, F_omega_rG, G_omega_rG, H_omega_rG = self.partial_coefficients_of_omega_r_wrt_G_values(V_inf, theta, alpha, G1, G2, G3, G4, G5, G6)
        A_omega_thetaG, B_omega_thetaG, C_omega_thetaG, D_omega_thetaG, E_omega_thetaG, F_omega_thetaG, G_omega_thetaG, H_omega_thetaG = self.partial_coefficients_of_omega_theta_wrt_G_values(V_inf, theta, alpha, G1, G2, G3, G4, G5, G6)
        AGr, BGr, CGr, DGr, EGr, FGr = self.partial_coefficients_of_G_wrt_r(r, theta, alpha, epsilon, R, r0, theta0)
        AGtheta, BGtheta, CGtheta, DGtheta, EGtheta, FGtheta = self.partial_coefficients_of_G_wrt_theta(r, theta, alpha, epsilon, R, r0, theta0)
        # calculate the A1 through A12 values
        A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12 = self.calc_A1_through_A12(V_inf, theta, alpha, G1, G2, G3, G4, G5, G6, A_omega_rG, B_omega_rG, C_omega_rG, D_omega_rG, E_omega_rG, F_omega_rG, G_omega_rG, H_omega_rG, A_omega_thetaG, B_omega_thetaG, C_omega_thetaG, D_omega_thetaG, E_omega_thetaG, F_omega_thetaG, G_omega_thetaG, H_omega_thetaG, AGr, BGr, CGr, DGr, EGr, FGr, AGtheta, BGtheta, CGtheta, DGtheta, EGtheta, FGtheta)
        # omega_r, omega_theta = Gamma*A1 + A2, Gamma*A3 + A4
        # partial_omega_r_wrt_r, partial_omega_r_wrt_theta, partial_omega_theta_wrt_theta, partial_omega_theta_wrt_r = self.chi_partial_derivatives(Gamma, A5, A6, A7, A8, A9, A10, A11, A12)
        B1, B2, B3, B4, B5, B6 = self.calc_B1_through_B6(r, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12)
        convective_acceleration = np.array([Gamma**2*B1 + Gamma*B2 + B3, Gamma**2*B4 + Gamma*B5 + B6])
        C1, C2, C3, C4, C5 = self.calc_C1_through_C5(B1, B2, B3, B4, B5, B6)
        return C1, C2, C3, C4, C5, convective_acceleration
    
    def convective_acceleration_squared(self, point_xy_in_Chi_plane, Gamma):
        """This function calculates the convective acceleration squared"""
        C1, C2, C3, C4, C5, chi_convective_acceleration = self.chi_convective_acceleration(point_xy_in_Chi_plane, Gamma)
        # convective_acceleration_squared = np.dot(chi_convective_acceleration, chi_convective_acceleration)
        convective_acceleration_squared = Gamma**4*C1 + Gamma**3*C2 + Gamma**2*C3 + Gamma*C4 + C5
        return convective_acceleration_squared
    
    def numerical_convective_acceleration(self, point_xy_in_Chi_plane, Gamma, step):
        """calculates the convective acceleration numerically"""
        Chi = point_xy_in_Chi_plane[0] + 1j*point_xy_in_Chi_plane[1]
        zeta = self.Chi_to_zeta(Chi)
        z = self.zeta_to_z(zeta)
        
        # r, theta = hlp.xy_to_r_theta(Chi.real, Chi.imag)
        # cartesian_velocity_chi = self.velocity_chi(point_xy_in_Chi_plane, Gamma)
        # polar_velocity_chi = self.polar_velocity(theta, cartesian_velocity_chi)
        # omega_r, omega_theta = polar_velocity_chi[0], polar_velocity_chi[1]
        # step_point = point_xy_in_Chi_plane

        r, theta = hlp.xy_to_r_theta(z.real, z.imag)
        point_xy_in_Z_plane = [z.real, z.imag]
        cartesian_z_velocity = self.velocity(point_xy_in_Z_plane, Gamma)
        polar_z_velocity = self.polar_velocity(theta, cartesian_z_velocity)
        omega_r, omega_theta = polar_z_velocity[0], polar_z_velocity[1]
        step_point = point_xy_in_Z_plane
        
        omega_r_plus_dr, omega_r_minus_dr, omega_r_plus_dtheta, omega_r_minus_dtheta, omega_theta_plus_dr, omega_theta_minus_dr, omega_theta_plus_dtheta, omega_theta_minus_dtheta = self.function_plus_minus_step_variable(step_point, Gamma, step)
        # calculate the partial derivatives of omega r with respect to r and theta
        partial_omega_r_wrt_r = hlp.central_difference(omega_r_plus_dr, omega_r_minus_dr, step)
        # print("numerical partial omega r wrt r: ", partial_omega_r_wrt_r)
        partial_omega_r_wrt_theta = hlp.central_difference(omega_r_plus_dtheta, omega_r_minus_dtheta, step)
        # print("numerical partial omega r wrt theta: ", partial_omega_r_wrt_theta)
        # calculate the partial derivatives of omega theta with respect to r and theta
        partial_omega_theta_wrt_r = hlp.central_difference(omega_theta_plus_dr, omega_theta_minus_dr, step)
        # print("numerical partial omega theta wrt r: ", partial_omega_theta_wrt_r)
        partial_omega_theta_wrt_theta = hlp.central_difference(omega_theta_plus_dtheta, omega_theta_minus_dtheta, step)
        # print("numerical partial omega theta wrt theta: ", partial_omega_theta_wrt_theta)
        # calculate the convective acceleration
        convective_acceleration_r = omega_r*partial_omega_r_wrt_r + (omega_theta/r)*partial_omega_r_wrt_theta - omega_theta**2/r
        convective_acceleration_theta = omega_r*partial_omega_theta_wrt_r + (omega_theta/r)*partial_omega_theta_wrt_theta + omega_r*omega_theta/r
        convective_acceleration = np.array([convective_acceleration_r, convective_acceleration_theta])
        return convective_acceleration
    
    def numerical_convective_acceleration_squared(self, numerical_convective_acceleration):
        """Calculates the convective acceleration squared numerically"""
        convective_acceleration_squared = np.dot(numerical_convective_acceleration, numerical_convective_acceleration)
        return convective_acceleration_squared
        
    def function_plus_minus_step_variable(self, point_xy, Gamma, stepsize):
        """takes in a function and returns the function plus and minus a stepsize"""
        r, theta = hlp.xy_to_r_theta(point_xy[0], point_xy[1])
        r_plus, r_minus = r + stepsize, r - stepsize
        theta_plus, theta_minus = theta + stepsize, theta - stepsize
        x_r_plus, x_r_minus, y_r_plus, y_r_minus = hlp.r_theta_to_xy(r_plus, theta)[0], hlp.r_theta_to_xy(r_minus, theta)[0], hlp.r_theta_to_xy(r_plus, theta)[1], hlp.r_theta_to_xy(r_minus, theta)[1]
        x_theta_plus, x_theta_minus, y_theta_plus, y_theta_minus = hlp.r_theta_to_xy(r, theta_plus)[0], hlp.r_theta_to_xy(r, theta_minus)[0], hlp.r_theta_to_xy(r, theta_plus)[1], hlp.r_theta_to_xy(r, theta_minus)[1]
        omega_xy_plus_r, omega_xy_minus_r = self.velocity([x_r_plus, y_r_plus], Gamma), self.velocity([x_r_minus, y_r_minus], Gamma)
        omega_xy_plus_theta, omega_xy_minus_theta = self.velocity([x_theta_plus, y_theta_plus], Gamma), self.velocity([x_theta_minus, y_theta_minus], Gamma)
        omega_r_plus_dr, omega_r_minus_dr = self.polar_velocity(theta, omega_xy_plus_r)[0], self.polar_velocity(theta, omega_xy_minus_r)[0]
        omega_r_plus_dtheta, omega_r_minus_dtheta = self.polar_velocity(theta, omega_xy_plus_theta)[0], self.polar_velocity(theta, omega_xy_minus_theta)[0]
        omega_theta_plus_dr, omega_theta_minus_dr = self.polar_velocity(theta, omega_xy_plus_r)[1], self.polar_velocity(theta, omega_xy_minus_r)[1]
        omega_theta_plus_dtheta, omega_theta_minus_dtheta = self.polar_velocity(theta, omega_xy_plus_theta)[1], self.polar_velocity(theta, omega_xy_minus_theta)[1]
        return omega_r_plus_dr, omega_r_minus_dr, omega_r_plus_dtheta, omega_r_minus_dtheta, omega_theta_plus_dr, omega_theta_minus_dr, omega_theta_plus_dtheta, omega_theta_minus_dtheta

    def convective_acceleration_squared_numerical(self, point_xy_in_Chi_plane, Gamma, step):
        """Calculates the convective acceleration squared numerically"""
        convective_acceleration = self.numerical_convective_acceleration(point_xy_in_Chi_plane, Gamma, step)
        convective_acceleration_squared = np.dot(convective_acceleration, convective_acceleration)
        return convective_acceleration_squared

    def numerical_appellian(self, Gamma_range: list, r_range: list, theta_range: list, is_analytic_conv_accel: bool, step): # uses analytic expression for convection acceleration
        """This function calculates the Appellian function numerically requires evenly spaced ranges for Gamma, r, and theta"""
        # create a meshgrid of r and theta values, the first value in r_range is the lower bound the second value is the upper bound, the third value is the increment size
        if is_analytic_conv_accel:
            print("ANALYTIC CONVECTIVE ACCELERATION")
            Gamma_values = hlp.list_to_range(Gamma_range)
            r_values = hlp.list_to_range(r_range)
            theta_values = hlp.list_to_range(theta_range)
            # calculate the area element         
            dr = r_values[1] - r_values[0] ## test dr values to see if there's some weird convergence thing going on. 
            print("dr: ", dr)
            dtheta = theta_values[1] - theta_values[0]
            print("dtheta: ", dtheta)
            # # The appellian sums the squared convective acceleration at each r and theta value. Plot the Appellian function with respect to each change in Gamma. 
            # this is a nested loop with Gamma as the outer loop, r as the middle loop, and theta as the inner loop
            Appellian_array = np.zeros((len(Gamma_values),1))
            xi_eta_values = np.zeros((len(r_values), len(theta_values), 2))
            Appellian_value = 0.0
            for i in range(len(Gamma_values)):
                for j in range(len(r_values)):
                    for k in range(len(theta_values)):
                        xi, eta = hlp.r_theta_to_xy(r_values[j], theta_values[k])
                        area_element = r_values[j]*dr*dtheta
                        Appellian_value += self.convective_acceleration_squared([xi, eta], Gamma_values[i])*area_element
                        # if j and k are the last values in the range, then the Appellian value is complete and should be appended to the Appellian list, then the appellian value should be reset to 0
                        if j == len(r_values)-1 and k == len(theta_values)-1:
                            xi_eta_values[j][k] = [xi, eta]
                            Appellian_array[i] = Appellian_value
                            Appellian_value = 0.0
                        else:
                            pass # pass means do nothing
            print("Min Appellian Value for Analytic Conv Accel: ", np.min(Appellian_array))
            index_min = np.argmin(Appellian_array)
            print("Index of Min Appellian Value: ", index_min)
            print("Gamma Value at Min Appellian Value for Analytic Conv Accel: ", Gamma_values[index_min]) 
            plt.plot(Gamma_values, Appellian_array)
            plt.xlabel("$\\Gamma$")
            plt.ylabel("S")
            plt.title("Appellian Function (Analytic Convective Acceleration)")
            plt.show()
        else:
            print("NUMERICAL CONVECTIVE ACCELERATION")
            Gamma_values = hlp.list_to_range(Gamma_range)
            r_values = hlp.list_to_range(r_range)
            theta_values = hlp.list_to_range(theta_range)
            # calculate the area element         
            dr = r_values[1] - r_values[0] ## test dr values to see if there's some weird convergence thing going on. 
            print("dr: ", dr)
            dtheta = theta_values[1] - theta_values[0]
            print("dtheta: ", dtheta)
            # # The appellian sums the squared convective acceleration at each r and theta value. Plot the Appellian function with respect to each change in Gamma. 
            # this is a nested loop with Gamma as the outer loop, r as the middle loop, and theta as the inner loop
            Appellian_array = np.zeros((len(Gamma_values),1))
            xi_eta_values = np.zeros((len(r_values), len(theta_values), 2))
            Appellian_value = 0.0
            for i in range(len(Gamma_values)):
                for j in range(len(r_values)):
                    for k in range(len(theta_values)):
                        xi, eta = hlp.r_theta_to_xy(r_values[j], theta_values[k])
                        area_element = r_values[j]*dr*dtheta
                        Appellian_value += self.convective_acceleration_squared_numerical([xi, eta], Gamma_values[i], step)*area_element
                        # if j and k are the last values in the range, then the Appellian value is complete and should be appended to the Appellian list, then the appellian value should be reset to 0
                        if j == len(r_values)-1 and k == len(theta_values)-1:
                            xi_eta_values[j][k] = [xi, eta]
                            Appellian_array[i] = Appellian_value
                            Appellian_value = 0.0
                        else:
                            pass # pass means do nothing

            print("Min Appellian Value for Numerical Conv Accel: ", np.min(Appellian_array))
            index_min = np.argmin(Appellian_array)
            print("Index of Min Appellian Value: ", index_min)
            print("Gamma Value at Min Appellian Value for Numerical Conv Accel: ", Gamma_values[index_min])
            # plot the Appellian function with respect to Gamma 
            plt.plot(Gamma_values, Appellian_array)
            plt.xlabel("$\\Gamma$")
            plt.ylabel("S")
            plt.title("Appellian Function (Numerical Convective Acceleration)")
            plt.show()
        return Appellian_array, xi_eta_values
    
    def r_theta_Chi_to_r_theta_zeta(self, r_Chi, theta_Chi):
        """This function takes in r and theta in the Chi plane and returns r and theta in the zeta plane"""
        xi_Chi, eta_Chi = hlp.r_theta_to_xy(r_Chi, theta_Chi)
        xi_zeta, eta_zeta = xi_Chi + self.zeta_center.real, eta_Chi + self.zeta_center.imag
        r_zeta, theta_zeta = hlp.xy_to_r_theta(xi_zeta, eta_zeta)
        return r_zeta, theta_zeta

    def zeta_to_z(self, zeta: complex):
        """This function takes in a zeta coordinate and returns the z coordinate"""
        if np.isclose(zeta.real, 0.0) and np.isclose(zeta.imag, 0.0):
            z = zeta
        else:
            z = zeta + (self.cylinder_radius - self.epsilon)**2/zeta # eq 96
        return z
    
    def zeta_to_Chi(self, zeta: complex):
        """This function takes in a zeta coordinate and returns the Chi coordinate"""
        Chi = zeta - self.zeta_center
        return Chi

    def Chi_to_zeta(self, Chi: complex):
        """This function takes in a Chi coordinate and returns the zeta coordinate"""
        zeta = Chi + self.zeta_center
        return zeta

    def z_to_zeta(self, z: complex): # eq 104 
        """This function takes in a z coordinate and returns the zeta coordinate"""
        z_1 = z**2 - 4*(self.cylinder_radius - self.epsilon)**2
        if z_1.real > 0:
            zeta = (z + np.sqrt(z_1))/2
            zeta_2 = (z - np.sqrt(z_1))/2
        elif z_1.real < 0:
            zeta = (z - 1j*np.sqrt(-z_1))/2
            zeta_2 = (z + 1j*np.sqrt(-z_1))/2
        elif z_1.imag >= 0:
            zeta = (z + np.sqrt(z_1))/2
            zeta_2 = (z - np.sqrt(z_1))/2
        else:
            zeta = (z - 1j*np.sqrt(-z_1))/2
            zeta_2 = (z + 1j*np.sqrt(-z_1))/2
        if abs(zeta_2 - self.zeta_center) > abs(zeta - self.zeta_center):
            zeta = zeta_2
        return zeta
    
    def calc_J_airfoil_zeta_center(self):
        """This function calculates the zeta center of the airfoil"""
        first = (-4*self.design_thickness/(3*np.sqrt(3)))
        second = 1j*self.design_CL/(2*np.pi*(1+4*self.design_thickness/(3*np.sqrt(3))))
        zeta_center = self.cylinder_radius*(first + second) # eq 152 in complex variables
        return zeta_center
    
    def calc_J_airfoil_epsilon(self):
        """This function calculates the epsilon of the airfoil"""
        epsilon = self.cylinder_radius - 1*np.sqrt(self.cylinder_radius**2-self.zeta_center.imag**2)-self.zeta_center.real # eq 113 in complex variables
        return epsilon
    
    def calc_circulation_J_airfoil(self):
        """This function calculates the circulation of the airfoil based on the Kutta condition"""
        gamma = 4*np.pi*self.freestream_velocity*(np.sin(self.angle_of_attack)*np.sqrt(self.cylinder_radius**2 - self.zeta_center.imag**2) + self.zeta_center.imag*np.cos(self.angle_of_attack)) # eq 122 in complex variables
        return gamma
    
    def calc_zeta_real_intercepts_J_airfoil(self):
        """This function calculates the real intercepts of the airfoil"""
        zeta_real_leading_edge = -1*np.sqrt(self.cylinder_radius**2 - self.zeta_center.imag**2) + self.zeta_center.real
        zeta_real_trailing_edge = np.sqrt(self.cylinder_radius**2 - self.zeta_center.imag**2) + self.zeta_center.real
        return zeta_real_leading_edge, zeta_real_trailing_edge
    
    def calc_zeta_real_intercepts(self):
        """This function calculates the real intercepts of the airfoil"""
        zeta_real_leading_edge = -1*np.sqrt(self.cylinder_radius**2 - self.zeta_center.imag**2) + self.zeta_center.real 
        zeta_real_trailing_edge = np.sqrt(self.cylinder_radius**2 - self.zeta_center.imag**2) + self.zeta_center.real 
        return zeta_real_leading_edge, zeta_real_trailing_edge
    
    def calc_J_airfoil_CL(self):
        """This function calculates the lift of the airfoil"""
        self.lift_j_airfoil = (2*np.pi*(np.sin(self.angle_of_attack)+(self.zeta_center.imag*np.cos(self.angle_of_attack))/np.sqrt(self.cylinder_radius**2 - self.zeta_center.imag**2)))/(1+self.zeta_center.real/(np.sqrt(self.cylinder_radius**2 - self.zeta_center.imag**2)-self.zeta_center.real)) # eq 124
        return self.lift_j_airfoil
    
    def calc_J_airfoil_Cmz(self, point_z_plane): # eq 141
        """This function calculates the moment coefficient of the airfoil""" 
        First = (np.sin(2*self.angle_of_attack)*np.pi/4 * ((self.cylinder_radius**2-self.zeta_center.imag**2-self.zeta_center.real**2)/(self.cylinder_radius**2-self.zeta_center.imag**2))**2 )
        Second = ((0.25*self.lift_j_airfoil))
        Third = ((point_z_plane.real-self.zeta_center.real)*np.cos(self.angle_of_attack) + (point_z_plane.imag-self.zeta_center.imag)*np.sin(self.angle_of_attack))/(self.cylinder_radius**2-self.zeta_center.imag**2)
        Fourth = (np.sqrt(self.cylinder_radius**2-self.zeta_center.imag**2)-self.zeta_center.real)
        Cmz = First + Second*Third*Fourth
        return Cmz
    
    def calc_J_airfoil_c4(self):
        """This function calculates the c4 location of the airfoil"""
        zeta_leading_edge, zeta_trailing_edge = self.calc_zeta_real_intercepts()
        z_leading_edge = self.zeta_to_z(zeta_leading_edge)
        z_trailing_edge = self.zeta_to_z(zeta_trailing_edge)
        c4 = (3*z_leading_edge+z_trailing_edge)/4
        return c4
    
    def shift_joukowski_airfoil(self):
        """This function finds shifts all the points back to where zeta_center = 0, finds how far to the left the leading edge is, and shifts all points by that much (so the leading edge is at 0)
        Then it finds the right most point, and finds the length from the right most point to the left most point, and scales all points by that much"""
        # shift all the points back to where zeta_center = 0 by shifting the x coordinates by the real part of zeta_center
        self.surface_points[:,0] = self.surface_points[:,0] - self.zeta_center.real
        # now shift the y coordinates by the imaginary part of zeta_center
        self.surface_points[:,1] = self.surface_points[:,1] - self.zeta_center.imag
        # find how far to the left the leading edge is
        leading_edge = np.min(self.surface_points[:,0])
        # shift all points by that much
        self.surface_points[:,0] = self.surface_points[:,0] - leading_edge
        # find the right most point
        trailing_edge = np.max(self.surface_points[:,0])
        self.surface_points = self.surface_points/trailing_edge
        return self.surface_points
    
    def make_surface_points_go_from_lower_trailing_to_upper_trailing(self):
        """This function removes duplicate points from self.surface_points while preserving order, and reorders 
        the surface points so that they go from the lower trailing edge to the upper trailing edge."""
        # Remove duplicates while preserving order
        unique_points = []
        seen = set()
        for point in self.surface_points:
            # Convert point to a tuple to make it hashable for the set
            point_tuple = tuple(point)
            if point_tuple not in seen:
                seen.add(point_tuple)
                unique_points.append(point)
        # Update surface_points to only unique points, maintaining order
        self.surface_points = np.array(unique_points)
        
    def output_J_airfoil(self):
        # flip the lower coordinates so that they go from the trailing edge to the leading edge
        self.lower_coords = np.flip(self.lower_coords, axis=0)
        # make total surface_points array that starts with lower, then does upper
        self.surface_points = np.concatenate((self.lower_coords, self.upper_coords), axis=0)
        self.make_surface_points_go_from_lower_trailing_to_upper_trailing()
        self.surface_points = self.shift_joukowski_airfoil()
        # if self.export_geometry is True, then export the geometry to a txt file
        print("Exporting geometry to txt file...")
        file_name = "Joukowski" + self.type + "_" + str(self.n_geom_points) + ".txt"
        # save text file with the geometry in double precision
        np.savetxt(file_name, self.surface_points, delimiter = ",", header = "x, y", comments = "")
        print("Geometry exported to", file_name)

    def test_appellian_stuff(self):
        zeta_test = 10 - 1j*4
        Gamma = self.circulation
        # z_test = cyl.zeta_to_z(zeta_test)
        chi_test = self.zeta_to_Chi(zeta_test)
        print("\n")
        z_test = cyl.zeta_to_z(zeta_test)
        complex_zeta_center = self.zeta_center.real + 1j*self.zeta_center.imag
        z_0 = cyl.zeta_to_z(complex_zeta_center)
        # print("z_0: ", z_0)
        zeta_test = [zeta_test.real, zeta_test.imag]
        theta_zeta = np.arctan2(zeta_test[1], zeta_test[0])
        z_test = [z_test.real, z_test.imag]
        # theta_z = np.arctan2(z_test[1], z_test[0])
        chi_test = [chi_test.real, chi_test.imag]
        theta_chi = np.arctan2(chi_test[1], chi_test[0])
        z_velocity_at_z_test = self.velocity(z_test, Gamma)
        z_split_velocity_at_z_test = self.split_velocity_z(z_test, Gamma)
        # print("z_velocity_at_z_test    :", z_velocity_at_z_test[0], z_velocity_at_z_test[1])
        velocity_chi_at_chi_test = self.velocity_chi(chi_test, Gamma)
        # print("velocity_chi_at_chi_test:", velocity_chi_at_chi_test[0], velocity_chi_at_chi_test[1])
        z_polar = self.polar_velocity(theta_zeta, z_velocity_at_z_test)
        Chi_polar = self.polar_velocity(theta_chi, velocity_chi_at_chi_test)
        z_polar_mag = np.sqrt(z_polar[0]**2 + z_polar[1]**2)
        # print("z_polar_mag  :", z_polar_mag)
        Chi_polar_mag = np.sqrt(Chi_polar[0]**2 + Chi_polar[1]**2)
        # print("Chi_polar_mag:", Chi_polar_mag)
        C1, C2, C3, C4, C5, chi_convective_acceleration = self.chi_convective_acceleration(chi_test, cyl.circulation)
        # print("convective_acceleration        : ", chi_convective_acceleration)
        convective_acceleration_squared = self.convective_acceleration_squared(chi_test, cyl.circulation)
        # print("convective_acceleration_squared: ", convective_acceleration_squared)
        acceleration = self.chi_acceleration(chi_test, self.circulation)
        # print("acceleration                   :", acceleration)
        acceleration_squared = self.chi_acceleration_squared(chi_test, self.circulation)
        # print("acceleration_squared           :", acceleration_squared)
        # print("\n")
        # compute numerical Appellian function
        Gamma_range = [4.5,5, 0.01]
        r_range = [self.cylinder_radius, 2*self.cylinder_radius, self.cylinder_radius]
        theta_range = [0, 2*np.pi, np.pi/4]
        step = 1e-12
        Appellian_values, xi_eta_values = cyl.numerical_appellian(Gamma_range, r_range, theta_range, False, step)
        # get numerical partials 
        # numerical_convective_acceleration = self.numerical_convective_acceleration(chi_test, cyl.circulation, step)
        # get analytical partials
        r0, theta0 = hlp.xy_to_r_theta(self.zeta_center.real, self.zeta_center.imag)
        r, theta = hlp.xy_to_r_theta(chi_test[0], chi_test[1])
        # d_omega_r_dr, d_omega_r_dtheta, d_omega_theta_dr, d_omega_theta_dtheta = self.all_partials(r,theta,self.angle_of_attack, self.epsilon, self.cylinder_radius, r0, theta0, Gamma)
        if 
        for i in range(len(xi_eta_values)):
            plt.plot(xi_eta_values[i][0], xi_eta_values[i][1], color='red')

        return Appellian_values, xi_eta_values

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
    # change legend parameters
    plt.rcParams["legend.fontsize"] = 17.0
    plt.rcParams["legend.frameon"] = True
    subdict = {
        "figsize" : (3.25,3.5),
        "constrained_layout" : True,
        "sharex" : True
    }
    cyl = cylinder("Joukowski_Cylinder.json")
    if cyl.is_test_appellian:
        test = cyl.test_appellian_stuff()

    plt.figure()
    cyl.get_full_geometry_zeta()
    cyl.plot_geometry_zeta()
    cyl.get_full_geometry()
    cyl.plot_geometry()
    cyl.get_and_plot_foci()
    # zeta_to_z_test_point = 1 + 1j
    if cyl.type == "airfoil":
        lift = cyl.calc_J_airfoil_CL()
        print("CL Joukowski: ", lift)
        Cmo = cyl.calc_J_airfoil_Cmz((0+1j*0))
        print("Cm0 Joukowski: ", Cmo)
        c4 = cyl.calc_J_airfoil_c4()
        print("C4 location Joukowski: ", c4)
        # now find the moment coefficient at c4
        Cmc4 = cyl.calc_J_airfoil_Cmz(c4)
        print("Cmc4 Joukowski: ", Cmc4)
        # now create the full geometry of the airfoil and export it to a text file
        cyl.output_J_airfoil()
        print("\n")
    if cyl.do_plot_streamlines:
        cyl.plot()
    # include y and x axes 
    plt.axhline(0, color='gray')
    plt.axvline(0, color='gray')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("$\\xi$/$R$")
    plt.ylabel("$\\eta$/$R$")
    plt.gca().xaxis.labelpad = 0.001
    plt.gca().yaxis.labelpad = -10
    plt.xlim(-cyl.original_cylinder_radius, cyl.original_cylinder_radius)
    plt.ylim(-cyl.original_cylinder_radius, cyl.original_cylinder_radius)
    plt.xticks(np.linspace(-1, 2, 4))  # 4 points: -1, 0, 1, 2
    plt.yticks(np.linspace(-1, 2, 4)) # the -1 is the bottom most point, and 2.0 is the top most point and the 1.0 is the increment.
    plt.text(-0.07, -0.01, str(int(-cyl.original_cylinder_radius)), transform=plt.gca().transAxes, fontsize=17, verticalalignment='top')
    epsilon_name = "epsilon_" + str(cyl.epsilon)
    zeta_0_name = "zeta_0_" + str(cyl.zeta_center)
    if cyl.plot_text:
        plt.text(0.05, 0.92, "$\\epsilon = $" + str(cyl.epsilon) + " $R$", transform=plt.gca().transAxes, fontsize=17, verticalalignment='top')
        plt.text(0.60, 0.96, "$\\xi_0 = $" + str(cyl.zeta_center.real) + " $R$", transform=plt.gca().transAxes, fontsize=17, verticalalignment='top')
        plt.text(0.60, 0.88, "$\\eta_0 = $" + str(cyl.zeta_center.imag) + " $R$", transform=plt.gca().transAxes, fontsize=17, verticalalignment='top')
    if cyl.save_fig:
        plt.savefig("Joukowski_Cylinder_" + epsilon_name + "_" +  zeta_0_name + "_" + ".svg")
    if cyl.show_fig:
        plt.show()


