import json
import numpy as np # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import scipy.integrate as spi # this is for numerical integration
# import bisection from scipy.optimize
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
            self.save_fig = input["plot"]["save_fig"]
            self.show_fig = input["plot"]["show_fig"]
            self.plot_text = input["plot"]["plot_text"]
            self.do_plot_streamlines = input["plot"]["do_plot_streamlines"]
            self.cylinder_radius = input["geometry"]["cylinder_radius"] # this is the radius of the cylinder in meters. We are normalizing everything by this value. This is used in the geometry function to calculate the geometry of the cylinder.
            self.original_cylinder_radius = self.cylinder_radius # this is the original cylinder radius in meters. We are normalizing everything by this value. This is used in the geometry function to calculate the geometry of the cylinder.
            self.cylinder_radius/=self.cylinder_radius # normalize the cylinder radius to 1. This makes the math easier. We are normalizing everything by the cylinder radius. This is used in the geometry function to calculate the geometry of the cylinder.
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
                print("zeta center", self.zeta_center)
                self.epsilon = self.calc_J_airfoil_epsilon() # uses the cylinder radius, and zeta center to calculate epsilon
                print("epsilon", self.epsilon)
                self.circulation = self.calc_circulation_J_airfoil() # uses the Kutta condition to calculate circulation
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
        upper = np.zeros((500,2))
        lower = np.zeros((500,2))
        camber = np.zeros((500,2))
        x_coords = np.linspace(self.x_leading_edge, self.x_trailing_edge, 500) # this is creating an array of x coordinates
        for i in range(500):
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
        plt.scatter(self.z_trailing_edge_focus, 0.0, color='black', marker='s', s = size, label = "J-sing")
        plt.scatter(self.z_leading_edge_focus, 0.0, color='black', marker='s', s = size)


    def velocity_zeta(self, point_xy): # A4 on project
        """
        Calculates the velocity at a given point in the flow field in cartesian coordinates using cylindrical velocity equations.
        Parameters:
        - point_xy (list): An xy coordinate.
        Returns:
        - velocity (list): The velocity at the given point.
        """
        z = point_xy[0] + 1j*point_xy[1]
        w_z = self.freestream_velocity*((np.exp(-1j*self.angle_of_attack)) + 1j*self.circulation/(2*np.pi*self.freestream_velocity*(z-self.zeta_center)) + -1*(np.exp(1j*self.angle_of_attack))*self.cylinder_radius**2/(z-self.zeta_center)**2)  
        velocity_complex = np.array([w_z.real, -w_z.imag])     
        return velocity_complex
    
    def velocity(self, point_xy_in_z_plane):
        """This function calculates the velocity at a given point in the flow field in the z plane"""
        z = point_xy_in_z_plane[0] + 1j*point_xy_in_z_plane[1]
        zeta = self.z_to_zeta(z)
        xi = zeta.real
        eta = zeta.imag
        r = np.sqrt(xi**2 + eta**2)
        theta = np.arctan2(eta, xi)
        velocity = self.freestream_velocity*(np.exp(-1j*self.angle_of_attack) + 1j*self.circulation/(2*np.pi*self.freestream_velocity*(zeta-self.zeta_center)) - np.exp(1j*self.angle_of_attack)*self.cylinder_radius**2/(zeta-self.zeta_center)**2) / (1 - (self.cylinder_radius-self.epsilon)**2/(zeta)**2) # eq 107
        velocity_complex = np.array([velocity.real, -velocity.imag])
        print("real velocity", velocity_complex[0])
        print("imag velocity", velocity_complex[1])
        # velocity_r = velocity_complex[0]*np.cos(theta) + velocity_complex[1]*np.sin(theta)
        # velocity_theta = velocity_complex[1]*np.cos(theta) - velocity_complex[0]*np.sin(theta)
        # print("radial velocity", velocity_r)
        # print("theta velocity", velocity_theta)
        return velocity_complex
    
    def Chi_velocity(self, point_xy_in_Chi_plane):
        """Start with a Chi value that is shifted from zeta_center"""
        Chi_real = point_xy_in_Chi_plane[0]
        Chi_imag = point_xy_in_Chi_plane[1]
        xi = Chi_real
        eta = Chi_imag
        r = np.sqrt(xi**2 + eta**2)
        theta = np.arctan2(eta, xi)
        zeta_center = self.zeta_center
        xio = zeta_center.real
        etao = zeta_center.imag
        r0 = np.sqrt(xio**2 + etao**2)
        theta0 = np.arctan2(etao, xio)
        V_inf = self.freestream_velocity
        Gamma = self.circulation
        R = self.cylinder_radius
        alpha = self.angle_of_attack
        epsilon = self.epsilon
        omega_chi_unsplit = V_inf*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf*(xi+1j*eta)) - np.exp(1j*alpha)*R**2/(xi+1j*eta)**2) / (1 - (R-epsilon)**2/(xi+xio+1j*(eta+etao))**2)
        velocity_complex = np.array([omega_chi_unsplit.real, -omega_chi_unsplit.imag])
        # G1 = eta/(xi**2 + eta**2)
        G1 = np.sin(theta)/r # this is the same as the above line
        # G2 = xi/(xi**2 + eta**2)
        G2 = np.cos(theta)/r # this is the same as the above line
        # G3 = R**2*(np.cos(alpha)*(xi**2-eta**2)+2*xi*eta*np.sin(alpha))/((xi**2-eta**2)**2+(4*xi**2*eta**2))
        G3 = R**2*(r**2*np.cos(alpha)*np.cos(2*theta)+r**2*np.sin(alpha)*np.sin(2*theta))/((r**2*np.cos(2*theta))**2+0.5*r**4*(1-np.cos(4*theta))) # this is the same as the above line
        # G4 = R**2*(np.sin(alpha)*(xi**2-eta**2)-2*xi*eta*np.cos(alpha))/((xi**2-eta**2)**2+(4*xi**2*eta**2))
        G4 = R**2*(r**2*np.sin(alpha)*np.cos(2*theta)-r**2*np.cos(alpha)*np.sin(2*theta))/((r**2*np.cos(2*theta))**2+0.5*r**4*(1-np.cos(4*theta))) # this is the same as the above line
        # G5 = 1 - ((R-epsilon)**2*((xi+xio)**2 - (eta+etao)**2))/(((xi+xio)**2 - (eta+etao)**2)**2 + (2*(xi+xio)*(eta+etao))**2)
        G5 = 1 - ((R-epsilon)**2*(r**2*np.cos(2*theta)+r0**2*np.cos(2*theta0)+2*r*r0*np.cos(theta+theta0)))/((r**2*np.cos(2*theta)+r0**2*np.cos(2*theta0)+2*r*r0*np.cos(theta+theta0))**2+(r**2*np.sin(2*theta)+r0**2*np.sin(2*theta0)+2*r*r0*np.sin(theta+theta0))**2) # this is the same as the above line
        # G6 = ((R-epsilon)**2*(2*(xi+xio)*(eta+etao)))/(((xi+xio)**2 - (eta+etao)**2)**2 + (2*(xi+xio)*(eta+etao))**2)
        G6 = ((R-epsilon)**2*(r**2*np.sin(2*theta)+r0**2*np.sin(2*theta0)+2*r*r0*np.sin(theta+theta0)))/((r**2*np.cos(2*theta)+r0**2*np.cos(2*theta0)+2*r*r0*np.cos(theta+theta0))**2+(r**2*np.sin(2*theta)+r0**2*np.sin(2*theta0)+2*r*r0*np.sin(theta+theta0))**2) # this is the same as the above line
        V_real = (Gamma/(2*np.pi))*((G1*G5+G2*G6)/(G5**2 + G6**2)) + V_inf*((G5*np.cos(alpha)-G6*np.sin(alpha)-G3*G5-G4*G6)/(G5**2 + G6**2))
        V_imag = (-1*Gamma/(2*np.pi))*((G2*G5-G1*G6)/(G5**2 + G6**2)) + V_inf*((G6*np.cos(alpha)+G5*np.sin(alpha)+G4*G5-G3*G6)/(G5**2 + G6**2))
        velocity_complex = np.array([V_real, V_imag])
        print("real velocity", V_real)
        print("imag velocity", V_imag)
        return velocity_complex

    def polar_velocity(self, theta, cartesian_velocity):
        """This function converts the cartesian velocity to polar velocity (can go from z, zeta, or chi plane to polar velocity)"""
        velocity_r = cartesian_velocity[0]*np.cos(theta) + cartesian_velocity[1]*np.sin(theta)
        print("\nradial velocity", velocity_r)
        velocity_theta = cartesian_velocity[1]*np.cos(theta) - cartesian_velocity[0]*np.sin(theta)
        print("theta velocity", velocity_theta)
        polar_velocity = np.array([velocity_r, velocity_theta])
        return polar_velocity
    
    def partials(self, point_xy_in_z_plane):
        z = point_xy_in_z_plane[0] + 1j*point_xy_in_z_plane[1]
        zeta = self.z_to_zeta(z)
        xi = zeta.real
        eta = zeta.imag
        zeta_center = self.zeta_center
        xio = zeta_center.real
        etao = zeta_center.imag
        r = np.sqrt(xi**2 + eta**2)
        r0 = np.sqrt(xio**2 + etao**2)
        theta = np.arctan2(eta, xi)
        theta0 = np.arctan2(etao, xio)
        V_inf = self.freestream_velocity
        Gamma = self.circulation
        R = self.cylinder_radius
        alpha = self.angle_of_attack
        epsilon = self.epsilon
        G1 = Gamma*(r*np.sin(theta)-r0*np.sin(theta0))/(2*np.pi*V_inf*(r**2 + r0**2 - 2*r*r0*np.cos(theta-theta0)))
        G2 = Gamma*(r*np.cos(theta)-r0*np.cos(theta0))/(2*np.pi*V_inf*(r**2 + r0**2 - 2*r*r0*np.cos(theta-theta0)))
        G3 =  R**2*(2*r**2*np.sin(theta)*np.sin(alpha - theta) + r**2*np.cos(alpha) - 2*r*r0*np.cos(-alpha + theta + theta0) + 2*r0**2*np.sin(theta0)*np.sin(alpha - theta0) + r0**2*np.cos(alpha))/((r**2*np.sin(2*theta) - 2*r*r0*np.sin(theta + theta0) + r0**2*np.sin(2*theta0))**2 + (r**2*np.cos(2*theta) - 2*r*r0*np.cos(theta + theta0) + r0**2*np.cos(2*theta0))**2)
        G4 = (R**2*(r**2*np.sin(alpha)-2*r**2*np.sin(theta)*np.cos(alpha-theta)+2*r*r0*np.sin(-alpha+theta+theta0)+r0**2*np.sin(alpha)-2*r0**2*np.sin(theta0)*np.cos(alpha-theta0)))/((r**2*np.cos(2*theta) - 2*r*r0*np.cos(theta+theta0) + r0**2*np.cos(2*theta0))**2 + (r**2*np.sin(2*theta) - 2*r*r0*np.sin(theta+theta0) + r0**2*np.sin(2*theta0))**2)
        G5 = 1 - ((R-epsilon)**2*(np.cos(2*theta)))/(r**2)
        G6 = ((R-epsilon)**2*(np.sin(2*theta)))/(r**2)
        #### omega_r partials with respect to G1, G2, G3, G4, G5, G6
        dr_d_G1 = V_inf*((G5*np.cos(theta))/(G5**2 + G6**2) - (G6*np.sin(theta))/(G5**2 + G6**2))
        dr_d_G2 = V_inf*((G6*np.cos(theta))/(G5**2 + G6**2) - (G5*np.sin(theta))/(G5**2 + G6**2))
        dr_d_G3 = -V_inf*((G5*np.cos(theta))/(G5**2 + G6**2) - (G6*np.sin(theta))/(G5**2 + G6**2))
        dr_d_G4 = -V_inf*((G6*np.cos(theta))/(G5**2 + G6**2) - (G5*np.sin(theta))/(G5**2 + G6**2))
        dr_d_G5 = V_inf*((np.cos(theta)*(G1 - G3 + np.cos(alpha)))/(G5**2 + G6**2) + (np.sin(theta)*(G4 - G2 + np.sin(alpha)))/(G5**2 + G6**2) - (2*G5*np.cos(theta)*(G5*(G1 - G3 + np.cos(alpha)) - G6*(G4 - G2 + np.sin(alpha))))/(G5**2 + G6**2)**2 + (2*G5*np.sin(theta)*(G6*(G1 - G3 + np.cos(alpha)) - G5*(G4 - G2 + np.sin(alpha))))/(G5**2 + G6**2)**2)
        dr_d_G6 = -V_inf*((np.sin(theta)*(G1 - G3 + np.cos(alpha)))/(G5**2 + G6**2) + (np.cos(theta)*(G4 - G2 + np.sin(alpha)))/(G5**2 + G6**2) + (2*G6*np.cos(theta)*(G5*(G1 - G3 + np.cos(alpha)) - G6*(G4 - G2 + np.sin(alpha))))/(G5**2 + G6**2)**2 - (2*G6*np.sin(theta)*(G6*(G1 - G3 + np.cos(alpha)) - G5*(G4 - G2 + np.sin(alpha))))/(G5**2 + G6**2)**2)
        #### omega_theta partials with respect to G1, G2, G3, G4, G5, G6
        dt_d_G1 = V_inf*((G6*np.cos(theta))/(G5**2 + G6**2) - (G5*np.sin(theta))/(G5**2 + G6**2))
        dt_d_G2 = -V_inf*((G5*np.cos(theta))/(G5**2 + G6**2) + (G6*np.sin(theta))/(G5**2 + G6**2))
        dt_d_G3 = -V_inf*((G6*np.cos(theta))/(G5**2 + G6**2) - (G5*np.sin(theta))/(G5**2 + G6**2))
        dt_d_G4 = V_inf*((G5*np.cos(theta))/(G5**2 + G6**2) + (G6*np.sin(theta))/(G5**2 + G6**2))
        dt_d_G5 = -V_inf*((np.sin(theta)*(G1 - G3 + np.cos(alpha)))/(G5**2 + G6**2) - (np.cos(theta)*(G4 - G2 + np.sin(alpha)))/(G5**2 + G6**2) + (2*G5*np.cos(theta)*(G6*(G1 - G3 + np.cos(alpha)) + G5*(G4 - G2 + np.sin(alpha))))/(G5**2 + G6**2)**2 - (2*G5*np.sin(theta)*(G5*(G1 - G3 + np.cos(alpha)) - G6*(G4 - G2 + np.sin(alpha))))/(G5**2 + G6**2)**2)
        dt_d_G6 = V_inf*((np.cos(theta)*(G1 - G3 + np.cos(alpha)))/(G5**2 + G6**2) + (np.sin(theta)*(G4 - G2 + np.sin(alpha)))/(G5**2 + G6**2) - (2*G6*np.cos(theta)*(G6*(G1 - G3 + np.cos(alpha)) + G5*(G4 - G2 + np.sin(alpha))))/(G5**2 + G6**2)**2 + (2*G6*np.sin(theta)*(G5*(G1 - G3 + np.cos(alpha)) - G6*(G4 - G2 + np.sin(alpha))))/(G5**2 + G6**2)**2)
        #### G partials with respect to r 
        dG1_d_r = (Gamma*np.sin(theta))/(2*np.pi*V_inf*(r**2 - 2*np.cos(theta - theta0)*r*r0 + r0**2)) - (Gamma*(2*r - 2*r0*np.cos(theta - theta0))*(r*np.sin(theta) - r0*np.sin(theta0)))/(2*np.pi*V_inf*(r**2 - 2*np.cos(theta - theta0)*r*r0 + r0**2)**2)
        dG2_d_r = (Gamma*np.cos(theta))/(2*np.pi*V_inf*(r**2 - 2*np.cos(theta - theta0)*r*r0 + r0**2)) - (Gamma*(2*r - 2*r0*np.cos(theta - theta0))*(r*np.cos(theta) - r0*np.cos(theta0)))/(2*np.pi*V_inf*(r**2 - 2*np.cos(theta - theta0)*r*r0 + r0**2)**2)
        dG3_d_r = (R**2*(2*r*np.cos(alpha) - 2*r0*np.cos(theta - alpha + theta0) + 4*r*np.sin(alpha - theta)*np.sin(theta)))/((np.cos(2*theta)*r**2 - 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 - 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2) - (R**2*(2*(2*r*np.cos(2*theta) - 2*r0*np.cos(theta + theta0))*(np.cos(2*theta)*r**2 - 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2) + 2*(2*r*np.sin(2*theta) - 2*r0*np.sin(theta + theta0))*(np.sin(2*theta)*r**2 - 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))*(r**2*np.cos(alpha) + r0**2*np.cos(alpha) + 2*r**2*np.sin(alpha - theta)*np.sin(theta) + 2*r0**2*np.sin(alpha - theta0)*np.sin(theta0) - 2*r*r0*np.cos(theta - alpha + theta0)))/((np.cos(2*theta)*r**2 - 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 - 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2)**2
        dG4_d_r = (R**2*(2*r0*np.sin(theta - alpha + theta0) + 2*r*np.sin(alpha) - 4*r*np.sin(theta)*np.cos(alpha - theta)))/((np.cos(2*theta)*r**2 - 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 - 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2) - (R**2*(2*(2*r*np.cos(2*theta) - 2*r0*np.cos(theta + theta0))*(np.cos(2*theta)*r**2 - 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2) + 2*(2*r*np.sin(2*theta) - 2*r0*np.sin(theta + theta0))*(np.sin(2*theta)*r**2 - 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))*(r**2*np.sin(alpha) + r0**2*np.sin(alpha) - 2*r**2*np.sin(theta)*np.cos(alpha - theta) - 2*r0**2*np.sin(theta0)*np.cos(alpha - theta0) + 2*r*r0*np.sin(theta - alpha + theta0)))/((np.cos(2*theta)*r**2 - 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 - 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2)**2
        dG5_d_r = (2*np.cos(2*theta)*(R - epsilon)**2)/r**3 
        dG6_d_r = -(2*np.sin(2*theta)*(R - epsilon)**2)/r**3
        #### G partials with respect to theta
        dG1_d_theta = (Gamma*r*np.cos(theta))/(2*np.pi*V_inf*(r**2 - 2*np.cos(theta - theta0)*r*r0 + r0**2)) - (Gamma*r*r0*np.sin(theta - theta0)*(r*np.sin(theta) - r0*np.sin(theta0)))/(np.pi*V_inf*(r**2 - 2*np.cos(theta - theta0)*r*r0 + r0**2)**2)
        dG2_d_theta = - (Gamma*r*np.sin(theta))/(2*np.pi*V_inf*(r**2 - 2*np.cos(theta - theta0)*r*r0 + r0**2)) - (Gamma*r*r0*np.sin(theta - theta0)*(r*np.cos(theta) - r0*np.cos(theta0)))/(np.pi*V_inf*(r**2 - 2*np.cos(theta - theta0)*r*r0 + r0**2)**2)
        dG3_d_theta = (R**2*(2*r**2*np.sin(alpha - theta)*np.cos(theta) - 2*r**2*np.sin(theta)*np.cos(alpha - theta) + 2*r*r0*np.sin(theta - alpha + theta0)))/((np.cos(2*theta)*r**2 - 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 - 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2) + (R**2*(2*(2*np.sin(2*theta)*r**2 - 2*r0*np.sin(theta + theta0)*r)*(np.cos(2*theta)*r**2 - 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2) - 2*(2*np.cos(2*theta)*r**2 - 2*r0*np.cos(theta + theta0)*r)*(np.sin(2*theta)*r**2 - 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))*(r**2*np.cos(alpha) + r0**2*np.cos(alpha) + 2*r**2*np.sin(alpha - theta)*np.sin(theta) + 2*r0**2*np.sin(alpha - theta0)*np.sin(theta0) - 2*r*r0*np.cos(theta - alpha + theta0)))/((np.cos(2*theta)*r**2 - 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 - 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2)**2
        dG4_d_theta = (R^2*(2*(2*sin(2*theta)*r^2 - 2*r_0*sin(theta + theta_0)*r)*(cos(2*theta)*r^2 - 2*cos(theta + theta_0)*r*r_0 + cos(2*theta_0)*r_0^2) - 2*(2*cos(2*theta)*r^2 - 2*r_0*cos(theta + theta_0)*r)*(sin(2*theta)*r^2 - 2*sin(theta + theta_0)*r*r_0 + sin(2*theta_0)*r_0^2))*(r^2*sin(alpha) + r_0^2*sin(alpha) - 2*r^2*sin(theta)*cos(alpha - theta) - 2*r_0^2*sin(theta_0)*cos(alpha - theta_0) + 2*r*r_0*sin(theta - alpha + theta_0)))/((cos(2*theta)*r^2 - 2*cos(theta + theta_0)*r*r_0 + cos(2*theta_0)*r_0^2)^2 + (sin(2*theta)*r^2 - 2*sin(theta + theta_0)*r*r_0 + sin(2*theta_0)*r_0^2)^2)^2 - (R^2*(2*r^2*cos(theta)*cos(alpha - theta) + 2*r^2*sin(alpha - theta)*sin(theta) - 2*r*r_0*cos(theta - alpha + theta_0)))/((cos(2*theta)*r^2 - 2*cos(theta + theta_0)*r*r_0 + cos(2*theta_0)*r_0^2)^2 + (sin(2*theta)*r^2 - 2*sin(theta + theta_0)*r*r_0 + sin(2*theta_0)*r_0^2)^2)
        dG5_d_theta = (2*sin(2*theta)*(R - epsilon)^2)/r^2
        dG6_d_theta = (2*cos(2*theta)*(R - epsilon)^2)/r^2

    def zeta_to_z(self, zeta: complex):
        """This function takes in a zeta coordinate and returns the z coordinate"""
        # if not np.isclose(zeta.real, 0.0) and not np.isclose(zeta.imag, 0.0):
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
        zeta_center = self.cylinder_radius*(first + second)
        return zeta_center
    
    def calc_J_airfoil_epsilon(self):
        """This function calculates the epsilon of the airfoil"""
        epsilon = self.cylinder_radius - 1*np.sqrt(self.cylinder_radius**2-self.zeta_center.imag**2)-self.zeta_center.real
        return epsilon
    
    def calc_circulation_J_airfoil(self):
        """This function calculates the circulation of the airfoil based on the Kutta condition"""
        gamma = 4*np.pi*self.freestream_velocity*(np.sin(self.angle_of_attack)*np.sqrt(self.cylinder_radius**2 - self.zeta_center.imag**2) + self.zeta_center.imag*np.cos(self.angle_of_attack))
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
    plt.text(-0.07, -0.01, str(-cyl.original_cylinder_radius), transform=plt.gca().transAxes, fontsize=17, verticalalignment='top')

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

    zeta_test = 10 - 1j*4
    # z_test = cyl.zeta_to_z(zeta_test)
    chi_test = cyl.zeta_to_Chi(zeta_test)
    print("\n")
    z_test = cyl.zeta_to_z(zeta_test)
    complex_zeta_center = cyl.zeta_center.real + 1j*cyl.zeta_center.imag
    z_0 = cyl.zeta_to_z(complex_zeta_center)
    # print("z_0: ", z_0)
    zeta_test = [zeta_test.real, zeta_test.imag]
    theta_zeta = np.arctan2(zeta_test[1], zeta_test[0])
    z_test = [z_test.real, z_test.imag]
    # theta_z = np.arctan2(z_test[1], z_test[0])
    chi_test = [chi_test.real, chi_test.imag]
    theta_chi = np.arctan2(chi_test[1], chi_test[0])
    z_velocity_at_z_test = cyl.velocity(z_test)
    Chi_velocity_at_z_test = cyl.Chi_velocity(chi_test)
    # chi_test = [chi_test[0]+cyl.zeta_center.real, chi_test[1]+cyl.zeta_center.imag]
    z_polar = cyl.polar_velocity(theta_zeta, z_velocity_at_z_test)
    Chi_polar = cyl.polar_velocity(theta_chi, Chi_velocity_at_z_test)

    z_polar_mag = np.sqrt(z_polar[0]**2 + z_polar[1]**2)
    print("z_polar_mag: ", z_polar_mag)
    Chi_polar_mag = np.sqrt(Chi_polar[0]**2 + Chi_polar[1]**2)
    print("Chi_polar_mag: ", Chi_polar_mag)
    # alternative_z_velocity_at_z_test = cyl.alternative_velocity(z_test)
    # polar_z_velocity_at_z_test = cyl.polar_velocity(z_test)
    print("\n")
