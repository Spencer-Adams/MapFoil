import json
import numpy as np # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import scipy.integrate as spi # this is for numerical integration
# import bisection from scipy.optimize
import helper as hlp
from complex_potential_flow_class import potential_flow_object  

class extras(potential_flow_object):
    """This is a class that creates a cylinder object and performs calculations specific to a cylinder"""
    def __init__(self, json_file):
        self.cyl_json_file = json_file
        self.parse_cylinder_json()
        super().__init__(json_file) # this is doing self.() of all the variables we want to in the super class

    def calc_z_G_values(self, xi, eta, alpha, epsilon, R, xio, etao):
        """Cartesian G values for the z plane"""
        G1 = (eta-etao)/((xi-xio)**2 + (eta-etao)**2)
        G2 = (xi-xio)/((xi-xio)**2 + (eta-etao)**2)
        G3 = R**2*(np.cos(alpha)*((xi-xio)**2-(eta-etao)**2)+2*np.sin(alpha)*((xi-xio)*(eta-etao)))/(((xi-xio)**2 -(eta-etao)**2)**2 + (2*(xi-xio)*(eta-etao))**2)
        G4 = R**2*(np.sin(alpha)*((xi-xio)**2-(eta-etao)**2)-2*np.cos(alpha)*((xi-xio)*(eta-etao)))/(((xi-xio)**2 -(eta-etao)**2)**2 + (2*(xi-xio)*(eta-etao))**2)
        G5 = 1 - ((xi**2 - eta**2)*(R-epsilon)**2)/(((xi**2 - eta**2)**2 + (2*xi*eta)**2))
        G6 = ((2*xi*eta)*(R-epsilon)**2)/(((xi**2 - eta**2)**2 + (2*xi*eta)**2))
        return G1, G2, G3, G4, G5, G6

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
        # xi, eta = point_xy_in_zeta_plane[0], point_xy_in_zeta_plane[1]
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
        convective_acceleration_squared = np.dot(chi_convective_acceleration, chi_convective_acceleration)
        # convective_acceleration_squared = Gamma**4*C1 + Gamma**3*C2 + Gamma**2*C3 + Gamma*C4 + C5
        return convective_acceleration_squared










    def split_velocity_z(self, point_xy_in_z_plane, Gamma):
        """This function calculates the velocity at a given point in the flow field in the z plane"""
        z = point_xy_in_z_plane[0] + 1j*point_xy_in_z_plane[1]
        zeta = self.z_to_zeta(z)
        xi, eta = zeta.real, zeta.imag
        G1, G2, G3, G4, G5, G6 = self.calc_z_G_values(xi, eta, self.angle_of_attack, self.epsilon, self.cylinder_radius, self.zeta_center.real, self.zeta_center.imag)
        omega_real = (Gamma/(2*np.pi))*((G1*G5+G2*G6)/(G5**2 + G6**2)) + self.freestream_velocity*((G5*np.cos(self.angle_of_attack)-G6*np.sin(self.angle_of_attack)-G3*G5-G4*G6)/(G5**2 + G6**2))
        omega_imag = -(Gamma/(2*np.pi))*((G2*G5-G1*G6)/(G5**2 + G6**2)) + self.freestream_velocity*((G6*np.cos(self.angle_of_attack)+G5*np.sin(self.angle_of_attack)+G4*G5-G3*G6)/(G5**2 + G6**2))
        return omega_real, omega_imag


    
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
    

    def velocity(self, point_xi_eta_in_z_plane, Gamma):
        """This function calculates the velocity at a given point in the flow field in the z plane"""
        z = point_xi_eta_in_z_plane[0] + 1j*point_xi_eta_in_z_plane[1]
        zeta = self.z_to_zeta(z)
        V_inf, epsilon, R, alpha, zeta0 = self.freestream_velocity, self.epsilon, self.cylinder_radius, self.angle_of_attack, self.zeta_center
        velocity = V_inf*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf*(zeta-zeta0)) - np.exp(1j*alpha)*R**2/(zeta-zeta0)**2) / (1 - (R-epsilon)**2/(zeta)**2) # eq 107
        velocity_complex = np.array([velocity.real, -velocity.imag])
        return velocity_complex

    def polar_vector(self, theta, cartesian_vector):
        """This function converts the cartesian velocity to polar velocity (can go from z, zeta, or chi plane to polar velocity)"""
        r = cartesian_vector[0]*np.cos(theta) + cartesian_vector[1]*np.sin(theta)
        # print("\nradial velocity", velocity_r)
        theta = cartesian_vector[1]*np.cos(theta) - cartesian_vector[0]*np.sin(theta)
        # print("theta velocity", velocity_theta)
        polar_velocity = np.array([r, theta])
        return polar_velocity
    
    def numerical_convective_acceleration(self, point_xi_eta_z, Gamma, step):
        """calculates the convective acceleration numerically in the z plane"""
        r, theta = hlp.xy_to_r_theta(point_xi_eta_z[0], point_xi_eta_z[1])
        cartesian_z_velocity = self.velocity(point_xi_eta_z, Gamma)
        polar_z_velocity = hlp.polar_vector(theta, cartesian_z_velocity)
        omega_r, omega_theta = polar_z_velocity[0], polar_z_velocity[1]
        step_point = point_xi_eta_z
        
        omega_r_plus_dr, omega_r_minus_dr, omega_r_plus_dtheta, omega_r_minus_dtheta, omega_theta_plus_dr, omega_theta_minus_dr, omega_theta_plus_dtheta, omega_theta_minus_dtheta = self.function_plus_minus_step_variable(step_point, Gamma, step)
        # calculate the partial derivatives of omega r with respect to r and theta
        partial_omega_r_wrt_r = hlp.central_difference(omega_r_plus_dr, omega_r_minus_dr, step)
        partial_omega_r_wrt_theta = hlp.central_difference(omega_r_plus_dtheta, omega_r_minus_dtheta, step)
        # calculate the partial derivatives of omega theta with respect to r and theta
        partial_omega_theta_wrt_r = hlp.central_difference(omega_theta_plus_dr, omega_theta_minus_dr, step)
        partial_omega_theta_wrt_theta = hlp.central_difference(omega_theta_plus_dtheta, omega_theta_minus_dtheta, step)
        # calculate the convective acceleration
        convective_acceleration_r = omega_r*partial_omega_r_wrt_r + (omega_theta/r)*partial_omega_r_wrt_theta - omega_theta**2/r
        convective_acceleration_theta = omega_r*partial_omega_theta_wrt_r + (omega_theta/r)*partial_omega_theta_wrt_theta + omega_r*omega_theta/r
        convective_acceleration = np.array([convective_acceleration_r, convective_acceleration_theta])
        return convective_acceleration
        
    def function_plus_minus_step_variable(self, point_xy, Gamma, stepsize):
        """takes in a function and returns the function plus and minus a stepsize"""
        r, theta = hlp.xy_to_r_theta(point_xy[0], point_xy[1])
        r_plus, r_minus = r + stepsize, r - stepsize
        theta_plus, theta_minus = theta + stepsize, theta - stepsize

        x_r_plus, x_r_minus, y_r_plus, y_r_minus = hlp.r_theta_to_xy(r_plus, theta)[0], hlp.r_theta_to_xy(r_minus, theta)[0], hlp.r_theta_to_xy(r_plus, theta)[1], hlp.r_theta_to_xy(r_minus, theta)[1]
        x_theta_plus, x_theta_minus, y_theta_plus, y_theta_minus = hlp.r_theta_to_xy(r, theta_plus)[0], hlp.r_theta_to_xy(r, theta_minus)[0], hlp.r_theta_to_xy(r, theta_plus)[1], hlp.r_theta_to_xy(r, theta_minus)[1]
        omega_xy_plus_r, omega_xy_minus_r = self.velocity([x_r_plus, y_r_plus], Gamma), self.velocity([x_r_minus, y_r_minus], Gamma)
        omega_xy_plus_theta, omega_xy_minus_theta = self.velocity([x_theta_plus, y_theta_plus], Gamma), self.velocity([x_theta_minus, y_theta_minus], Gamma)
        omega_r_plus_dr, omega_r_minus_dr = hlp.polar_vector(theta, omega_xy_plus_r)[0], hlp.polar_vector(theta, omega_xy_minus_r)[0]
        omega_r_plus_dtheta, omega_r_minus_dtheta = hlp.polar_vector(theta, omega_xy_plus_theta)[0], hlp.polar_vector(theta, omega_xy_minus_theta)[0]
        omega_theta_plus_dr, omega_theta_minus_dr = hlp.polar_vector(theta, omega_xy_plus_r)[1], hlp.polar_vector(theta, omega_xy_minus_r)[1]
        omega_theta_plus_dtheta, omega_theta_minus_dtheta = hlp.polar_vector(theta, omega_xy_plus_theta)[1], hlp.polar_vector(theta, omega_xy_minus_theta)[1]
        return omega_r_plus_dr, omega_r_minus_dr, omega_r_plus_dtheta, omega_r_minus_dtheta, omega_theta_plus_dr, omega_theta_minus_dr, omega_theta_plus_dtheta, omega_theta_minus_dtheta

    def convective_acceleration_squared_numerical(self, point_xi_eta, Gamma, step):
        """Calculates the convective acceleration squared numerically xi and eta are the coordinates in the plane of interest"""
        convective_acceleration = self.numerical_convective_acceleration(point_xi_eta, Gamma, step)
        convective_acceleration_squared = np.dot(convective_acceleration, convective_acceleration)
        return convective_acceleration_squared
    
    def numerical_appellian(self, Gamma_range: list, r_range: list, theta_range: list, is_analytic_conv_accel: bool, step):
        """This function calculates the Appellian function numerically requires evenly spaced ranges for Gamma, r, and theta in Chi"""
        # create a meshgrid of r and theta values, the first value in r_range is the lower bound the second value is the upper bound, the third value is the increment size
        Gamma_values = hlp.list_to_range(Gamma_range)
        r_values = hlp.list_to_range(r_range)
        theta_values = hlp.list_to_range(theta_range)
        # calculate the area element 
        if len(r_values) == 1:
            dr = 1.0
        else:        
            dr = r_values[1] - r_values[0]
        if len(theta_values) == 1:
            dtheta = 1.0
        else:
            dtheta = theta_values[1] - theta_values[0]
        # now get dr and dtheta in z plane using the zeta to z function 
        d_zeta = hlp.r_theta_to_xy(dr, dtheta)[0] + 1j*hlp.r_theta_to_xy(dr, dtheta)[1]
        d_z = self.zeta_to_z(d_zeta)
        d_xi_z, d_eta_z = d_z.real, d_z.imag
        # now these are the dr and dtheta values in the z plane
        dr, dtheta = hlp.xy_to_r_theta(d_xi_z, d_eta_z)
        # The appellian sums the squared convective acceleration at each r and theta value. Plot the Appellian function with respect to each change in Gamma.
        Appellian_array = np.zeros((len(Gamma_values), 2))
        xi_eta_values = np.zeros((len(r_values)*len(theta_values), 2))
        Appellian_value = 0.0
        index = 0
        print("NUMERICAL CONVECTIVE ACCELERATION")
        for i in range(len(Gamma_values)):
            for j in range(len(r_values)):
                for k in range(len(theta_values)):
                    Chi = hlp.r_theta_to_xy(r_values[j], theta_values[k])[0] + 1j*hlp.r_theta_to_xy(r_values[j], theta_values[k])[1] # the r, theta point in the Chi plane is converted to a complex number
                    zeta = self.Chi_to_zeta(Chi)
                    z = self.zeta_to_z(zeta)
                    r_z = hlp.xy_to_r_theta(z.real, z.imag)[0]
                    area_element = r_z * dr * dtheta
                    convective_acceleration = self.numerical_convective_acceleration([z.real, z.imag], Gamma_values[i], step)
                    Appellian_value += np.dot(convective_acceleration, convective_acceleration)*area_element
                    if i == 0:  # Save xi and eta values only once
                        xi_eta_values[index] = [z.real, z.imag]
                        index += 1
            Appellian_array[i] = [Gamma_values[i], 0.5*Appellian_value]
            Appellian_value = 0.0
        return Appellian_array, xi_eta_values
    
    def zeta_to_Chi(self, zeta: complex):
        """This function takes in a zeta coordinate and returns the Chi coordinate"""
        Chi = zeta - self.zeta_center
        return Chi

    def Chi_to_zeta(self, Chi: complex):
        """This function takes in a Chi coordinate and returns the zeta coordinate"""
        zeta = Chi + self.zeta_center
        return zeta
    
    def zeta_to_z(self, zeta: complex):
        """This function takes in a zeta coordinate and returns the z coordinate"""
        if np.isclose(zeta.real, 0.0) and np.isclose(zeta.imag, 0.0):
            z = zeta
        else:
            z = zeta + (self.cylinder_radius - self.epsilon)**2/zeta # eq 96
        return z
    
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

    def calc_Chi_G_values(self, r, theta, alpha, epsilon, R, r0, theta0):
        """takes in r, theta, alpha, epsilon, R, r0, theta0 and calculates the G values"""
        G1 = np.sin(theta)/r
        G2 = np.cos(theta)/r
        G3 = R**2*np.cos(alpha-2*theta)/r**2
        G4 = R**2*np.sin(alpha-2*theta)/r**2
        G5 = 1 - ((R-epsilon)**2*(r**2*np.cos(2*theta)+r0**2*np.cos(2*theta0)+2*r*r0*np.cos(theta+theta0)))/((r**2*np.cos(2*theta)+r0**2*np.cos(2*theta0)+2*r*r0*np.cos(theta+theta0))**2+(r**2*np.sin(2*theta)+r0**2*np.sin(2*theta0)+2*r*r0*np.sin(theta+theta0))**2)
        G6 = ((R-epsilon)**2*(r**2*np.sin(2*theta)+r0**2*np.sin(2*theta0)+2*r*r0*np.sin(theta+theta0)))/((r**2*np.cos(2*theta)+r0**2*np.cos(2*theta0)+2*r*r0*np.cos(theta+theta0))**2+(r**2*np.sin(2*theta)+r0**2*np.sin(2*theta0)+2*r*r0*np.sin(theta+theta0))**2)
        return G1, G2, G3, G4, G5, G6
    
    def calc_z_G_values(self, r, theta, alpha, epsilon, R, r_0, theta_0):
        """takes in r, theta, alpha, epsilon, R, r0, theta0 and calculates the G values"""
        G1 = (r*np.sin(theta) - r_0*np.sin(theta_0))/(r**2 + r_0**2 - 2*r*r_0*np.cos(theta - theta_0))
        G2 = (r*np.cos(theta) - r_0*np.cos(theta_0))/(r**2 + r_0**2 - 2*r*r_0*np.cos(theta - theta_0))
        G3 = (R**2*(np.cos(alpha)*((r*np.cos(theta)-r_0*np.cos(theta_0))**2-(r*np.sin(theta)-r_0*np.sin(theta_0))**2)+2*np.sin(alpha)*((r*np.cos(theta)-r_0*np.cos(theta_0))*(r*np.sin(theta)-r_0*np.sin(theta_0)))))/(((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2+(2*(r*np.cos(theta)-r_0*np.cos(theta_0))*(r*np.sin(theta)-r_0*np.sin(theta_0)))**2)
        G4 = (R**2*(np.sin(alpha)*((r*np.cos(theta)-r_0*np.cos(theta_0))**2-(r*np.sin(theta)-r_0*np.sin(theta_0))**2)-2*np.cos(alpha)*((r*np.cos(theta)-r_0*np.cos(theta_0))*(r*np.sin(theta)-r_0*np.sin(theta_0)))))/(((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2+(2*(r*np.cos(theta)-r_0*np.cos(theta_0))*(r*np.sin(theta)-r_0*np.sin(theta_0)))**2)
        G5 = 1 - (np.cos(2*theta)*(R - epsilon)**2)/r**2
        G6 = (np.sin(2*theta)*(R - epsilon)**2)/r**2
        return G1, G2, G3, G4, G5, G6
    
    def partial_coefficients_of_omega_r_wrt_G_Chi(self, V_inf, theta, alpha, G1, G2, G3, G4, G5, G6):
        """This function retrieves the partial derivatives of omega r with respect to G values"""
        # G1, G2, G3, G4, G5, G6 = self.calc_Chi_G_values(r, theta, alpha, epsilon, R, r0, theta0)
        # omega r partials with respect to G1, G2, G3, G4, G5, G6
        A_omega_rG = 1/(2*np.pi)*((G5*np.cos(theta))/(G5**2 + G6**2) + (G6*np.sin(theta))/(G5**2 + G6**2))
        B_omega_rG = 1/(2*np.pi)*((G6*np.cos(theta))/(G5**2 + G6**2) - (G5*np.sin(theta))/(G5**2 + G6**2))
        C_omega_rG = - V_inf*(G5*np.cos(theta)/(G5**2 + G6**2) + G6*np.sin(theta)/(G5**2 + G6**2))
        D_omega_rG = V_inf*(G5*np.sin(theta)/(G5**2 + G6**2) - G6*np.cos(theta)/(G5**2 + G6**2))
        E_omega_rG = (G1*np.cos(theta))/(2*np.pi*(G5**2 + G6**2)) - (G2*np.sin(theta))/(2*np.pi*(G5**2 + G6**2)) - (G5*np.cos(theta)*(G1*G5 + G2*G6))/(np.pi*(G5**2 + G6**2)**2) - (G5*np.sin(theta)*(G1*G6 - G2*G5))/(np.pi*(G5**2 + G6**2)**2)
        F_omega_rG = V_inf*((np.sin(theta)*(G4+np.sin(alpha)))/(G5**2+G6**2)-(np.cos(theta)*(G3-np.cos(alpha)))/(G5**2+G6**2)+(2*G5*np.cos(theta)*(G6*np.sin(alpha)-G5*np.cos(alpha)+G3*G5+G4*G6))/(G5**2+G6**2)**2-(2*G5*np.sin(theta)*(G6*np.cos(alpha)+G5*np.sin(alpha)-G3*G6+G4*G5))/(G5**2+G6**2)**2)
        G_omega_rG = (G2*np.cos(theta))/(2*np.pi*(G5**2 + G6**2)) + (G1*np.sin(theta)/(2*np.pi*(G5**2+G6**2))) - (G6*np.cos(theta)*(G1*G5+G2*G6))/(np.pi*(G5**2+G6**2)**2) - (G6*np.sin(theta)*(G1*G6-G2*G5))/(np.pi*(G5**2+G6**2)**2)
        H_omega_rG = V_inf*((2*G6*np.cos(theta)*(G6*np.sin(alpha)-G5*np.cos(alpha)+G3*G5+G4*G6))/(G5**2+G6**2)**2-(np.sin(theta)*(G3-np.cos(alpha)))/(G5**2+G6**2)-(np.cos(theta)*(G4+np.sin(alpha)))/(G5**2+G6**2)-(2*G6*np.sin(theta)*(G6*np.cos(alpha)+G5*np.sin(alpha)-G3*G6+G4*G5))/(G5**2+G6**2)**2)
        return A_omega_rG, B_omega_rG, C_omega_rG, D_omega_rG, E_omega_rG, F_omega_rG, G_omega_rG, H_omega_rG
    
    def partial_coefficients_of_omega_r_wrt_G_z(self, V_inf, theta, alpha, G1, G2, G3, G4, G5, G6):
        """This function retrieves the partial derivatives of omega r with respect to G values"""
        A_omega_rG = 1/(2*np.pi)*((G5*np.cos(theta))/(G5**2 + G6**2) + (G6*np.sin(theta))/(G5**2 + G6**2))
        B_omega_rG = 1/(2*np.pi)*((G6*np.cos(theta))/(G5**2 + G6**2) - (G5*np.sin(theta))/(G5**2 + G6**2))
        C_omega_rG = - V_inf*(G5*np.cos(theta)/(G5**2 + G6**2) + G6*np.sin(theta)/(G5**2 + G6**2))
        D_omega_rG = V_inf*(G5*np.sin(theta)/(G5**2 + G6**2) - G6*np.cos(theta)/(G5**2 + G6**2))
        E_omega_rG = (G1*np.cos(theta))/(2*np.pi*(G5**2 + G6**2)) - (G2*np.sin(theta))/(2*np.pi*(G5**2 + G6**2)) - (G5*np.cos(theta)*(G1*G5 + G2*G6))/(np.pi*(G5**2 + G6**2)**2) - (G5*np.sin(theta)*(G1*G6 - G2*G5))/(np.pi*(G5**2 + G6**2)**2)
        F_omega_rG = V_inf*((np.sin(theta)*(G4+np.sin(alpha)))/(G5**2+G6**2)-(np.cos(theta)*(G3-np.cos(alpha)))/(G5**2+G6**2)+(2*G5*np.cos(theta)*(G6*np.sin(alpha)-G5*np.cos(alpha)+G3*G5+G4*G6))/(G5**2+G6**2)**2-(2*G5*np.sin(theta)*(G6*np.cos(alpha)+G5*np.sin(alpha)-G3*G6+G4*G5))/(G5**2+G6**2)**2)
        G_omega_rG = (G2*np.cos(theta))/(2*np.pi*(G5**2 + G6**2)) + (G1*np.sin(theta)/(2*np.pi*(G5**2+G6**2))) - (G6*np.cos(theta)*(G1*G5+G2*G6))/(np.pi*(G5**2+G6**2)**2) - (G6*np.sin(theta)*(G1*G6-G2*G5))/(np.pi*(G5**2+G6**2)**2)
        H_omega_rG = V_inf*((2*G6*np.cos(theta)*(G6*np.sin(alpha)-G5*np.cos(alpha)+G3*G5+G4*G6))/(G5**2+G6**2)**2-(np.sin(theta)*(G3-np.cos(alpha)))/(G5**2+G6**2)-(np.cos(theta)*(G4+np.sin(alpha)))/(G5**2+G6**2)-(2*G6*np.sin(theta)*(G6*np.cos(alpha)+G5*np.sin(alpha)-G3*G6+G4*G5))/(G5**2+G6**2)**2)
        return A_omega_rG, B_omega_rG, C_omega_rG, D_omega_rG, E_omega_rG, F_omega_rG, G_omega_rG, H_omega_rG

    def partial_coefficients_of_omega_theta_wrt_G_Chi(self, V_inf, theta, alpha, G1, G2, G3, G4, G5, G6):
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
    
    def partial_coefficients_of_omega_theta_wrt_G_z(self, V_inf, theta, alpha, G1, G2, G3, G4, G5, G6):
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
    
    def partial_coefficients_of_G_wrt_r_Chi(self, r, theta, alpha, epsilon, R, r0, theta0):
        """This function retrieves the partial derivatives of G values with respect to r"""
        AGr = -np.sin(theta)/r**2
        BGr = -np.cos(theta)/r**2
        CGr = (R**2*(r**2*np.cos(2*theta)*np.cos(alpha) + r**2*np.sin(2*theta)*np.sin(alpha))*(2*r**3*(np.cos(4*theta) - 1) - 4*r**3*np.cos(2*theta)**2))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2)**2 - (R**2*(2*r*np.cos(2*theta)*np.cos(alpha) + 2*r*np.sin(2*theta)*np.sin(alpha)))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2)
        DGr = (R**2*(r**2*np.cos(2*theta)*np.sin(alpha) - r**2*np.sin(2*theta)*np.cos(alpha))*(2*r**3*(np.cos(4*theta) - 1) - 4*r**3*np.cos(2*theta)**2))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2)**2 - (R**2*(2*r*np.cos(2*theta)*np.sin(alpha) - 2*r*np.sin(2*theta)*np.cos(alpha)))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2)
        EGr = ((R - epsilon)**2*(2*(2*r*np.cos(2*theta) + 2*r0*np.cos(theta + theta0))*(np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2) + 2*(2*r*np.sin(2*theta) + 2*r0*np.sin(theta + theta0))*(np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))*(np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2)**2 - ((R - epsilon)**2*(2*r*np.cos(2*theta) + 2*r0*np.cos(theta + theta0)))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2)
        FGr= ((2*r*np.sin(2*theta) + 2*r0*np.sin(theta + theta0))*(R - epsilon)**2)/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2) - ((R - epsilon)**2*(2*(2*r*np.cos(2*theta) + 2*r0*np.cos(theta + theta0))*(np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2) + 2*(2*r*np.sin(2*theta) + 2*r0*np.sin(theta + theta0))*(np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))*(np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2)**2
        return AGr, BGr, CGr, DGr, EGr, FGr
    
    def partial_coefficients_of_G_wrt_r_z(self, r, theta, alpha, epsilon, R, r_0, theta_0):
        """"""
        AGr = (r_0**2*np.sin(theta - 2*theta_0) - r**2*np.sin(theta) + 2*r*r_0*np.sin(theta_0))/(r**2 + r_0**2 - 2*r*r_0*np.cos(theta - theta_0))**2
        BGr = -(r**2*np.cos(theta) + r_0**2*np.cos(theta - 2*theta_0) - 2*r*r_0*np.cos(theta_0))/(r**2 + r_0**2 - 2*r*r_0*np.cos(theta - theta_0))**2
        CGr = - (R**2*(2*r_0*np.cos(theta - alpha + theta_0) - 2*r*np.cos(alpha - 2*theta)))/(((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2 + 4*(r*np.cos(theta) - r_0*np.cos(theta_0))**2*(r*np.sin(theta) - r_0*np.sin(theta_0))**2) - (R**2*(np.cos(alpha)*((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2) + 2*np.sin(alpha)*(r*np.cos(theta) - r_0*np.cos(theta_0))*(r*np.sin(theta) - r_0*np.sin(theta_0)))*(8*r*r_0**2 - 4*r_0**3*np.cos(theta - theta_0) + 4*r**3 + 4*r*r_0**2*np.cos(2*theta - 2*theta_0) - 12*r**2*r_0*np.cos(theta - theta_0)))/(((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2 + 4*(r*np.cos(theta) - r_0*np.cos(theta_0))**2*(r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2
        DGr = (R**2*(2*r_0*np.sin(theta - alpha + theta_0) + 2*r*np.sin(alpha - 2*theta)))/(((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2 + 4*(r*np.cos(theta) - r_0*np.cos(theta_0))**2*(r*np.sin(theta) - r_0*np.sin(theta_0))**2) - (R**2*(np.sin(alpha)*((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2) - 2*np.cos(alpha)*(r*np.cos(theta) - r_0*np.cos(theta_0))*(r*np.sin(theta) - r_0*np.sin(theta_0)))*(8*r*r_0**2 - 4*r_0**3*np.cos(theta - theta_0) + 4*r**3 + 4*r*r_0**2*np.cos(2*theta - 2*theta_0) - 12*r**2*r_0*np.cos(theta - theta_0)))/(((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2 + 4*(r*np.cos(theta) - r_0*np.cos(theta_0))**2*(r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2
        EGr = (2*np.cos(2*theta)*(R - epsilon)**2)/r**3
        FGr = -(2*np.sin(2*theta)*(R - epsilon)**2)/r**3
        return AGr, BGr, CGr, DGr, EGr, FGr

    def partial_coefficients_of_G_wrt_theta_Chi(self, r, theta, alpha, epsilon, R, r0, theta0):
        AGtheta = np.cos(theta)/r
        BGtheta  = -np.sin(theta)/r
        CGtheta = -(R**2*(2*r**2*np.cos(2*theta)*np.sin(alpha) - 2*r**2*np.sin(2*theta)*np.cos(alpha)))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2) - (R**2*(2*r**4*np.sin(4*theta) - 4*r**4*np.cos(2*theta)*np.sin(2*theta))*(r**2*np.cos(2*theta)*np.cos(alpha) + r**2*np.sin(2*theta)*np.sin(alpha)))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2)**2
        DGtheta = (R**2*(2*r**2*np.cos(2*theta)*np.cos(alpha) + 2*r**2*np.sin(2*theta)*np.sin(alpha)))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2) - (R**2*(2*r**4*np.sin(4*theta) - 4*r**4*np.cos(2*theta)*np.sin(2*theta))*(r**2*np.cos(2*theta)*np.sin(alpha) - r**2*np.sin(2*theta)*np.cos(alpha)))/((r**4*(np.cos(4*theta) - 1))/2 - r**4*np.cos(2*theta)**2)**2
        EGtheta = ((R - epsilon)**2*(2*np.sin(2*theta)*r**2 + 2*r0*np.sin(theta + theta0)*r))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2) - ((2*(2*np.sin(2*theta)*r**2 + 2*r0*np.sin(theta + theta0)*r)*(np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2) - 2*(2*np.cos(2*theta)*r**2 + 2*r0*np.cos(theta + theta0)*r)*(np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))*(R - epsilon)**2*(np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2)**2
        FGtheta = ((R - epsilon)**2*(2*np.cos(2*theta)*r**2 + 2*r0*np.cos(theta + theta0)*r))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2) + ((2*(2*np.sin(2*theta)*r**2 + 2*r0*np.sin(theta + theta0)*r)*(np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2) - 2*(2*np.cos(2*theta)*r**2 + 2*r0*np.cos(theta + theta0)*r)*(np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))*(R - epsilon)**2*(np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2))/((np.cos(2*theta)*r**2 + 2*np.cos(theta + theta0)*r*r0 + np.cos(2*theta0)*r0**2)**2 + (np.sin(2*theta)*r**2 + 2*np.sin(theta + theta0)*r*r0 + np.sin(2*theta0)*r0**2)**2)**2
        return AGtheta, BGtheta, CGtheta, DGtheta, EGtheta, FGtheta
    
    def partial_coefficients_of_G_wrt_theta_z(self, r, theta, alpha, epsilon, R, r_0, theta_0):
        """"""
        AGtheta = (r*(r**2*np.cos(theta) + r_0**2*np.cos(theta - 2*theta_0) - 2*r*r_0*np.cos(theta_0)))/(r**2 + r_0**2 - 2*r*r_0*np.cos(theta - theta_0))**2
        BGtheta = (r*(r_0**2*np.sin(theta - 2*theta_0) - r**2*np.sin(theta) + 2*r*r_0*np.sin(theta_0)))/(r**2 + r_0**2 - 2*r*r_0*np.cos(theta - theta_0))**2
        CGtheta = (2*R**2*r*(r_0*np.sin(theta - alpha + theta_0) + r*np.sin(alpha - 2*theta)))/(((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2 + 4*(r*np.cos(theta) - r_0*np.cos(theta_0))**2*(r*np.sin(theta) - r_0*np.sin(theta_0))**2) - (4*R**2*r*r_0*(np.cos(alpha)*((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2) + 2*np.sin(alpha)*(r*np.cos(theta) - r_0*np.cos(theta_0))*(r*np.sin(theta) - r_0*np.sin(theta_0)))*(r**2*np.sin(theta - theta_0) + r_0**2*np.sin(theta - theta_0) - r*r_0*np.sin(2*theta - 2*theta_0)))/(((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2 + 4*(r*np.cos(theta) - r_0*np.cos(theta_0))**2*(r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2
        DGtheta = (2*R**2*r*(r_0*np.cos(theta - alpha + theta_0) - r*np.cos(alpha - 2*theta)))/(((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2 + 4*(r*np.cos(theta) - r_0*np.cos(theta_0))**2*(r*np.sin(theta) - r_0*np.sin(theta_0))**2) - (4*R**2*r*r_0*(np.sin(alpha)*((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2) - 2*np.cos(alpha)*(r*np.cos(theta) - r_0*np.cos(theta_0))*(r*np.sin(theta) - r_0*np.sin(theta_0)))*(r**2*np.sin(theta - theta_0) + r_0**2*np.sin(theta - theta_0) - r*r_0*np.sin(2*theta - 2*theta_0)))/(((r*np.cos(theta) - r_0*np.cos(theta_0))**2 - (r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2 + 4*(r*np.cos(theta) - r_0*np.cos(theta_0))**2*(r*np.sin(theta) - r_0*np.sin(theta_0))**2)**2
        EGtheta = (2*np.sin(2*theta)*(R - epsilon)**2)/r**2
        FGtheta = (2*np.cos(2*theta)*(R - epsilon)**2)/r**2
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
        # xi, eta = point_xy_in_zeta_plane[0], point_xy_in_zeta_plane[1]
        r, theta = hlp.xy_to_r_theta(xi, eta)
        xio, etao = self.zeta_center.real, self.zeta_center.imag
        r0, theta0 = hlp.xy_to_r_theta(xio, etao)
        V_inf, R, alpha, epsilon = self.freestream_velocity, self.cylinder_radius, self.angle_of_attack, self.epsilon
        # omega_chi_unsplit = V_inf*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf*(xi+1j*eta)) - np.exp(1j*alpha)*R**2/(xi+1j*eta)**2) / (1 - (R-epsilon)**2/(xi+xio+1j*(eta+etao))**2)
        G1, G2, G3, G4, G5, G6 = self.calc_Chi_G_values(r, theta, alpha, epsilon, R, r0, theta0)
        # velocity_r = np.cos(theta)*Gamma/(2*np.pi)*(G1*G5+G2*G6)/(G5**2 + G6**2)+np.cos(theta)*V_inf*(G5*np.cos(alpha)-G6*np.sin(alpha)-G3*G5-G4*G6)/(G5**2 + G6**2) - np.sin(theta)*Gamma/(2*np.pi)*(G2*G5-G1*G6)/(G5**2 + G6**2) + np.sin(theta)*V_inf*(G6*np.cos(alpha)+G5*np.sin(alpha)+G4*G5-G3*G6)/(G5**2 + G6**2)
        # velocity_theta = np.cos(theta)*V_inf*(G6*np.cos(alpha)+G5*np.sin(alpha)+G4*G5-G3*G6)/(G5**2 + G6**2) - np.cos(theta)*Gamma/(2*np.pi)*(G2*G5-G1*G6)/(G5**2 + G6**2) - np.sin(theta)*Gamma/(2*np.pi)*(G1*G5+G2*G6)/(G5**2 + G6**2) - np.sin(theta)*V_inf*(G5*np.cos(alpha)-G6*np.sin(alpha)-G3*G5-G4*G6)/(G5**2 + G6**2)
        A_omega_rG, B_omega_rG, C_omega_rG, D_omega_rG, E_omega_rG, F_omega_rG, G_omega_rG, H_omega_rG = self.partial_coefficients_of_omega_r_wrt_G_Chi(V_inf, theta, alpha, G1, G2, G3, G4, G5, G6)
        A_omega_thetaG, B_omega_thetaG, C_omega_thetaG, D_omega_thetaG, E_omega_thetaG, F_omega_thetaG, G_omega_thetaG, H_omega_thetaG = self.partial_coefficients_of_omega_theta_wrt_G_Chi(V_inf, theta, alpha, G1, G2, G3, G4, G5, G6)
        AGr, BGr, CGr, DGr, EGr, FGr = self.partial_coefficients_of_G_wrt_r_Chi(r, theta, alpha, epsilon, R, r0, theta0)
        AGtheta, BGtheta, CGtheta, DGtheta, EGtheta, FGtheta = self.partial_coefficients_of_G_wrt_theta_Chi(r, theta, alpha, epsilon, R, r0, theta0)
        # calculate the A1 through A12 values
        A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12 = self.calc_A1_through_A12(V_inf, theta, alpha, G1, G2, G3, G4, G5, G6, A_omega_rG, B_omega_rG, C_omega_rG, D_omega_rG, E_omega_rG, F_omega_rG, G_omega_rG, H_omega_rG, A_omega_thetaG, B_omega_thetaG, C_omega_thetaG, D_omega_thetaG, E_omega_thetaG, F_omega_thetaG, G_omega_thetaG, H_omega_thetaG, AGr, BGr, CGr, DGr, EGr, FGr, AGtheta, BGtheta, CGtheta, DGtheta, EGtheta, FGtheta)
        # omega_r, omega_theta = Gamma*A1 + A2, Gamma*A3 + A4
        # partial_omega_r_wrt_r, partial_omega_r_wrt_theta, partial_omega_theta_wrt_theta, partial_omega_theta_wrt_r = self.chi_partial_derivatives(Gamma, A5, A6, A7, A8, A9, A10, A11, A12)
        B1, B2, B3, B4, B5, B6 = self.calc_B1_through_B6(r, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12)
        convective_acceleration = np.array([Gamma**2*B1 + Gamma*B2 + B3, Gamma**2*B4 + Gamma*B5 + B6])
        C1, C2, C3, C4, C5 = self.calc_C1_through_C5(B1, B2, B3, B4, B5, B6)
        return convective_acceleration
    
    def z_convective_acceleration(self, point_xy_in_z_plane, Gamma):
        """Start with a z value that is shifted from z_center"""
        z = point_xy_in_z_plane[0] + 1j*point_xy_in_z_plane[1]
        zeta = self.z_to_zeta(z, self.epsilon)
        r, theta = hlp.xy_to_r_theta(zeta.real, zeta.imag)
        xio, etao = self.zeta_center.real, self.zeta_center.imag
        r0, theta0 = hlp.xy_to_r_theta(xio, etao)
        V_inf, R, alpha, epsilon = self.freestream_velocity, self.cylinder_radius, self.angle_of_attack, self.epsilon
        G1, G2, G3, G4, G5, G6 = self.calc_z_G_values(r, theta, alpha, epsilon, R, r0, theta0)
        A_omega_rG, B_omega_rG, C_omega_rG, D_omega_rG, E_omega_rG, F_omega_rG, G_omega_rG, H_omega_rG = self.partial_coefficients_of_omega_r_wrt_G_z(V_inf, theta, alpha, G1, G2, G3, G4, G5, G6)
        A_omega_thetaG, B_omega_thetaG, C_omega_thetaG, D_omega_thetaG, E_omega_thetaG, F_omega_thetaG, G_omega_thetaG, H_omega_thetaG = self.partial_coefficients_of_omega_theta_wrt_G_z(V_inf, theta, alpha, G1, G2, G3, G4, G5, G6)
        AGr, BGr, CGr, DGr, EGr, FGr = self.partial_coefficients_of_G_wrt_r_z(r, theta, alpha, epsilon, R, r0, theta0)
        AGtheta, BGtheta, CGtheta, DGtheta, EGtheta, FGtheta = self.partial_coefficients_of_G_wrt_theta_z(r, theta, alpha, epsilon, R, r0, theta0)
        # calculate the A1 through A12 values
        A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12 = self.calc_A1_through_A12(V_inf, theta, alpha, G1, G2, G3, G4, G5, G6, A_omega_rG, B_omega_rG, C_omega_rG, D_omega_rG, E_omega_rG, F_omega_rG, G_omega_rG, H_omega_rG, A_omega_thetaG, B_omega_thetaG, C_omega_thetaG, D_omega_thetaG, E_omega_thetaG, F_omega_thetaG, G_omega_thetaG, H_omega_thetaG, AGr, BGr, CGr, DGr, EGr, FGr, AGtheta, BGtheta, CGtheta, DGtheta, EGtheta, FGtheta)
        B1, B2, B3, B4, B5, B6 = self.calc_B1_through_B6(r, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12)
        convective_acceleration = np.array([Gamma**2*B1 + Gamma*B2 + B3, Gamma**2*B4 + Gamma*B5 + B6])
        C1, C2, C3, C4, C5 = self.calc_C1_through_C5(B1, B2, B3, B4, B5, B6)
        return convective_acceleration
    
    def appellian_acceleration_loop(self, Gamma: float, r_values: np.array, theta_values: np.array, dr: float, dtheta: float, is_Chi: bool, acceleration: callable):
        """"""
        Gamma_values = np.array([Gamma])
        # index = 0
        Appellian_value = 0.0
        Appellian_array = np.zeros((len(Gamma_values), 2))
        xi_eta_values = np.zeros((len(r_values)*len(theta_values), 2))
        if is_Chi:
                for j in range(len(r_values)):
                    for k in range(len(theta_values)):
                        Chi_val = hlp.r_theta_to_xy(r_values[j], theta_values[k])[0] + 1j*hlp.r_theta_to_xy(r_values[j], theta_values[k])[1] # the r, theta point in the Chi plane is converted to a complex number
                        # print("Chi_val", Chi_val)
                        area_element = r_values[j]*dr*dtheta
                        # print("area element", area_element)
                        # print("\n")
                        convective_acceleration = acceleration([Chi_val.real, Chi_val.imag], Gamma)
                        Appellian_value += np.dot(convective_acceleration, convective_acceleration)*area_element
                        # xi_eta_values[index] = [Chi_val.real, Chi_val.imag]
                        # index += 1
                Appellian_array[0] = [Gamma, 0.5*Appellian_value]
        else:
                for j in range(len(r_values)):
                    for k in range(len(theta_values)):
                        Chi = hlp.r_theta_to_xy(r_values[j], theta_values[k])[0] + 1j*hlp.r_theta_to_xy(r_values[j], theta_values[k])[1] # the r, theta point in the Chi plane is converted to a complex number
                        zeta = self.Chi_to_zeta(Chi)
                        z = self.zeta_to_z(zeta, self.epsilon)
                        r_z = hlp.xy_to_r_theta(z.real, z.imag)[0]
                        area_element = r_z * dr * dtheta
                        convective_acceleration = acceleration([z.real, z.imag], Gamma)
                        Appellian_value += np.dot(convective_acceleration, convective_acceleration)*area_element
                        # index += 1
                Appellian_array[0] = [Gamma, 0.5*Appellian_value]
                # Appellian_value = 0.0
                # index = 0
        Appellian_array = 0.5*Appellian_value
        return Appellian_array
    
    def conv_accel_from_equations(Gamma, r, R, theta, r_0, theta_0, V_inf, epsilon, alpha, velocity):
        term1 = np.exp(-1j * alpha) + 1j * (Gamma / (2 * np.pi * V_inf)) * (1 / (r * np.exp(1j * theta))) - (R**2 * np.exp(1j * alpha)) / ((r * np.exp(1j * theta))**2)
        # print("term1", term1)
        term2 = (-1j * (Gamma / (2 * np.pi * V_inf)) * (1 / ((r * np.exp(1j * theta))**2))) + (2 * R**2 * np.exp(1j * alpha) / ((r * np.exp(1j * theta))**3))
        # print("term2", term2)
        term3 = 2 * (R - epsilon)**2 / ((r * np.exp(1j * theta)) + (r_0 * np.exp(1j * theta_0)))**3
        # print("term3", term3)
        term4 = (1 - ((R - epsilon)**2 / ((r * np.exp(1j * theta)) + (r_0 * np.exp(1j * theta_0)))**2))
        # print("term4", term4)
        
        # Compute grad Cp using the equation in the image
        conv_accel = velocity**2 * term1 / term4 * (term2*term4 - (term1 * term3)) / term4**3
        # conv_accel = velocity**2 * term1 / term4 * (term2*term4 - (term1 * term3)) / term4**3
        
        # Convert to real-valued pressure gradient vector
        conv_accel_complex = np.array([conv_accel.real, -conv_accel.imag])
        # conv_accel_complex = np.linalg.norm(conv_accel_complex)
        return conv_accel_complex
    
    def compare_correct_and_incorrect_z_selection(self):
        """This function plots the zeta surface points using the correct and incorrect z selection methods"""
        # first plot the correct zeta surface points
        upper,lower,_= self.get_full_geometry()
        upper_complex = upper[:,0] + 1j*upper[:,1] # the type of upper_complex is an array of complex numbers
        lower_complex = lower[:,0] + 1j*lower[:,1] # the type of lower_complex is an array of complex numbers

        upper_complex_zeta = np.zeros(len(upper_complex), dtype=complex)
        lower_complex_zeta = np.zeros(len(lower_complex), dtype=complex)
        upper_complex_zeta_incorrect = np.zeros(len(upper_complex), dtype=complex)
        lower_complex_zeta_incorrect = np.zeros(len(lower_complex), dtype=complex)
        for i in range(len(upper_complex)):
            upper_complex_zeta[i] = self.z_to_zeta(upper_complex[i], self.epsilon)
        for j in range(len(lower_complex)):
            lower_complex_zeta[j] = self.z_to_zeta(lower_complex[j], self.epsilon)
        for k in range(len(upper_complex)):
            upper_complex_zeta_incorrect[k] = self.z_to_zeta_incorrect(upper_complex[k], self.epsilon)
        for l in range(len(lower_complex)):
            lower_complex_zeta_incorrect[l] = self.z_to_zeta_incorrect(lower_complex[l], self.epsilon)

        # now create arrays that hold the real and imaginary parts as x and y components for the complex zeta coordinates for the correct and incorrect methods
        upper_zeta = np.zeros((len(upper_complex_zeta), 2))
        lower_zeta = np.zeros((len(lower_complex_zeta), 2))
        upper_zeta_incorrect = np.zeros((len(upper_complex_zeta_incorrect), 2))
        lower_zeta_incorrect = np.zeros((len(lower_complex_zeta_incorrect), 2))
        for i in range(len(upper_complex_zeta)):
            upper_zeta[i] = [upper_complex_zeta[i].real, upper_complex_zeta[i].imag]
        for j in range(len(lower_complex_zeta)):
            lower_zeta[j] = [lower_complex_zeta[j].real, lower_complex_zeta[j].imag]
        for k in range(len(upper_complex_zeta_incorrect)):
            upper_zeta_incorrect[k] = [upper_complex_zeta_incorrect[k].real, upper_complex_zeta_incorrect[k].imag]
        for l in range(len(lower_complex_zeta_incorrect)):
            lower_zeta_incorrect[l] = [lower_complex_zeta_incorrect[l].real, lower_complex_zeta_incorrect[l].imag]      
        # now plot z, zeta, and zeta incorrect on the same plot
        plt.figure()
        plt.plot(upper[:,0], upper[:,1], label="$z$", color="black")
        plt.plot(lower[:,0], lower[:,1], color="black")
        plt.plot(upper_zeta[:,0], upper_zeta[:,1],linestyle="--", label="$\\zeta$", color="black")
        plt.plot(lower_zeta[:,0], lower_zeta[:,1],linestyle="--", color="black")
        plt.plot(upper_zeta_incorrect[:,0], upper_zeta_incorrect[:,1], label="$\\zeta$ incorrect", linestyle="dotted", color="black")
        plt.plot(lower_zeta_incorrect[:,0], lower_zeta_incorrect[:,1], linestyle="dotted", color="black")
        # plt.axhline(0, color='gray')
        # plt.axvline(0, color='gray')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("$\\xi$/$R$")
        plt.ylabel("$\\eta$/$R$")
        plt.gca().xaxis.labelpad = 0.001
        plt.gca().yaxis.labelpad = -10
        plt.xlim(self.plot_x_lower_lim, self.plot_x_upper_lim)
        plt.ylim(self.plot_x_lower_lim, self.plot_x_upper_lim)
        x_tick_length = int(self.plot_x_upper_lim - self.plot_x_lower_lim)
        y_tick_length = int(self.plot_x_upper_lim - self.plot_x_lower_lim)
        x_ticks = np.linspace(self.plot_x_lower_lim, self.plot_x_upper_lim, x_tick_length + 1)[1:] # ticks everywhere except for the first element
        y_ticks = np.linspace(self.plot_x_lower_lim, self.plot_x_upper_lim, y_tick_length + 1)[1:] # ticks everywhere except for the first element
        plt.xticks(x_ticks)
        plt.yticks(y_ticks) 
        plt.text(-0.07, -0.01, str(int(self.plot_x_lower_lim)), transform=plt.gca().transAxes, fontsize=17, verticalalignment='top')
        if self.plot_text:
            plt.text(0.05, 0.92, "$\\epsilon = $" + str(self.epsilon) + " $R$", transform=plt.gca().transAxes, fontsize=17, verticalalignment='top')
            plt.text(0.60, 0.96, "$\\xi_0 = $" + str(self.zeta_center.real) + " $R$", transform=plt.gca().transAxes, fontsize=17, verticalalignment='top')
            plt.text(0.60, 0.88, "$\\eta_0 = $" + str(self.zeta_center.imag) + " $R$", transform=plt.gca().transAxes, fontsize=17, verticalalignment='top')
        # plt.legend(fontsize=12.0, loc="lower right")
        plt.show()

    def calc_Chi_G_values(self, r, theta, alpha, epsilon, R, r0, theta0):
        """takes in r, theta, alpha, epsilon, R, r0, theta0 and calculates the G values"""
        G1 = np.sin(theta)/r
        G2 = np.cos(theta)/r
        G3 = R**2*np.cos(alpha-2*theta)/r**2
        G4 = R**2*np.sin(alpha-2*theta)/r**2
        G5 = 1 - ((R-epsilon)**2*(r**2*np.cos(2*theta)+r0**2*np.cos(2*theta0)+2*r*r0*np.cos(theta+theta0)))/((r**2*np.cos(2*theta)+r0**2*np.cos(2*theta0)+2*r*r0*np.cos(theta+theta0))**2+(r**2*np.sin(2*theta)+r0**2*np.sin(2*theta0)+2*r*r0*np.sin(theta+theta0))**2)
        G6 = ((R-epsilon)**2*(r**2*np.sin(2*theta)+r0**2*np.sin(2*theta0)+2*r*r0*np.sin(theta+theta0)))/((r**2*np.cos(2*theta)+r0**2*np.cos(2*theta0)+2*r*r0*np.cos(theta+theta0))**2+(r**2*np.sin(2*theta)+r0**2*np.sin(2*theta0)+2*r*r0*np.sin(theta+theta0))**2)
        return G1, G2, G3, G4, G5, G6
    
    def z_analytic_acceleration(self, point_xi_eta_in_z_plane, Gamma):
        """This function also calculates the acceleration at a given point in the flow field in the z plane according"""
        z = point_xi_eta_in_z_plane[0] + 1j*point_xi_eta_in_z_plane[1]
        zeta = self.z_to_zeta(z, self.epsilon)
        V_inf, epsilon, R, alpha, zeta0 = self.freestream_velocity, self.epsilon, self.cylinder_radius, self.angle_of_attack, self.zeta_center
        plus = (z + np.sqrt(z**2 - 4*(R-epsilon)**2))/2
        minus = (z - np.sqrt(z**2 - 4*(R-epsilon)**2))/2
        if zeta == plus:
            dzeta_dz_squared = (z+1)/(4*(2*epsilon-2*R+z)*(2*R-2*epsilon+z))
            d2zeta_dz2 = -(2*(R-epsilon)**2)/(z**2-4*(R-epsilon)**2)**(3/2)
        elif zeta == minus:
            dzeta_dz_squared = (1-z)/(4*(2*epsilon-2*R+z)*(2*R-2*epsilon+z))
            d2zeta_dz2 = (2*(R-epsilon)**2)/(z**2-4*(R-epsilon)**2)**(3/2)
        acceleration_1 = V_inf*(-1j*Gamma/(2*np.pi*V_inf*(zeta-zeta0)**2) + 2*np.exp(1j*alpha)*R**2/((zeta-zeta0)**3))*dzeta_dz_squared
        acceleration_2 = V_inf*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf*(zeta-zeta0)) - np.exp(1j*alpha)*R**2/(zeta-zeta0)**2)*d2zeta_dz2
        acceleration = acceleration_1 + acceleration_2
        # acceleration *= 2*V_inf*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf*(zeta-zeta0)) - np.exp(1j*alpha)*R**2/(zeta-zeta0)**2) / (1 - (R-epsilon)**2/(zeta)**2)
        acceleration_complex = np.array([acceleration.real, -acceleration.imag])
        theta = np.arctan2(point_xi_eta_in_z_plane[1], point_xi_eta_in_z_plane[0])
        polar_acceleration = hlp.polar_vector(theta, acceleration_complex)
        return polar_acceleration
    
    def z_to_zeta_incorrect(self, z: complex, epsilon: float):
        """This function returns the opposite root that z_to_zeta returns"""
        zeta_correct = self.z_to_zeta(z, epsilon)
        z_1 = z**2 - 4*(self.cylinder_radius - epsilon)**2
        zeta_option_1 = (z + np.sqrt(z_1))/2
        zeta_option_2 = (z - np.sqrt(z_1))/2
        # check if the zeta coordinate is the same as either of the options
        if np.isclose(zeta_correct, zeta_option_1):
            zeta = zeta_option_2
        elif np.isclose(zeta_correct, zeta_option_2):
            zeta = zeta_option_1
        else:
            raise ValueError("The zeta coordinate does not match either of the options")
        return zeta
    
    def r_theta_Chi_to_r_theta_zeta(self, r_Chi, theta_Chi):
        """This function takes in r and theta in the Chi plane and returns r and theta in the zeta plane"""
        xi_Chi, eta_Chi = hlp.r_theta_to_xy(r_Chi, theta_Chi)
        xi_zeta, eta_zeta = xi_Chi + self.zeta_center.real, eta_Chi + self.zeta_center.imag
        r_zeta, theta_zeta = hlp.xy_to_r_theta(xi_zeta, eta_zeta)
        return r_zeta, theta_zeta
    
    def r_theta_zeta_to_r_theta_z(self, r_zeta, theta_zeta):
        """This function takes in r and theta in the zeta plane and returns r and theta in the z plane"""
        xi_zeta, eta_zeta = hlp.r_theta_to_xy(r_zeta, theta_zeta)
        point_xi_eta_in_zeta = xi_zeta + 1j*eta_zeta
        z = self.Joukowski_zeta_to_z(point_xi_eta_in_zeta, self.epsilon)
        r_z, theta_z = hlp.xy_to_r_theta(z.real, z.imag)
        return r_z, theta_z
    
    def velocity_chi(self, point_xy_in_Chi_plane, Gamma):
        """Start with a Chi value that is shifted from zeta_center"""
        xi, eta = point_xy_in_Chi_plane[0], point_xy_in_Chi_plane[1]
        zeta0 = self.zeta_center
        Chi = xi + 1j*eta
        zeta = Chi + zeta0
        r, theta = hlp.xy_to_r_theta(xi, eta)
        zeta_center = self.zeta_center
        # xio, etao = zeta_center.real, zeta_center.imag
        # r0, theta0 = hlp.xy_to_r_theta(xio, etao)
        V_inf, R, alpha, epsilon = self.freestream_velocity, self.cylinder_radius, self.angle_of_attack, self.epsilon
        # G1, G2, G3, G4, G5, G6 = self.calc_Chi_G_values(r, theta, alpha, epsilon, R, r0, theta0)
        # V_real = (Gamma/(2*np.pi))*((G1*G5+G2*G6)/(G5**2 + G6**2)) + V_inf*((G5*np.cos(alpha)-G6*np.sin(alpha)-G3*G5-G4*G6)/(G5**2 + G6**2))
        # V_imag = (-1*Gamma/(2*np.pi))*((G2*G5-G1*G6)/(G5**2 + G6**2)) + V_inf*((G6*np.cos(alpha)+G5*np.sin(alpha)+G4*G5-G3*G6)/(G5**2 + G6**2))
        # velocity_complex = np.array([V_real, V_imag])
        velocity = V_inf*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf*(Chi)) - np.exp(1j*alpha)*R**2/(Chi)**2) / (1 - (R-epsilon)**2/(Chi+zeta0)**2) # eq 107
        velocity_complex = np.array([velocity.real, -velocity.imag])
        return velocity_complex
    
    def d2phi_dzeta2(self, point_r_theta_in_Chi, Gamma):
        V_inf, epsilon, R, alpha, zeta_0 = self.freestream_velocity, self.epsilon, self.cylinder_radius, self.angle_of_attack, self.zeta_center
        xi_0, eta_0 = zeta_0.real, zeta_0.imag
        r_0, theta_0 = hlp.xy_to_r_theta(xi_0, eta_0)
        r, theta = point_r_theta_in_Chi[0], point_r_theta_in_Chi[1]
        d2phi_dzeta2 = V_inf*(-1j*Gamma/(2*np.pi*V_inf*(r*np.exp(1j*theta))**2) + 2*np.exp(1j*alpha)*R**2/((r*np.exp(1j*theta))**3))
        d2phi_dzeta2_comp_conj = np.conj(d2phi_dzeta2)
        d2phi_dzeta2_squared = d2phi_dzeta2 * d2phi_dzeta2_comp_conj
        # print("d2phi_dzeta2\n", d2phi_dzeta2_squared)
        return d2phi_dzeta2_squared
    
    def grid_plots(self):
        # Reset the plot settings
        cyl.reset_plot_stuff()

        # Parameters for the polar grid
        R = cyl.cylinder_radius  # Cylinder radius
        zeta_center = cyl.zeta_center  # Center of the cylinder in the zeta plane
        theta_increment = np.pi / 32  # Spacing between radial lines
        r_increment = 0.1 * R  # Increment for radial spacing
        max_radius = 6.0 * R  # Maximum radius for the grid

        # Generate theta and r values
        theta_vals = np.arange(0, 2 * np.pi, theta_increment)  # Angles for radial lines
        r_vals = np.arange(R, max_radius + r_increment, r_increment)  # Radii for circular lines

        # Arrays to store the lines
        radial_lines = []
        circular_lines = []

        # Create radial lines in the zeta plane
        for theta in theta_vals:
            line = np.array([[zeta_center.real + r * np.cos(theta), zeta_center.imag + r * np.sin(theta)] for r in np.linspace(r_vals[0], r_vals[-1], 1000)])  # Offset by zeta_center
            radial_lines.append(line)  # Save the line
            plt.plot(line[:, 0], line[:, 1], color='black')  # Plot the radial line

        # Create circular lines in the zeta plane
        for r in r_vals:
            line = np.array([[zeta_center.real + r * np.cos(theta), zeta_center.imag + r * np.sin(theta)] for theta in np.linspace(0, 2 * np.pi, 1000)])  # Offset by zeta_center
            circular_lines.append(line)  # Save the line
            plt.plot(line[:, 0], line[:, 1], color='black')  # Plot the circular line

        # Plot the geometry of the circular cylinder in the zeta plane
        plt.plot(cyl.upper_zeta_coords[:, 0], cyl.upper_zeta_coords[:, 1], color='black')
        plt.plot(cyl.lower_zeta_coords[:, 0], cyl.lower_zeta_coords[:, 1], color='black')

        # plt.show()

        # Transform the grid to the z-plane
        plt.figure()
        cyl.reset_plot_stuff()

        # Arrays to store the transformed lines
        transformed_radial_lines = []
        transformed_circular_lines = []

        # Transform radial lines to the z-plane
        for line in radial_lines:
            transformed_line = []
            for point in line:
                zeta = point[0] + 1j * point[1]  # Convert to complex zeta
                z = cyl.zeta_to_z(zeta, cyl.epsilon)  # Transform to z-plane
                transformed_line.append([z.real, z.imag])  # Store the transformed point
            transformed_line = np.array(transformed_line)  # Convert to numpy array
            transformed_radial_lines.append(transformed_line)  # Save the transformed line
            plt.plot(transformed_line[:, 0], transformed_line[:, 1], color='black')  # Plot the transformed radial line

        # Transform circular lines to the z-plane
        for line in circular_lines:
            transformed_line = []
            for point in line:
                zeta = point[0] + 1j * point[1]  # Convert to complex zeta
                z = cyl.zeta_to_z(zeta, cyl.epsilon)  # Transform to z-plane
                transformed_line.append([z.real, z.imag])  # Store the transformed point
            transformed_line = np.array(transformed_line)  # Convert to numpy array
            transformed_circular_lines.append(transformed_line)  # Save the transformed line
            plt.plot(transformed_line[:, 0], transformed_line[:, 1], color='black')  # Plot the transformed circular line

        # Plot the geometry of the shape in the z-plane
        plt.plot(cyl.upper_coords[:, 0], cyl.upper_coords[:, 1], color='black')
        plt.plot(cyl.lower_coords[:, 0], cyl.lower_coords[:, 1], color='black')

        plt.show()

    def reset_plot_stuff(self):
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("$\\xi$/$R$")
        plt.ylabel("$\\eta$/$R$")
        plt.gca().xaxis.labelpad = 0.001
        plt.gca().yaxis.labelpad = -10
        plt.xlim(self.plot_x_lower_lim, self.plot_x_upper_lim)
        plt.ylim(self.plot_x_lower_lim, self.plot_x_upper_lim)
        x_tick_length = int(self.plot_x_upper_lim - self.plot_x_lower_lim)
        y_tick_length = int(self.plot_x_upper_lim - self.plot_x_lower_lim)
        x_ticks = np.linspace(self.plot_x_lower_lim, self.plot_x_upper_lim, x_tick_length + 1)[1:] # ticks everywhere except for the first element
        y_ticks = np.linspace(self.plot_x_lower_lim, self.plot_x_upper_lim, y_tick_length + 1)[1:] # ticks everywhere except for the first element
        plt.xticks(x_ticks)
        plt.yticks(y_ticks) 
        plt.text(-0.07, -0.01, str(int(self.plot_x_lower_lim)), transform=plt.gca().transAxes, fontsize=17, verticalalignment='top')

    def analytic_conv_accel_square_comp_conj_wiki(self, point_r_theta_chi, Gamma):
        """This function calculates the pressure gradient at a given point in the flow field in the z plane"""
        point_xi_eta_chi = hlp.r_theta_to_xy(point_r_theta_chi[0], point_r_theta_chi[1])
        Chi = point_xi_eta_chi[0] + 1j*point_xi_eta_chi[1]
        zeta = self.Chi_to_zeta(Chi)
        z = self.zeta_to_z(zeta, self.epsilon)
        V_inf, epsilon, R, alpha, zeta0 = self.freestream_velocity, self.epsilon, self.cylinder_radius, self.angle_of_attack, self.zeta_center
        plus = (z + np.sqrt(z**2 - 4*(R-epsilon)**2))/2
        minus = (z - np.sqrt(z**2 - 4*(R-epsilon)**2))/2
        omega_z = V_inf*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf*(zeta-zeta0)) - np.exp(1j*alpha)*R**2/(zeta-zeta0)**2) / (1 - (R-epsilon)**2/(zeta)**2)
        dPhi_dzeta = V_inf*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf*(zeta-zeta0)) - np.exp(1j*alpha)*R**2/(zeta-zeta0)**2)
        d2Phi_dzeta2 = V_inf*(-1j*Gamma/(2*np.pi*V_inf*(zeta-zeta0)**2) + 2*np.exp(1j*alpha)*R**2/((zeta-zeta0)**3))
        if np.isclose(zeta, plus, atol=1e-10):
            dzeta_dz_squared = ((1/2)+(z)/(2*np.sqrt(z**2-4*(R-epsilon)**2)))**2
            d2zeta_dz2 = -(2*(R-epsilon)**2)/(z**2-4*(R-epsilon)**2)**(3/2)
        elif np.isclose(zeta, minus, atol=1e-10):
            dzeta_dz_squared = ((1/2)-(z)/(2*np.sqrt(z**2-4*(R-epsilon)**2)))**2
            d2zeta_dz2 = (2*(R-epsilon)**2)/(z**2-4*(R-epsilon)**2)**(3/2)
        else:
            print("zeta: ", zeta)
            print("plus: ", plus)
            print("minus: ", minus)
            raise ValueError("zeta is not equal to plus or minus")
        convective_acceleration = omega_z*(d2Phi_dzeta2*dzeta_dz_squared + dPhi_dzeta*d2zeta_dz2)
        convective_acceleration_comp_conj = np.conj(convective_acceleration)
        convective_acceleration_squared = convective_acceleration * convective_acceleration_comp_conj
        return np.real(convective_acceleration_squared)
    
    def taha_analytic_conv_accel_square_comp_conj(self, point_r_theta_in_Chi, Gamma): # need to adjust so that it uses the Taha transformation
        """This function calculates the pressure gradient at a given point in the flow field in the z plane"""
        V_inf, epsilon, R, alpha, zeta_0 = self.freestream_velocity, self.epsilon, self.cylinder_radius, self.angle_of_attack, self.zeta_center
        xi_0, eta_0 = zeta_0.real, zeta_0.imag
        r_0, theta_0 = hlp.xy_to_r_theta(xi_0, eta_0)
        r, theta = point_r_theta_in_Chi[0], point_r_theta_in_Chi[1]
        xi_chi, eta_chi = hlp.r_theta_to_xy(r, theta)
        chi = xi_chi + 1j*eta_chi
        zeta = self.Chi_to_zeta(chi)
        G_of_zeta = 1/(1-(R-epsilon)**2/(r*np.exp(1j*theta)+r_0*np.exp(1j*theta_0))**2)
        # G_of_zeta = 1/self.zeta_to_z(zeta)
        G_of_zeta_comp_conj = np.conj(G_of_zeta)
        G_of_zeta_squared = G_of_zeta * G_of_zeta_comp_conj
        # derivative of G with respect to zeta
        dG_dzeta = 4
        dG_dzeta_comp_conj = np.conj(dG_dzeta)
        omega_of_zeta = V_inf*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf*(r*np.exp(1j*theta))) - np.exp(1j*alpha)*R**2/(r*np.exp(1j*theta))**2)
        omega_of_zeta_comp_conj = np.conj(omega_of_zeta)
        omega_of_zeta_squared = omega_of_zeta * omega_of_zeta_comp_conj 
        d_omega_of_zeta_dzeta = V_inf*(-1j*Gamma/(2*np.pi*V_inf*(r*np.exp(1j*theta))**2) + 2*np.exp(1j*alpha)*R**2/((r*np.exp(1j*theta))**3))
        d_omega_of_zeta_dzeta_comp_conj = np.conj(d_omega_of_zeta_dzeta)
        a_of_zeta = omega_of_zeta*d_omega_of_zeta_dzeta_comp_conj
        integrand = G_of_zeta_squared * (G_of_zeta_comp_conj * a_of_zeta + omega_of_zeta_squared*dG_dzeta_comp_conj)
        integrand_comp_conj = np.conj(integrand)
        conv_accel_squared = integrand * integrand_comp_conj
        # self.tau = 0.1
        # self.epsilon = self.calc_taha_epsilon_from_tau()
        # self.C = self.calc_taha_C_from_epsilon()
        return np.real(conv_accel_squared)
    
    def analytic_conv_accel_for_line_int_comp_conj(self, point_r_theta_in_Chi, Gamma, step = 1e-14):
        """"""
        V_inf, epsilon, R, alpha, zeta_0 = self.freestream_velocity, self.epsilon, self.cylinder_radius, self.angle_of_attack, self.zeta_center
        r, theta = point_r_theta_in_Chi[0], point_r_theta_in_Chi[1]
        xi_Chi, eta_Chi = hlp.r_theta_to_xy(r, theta)
        Chi = xi_Chi + 1j*eta_Chi
        r_0, theta_0 = hlp.xy_to_r_theta(zeta_0.real, zeta_0.imag)
        A = Gamma/(2*np.pi*V_inf)
        B = R**2
        C = (R-epsilon)**2
        conv_accel_squared = ((np.exp(-alpha*1j)+(A*np.exp(-theta*1j)*1j)/r - (B*np.exp(alpha*1j)*np.exp(-theta*2*1j))/r**2)**2*(- np.exp(alpha*1j)+(A*np.exp(theta*1j)*1j)/r \
                            + (B*np.exp(-alpha*1j)*np.exp(theta*2*1j))/r**2)**2)/((C/(r*np.exp(-theta*1j) + r_0*np.exp(-theta_0*1j))**2 - 1)**2*(C/(r*np.exp(theta*1j) + \
                            r_0*np.exp(theta_0*1j))**2 - 1)**2) - (2*r*((A*np.exp(-theta*1j)*1j)/r**2 - (2*B*np.exp(alpha*1j)*np.exp(-theta*2*1j))/r**3)*(np.exp(-alpha*1j) + \
                            (A*np.exp(-theta*1j)*1j)/r - (B*np.exp(alpha*1j)*np.exp(-theta*2*1j))/r**2)*(- np.exp(alpha*1j) + (A*np.exp(theta*1j)*1j)/r + (B*np.exp(-alpha*1j)* \
                            np.exp(theta*2*1j))/r**2)**2)/((C/(r*np.exp(-theta*1j) + r_0*np.exp(-theta_0*1j))**2 - 1)**2*(C/(r*np.exp(theta*1j) + r_0*np.exp(theta_0*1j))**2 - 1)**2) -\
                            (2*r*((A*np.exp(theta*1j)*1j)/r**2 + (2*B*np.exp(-alpha*1j)*np.exp(theta*2*1j))/r**3)*(np.exp(-alpha*1j) + (A*np.exp(-theta*1j)*1j)/r - (B*np.exp(alpha*1j)*\
                            np.exp(-theta*2*1j))/r**2)**2*(- np.exp(alpha*1j) + (A*np.exp(theta*1j)*1j)/r + (B*np.exp(-alpha*1j)*np.exp(theta*2*1j))/r**2))/((C/(r*np.exp(-theta*1j) + \
                            r_0*np.exp(-theta_0*1j))**2 - 1)**2*(C/(r*np.exp(theta*1j) + r_0*np.exp(theta_0*1j))**2 - 1)**2) + (4*C*r*np.exp(-theta*1j)*(np.exp(-alpha*1j) + (A*np.exp(-theta*1j)*1j)/r \
                            - (B*np.exp(alpha*1j)*np.exp(-theta*2*1j))/r**2)**2*(- np.exp(alpha*1j) + (A*np.exp(theta*1j)*1j)/r + (B*np.exp(-alpha*1j)*np.exp(theta*2*1j))/r**2)**2)/((r*np.exp(-theta*1j) \
                            + r_0*np.exp(-theta_0*1j))**3*(C/(r*np.exp(-theta*1j) + r_0*np.exp(-theta_0*1j))**2 - 1)**3*(C/(r*np.exp(theta*1j) + r_0*np.exp(theta_0*1j))**2 - 1)**2) + \
                            (4*C*r*np.exp(theta*1j)*(np.exp(-alpha*1j) + (A*np.exp(-theta*1j)*1j)/r - (B*np.exp(alpha*1j)*np.exp(-theta*2*1j))/r**2)**2*(- np.exp(alpha*1j) + (A*np.exp(theta*1j)*1j)/r \
                            + (B*np.exp(-alpha*1j)*np.exp(theta*2*1j))/r**2)**2)/((r*np.exp(theta*1j) + r_0*np.exp(theta_0*1j))**3*(C/(r*np.exp(-theta*1j) + r_0*np.exp(-theta_0*1j))**2 - 1)**2*\
                            (C/(r*np.exp(theta*1j) + r_0*np.exp(theta_0*1j))**2 - 1)**3)
        # print("conv_accel_squared", conv_accel_squared)
        return np.real(conv_accel_squared)
    

    def appellian_acceleration_loop(self, Gamma: float, r_values: np.array, theta_values: np.array, dr: float, dtheta: float, D, acceleration: callable):
        """"""
        Appellian_value = 0.0
        if self.appellian_is_area_integral: # using complex conjugate
            # if abs(D) >= 0.001 and r_values[0] != r_values[-1]:
            #     r_min, r_max = r_values[0], r_values[-1]
            #     theta_min, theta_max = theta_values[0], theta_values[-1]

            #     # Define the integrand for dblquad
            #     def integrand(theta, r):
            #         accel = acceleration([r, theta], Gamma)
            #         return accel * r  # Include the r term for the area element

            #     # Perform the double integration
            #     integral, _ = dblquad(integrand, r_min, r_max, lambda r: theta_min, lambda r: theta_max)

            #     # Scale the result
            #     Appellian_value = integral * 0.5
            #     return Appellian_value
            # elif abs(D) >= 0.001 and r_values[0] == r_values[-1]:
            #     def integrand(theta, r, Gamma):
            #         """Define the integrand for the line integral."""
            #         accel = acceleration([r, theta], Gamma)
            #         return accel # Include the r term for the area element

            #     # Use the single value of r from r_values
            #     r = r_values[0]#+0.005*self.cylinder_radius  # Assuming r_values contains only one value
            #     theta_start = theta_values[0]
            #     theta_end = theta_values[-1]

            #     # Perform the integration over theta
            #     integral, _ = quad(integrand, theta_start, theta_end, args=(r, Gamma))

            #     # The result of the line integral
            #     Appellian_value = integral*(0.5)
            #     return Appellian_value
            # else:
            for j in range(len(r_values)):
                for k in range(len(theta_values)):
                    area_element = r_values[j]*dr*dtheta
                    accel = acceleration([r_values[j], theta_values[k]], Gamma)
                    Appellian_value += accel*area_element
            return Appellian_value*(0.5)
                # Define the integration bounds
        elif self.appellian_is_line_integral:
        #     if abs(D) > 0.001:
        #         def integrand(theta, r, Gamma):
        #             """Define the integrand for the line integral."""
        #             accel = acceleration([r, theta], Gamma)
        #             return accel # Include the r term for the area element

        #         # Use the single value of r from r_values
        #         r = r_values[0]#+0.005*self.cylinder_radius  # Assuming r_values contains only one value
        #         theta_start = theta_values[0]
        #         theta_end = theta_values[-1]

        #         # Perform the integration over theta
        #         integral, _ = quad(integrand, theta_start, theta_end, args=(r, Gamma))

        #         # The result of the line integral
        #         Appellian_value = integral*(-0.03125) # 1/32
        #         return Appellian_value
        #     else:
            for k in range(len(theta_values)):
                area_element = r_values[0]*dtheta
                accel = acceleration([r_values[0], theta_values[k]], Gamma)
                Appellian_value += accel*area_element
            return Appellian_value*(-0.03125) # 1/32
        else:
            for j in range(len(r_values)):
                for k in range(len(theta_values)):
                    Chi = hlp.r_theta_to_xy(r_values[j], theta_values[k])[0] + 1j*hlp.r_theta_to_xy(r_values[j], theta_values[k])[1] # the r, theta point in the Chi plane is converted to a complex number
                    zeta = self.Chi_to_zeta(Chi)
                    z = self.zeta_to_z(zeta, self.epsilon)
                    r_z = hlp.xy_to_r_theta(z.real, z.imag)[0]
                    area_element = r_z * dr * dtheta
                    convective_acceleration = acceleration([z.real, z.imag], Gamma)
                    Appellian_value += np.dot(convective_acceleration, convective_acceleration)*area_element
        return Appellian_value*0.5

    if __name__ == "__main__":
        four = 4 # placeholder 
        # zeta_test = 4 - 1j*4
        # Gamma = self.circulation
        # # z_test = cyl.zeta_to_z(zeta_test)
        # chi_test = self.zeta_to_Chi(zeta_test)
        # print("\n")
        # z_test = cyl.zeta_to_z(zeta_test)
        # complex_zeta_center = self.zeta_center.real + 1j*self.zeta_center.imag
        # z_0 = cyl.zeta_to_z(complex_zeta_center)
        # # print("z_0: ", z_0)
        # zeta_test = [zeta_test.real, zeta_test.imag]
        # theta_zeta = np.arctan2(zeta_test[1], zeta_test[0])
        # z_test = [z_test.real, z_test.imag]
        # # theta_z = np.arctan2(z_test[1], z_test[0])
        # chi_test = [chi_test.real, chi_test.imag]
        # theta_chi = np.arctan2(chi_test[1], chi_test[0])
        # z_velocity_at_z_test = self.velocity(z_test, Gamma)
        # z_polar = hlp.polar_vector(theta_zeta, z_velocity_at_z_test)
        # # Chi_polar = hlp.polar_vector(theta_chi, velocity_chi_at_chi_test)
        # z_polar_mag = np.sqrt(z_polar[0]**2 + z_polar[1]**2)
        
        # diffs between acceleration methods
            # counter = 0
    # decimal_places = 14
    # num_points = len(r_vals) * len(theta_vals)
    # tolerance = 10**-decimal_places  # Absolute tolerance for comparison
    # print("Number of points to be compared: ", num_points)
    # print("Comparison out to " + str(decimal_places) + " decimal places")
            # for i in range(len(r_vals)):
    #     for j in range(len(theta_vals)):
    #         conv_accel_squared_comp_conj = cyl.taha_analytic_conv_accel_square_comp_conj([r_vals[i], theta_vals[j]], cyl.circulation)
    #         alternative_accel_squared_comp_conj = cyl.analytic_conv_accel_square_comp_conj([r_vals[i], theta_vals[j]], cyl.circulation)
            
    #         # Check if the values are equal within the specified decimal places
    #         if not np.isclose(conv_accel_squared_comp_conj, alternative_accel_squared_comp_conj, atol=tolerance): # atol is the absolute tolerance which is calculated as atol = alt
    #             # compare the absolute tolerance to the result
    #             # print("conv_accel_squared_comp_conj: ", conv_accel_squared_comp_conj)
    #             # print("alternative_accel_squared_comp_conj: ", alternative_accel_squared_comp_conj)
    #             # print("Difference: ", abs(conv_accel_squared_comp_conj - alternative_accel_squared_comp_conj))
                
    #             print("Conv Accel Squared Comp Conj:        ", conv_accel_squared_comp_conj)
    #             print("Alternative Accel Squared Comp Conj: ", alternative_accel_squared_comp_conj)
    #             print("r_vals[i]: ", r_vals[i])
    #             print("theta_vals[j]: ", theta_vals[j])
    #             print("\n")
    #             counter += 1

    # print("counter: ", counter)
    # if counter == 0:
    #     print("All values are equal between the two acceleration methods out to " + str(decimal_places) + " decimal places")

    # diffs between S values
        # print("\n")
    # r_range = [cyl.cylinder_radius, 8*cyl.cylinder_radius, 0.01]
    # r_vals = hlp.list_to_range(r_range)
    # theta_range = [0, 2*np.pi, np.pi/1000]
    # theta_vals = hlp.list_to_range(theta_range)
    # Gamma = 0.376992934959691
    # S = cyl.numerically_integrate_appellian(Gamma, r_vals, theta_vals)
    # print("Gamma: ", Gamma)
    # print("r_end: ", r_range[1])
    # print("r_step: ", r_range[2])
    # print("theta_step: ", theta_range[2])
    # print("S: ", S)
    # print("\n")