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
from complex_potential_flow_class import potential_flow_object
# import tqdm for progress bar
from tqdm import tqdm # type: ignore
import os # type: ignore


class try_it(potential_flow_object):
    def __init__(self, json_file):
        print("Initializing try_it class")
        self.zeta_center = -0.09 + 1j*0.0
        self.D = 0.1
        self.freestream_velocity = 10.0
        self.angle_of_attack = 5.0
        self.cylinder_radius = 1.0
        self.J_epsilon = self.calc_J_airfoil_epsilon()
        self.epsilon = self.calc_epsilon_from_D(self.D)
        self.circulation = 10

    def analytic_conv_accel_square_comp_conj(self,point_r_theta_in_Chi,Gamma):
        r, theta = point_r_theta_in_Chi[0], point_r_theta_in_Chi[1]
        xi_chi, eta_chi = hlp.r_theta_to_xy(r, theta)
        chi = xi_chi + 1j*eta_chi
        zeta = self.Chi_to_zeta(chi)
        dphi_dzeta, d2Phi_dzeta2, dz_dzeta, d2z_dzeta2 = self.dPhi_dzeta(zeta, Gamma), self.d2Phi_dzeta2(zeta, Gamma), self.dZ_dzeta(zeta), self.d2Z_dzeta2(zeta)
        # print("dphi_dzeta", dphi_dzeta)
        # print("d2Phi_dzeta2", d2Phi_dzeta2)
        # print("dz_dzeta", dz_dzeta)
        # print("d2z_dzeta2", d2z_dzeta2)
        dz_dzeta2 = dz_dzeta * np.conj(dz_dzeta)
        dz_dzeta4 = dz_dzeta**2*np.conj(dz_dzeta**2)
        # conv_accel = dphi_dzeta*(d2Phi_dzeta2*dz_dzeta - dphi_dzeta*d2z_dzeta2)/dz_dzeta2
        conv_accel = dphi_dzeta*(d2Phi_dzeta2*dz_dzeta - dphi_dzeta*d2z_dzeta2)/dz_dzeta4
        conv_accel_comp_conj = np.conj(conv_accel)
        conv_accel_squared = conv_accel * conv_accel_comp_conj * dz_dzeta2 
        return np.real(conv_accel_squared)
    
    def taha_analytic_conv_accel_square_comp_conj(self, point_r_theta_in_Chi, Gamma): # need to adjust so that it uses the Taha transformation
        """This function calculates the pressure gradient at a given point in the flow field in the z plane"""
        r, theta = point_r_theta_in_Chi[0], point_r_theta_in_Chi[1]
        xi_chi, eta_chi = hlp.r_theta_to_xy(r, theta)
        chi = xi_chi + 1j*eta_chi
        zeta = self.Chi_to_zeta(chi)
        dphi_dzeta, d2Phi_dzeta2, dz_dzeta, d2z_dzeta2 = self.dPhi_dzeta(zeta, Gamma), self.d2Phi_dzeta2(zeta, Gamma), self.dZ_dzeta(zeta), self.d2Z_dzeta2(zeta)
        G_of_zeta = 1/dz_dzeta
        G_of_zeta_comp_conj = np.conj(G_of_zeta)
        G_of_zeta_squared = G_of_zeta * G_of_zeta_comp_conj
        R, epsilon = self.cylinder_radius, self.epsilon
        dG_dzeta = -2*(R-epsilon)**2/((1-(R-epsilon)**2/zeta**2)**2*zeta**3)
        dG_dzeta_comp_conj = np.conj(dG_dzeta)
        omega_of_zeta = dphi_dzeta
        omega_of_zeta_comp_conj = np.conj(omega_of_zeta)
        omega_of_zeta_squared = omega_of_zeta * omega_of_zeta_comp_conj 
        a_of_zeta = d2Phi_dzeta2
        integrand = G_of_zeta_comp_conj * a_of_zeta + omega_of_zeta_squared*dG_dzeta_comp_conj
        integrand_comp_conj = np.conj(integrand)
        conv_accel_squared =  G_of_zeta_squared * integrand * integrand_comp_conj 
        return np.real(conv_accel_squared)
    
    def dPhi_dzeta(self, zeta, Gamma):
        """This function calculates the derivative of Phi with respect to zeta for a Joukowski transformation"""
        V_inf, R, alpha, zeta0 = self.freestream_velocity, self.cylinder_radius, self.angle_of_attack, self.zeta_center
        return V_inf*(np.exp(-1j*alpha) + 1j*Gamma/(2*np.pi*V_inf*(zeta-zeta0)) - np.exp(1j*alpha)*R**2/(zeta-zeta0)**2)
    
    def d2Phi_dzeta2(self, zeta, Gamma):
        """This function calculates the second derivative of Phi with respect to zeta for a Joukowski transformation"""
        V_inf, R, alpha, zeta0 = self.freestream_velocity, self.cylinder_radius, self.angle_of_attack, self.zeta_center
        return V_inf*(-1j*Gamma/(2*np.pi*V_inf*(zeta-zeta0)**2) + 2*np.exp(1j*alpha)*R**2/((zeta-zeta0)**3))
    
    def dZ_dzeta(self, zeta):
        """This function calculates the derivative of Z with respect to zeta for a Joukowski transformation"""
        epsilon, R, = self.epsilon, self.cylinder_radius
        return 1 - (R-epsilon)**2/(zeta)**2
    
    def d2Z_dzeta2(self, zeta):
        """This function calculates the second derivative of Z with respect to zeta for a Joukowski transformation"""
        epsilon, R = self.epsilon, self.cylinder_radius
        return 2*(R-epsilon)**2/(zeta)**3
    
    def calc_epsilon_from_D(self, D: float):
        """This function calculates epsilon from D"""
        epsilon_o = self.calc_J_airfoil_epsilon()
        epsilon = D*(1-epsilon_o)+epsilon_o
        return epsilon
    
    def calc_J_airfoil_epsilon(self):
        """This function calculates the epsilon of the airfoil"""
        epsilon = self.cylinder_radius - 1*np.sqrt(self.cylinder_radius**2-self.zeta_center.imag**2)-self.zeta_center.real # eq 113 in complex variables
        return epsilon
    
    def Joukowski_z_to_zeta(self, z: complex, epsilon: float, D=0.0): # eq 104 
        """This function takes in a z coordinate and returns the zeta coordinate"""
        z_1 = z**2 - 4*(self.cylinder_radius - epsilon)**2
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
    
    def zeta_to_Chi(self, zeta: complex):
        """This function takes in a zeta coordinate and returns the Chi coordinate"""
        Chi = zeta - self.zeta_center
        return Chi
    
    def Chi_to_zeta(self, Chi: complex):
        """This function takes in a Chi coordinate and returns the zeta coordinate"""
        zeta = Chi + self.zeta_center
        return zeta
    

if __name__ == "__main__":
    cyl = try_it("Joukowski_Cylinder.json")


    # test_point_in_z = cyl.lower_coords[5]
    test_point_in_z = [0.5, 0.5]
    test_point_in_zeta = cyl.Joukowski_z_to_zeta(test_point_in_z[0] + 1j*test_point_in_z[1], cyl.epsilon)
    test_point_in_Chi = cyl.zeta_to_Chi(test_point_in_zeta)
    r_chi, theta_chi = hlp.xy_to_r_theta(test_point_in_Chi.real, test_point_in_Chi.imag)
    r_0, theta_0 = hlp.xy_to_r_theta(cyl.zeta_center.real, cyl.zeta_center.imag)
    test_point_in_Chi = [test_point_in_Chi.real, test_point_in_Chi.imag]
    test_point_in_zeta = [test_point_in_zeta.real, test_point_in_zeta.imag]


    Taha_conv_accel_squared_comp_conj = cyl.taha_analytic_conv_accel_square_comp_conj([r_chi,theta_chi], cyl.circulation)
    print("Taha Analytic Convective Acceleration squared is    ", Taha_conv_accel_squared_comp_conj)
    # cyl.epsilon = equivalent_Joukowski_epsilon
    # cyl.transformation_type = "joukowski"
    # cyl.zeta_to_z, cyl.z_to_zeta, cyl.dZ_dzeta, cyl.d2Z_dzeta2 = cyl.general_zeta_to_z_transformation()
    Spencer_accel_squared_comp_conj = cyl.analytic_conv_accel_square_comp_conj([r_chi,theta_chi], cyl.circulation)
    print("Spencer Alternat Convective Acceleration squared is ", Spencer_accel_squared_comp_conj)
    print("\n")
