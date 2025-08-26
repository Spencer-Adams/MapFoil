import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
import helper as hlp 

epsilon = 1.0#0.08
freestream_velocity = 1.0
angle_of_attack = 0.0#5 * np.pi / 180  # Convert degrees to radians
zeta_center = -0.09 + 1j*0.0 
Gamma = 0.0
cylinder_radius = 1.0

def z_to_zeta(z: complex, epsilon: float, D=0.0): # eq 104 
    """This function takes in a z coordinate and returns the zeta coordinate"""
    z_1 = z**2 - 4*(cylinder_radius - epsilon)**2
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
    if abs(zeta_2 - zeta_center) > abs(zeta - zeta_center):
        zeta = zeta_2
    return zeta

def Chi_to_zeta(Chi: complex):
    """This function takes in a Chi coordinate and returns the zeta coordinate"""
    zeta = Chi + zeta_center
    return zeta

def zeta_to_z(zeta: complex, epsilon: float):
    """This function takes in a zeta coordinate and returns the z coordinate"""
    if np.isclose(zeta.real, 0.0) and np.isclose(zeta.imag, 0.0):
        z = zeta
    else:
        z = zeta + (cylinder_radius - epsilon)**2/zeta # eq 96
    return z


def potential(point_xi_eta_in_z_plane, Gamma, C = 300.0):
    """This function calculates the potential at a given point in the flow field in the z plane"""
    z = point_xi_eta_in_z_plane[0] + 1j*point_xi_eta_in_z_plane[1]
    zeta = z_to_zeta(z, epsilon)
    Potential = freestream_velocity*(zeta*np.exp(-1j*angle_of_attack) + 1j*Gamma/(2*np.pi*freestream_velocity)*np.log(zeta-zeta_center) + np.exp(1j*angle_of_attack)*cylinder_radius**2/(zeta-zeta_center)) + C
    Potential_complex = np.array([Potential.real, Potential.imag])  # the minus sign is because we are using the complex potential
    return Potential_complex


# Step 1: Setup polar grid in chi-plane (centered at origin)
r_first = np.linspace(cylinder_radius, cylinder_radius + 0.1, 100)  # Start just outside the cylinder
r_second = np.linspace(cylinder_radius + 0.2, 10.0, 100)  # Extend to a larger radius
r = np.concatenate((r_first, r_second))
# r = np.linspace(cylinder_radius, 10.0, 100)
theta = np.linspace(0, 2 * np.pi, 100)
# remove the last point to avoid duplication at the end of the circle
r = r[1:]  # Remove the last point to avoid duplication at the end of the circle

# Step 2: Convert polar to Cartesian coordinates in chi-plane
chi = np.zeros(len(r) * len(theta), dtype=complex)
for i in range(len(chi)):
    x_chi = r[i // len(theta)] * np.cos(theta[i % len(theta)])
    y_chi = r[i // len(theta)] * np.sin(theta[i % len(theta)])
    chi[i] = x_chi + 1j * y_chi  

# Step 3: Map from chi → zeta (elementwise, since z_to_zeta expects a scalar)
zeta = np.zeros_like(chi, dtype=complex)
for i in range(chi.shape[0]):
    zeta[i] = Chi_to_zeta(chi[i])  

# Step 4: Map from zeta → z (elementwise)
z = np.zeros_like(zeta, dtype=complex)
for i in range(len(zeta)):
    z[i] = zeta_to_z(zeta[i], epsilon)   

# Step 5: Extract grid in the z-plane
x = z.real
y = z.imag

# Reshape to 2D grids for contour plotting
n_theta = len(theta)
n_r = len(r)
x_grid = x.reshape((n_theta, n_r))
y_grid = y.reshape((n_theta, n_r))

# Compute phi, psi over the mapped grid
phi = np.zeros_like(x_grid)
psi = np.zeros_like(x_grid)
for i in range(n_theta):
    for j in range(n_r):
        phi_psi = potential([x_grid[i, j], y_grid[i, j]], Gamma)
        phi[i, j] = phi_psi[0]
        psi[i, j] = phi_psi[1]

# Plot streamlines (psi) and equipotentials (phi)
plt.figure(figsize=(8, 6))
plt.contour(x_grid, y_grid, psi, levels=100, colors='black', linewidths=0.5, linestyles = "solid")  # Streamlines
plt.contour(x_grid, y_grid, phi, levels=100, colors='gray', linestyles='dashed')  # Equipotentials
plt.axis('equal')
plt.title("Streamlines (black) and Equipotentials (dashed gray)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()