import numpy as np # type: ignore
import os # type: ignore
import matplotlib.pyplot as plt  # type: ignore

def read_in_csv(file_path):
    """
    Reads a CSV file and returns the data as a list of lists.
    
    :param file_path: Path to the CSV file.
    :return: List of lists containing the data from the CSV file.
    """
    with open(file_path, 'r') as file:
        data = [line.strip().split(',') for line in file.readlines()] 
    # Convert the data to a numpy array for easier manipulation (skipping headers)
    # Assuming the first row is headers, we skip it and convert the rest to float
    # data = data[1:]  # Skip the header row
    data = np.array(data)
    return data

def calc_Vx_Vy_Vmag(Momentum_x, Momentum_y, rho):
    """
    Calculates the velocity components V_x, V_y and the magnitude V_mag.
    
    :param Momentum_x: List of x-components of momentum.
    :param Momentum_y: List of y-components of momentum.
    :param rho: Density of the fluid.
    :return: Tuple containing V_x, V_y, and V_mag.
    """
    V_x = Momentum_x/rho
    V_y = Momentum_y/rho
    V_mag = np.sqrt(V_x**2 + V_y**2)
    return V_x, V_y, V_mag

def calc_rho_e(energy_per_mass, rho, V_mag):
    """
    Calculates the product of density and specific energy (rho * e).
    :param energy_per_mass: Specific energy per unit mass.
    :param rho: Density of the fluid.
    :param V_mag: Magnitude of velocity.
    :return: rho * e 
    """
    return energy_per_mass - 0.5 * rho * V_mag**2

def calc_pressure(rho_e, Gamma):
    """
    Calculates dimensional pressure P.
    :param rho_e: Product of density and specific energy (rho * e).
    :param Gamma: Specific heat ratio. 1.4 for air in most cases.
    :return: Dimensional pressure P.
    """
    return (Gamma - 1) * rho_e

def calc_cp(P, P_inf, V_inf, rho_inf):
    """
    Calculates the coefficient of pressure C_p.
    :param P: Dimensional pressure.
    :param P_inf: Free stream pressure.
    :param V_inf: Free stream velocity.
    :param rho_inf: Free stream density.
    :return: Coefficient of pressure C_p.
    """
    return (P - P_inf) / (0.5 * rho_inf * V_inf**2)

def write_out_adjusted_csv(file_path, P_inf=101325, V_inf=265.057, rho_inf=1.29225):
    """Takes the csv data, and adds columns for V_x, V_y, V_mag, rho*e, P, and C_p, and writes to a new _adjusted file."""
    import os
    base, ext = os.path.splitext(file_path)
    new_file_path = base + "_adjusted" + ext
    array = read_in_csv(file_path)
    # Ensure the new file path does not already exist
    # if os.path.exists(new_file_path):
        # raise FileExistsError(f"The file {new_file_path} already exists. Please choose a different name or delete the existing file.")
    with open(new_file_path, 'w') as file:
        # Write the header
        file.write(','.join(array[0]) + ',V_x,V_y,V_mag,rho*e,P,C_p\n')
        # Write the data, skipping the header row
        for row in array[1:]:
            Momentum_x = float(row[4])
            Momentum_y = float(row[5])
            rho = float(row[3])
            energy_per_mass = float(row[6])
            V_x, V_y, V_mag = calc_Vx_Vy_Vmag(Momentum_x, Momentum_y, rho)
            rho_e = calc_rho_e(energy_per_mass, rho, V_mag)
            P = calc_pressure(rho_e, 1.4)
            C_p = calc_cp(P, P_inf, V_inf, rho_inf)
            # adjust the array to include the new values
            file.write(','.join(map(str, row)) + f",{V_x},{V_y},{V_mag},{rho_e},{P},{C_p}\n")

def plot_pressure_distribution(file_path):
    """
    Plots the pressure distribution from the CSV file.
    
    :param file_path: Path to the CSV file.
    """
    data = read_in_csv(file_path)
    # Assuming the first column is x-coordinates and the last column is C_p
    x_coords = [float(row[1]) for row in data[1:]]
    C_p = [float(row[-1]) for row in data[1:]]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, C_p, marker='o')
    plt.title('Pressure Distribution')
    plt.xlabel('$x/c$')
    plt.xlim(0, 1)  # Assuming x/c ranges from 0 to 1
    # set y-axis rotation to 0 for better readability
    plt.ylabel('$C_p$', rotation=0)
    # invert the y-axis to have higher pressure at the top
    plt.gca().invert_yaxis()
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Add a horizontal line at C_p = 0
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # ## initialize the cylinder object
    # plt.rcParams["font.family"] = "Serif"
    # plt.rcParams["font.size"] = 17.0
    # plt.rcParams["axes.labelsize"] = 17.0
    # plt.rcParams['lines.linewidth'] = 1.0 # 1.0
    # plt.rcParams["xtick.minor.visible"] = True 
    # plt.rcParams["ytick.minor.visible"] = True
    # plt.rcParams["xtick.direction"] = plt.rcParams["ytick.direction"] = "in"
    # plt.rcParams["xtick.bottom"] = plt.rcParams["xtick.top"] = True
    # plt.rcParams["ytick.left"] = plt.rcParams["ytick.right"] = True
    # plt.rcParams["xtick.major.width"] = plt.rcParams["ytick.major.width"] = 0.75
    # plt.rcParams["xtick.minor.width"] = plt.rcParams["ytick.minor.width"] = 0.75
    # plt.rcParams["xtick.major.size"] = plt.rcParams["ytick.major.size"] = 5.0
    # plt.rcParams["xtick.minor.size"] = plt.rcParams["ytick.minor.size"] = 2.5
    # plt.rcParams["mathtext.fontset"] = "dejavuserif"
    # plt.rcParams['figure.dpi'] = 300.0
    # ## change legend parameters
    # plt.rcParams["legend.fontsize"] = 17.0
    # plt.rcParams["legend.frameon"] = True
    # subdict = {
    #     "figsize" : (3.25,3.5),
    #     "constrained_layout" : True,
    #     "sharex" : True
    # }
    # Example usage
    file_path = 'C:/Users/Spencer/repos/CFD/win64/bin/quick_start_directory/surface_flow.csv'  # Replace with your actual file path
    # Write out the adjusted CSV
    P_inf = 101325  # Free stream pressure in Pascals
    V_inf = 265.057  # Free stream velocity in m/s
    rho_inf = 1.29225  # Free stream density in kg/m^3
    write_out_adjusted_csv(file_path, P_inf, V_inf, rho_inf)
    print(f"Adjusted data written to {file_path}_adjusted.csv")
    # Plot the pressure distribution
    base, ext = os.path.splitext(file_path)
    adjusted_file_path = base + "_adjusted" + ext
    plot_pressure_distribution(adjusted_file_path)