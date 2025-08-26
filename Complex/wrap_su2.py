import os # type: ignore
import csv # type: ignore

def update_su2_config_mesh(cfg_path, new_mesh_filename, output_path=None):
    """
    Update the MESH_FILENAME in an SU2 config file.

    Parameters:
        cfg_path (str): Path to the original SU2 config file.
        new_mesh_filename (str): New SU2 mesh file name (with .su2 extension).
        output_path (str, optional): Path to save the modified config file.
                                     If None, overwrites the original file.

    Returns:
        None
    """
    with open(cfg_path, 'r') as f:
        lines = f.readlines()

    updated_lines = []
    mesh_updated = False
    for line in lines:
        if line.strip().startswith("MESH_FILENAME"):
            updated_line = f"MESH_FILENAME= {new_mesh_filename}\n"
            updated_lines.append(updated_line)
            mesh_updated = True
        else:
            updated_lines.append(line)

    if not mesh_updated:
        raise ValueError("MESH_FILENAME key not found in the configuration file.")

    # Save to output path or overwrite
    output = output_path if output_path is not None else cfg_path
    with open(output, 'w') as f:
        f.writelines(updated_lines)

def run_su2_solver(cfg_path):
    """
    Run the SU2 solver using the provided configuration file.

    Parameters:
        cfg_path (str): Path to the SU2 config file.

    Returns:
        None
    """
    os.system(f"SU2_CFD {cfg_path}")

if __name__ == "__main__":
    grid_convergence = False
    # folder_names = ["D_0.000_grid_conv", "D_0.001_grid_conv", "D_0.010_grid_conv", "D_0.100_grid_conv", "D_0.200_grid_conv","D_0.300_grid_conv", "D_0.400_grid_conv", "D_0.500_grid_conv", "D_0.600_grid_conv", "D_0.700_grid_conv", "D_0.800_grid_conv","D_0.900_grid_conv", "D_1.000_grid_conv"]
    folder_names = ["D_0.001_grid_conv"]
    if grid_convergence:
        for folder in folder_names:
            # loop through the files in the folder
            for file in os.listdir(folder):
                if file.endswith(".su2"):
                    cfg_path = "inv_conformal_map.cfg"
                    new_mesh_filename = f"{folder}/{file}"
                    update_su2_config_mesh(cfg_path, new_mesh_filename)
                    run_su2_solver(cfg_path)
                    dat_file_name = "forces_breakdown.dat"
                    # now make a copy of the dat file, move it to the folder/dat_folder and rename the dat file to be file but replace .su2 with .dat
                    dat_folder = folder + "/dat_folder"
                    os.makedirs(dat_folder, exist_ok=True)
                    os.system(f"cp {dat_file_name} {dat_folder}/{file.replace('.su2', '.dat')}")
    
    else:
        # loop through the dat files in each folder and extract the CL from each forces_breakdown.dat file
        for folder in folder_names:
            dat_folder = os.path.join(folder, "dat_folder")
            cl_rows = []
            for dat_file in os.listdir(dat_folder):
                if dat_file.endswith(".dat"):
                    dat_path = os.path.join(dat_folder, dat_file)
                    with open(dat_path, 'r') as f:
                        for line in f:
                            if line.strip().startswith("Total CL:"):
                                cl_value = float(line.split()[2])
                                cl_rows.append({"file": dat_file, "CL": cl_value})
                                break  # Only take the first occurrence per file

            # Write results to a CSV for this folder
            csv_filename = os.path.join(folder, "CL_results.csv")
            with open(csv_filename, "w", newline='') as csvfile:
                fieldnames = ["file", "CL"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in cl_rows:
                    writer.writerow(row)

            print(f"Exported CL values to {csv_filename}")