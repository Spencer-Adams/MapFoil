import os
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import re

plt.rcParams["font.family"] = "Serif"
plt.rcParams["font.size"] = 17.0
plt.rcParams["axes.labelsize"] = 17.0
plt.rcParams['lines.linewidth'] = 1.0 # 1.0
plt.rcParams["xtick.minor.visible"] = True 
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.direction"] = plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.bottom"] = True
plt.rcParams["xtick.top"]    = False
plt.rcParams["ytick.left"]   = True
plt.rcParams["ytick.right"]  = False
plt.rcParams["xtick.major.width"] = plt.rcParams["ytick.major.width"] = 0.75
plt.rcParams["xtick.minor.width"] = plt.rcParams["ytick.minor.width"] = 0.75
plt.rcParams["xtick.major.size"] = plt.rcParams["ytick.major.size"] = 5.0
plt.rcParams["xtick.minor.size"] = plt.rcParams["ytick.minor.size"] = 2.5
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams['figure.dpi'] = 300.0
## change legend parameters
plt.rcParams["legend.fontsize"] = 17.0
plt.rcParams["legend.frameon"] = True
subdict = {"figsize" : (3.25,3.5),"constrained_layout" : True,"sharex" : True}
all_linestyles = [
    "solid",            # solid
    (0, (5, 5)),        # dashed
    "dashdot",          # dash-dot (standard Matplotlib style)
    (0, (3, 5, 1, 5)),  # dash-dot with spacing
    (0, (1, 5)),        # dotted (sparse)
    (0, (1, 3)),        # dotted (denser)
    (0, (1, 1))         # very fine dotted
]
color = ["black"]

#### First Plot (Numerical vs Analytic line integral Grid Conv Comparison at D = 0.1 without camber)####### 
# file_location = "Grid_conv/figures/integral_line/base_plots_num_vs_analytic"
# x_is_log_scale = True
# y_is_log_scale = True
# labels = ["Analytical", "Numerical"] # labels for the numerical vs analytic first plot
# is_legend = True
# is_legend_below_fig = False
# bbox_to_anchor = (0.5, -0.5)
# x_axis_title = "Surface Points"
# y_axis_title = "$\\varepsilon_a$"
# labelpad = -1
# x_limit = (10, 1e6)
# x_ticks = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
# x_tick_labels = ["$10^1$", "$10^2$", "$10^3$", "$10^4$", "$10^5$", "$10^6$"]
# y_limit = (1e-14, 100)
# y_ticks = [1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2]
# y_tick_labels = ["$10^{-14}$", "$10^{-12}$", "$10^{-10}$", "$10^{-8}$", "$10^{-6}$", "$10^{-4}$", "$10^{-2}$", "$10^{0}$", "$10^{2}$"]
# is_show_plot = False
# is_save_plot = True
# plot_name = "D_0.1_analytic_vs_numerical_line_integral_comparison"
# first_column_for_plotting = 0
# second_column_for_plotting = 1
# number_of_rows_to_skip = 2

##### Second Plot (Analytic line integral Grid Conv of D = [0.001,0.01,0.1,0.2,0.4,0.8,1.0] without camber)####### 
# file_location = "Grid_conv/figures/integral_line/uncambered"
# x_is_log_scale = True
# y_is_log_scale = True
# labels = ["D = 0.001", "D = 0.01", "D = 0.1", "D = 0.2", "D = 0.4", "D = 0.8", "D = 1.0"] # labels for the different D values second plot
# is_legend = True
# is_legend_below_fig = True
# bbox_to_anchor = (0.5, -0.4)
# x_axis_title = "Surface Points"
# y_axis_title = "$\\varepsilon_a$"
# labelpad = -1
# x_limit = (10, 1e6)
# x_ticks = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
# x_tick_labels = ["$10^1$", "$10^2$", "$10^3$", "$10^4$", "$10^5$", "$10^6$"]
# y_limit = (1e-16, 100)
# y_ticks = [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2]
# y_tick_labels = ["$10^{-16}$", "$10^{-14}$", "$10^{-12}$", "$10^{-10}$", "$10^{-8}$", "$10^{-6}$", "$10^{-4}$", "$10^{-2}$", "$10^{0}$", "$10^{2}$"]
# is_move_x_tick_label_right = False
# is_show_plot = False
# is_save_plot = True
# plot_name = "Grid_conv_for_different_D_values"
# first_column_for_plotting = 0
# second_column_for_plotting = 1
# number_of_rows_to_skip = 2

##### Third Plot (Analytic line integral Showing S for D = [0.001,0.01,0.1,0.2,0.4,0.8,1.0] without camber)####### 
# file_location = "Grid_conv/figures/integral_line/uncambered"
# x_is_log_scale = True
# y_is_log_scale = False
# labels = ["D = 0.001", "D = 0.01", "D = 0.1", "D = 0.2", "D = 0.4", "D = 0.8", "D = 1.0"] # labels for the different D values second plot
# is_legend = True
# is_legend_below_fig = True
# bbox_to_anchor = (0.5, -0.4)
# x_axis_title = "Surface Points"
# y_axis_title = "$\\hat{S}$"
# labelpad = -1
# x_limit = (10, 1e6)
# x_ticks = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
# x_tick_labels = ["$10^1$", "$10^2$", "$10^3$", "$10^4$", "$10^5$", "$10^6$"]
# is_move_x_tick_label_right = True
# y_limit = (0.0, 6.0)
# y_ticks = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
# y_tick_labels = ["$0$", "$1$", "$2$", "$3$", "$4$", "$5$", "$6$"]
# is_show_plot = False
# is_save_plot = True
# plot_name = "appellian_for_different_D_values"
# first_column_for_plotting = 0
# second_column_for_plotting = 3
# number_of_rows_to_skip = 1

# # ##### Fourth Plot (Analytic line integral Grid Conv of D = [0.001,0.01,0.1,0.2,0.4,0.8,1.0] with camber)####### 
# file_location = "Grid_conv/figures/integral_line/cambered"
# x_is_log_scale = True
# y_is_log_scale = True
# labels = ["D = 0.001", "D = 0.01", "D = 0.1", "D = 0.2", "D = 0.4", "D = 0.8", "D = 1.0"] # labels for the different D values second plot
# is_legend = True
# is_legend_below_fig = True
# bbox_to_anchor = (0.5, -0.4)
# x_axis_title = "Surface Points"
# y_axis_title = "$\\varepsilon_a$"
# labelpad = -1
# x_limit = (10, 1e6)
# x_ticks = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
# x_tick_labels = ["$10^1$", "$10^2$", "$10^3$", "$10^4$", "$10^5$", "$10^6$"]
# is_move_x_tick_label_right = False
# y_limit = (1e-16, 100)
# y_ticks = [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2]
# y_tick_labels = ["$10^{-16}$", "$10^{-14}$", "$10^{-12}$", "$10^{-10}$", "$10^{-8}$", "$10^{-6}$", "$10^{-4}$", "$10^{-2}$", "$10^{0}$", "$10^{2}$"]
# is_show_plot = False
# is_save_plot = True
# plot_name = "Grid_conv_for_different_D_values"
# first_column_for_plotting = 0
# second_column_for_plotting = 1
# number_of_rows_to_skip = 2

# #### Fifth Plot (Analytic line integral Showing S for D = [0.001,0.01,0.1,0.2,0.4,0.8,1.0] with camber)####### 
file_location = "Grid_conv/figures/integral_line/cambered"
x_is_log_scale = True
y_is_log_scale = False
labels = ["D = 0.001", "D = 0.01", "D = 0.1", "D = 0.2", "D = 0.4", "D = 0.8", "D = 1.0"] # labels for the different D values second plot
is_legend = True
is_legend_below_fig = True
bbox_to_anchor = (0.5, -0.4)
x_axis_title = "Surface Points"
y_axis_title = "$\\hat{S}$"
labelpad = -1
x_limit = (10, 1e6)
x_ticks = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
x_tick_labels = ["$10^1$", "$10^2$", "$10^3$", "$10^4$", "$10^5$", "$10^6$"]
is_move_x_tick_label_right = True
is_move_first_x_tick_label_right = True
y_limit = (0.0, 6.0)
y_ticks = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
y_tick_labels = ["$0$", "$1$", "$2$", "$3$", "$4$", "$5$", "$6$"]
is_show_plot = False
is_save_plot = True
plot_name = "appellian_for_different_D_values"
first_column_for_plotting = 0
second_column_for_plotting = 3
number_of_rows_to_skip = 1

def read_file(file):
    """reads in a file and returns its content as a numpy array"""
    # reads in the file, skips the user-specified number of rows, and returns the data
    return np.loadtxt(file, delimiter=",", skiprows=number_of_rows_to_skip)

def plot_files(file_location, first_column_for_plotting: int = 0, second_column_for_plotting: int = 1, is_legend: bool = True, is_legend_below_fig: bool = False, bbox_to_anchor: tuple = (0.5, -0.5), is_move_x_tick_label_right = False):
    files = [f for f in os.listdir(file_location) if f.endswith(".csv") or f.endswith(".txt")]
    files_sorted = sorted(files, key=lambda x: int(x.split('_')[0]))
    local_labels = labels.copy()
    n_lines = len(all_linestyles)
    for i, filename in enumerate(files_sorted):
        file_path = os.path.join(file_location, filename)
        data = read_file(file_path)
        if y_is_log_scale:
            y_data = data[:, second_column_for_plotting]
            # Use np.isclose to 0 with high precision (16 digits)
            y_data = np.where(np.isclose(y_data, 0.0, atol=1e-16), 2e-16, y_data)
            data = data.copy()
            data[:, second_column_for_plotting] = y_data
        if len(files_sorted) == len(local_labels):
            linestyle = all_linestyles[i % n_lines]
            if i < n_lines:
                plt.plot(
                    data[:, first_column_for_plotting],
                    data[:, second_column_for_plotting],
                    label=local_labels[i],
                    color="black",
                    linestyle=linestyle,
                )
            else:
                plt.plot(
                    data[:, first_column_for_plotting],
                    data[:, second_column_for_plotting],
                    label=local_labels[i],
                    color="gray",
                    linestyle=linestyle,
                )
        else:
            raise ValueError("Mismatch between number of files and labels")
    plt.ylabel(y_axis_title, rotation=0)
    plt.ylabel(y_axis_title, labelpad=labelpad)
    plt.xlabel(x_axis_title, labelpad=labelpad)
    plt.xlim(x_limit)
    plt.ylim(y_limit)
    if x_is_log_scale:
        plt.xscale("log")
        plt.xticks(x_ticks, labels=x_tick_labels)
    if y_is_log_scale:
        plt.yscale("log")
        plt.yticks(y_ticks, labels=y_tick_labels)
    if is_legend:
        handles, legend_labels = plt.gca().get_legend_handles_labels()
        if len(handles) == 7:
            # Reorder: group by rows instead of columns for 3 columns
            reordered = [
                handles[0], handles[3], handles[6],
                handles[1], handles[4],
                handles[2], handles[5]
            ]
            reordered_labels = [
                legend_labels[0], legend_labels[3], legend_labels[6],
                legend_labels[1], legend_labels[4],
                legend_labels[2], legend_labels[5]
            ]
            if is_legend_below_fig:
                plt.legend(
                    reordered, reordered_labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, -0.5),  # Move legend further down
                    ncol=3
                )
            else:
                plt.legend(reordered, reordered_labels, ncol=3)
        else:
            # Default legend order for any other number of curves
            if is_legend_below_fig:
                plt.legend(loc="lower center", bbox_to_anchor=bbox_to_anchor, ncol=1)
            else:
                plt.legend(ncol=1)
    if is_move_x_tick_label_right:
        ax = plt.gca()
        ticklabels = ax.get_xticklabels()
        xticks = ax.get_xticks()
        if ticklabels and len(xticks) > 0:
            # Remove the first label
            new_labels = ["" if i == 0 else label.get_text() for i, label in enumerate(ticklabels)]
            ax.set_xticklabels(new_labels)
            # Add custom text at a shifted position
            # You can adjust the y value (vertical position) as needed
            # Use the same y as the original tick label
            x, y = ticklabels[0].get_position()
            ax.text(xticks[0] + 0.05 * (xticks[1] - xticks[0]), y-0.28, x_tick_labels[0], ha='center', va='center')
            # move yaxis title down a bit
            ax.yaxis.label.set_position((ax.yaxis.label.get_position()[0], ax.yaxis.label.get_position()[1] + 0.05))
    if is_save_plot:
        plt.savefig(os.path.join(file_location, f"{plot_name}.svg"), bbox_inches="tight")
    if is_show_plot:
        plt.show()

# run the plotting function
plot_files(file_location, first_column_for_plotting=first_column_for_plotting, second_column_for_plotting=second_column_for_plotting, is_legend=is_legend, is_legend_below_fig=is_legend_below_fig, bbox_to_anchor=bbox_to_anchor, is_move_x_tick_label_right=is_move_x_tick_label_right)