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
from Joukowski_Cylinder import cylinder
# import tqdm for progress bar
from tqdm import tqdm # type: ignore
import os # type: ignore
import matplotlib.patches as patches  # type: ignore

# make a class which inherits from potential_flow_object and cylinder classes
class pmpg(potential_flow_object, cylinder):
    def __init__(self, **kwargs):
        potential_flow_object.__init__(self, **kwargs)
        cylinder.__init__(self, **kwargs)

    def parse_json(self):
        """This function reads the json file and stores the cylinder specific data in a dictionary"""
        with open(self.cyl_json_file, 'r') as json_handle:
            input = json.load(json_handle)
            print("\n")