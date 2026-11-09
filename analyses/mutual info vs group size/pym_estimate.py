import os, sys
cur_directory = os.getcwd()
project_folder = os.path.abspath(os.path.join(cur_directory, os.pardir, os.pardir))
save_directory = os.path.join(cur_directory, 'saved')
sys.path.append(os.path.join(project_folder, 'analysis-suite'))

import analyses as als
from scipy import io as io
import numpy as np
from argparse import Namespace




mat_data = io.loadmat('./pym scaling.mat')
mat = Namespace(**mat_data)
N_group_sizes = mat.N_group_sizes[0,0]
group_sizes = mat.group_sizes[0]
samples_per_group = mat.samples_per_group[0,0]
I_array = np.zeros((N_group_sizes, samples_per_group))
I_var_array = np.zeros((N_group_sizes, samples_per_group))
I_shuffled_array = np.zeros((N_group_sizes, samples_per_group))
I_shuffled_var_array = np.zeros((N_group_sizes, samples_per_group))
for n in range(N_group_sizes):
    for t in range(samples_per_group):
        I_array[n,t] = mat.output_data[n,t][0,0][6]
        I_var_array[n,t] = mat.output_data[n,t][0,0][7]
        I_shuffled_array[n,t] = mat.output_data[n,t][0,0][14]
        I_shuffled_var_array[n,t] = mat.output_data[n,t][0,0][15]

I = np.sum(I_array, axis=1) / samples_per_group
I_var = (np.sum((I_array - I[:,None])**2, axis=1) + np.sum(I_var_array, axis=1)) / samples_per_group
I_shuffled = np.sum(I_shuffled_array, axis=1) / samples_per_group
I_shuffled_var = (np.sum((I_shuffled_array - I_shuffled[:,None])**2, axis=1) + np.sum(I_shuffled_var_array, axis=1)) / samples_per_group

def plot_results(ax):
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()
    
    # good_x = abs(I_shuffled) < np.sqrt(I_shuffled_var)
    good_x = I_shuffled < 0.01
    
    bad_x = ~ good_x

    ax.errorbar(group_sizes[good_x], I[good_x], np.sqrt(I_var + I_shuffled_var)[good_x], color = (0.3,0.3,0.3), label="PYM")
    ax.errorbar(group_sizes[bad_x], I[bad_x], np.sqrt(I_var + I_shuffled_var)[bad_x], color = (0.8,0.8,0.8), label="PYM after breakdown")
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)

def plot_linear_regress(ax):
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()
    
    # good_x = abs(I_shuffled) < np.sqrt(I_shuffled_var)
    good_x = I_shuffled < 0.01
    
    bad_x = ~ good_x
    
    x = group_sizes
    out = als.linear_regress(x[good_x], I_array[good_x,:], np.sqrt(I_var_array[good_x,:]))
    ax.plot(x, out["fit"](x), color=(0.3,0.3,0.3))
    # ax.fill_between(x, out["fit"](x) - out["fit_err"](x), out["fit"](x) + out["fit_err"](x), color=(0.3,0.3,0.3,0.3), label="PYM")
    ax.errorbar(group_sizes[good_x], I[good_x], np.sqrt(I_var + I_shuffled_var)[good_x], linestyle='', color=(0.3,0.3,0.3))

    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)



# from matplotlib import pyplot as plt
# plt.errorbar(group_sizes, I - I_sh, np.sqrt(I_var + I_sh_var))
# plt.show()