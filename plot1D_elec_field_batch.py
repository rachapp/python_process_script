# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 23:13:24 2023

@author: Racha
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Don't display the plot
import tables
import re
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator

def extract_number(filename):
    match = re.search(r'(\d+)\.h5$', filename)  # Find the last sequence of numbers before .h5
    if match:
        return int(match.group(1))  # Convert to integer and return
    else:
        return None  # Return None if no number found

# und_numbers = 1
und_numbers = sys.argv[1]

file_numbers = 40 * int(und_numbers) + np.arange(0,40)

z2_lim = (179, 190)
p_j_batch = []
p_j_global_max = []
m_Z_batch = []
net_sums_batch = []
net_sums_global_max = []
num_e_batch = []
num_e_global_max = []
# Define bin edges with fixed width
d_bin = 0.05 # in unit of z2
MPperWave = 800
norm_num = MPperWave*d_bin
bin_edges = np.arange(z2_lim[0], z2_lim[1], d_bin)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

print("Reading files...")
for num in file_numbers:
    # electrons_file = f"D://Puffin_results//gamma_100_rho0.079_helical//SSS_e_{num}.h5"
    electrons_file = sys.argv[2] + f'{num}.h5'
    # print(electrons_file)
    h5e = tables.open_file(electrons_file, mode='r')
    Electrons = h5e.root.electrons.read()
    m_Z = Electrons[:, 2]
    m_GAMMA = Electrons[:, 5]

    # Calculate energy spread
    gamma_r = h5e.root.runInfo._v_attrs.gamma_r
    gamma_j = gamma_r * m_GAMMA
    rho = h5e.root.runInfo._v_attrs.rho
    
    energy_spread = (gamma_j - gamma_r) * 100 / gamma_r

    p_j = (gamma_j - gamma_r) / (rho * gamma_r)
    
    filtered_indices = np.where((m_Z >= z2_lim[0]) & (m_Z <= z2_lim[1]))
    filtered_m_Z = m_Z[filtered_indices]
    filtered_p_j = p_j[filtered_indices]
    
    num_bins = len(bin_edges) - 1
    net_sums = np.zeros(num_bins)
    num_e = np.zeros(num_bins)
    for k in range(num_bins):
        in_bin = (filtered_m_Z >= bin_edges[k]) & (filtered_m_Z < bin_edges[k+1])
        # count_true = np.sum(in_bin)
        energies_in_bin = filtered_p_j[in_bin]
        
        num_e[k] = np.sum(in_bin)/norm_num
        net_sums[k] = np.sum(energies_in_bin)/norm_num
    
    num_e_batch.append(num_e)
    num_e_global_max.append(np.max(num_e))
    
    net_sums_batch.append(net_sums)
    net_sums_global_max.append(np.max(np.abs(net_sums)))
   
    
    p_j_batch.append(filtered_p_j)
    p_j_global_max.append(np.max(np.abs(filtered_p_j)))
    m_Z_batch.append(filtered_m_Z)
    
    h5e.close()
    
print("___Read Done___")

print("Getting Max for plot...")
p_j_global_max = np.array(p_j_global_max)
p_j_ylim = 0.5 * np.ceil(np.max(p_j_global_max)/0.5)

net_sums_global_max = np.array(net_sums_global_max)
net_sums_lim =  0.5 * np.ceil(np.max(net_sums_global_max) /0.5 )

num_e_global_max = np.array(num_e_global_max)
num_e_lim = 1 +  0.5 * np.ceil(np.max(num_e_global_max)/0.5)

print(f"pj_lim = {p_j_ylim}, net_pj_lim = {net_sums_lim}, num_e_lim = {num_e_lim}")

for i, num in enumerate(file_numbers):
    # aperp_file = f"D://Puffin_results//gamma_100_rho0.079_helical//SSS_ap_{num}.h5"
    aperp_file = sys.argv[3] + f'{num}.h5'
    # print(aperp_file)
    h5f = tables.open_file(aperp_file, mode='r')
    
    # Calculate z2_bar
    nz = h5f.root.runInfo._v_attrs.nZ2
    meshsizeZ2 = h5f.root.runInfo._v_attrs.sLengthOfElmZ2
    z2_bar = np.linspace(0, meshsizeZ2 * (nz - 1.0), nz)
    
    z2_lo = int(np.floor(z2_lim[0]/meshsizeZ2))
    z2_hi = int(np.floor(z2_lim[1]/meshsizeZ2))
    
    aperp_x = h5f.root.aperp[0][z2_lo:z2_hi]
    aperp_y = h5f.root.aperp[1][z2_lo:z2_hi]
    intensity = aperp_x**2 + aperp_y**2
    z2axis = z2_bar[z2_lo:z2_hi]
    plt.rcParams.update({'font.size': 16})
    # Create a figure and set up a gridspec layout
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[4, 3, 5])
    
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(z2axis, intensity)
    ax0.set_ylim(0)
    ax0.set_xlim(z2_lim)
    ax0.set_ylabel(r'$|A|^2$')
    ax0.xaxis.set_ticklabels([]) 
    ax0.grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=0.3)
    ax0.tick_params(direction='in')
    x_spacing_value = 1  # Adjust as needed
    ax0.xaxis.set_major_locator(MultipleLocator(x_spacing_value))
    
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(z2axis, aperp_x, label='$A_x$')
    ax1.plot(z2axis, aperp_y, label='$A_y$')
    ax1.set_ylabel(r'$A$')
    ax1.set_xlim(z2_lim)
    ax1.xaxis.set_ticklabels([])  # remove x-ticks for ax0 since it shares the x-axis with ax1
    # Control the x-axis tick (and grid) spacing
    ax1.grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=0.3)
    ax1.tick_params(direction='in')
    ax1.xaxis.set_major_locator(MultipleLocator(x_spacing_value))
    ax1.legend(loc='upper right', framealpha=0, edgecolor='none')
    
    ax1_2 = ax1.twinx()
    ax1_2.bar(bin_centers, num_e_batch[i], width=(bin_edges[1] - bin_edges[0]), align='center', color='tab:green', alpha=0.4)
    ax1_2.set_ylim(0, num_e_lim)
    ax1_2.set_ylabel('norm. electrons density')
    
    print(f'plotting file_num = {num}')
    z_bar = num/40.0
    ax2 = fig.add_subplot(gs[2])
    ax2.text(0.02, 0.98, r'$\bar{z} =$' + f'{z_bar:.3f}', transform=ax2.transAxes,
             verticalalignment='top', horizontalalignment='left')
    ax2.scatter(m_Z_batch[i], p_j_batch[i], s=0.5 , marker='o', color='black', label='$p_j$')
    
    x_ticks = np.arange(z2_lim[0], z2_lim[1]+0.5, 1)
    ax2.set_xticks(x_ticks)
    
    ax2.set_ylim(-p_j_ylim, p_j_ylim)
    ax2.set_xlim(z2_lim)
    ax2.set_ylabel(r'$p_j$')
    ax2.set_xlabel(r'$\bar{z}_2$')
    ax2.grid(True, which='both', axis='x', color='gray', linestyle='--', linewidth=0.3)
    ax2.tick_params(direction='in')
    ax2.xaxis.set_major_locator(MultipleLocator(x_spacing_value))
    
    ax2_2 = ax2.twinx()
    ax2_2.bar(bin_centers, net_sums_batch[i], width=(bin_edges[1] - bin_edges[0]), align='center', alpha=0.5, color=np.where(net_sums_batch[i] >= 0, 'tab:blue', 'tab:red'))
    ax2_2.set_xlim(z2_lim)
    ax2_2.set_ylim(-net_sums_lim, net_sums_lim)
    ax2_2.tick_params(direction='in')
    ax2_2.set_ylabel(r'$\Sigma p_j$')
    
    fig.savefig(aperp_file[:-3] + '_elec.png', pad_inches = 0.1, dpi=150)
    # Close the input files
    h5f.close()
print(f'Undulator {und_numbers} DONE')