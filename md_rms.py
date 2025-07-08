#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: My Nguyen
Email: mynguyen@usc.edu
Date: March 27, 2025
Description: Align trajectories and calculate RMSD and RMSF.

Dependencies:
    - numpy
    - matplotlib
    - mdtraj

Usage:
    python md_rms.py

License: MIT License
"""
import mdtraj as md

import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
matplotlib.rc('font', size=9)
matplotlib.rc('axes', titlesize=9)
colors = ['#6495ED','r','#6da81b','#483D8B','#FF8C00','#2E8B57','#800080','#008B8B','#949c2d', '#a34a17','#c43b99','#949c2d','#1E90FF']
plt.rc('axes', prop_cycle=cycler('color', colors))

# =============================================
#                    Input
# =============================================
align_traj = True
coordfiles = ["../trajcat_Set_0_0_0/step7_cat_568ns_notwater.xtc",
              "../trajcat_Set_0_0_1/step7_cat_450ns_notwater.xtc",
              ]
# 1 ns = 10 frames
dt = 0.1 # time step in ns, used to convert time to frames
topfile = "../trajcat_Set_0_0_0/step7_cat_notwater.pdb"
#topfile = "../trajcat_Set_0_0_0/step7_cat.pdb"
warmup = 0 # number of frames to discard before stride
stride = 5

# receptor
rec_chainid = 0
rec_resid0 = 47 # index of the first residue of the receptor based on UnitProt sequence
# ligand
lig_chainid = 4
# other proteins (G protein/ Arrestin)
prot_chainid = [1,2,3] # other protein chains if any, optional
prot_resid0 = [11,14,1] # index of the first residue of proteins based on UnitProt sequence

# Select the C-alpha atoms of the receptor for alignment,
# TM region: resid 55-335
resid_start_correct = 55 # based on UnitProt sequence
resid_end_correct = 335  # based on UnitProt sequence

# receptor domains {domain: unitprot resid range}
# https://gpcrdb.org/protein/P30411/
rec_domains = {'N-term' : [1,54],
               'TM1'    : [55, 85],
               'ICL1'   : [86, 89],
               'TM2'    : [90, 120],
               'ECL1'   : [121, 125],
               'TM3'    : [126, 161],
               'ICL2'   : [162, 169],
               'TM4'    : [170, 196],
               'ECL2'   : [197, 216],
               'TM5'    : [217, 258],
               'ICL3'   : [259, 261],
               'TM6'    : [262, 299],
               'ECL3'   : [300, 302],
               'TM7'    : [303, 335],
               'H8'     : [336, 346], 
               'C-term' : [347, 391]}

domain_color_map = {'N-term' : '#57b2f2',
                 'TM1'     : '#b17cf7',
                 'TM2'     : '#b17cf7',
                 'TM3'     : '#b17cf7',
                 'TM4'     : '#b17cf7',
                 'TM5'     : '#b17cf7',
                 'TM6'     : '#b17cf7',
                 'TM7'     : '#b17cf7',
                 'H8'      : '#94fa61',
                 'C-term' : '#f54949',}
# ==========================================================================================
rmsd_rec_list = []
rmsd_comp_list = []
rmsd_lig_list = []
time_list = []
rmsf_list = []

for i,coordfile in enumerate(coordfiles):
    # =============================================
    #               Load trajectory
    # =============================================
    print(f"\n... Loading Trajectory {i+1}/{len(coordfiles)} ...", flush=True)
    traj = md.load(coordfile,top=topfile,stride=stride)
    traj = traj[int(warmup/stride):]
    top = traj.topology
    time = np.arange(traj.n_frames) * dt * stride + warmup * dt
    print("... Done Loading ...", flush=True)
    print(f"Loaded {traj.n_frames} frames, {time[0]}-{time[-1]} ns, {traj.n_atoms} atoms, {traj.n_residues} residues", flush=True)

    # Recenter and apply periodic boundary conditions if molecules are outside of the box
    # print("... Applying periodic boundary condition ...", flush=True)
    # traj = traj.image_molecules()

    # Collect residue indices
    rec_resid = [r.index for r in top.residues if r.chain.index == rec_chainid]
    resid_delta = rec_resid0 - rec_resid[0]  # adjust for the first residue index
    rec_resid_correct = [r + resid_delta for r in rec_resid] # corrected resid based on UnitProt sequence
    lig_resid = [r.index for r in top.residues if r.chain.index == lig_chainid]

    # =============================================
    #               Align Trajectory
    # =============================================
    # Select the C-alpha atoms of the receptor for alignment,
    # convert to mdtraj indices
    resid_start = max(resid_start_correct - resid_delta, rec_resid[0])  # ensure it does not go below the first residue
    resid_end = min(resid_end_correct - resid_delta, rec_resid[-1])
    ca_atoms_TM = top.select(f'name CA and resid {resid_start} to {resid_end}')

    # Align the trajectory to the C-alpha atoms of the receptor
    if align_traj:
        print("... Superimposing CA atoms of TMs ...", flush=True)
        traj.superpose(traj, 0, atom_indices=ca_atoms_TM)
        # Save the aligned trajectory
        name = coordfile.split("/")[-1]
        name = ''.join(name.split(".")[:-1])
        traj.save(f"{name}_aligned_stride{stride}_{i}.xtc")
    else:
        print("... Skipping superimposing ...", flush=True)
    # =============================================
    #               Calculate RMSD
    # =============================================
    print("... Calculating RMSD ...", flush=True)

    # Calculate RMSD of the receptor based on C-alpha atoms
    ca_atoms_rec = top.select(f'name CA and chainid {rec_chainid}')  
    rmsd_rec = md.rmsd(traj, traj, frame=0, atom_indices=ca_atoms_rec) * 10
    # Save RMSD to text file
    # concat time with rmsd for better readability
    data = np.column_stack((time, rmsd_rec))
    np.savetxt(f"rmsd_rec_traj{i}.txt", data, header="Time_(ns)  RMSD_(Angstrom)", fmt='%.6f')

    # Calculate RMSD of complex receptor and proteins
    sel = f'name CA and (chainid {rec_chainid}' 
    for j, chainid in enumerate(prot_chainid):
        sel += f' {chainid}'
    ca_atoms_complex = top.select(sel + ')')
    rmsd_complex = md.rmsd(traj, traj, frame=0, atom_indices=ca_atoms_complex) * 10
    # Save RMSD to text file
    data = np.column_stack((time, rmsd_complex))
    np.savetxt(f"rmsd_complex_traj{i}.txt", data, header="Time_(ns)  RMSD_(Angstrom)", fmt='%.6f')
    
    # Calculate RMSD of the ligand
    sel = f'chainid {lig_chainid} and not element H'
    rmsd_lig = md.rmsd(traj, traj, frame=0, atom_indices=top.select(sel)) * 10
    # Save RMSD to text file
    data = np.column_stack((time, rmsd_lig))
    np.savetxt(f"rmsd_lig_traj{i}.txt", data, header="Time_(ns)  RMSD_(Angstrom)", fmt='%.6f')

    # Store to lists for plotting later
    rmsd_rec_list.append(rmsd_rec)
    rmsd_comp_list.append(rmsd_complex)
    rmsd_lig_list.append(rmsd_lig)
    time_list.append(time)

    # =============================================
    #               Calculate RMSF
    # =============================================
    print("... Calculating RMSF ...", flush=True)
    # Calculate the mean structure (average structure)
    traj_mean = md.Trajectory(traj.xyz.mean(axis=0)[np.newaxis,:,:], top)
    rmsf = md.rmsf(traj, traj_mean, atom_indices=ca_atoms_rec) * 10
    
    # Save RMSF to text file
    data = np.column_stack((rec_resid_correct, rmsf))
    np.savetxt(f"rmsf_traj{i}.txt", data, header="Residue  RMSF_(Angstrom)", fmt='%d %.6f')

    # Store to lists for plotting later
    rmsf_list.append(rmsf)

# =============================================================
#               Plot RMSD series and histogram
# =============================================================
def plot_rmsd(rmsd_list, time_list, title='rmsd',hist_range=[0,10]):

    # Time series
    plt.figure(figsize=((6,4)))
    for i, rmsd in enumerate(rmsd_list):
        plt.plot(time_list[i], rmsd, lw=1, label=f'Traj {i}')
    plt.xlabel('Time (ns)')
    plt.ylabel('RMSD ($\AA$)')
    plt.xlim(0, max(max(time) for time in time_list))
    plt.ylim(0)
    plt.legend(loc='best',prop={'size':6})
    plt.savefig(f"{title}.png",dpi=500, bbox_inches='tight')
    # Average RMSD over all trajectories
    if len(rmsd_list) > 1:
        # truncate to the shortest trajectory length
        min_length = min(len(rmsd) for rmsd in rmsd_list)
        rmsd_list_trunc = [rmsd[:min_length] for rmsd in rmsd_list]
        time_list_trunc = [time[:min_length] for time in time_list]
        rmsd_mean = np.mean(rmsd_list_trunc, axis=0)
        
        plt.figure(figsize=((6,4)))
        plt.plot(time_list_trunc[0], rmsd_mean, '-k')
        plt.xlabel('Time (ns)')
        plt.ylabel('RMSD ($\AA$)')
        plt.xlim(0, max(time_list_trunc[0]))
        plt.ylim(0)
        plt.savefig(f"{title}_mean.png",dpi=500, bbox_inches='tight')
    
    # Histogram
    bin_size = 0.1  # bin size in Angstroms
    n_bins = int((hist_range[1] - hist_range[0])/ bin_size)  # number of bins based on range
    plt.figure(figsize=((6,4)))
    for i, rmsd in enumerate(rmsd_list):    
        plt.hist(rmsd, bins=n_bins, alpha=0.4, density=True, range=hist_range, rwidth=0.9,  label=f'Traj {i}')
    plt.xlabel('RMSD ($\AA$)')
    plt.ylabel('Probability Density')
    plt.xlim(hist_range[0], hist_range[1])
    plt.legend(loc='best',prop={'size':6})
    plt.savefig(f"{title}_hist.png",dpi=500, bbox_inches='tight')
    
    if len(rmsd_list) > 1:
        rmsd_all = np.concatenate(rmsd_list)  # concatenate all RMSD values
        plt.figure(figsize=((6,4)))
        plt.hist(rmsd_all, bins=n_bins, alpha=0.7, density=True, range=hist_range, rwidth=0.9)
        plt.xlabel('RMSD ($\AA$)')
        plt.ylabel('Probability Density')
        plt.xlim(hist_range[0], hist_range[1])
        plt.savefig(f"{title}_hist_all.png",dpi=500, bbox_inches='tight')

plot_rmsd(rmsd_rec_list, time_list, title='rmsd_rec')
plot_rmsd(rmsd_comp_list, time_list, title='rmsd_complex')
plot_rmsd(rmsd_lig_list, time_list, title='rmsd_lig', hist_range=[0,5])

# =============================================
#               Plot RMSF
# =============================================
plt.figure(figsize=((10,4)))
for i, rmsf in enumerate(rmsf_list):
    plt.plot(rec_resid_correct, rmsf, lw=1, label=f'Traj {i}')
plt.ylim(0, 5)
ymax = plt.gca().get_ylim()[1]
# Shade the receptor domains
for domain, resid_range in rec_domains.items():
    # label domain with text on plot
    if domain == 'N-term':
        x = (max(rec_resid[0]+resid_delta,resid_range[0]) + resid_range[1]) / 2 
    elif domain == 'C-term':
        x = (resid_range[0] + min(rec_resid[-1]+resid_delta, resid_range[1])) / 2
    else:
        x = (resid_range[0] + resid_range[1]) / 2 
    if resid_range[0] >= rec_resid[0] or resid_range[1] <= rec_resid[-1]:
        plt.text(x, ymax*0.9, domain, color='k', horizontalalignment='center',
                 verticalalignment='center', fontsize=6, rotation=90)    
    if domain in ['N-term', 'C-term', 'TM1', 'TM2', 'TM3', 'TM4', 'TM5', 'TM6', 'TM7','H8']:
        plt.axvspan(resid_range[0]-0.5, resid_range[1]+0.5, 
                alpha=0.2, color = domain_color_map[domain], ec='None')
plt.xlim(rec_resid_correct[0], rec_resid_correct[-1])      
plt.xlabel('Residue Index')
plt.ylabel('RMSF ($\AA$)')
plt.legend(loc='lower right',prop={'size':6})
plt.savefig("rmsf.png",dpi=500, bbox_inches='tight')

# Average RMSF over all trajectories
if len(rmsf_list) > 1:
    rmsf_mean = np.mean(rmsf_list, axis=0)
    plt.figure(figsize=((10,4)))
    plt.plot(rec_resid_correct, rmsf_mean, '-k')
    plt.ylim(0, 5)

    # Shade the receptor domains
    for domain, resid_range in rec_domains.items():
        # label domain with text on plot
        if domain == 'N-term':
            x = (max(rec_resid[0]+resid_delta,resid_range[0]) + resid_range[1]) / 2 
        elif domain == 'C-term':
            x = (resid_range[0] + min(rec_resid[-1]+resid_delta, resid_range[1])) / 2
        else:
            x = (resid_range[0] + resid_range[1]) / 2 
        if resid_range[0] >= rec_resid[0] or resid_range[1] <= rec_resid[-1]:
            plt.text(x, ymax*0.9, domain, color='k', horizontalalignment='center',
                    verticalalignment='center', fontsize=6, rotation=90)    
        if domain in ['N-term', 'C-term', 'TM1', 'TM2', 'TM3', 'TM4', 'TM5', 'TM6', 'TM7','H8']:
            plt.axvspan(resid_range[0]-0.5, resid_range[1]+0.5, 
                    alpha=0.2, color = domain_color_map[domain], ec='None')

    plt.xlim(rec_resid_correct[0], rec_resid_correct[-1])
    plt.xlabel('Residue Index')
    plt.ylabel('RMSF ($\AA$)')
    plt.savefig("rmsf_mean.png",dpi=500, bbox_inches='tight')
print("\n... Done ...", flush=True)
