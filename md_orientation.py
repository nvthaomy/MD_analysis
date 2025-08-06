#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: My Nguyen
Email: mynguyen@usc.edu
Date: March 27, 2025
Description: 
    - Align trajectories to the reference receptor structure 
    - Calculate rotation of principle axes of arrestin

Dependencies:
    - numpy
    - mdtraj

Usage:
    python align_traj.py

License: MIT License
"""
import mdtraj as md

import time as time_py
import numpy as np
from scipy.spatial.transform import Rotation
import multiprocessing as mp

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', size=9)
matplotlib.rc('axes', titlesize=9)

import seaborn as sns
import pandas as pd

from sklearn.decomposition import PCA
# =============================================
#                    Input
# =============================================
skip_save_align = True # skip saving if already generated aligned structure
coordfiles = ["../trajcat_Set_0_0_0/step7_cat_1200ns.xtc",
              "../trajcat_Set_0_0_1/step7_cat_1200ns.xtc",
              "../trajcat_Set_0_0_2/step7_cat_1200ns.xtc",
              "../trajcat_Set_0_0_3/step7_cat_1200ns.xtc",
              "../trajcat_Set_0_0_4/step7_cat_1200ns.xtc",
              "../trajcat_Set_0_0_5/step7_cat_1200ns.xtc"
              ]
# 1 ns = 10 frames
dt = 0.2 # time step in ns, 0.1 if aligning trajectories from ../trajcat_Set_0_0_0/step7_cat_1200ns.xtc, adjust accordingly if using aligned trajectories
topfile = "../trajcat_Set_0_0_0/step7_cat_notwater.pdb"
ref_file = "../trajcat_Set_0_0_0/step7_cat_notwater.pdb" # use for aligning and PC analysis of arrestin rotation
stride = 1

# receptor
rec_chainid = 0
rec_resid0 = 47 # index of the first residue of the receptor based on UnitProt sequence

# Select the C-alpha atoms of the receptor for alignment,
# TM region: resid 55-335
resid_start_correct = 55 # based on UnitProt sequence
resid_end_correct = 335  # based on UnitProt sequence

# arrestin
arr_chainid = 1 # None to skip rotation analysis on Arrestin
arr_resid0 = 1

# tasks to perform
align_traj = False
rotation_analysis = True

# if align_traj is False, aligned trajectories are used
aligned_trajs = ['./step7_cat_1200ns_notwater_aligned_stride2_0.xtc',
                 './step7_cat_1200ns_notwater_aligned_stride2_1.xtc',
                 './step7_cat_1200ns_notwater_aligned_stride2_2.xtc',
                 './step7_cat_1017ns_notwater_aligned_stride2_3.xtc',
                 './step7_cat_1134ns_notwater_aligned_stride2_4.xtc',
                 './step7_cat_749ns_notwater_aligned_stride2_5.xtc']

# =============================================
ref = md.load(ref_file)

def write_traj_with_pc_axes(traj, principal_axes_list, atom_indices_for_pca, filename, scale=3.0):
    """Write a trajectory file containing principal component axes as pseudo-atoms, ignore all water
        pca_variance: variance explained by each of the principal axes
        scale: scale factor for the principal axes to make them visible
        visualize PC in VMD as arrows: https://www.ks.uiuc.edu/Research/vmd/current/ug/node127.html"""
    traj = traj.atom_slice(traj.top.select("not water"))
    xyz = traj.xyz
    ul = traj.unitcell_lengths
    ua = traj.unitcell_angles
    # add 4 pseudo atoms to topology
    top = traj.topology.copy()
    res = top.add_residue("PC", top.add_chain("PCA"))
    for i in range(4):
        top.add_atom(f"AXI{i}", md.element.get_by_symbol('C'), res)
    for i in range(1,4):
        top.add_bond(res.atom(0), res.atom(i))
    # center of mass
    com = md.compute_center_of_mass(traj.atom_slice(atom_indices_for_pca))
    com = com[:,np.newaxis,:]  # add a new axis to com
    principal_axes = np.array(principal_axes_list) * scale + com
    # concatenate the original traj with principal axes origin (com) and endpoints, along the atom axis
    xyz_new = np.concatenate((xyz, com, principal_axes), axis=1)
    traj_new = md.Trajectory(xyz=xyz_new, topology=top, unitcell_lengths=ul, unitcell_angles=ua)
    traj_new.save(filename+'.xtc')
    traj_new[0].save(filename+'.pdb')

def compute_principal_axes(frame,atom_indices,nose_atom_indices=None,bottom_atom_indices=None):
    """Compute the principal axes (eigenvectors) position array for a given structure.
        nose_atom_indices, bottom_atom_indices: indices of atoms indicates direction where the PC1/PC2 points"""
    traj_slice = frame.atom_slice(atom_indices)
    com = md.compute_center_of_mass(traj_slice)
    positions = traj_slice.xyz - com  # Center the structure
    positions = np.squeeze(positions)
    pca = PCA(n_components=3)
    pca.fit(positions)
    pc1, pc2, pc3 = pca.components_
    # make sure that PC1 always points to the nose residues
    com_nose = md.compute_center_of_mass(frame.atom_slice(nose_atom_indices))
    nose_vector = np.squeeze(com_nose)
    # cosine of angle between PC1 and nose_vector
    cos_theta = np.dot(pc1, nose_vector) / (np.linalg.norm(pc1) * np.linalg.norm(nose_vector))
    if cos_theta < 0:
        pc1 = -pc1
    # make sure that PC2 always points to the bottom residues
    com_bottom = md.compute_center_of_mass(frame.atom_slice(bottom_atom_indices))
    bottom_vector = np.squeeze(com_bottom)
    # cosine of angle between PC2 and bottom_vector
    cos_theta = np.dot(pc2, bottom_vector) / (np.linalg.norm(pc2) * np.linalg.norm(bottom_vector))
    # change direction of PC3 to follow right hand rule
    pc3 = np.cross(pc1, pc2)
    return np.array([pc1, pc2, pc3]), pca.explained_variance_

def compute_angle(v1, v2):
    """Compute angle between two vectors in degrees."""
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))  # Clip to avoid numerical errors

def plot_hist(angles, plotname, xlabels='x', ylabels='Frequency'):
    """plot distribution of 3 angle components"""

    plt.figure(figsize=(10, 5))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.hist(angles[:,i], bins=30, alpha=0.5, color='blue')
        plt.vlines(np.mean(angles[:,i]), plt.gca().get_ylim()[0], plt.gca().get_ylim()[1], color='k', linestyle='dashed', linewidth=1)
        if isinstance(xlabels, list):
            plt.xlabel(xlabels[i])
        else:
            plt.xlabel(xlabels)
        if isinstance(ylabels, list):
            plt.ylabel(ylabels[i])
        else:
            plt.ylabel(ylabels)
    plt.tight_layout()
    plt.savefig(f"{plotname}.png", dpi=500, bbox_inches='tight')

def plot_scatter(x,y, plotname, xlabel="RMSD ($\AA$)", ylabel='y'):
    """plot scatter plot of 2 data with marginal histograms"""
    df = pd.DataFrame({xlabel: x, ylabel: y})
    g = sns.JointGrid(data=df,x=xlabel, y=ylabel)
    g.plot_joint(sns.scatterplot, alpha=0.1, color='blue')
    #g.plot_marginals(sns.histplot, kde=True, color='blue')
    g.plot_marginals(sns.kdeplot, fill=True, color='blue')
    g.savefig(f"{plotname}.png", dpi=500) 

 if align_traj:
    for i,coordfile in enumerate(coordfiles):
    # =============================================
    #               Load trajectory
    # =============================================
   
        print(f"\n... Loading Trajectory {i+1}/{len(coordfiles)} ...", flush=True)
        traj = md.load(coordfile,top=topfile,stride=stride)
        top = traj.topology
        time = np.arange(traj.n_frames) * dt * stride 
        print("... Done Loading ...", flush=True)
        print(f"Loaded {traj.n_frames} frames, {time[0]}-{time[-1]} ns, {traj.n_atoms} atoms, {traj.n_residues} residues", flush=True)

        # Recenter and apply periodic boundary conditions if molecules are outside of the box
        # print("... Applying periodic boundary condition ...", flush=True)
        # traj = traj.image_molecules()

        # Collect residue indices
        rec_resid = [r.index for r in top.residues if r.chain.index == rec_chainid]
        resid_delta = rec_resid0 - rec_resid[0]  # adjust for the first residue index
        rec_resid_correct = [r + resid_delta for r in rec_resid] # corrected resid based on UnitProt sequence
        
        # =============================================
        #               Align Trajectory
        # =============================================
        # Select the C-alpha atoms of the receptor for alignment,
        # convert to mdtraj indices
        resid_start = max(resid_start_correct - resid_delta, rec_resid[0])  # ensure it does not go below the first residue
        resid_end = min(resid_end_correct - resid_delta, rec_resid[-1])
        ca_atoms_TM = top.select(f'name CA and resid {resid_start} to {resid_end}')

        # Align the trajectory to the C-alpha atoms of the receptor
        print("... Superimposing CA atoms of TMs ...", flush=True)
        traj.superpose(ref, 0, atom_indices=ca_atoms_TM)

        # =============================================
        #               Save Trajectory
        # =============================================
        # get name of coordfile
        if not skip_save_align:
            name = coordfile.split("/")[-1]
            name = ''.join(name.split(".")[:-1])
            traj.save(f"{name}_aligned_stride{stride}_{i}.xtc")
else:
    for i,coordfile in enumerate(aligned_trajs):
        print(f"\n... Loading Aligned Trajectory {i+1}/{len(aligned_trajs)} ...", flush=True)
        traj = md.load(coordfile,top=topfile,stride=stride)
        top = traj.topology
        time = np.arange(traj.n_frames) * dt * stride
        print("... Done Loading ...", flush=True)
        print(f"Loaded {traj.n_frames} frames, {time[0]}-{time[-1]} ns, {traj.n_atoms} atoms, {traj.n_residues} residues", flush=True)

    # ===================================================================
    #           PCA analysis of Arrestin on aligned trajectory
    # ===================================================================
    if rotation_analysis and arr_chainid is not None: 
        
        print(f"... PCA analysis on Arrestin ...", flush=True)
        # Get arrestin atoms
        arr_atoms = traj.top.select(f"name CA and chainid {arr_chainid}")
        arr_resid = [r.index for r in top.residues if r.chain.index == arr_chainid]
        arr_delta = arr_resid0 - arr_resid[0]  # adjust for the first residue index
        arr_resid_correct = [r + arr_delta for r in arr_resid] # corrected resid based on UnitProt sequence
        # nose atoms in Arrestin, belongs to residues 232, 233, 235, 236, 344, 345
        nose_res = [arr_resid[arr_resid_correct.index(r)] for r in [232, 233, 235, 236, 344, 345]]
        nose_atoms = traj.top.select(f"resid {' '.join(map(str,nose_res))} and name CA")
        # bottom atoms in Arrestin, belongs to residues 27, 28, 170, 171
        bot_res = [arr_resid[arr_resid_correct.index(r)] for r in [27, 28, 170, 171]]
        bot_atoms = traj.top.select(f"resid {' '.join(map(str,bot_res))} and name CA")

        # Compute principal axes for reference structure
        reference_axes,pca_variance = compute_principal_axes(ref,arr_atoms,nose_atoms,bot_atoms)
        write_traj_with_pc_axes(ref, reference_axes, arr_atoms, 'ref_pca')

        # Compute principal axes for each frame 
        def process_frame(frame_index):
            frame = traj[frame_index]
            principal_axes,_ = compute_principal_axes(frame,arr_atoms,nose_atoms,bot_atoms)
            # Compute rotation matrix
            R = np.dot(principal_axes, reference_axes.T)
            # Convert a rotation matrix to Euler angles (XYZ convention)
            r = Rotation.from_matrix(R)
            r = r.as_euler('xyz', degrees=True)
            # compute angle deviation
            angle1 = compute_angle(principal_axes[:, 0], reference_axes[:, 0])
            angle2 = compute_angle(principal_axes[:, 1], reference_axes[:, 1])
            angle3 = compute_angle(principal_axes[:, 2], reference_axes[:, 2])
            return frame_index, principal_axes, r, [angle1, angle2, angle3]
        
        # Create multiprocessing pool
        num_workers = mp.cpu_count()  # Use all available CPUs
        print(f"... Calculating principal axes using {num_workers} CPU(s)...", flush=True)
        t0 = time_py.time()
        with mp.Pool(num_workers) as pool:
            results = pool.map(process_frame, range(traj.n_frames))
        t1 = time_py.time()
        print(f"... Done calculating in {(t1 - t0)/60:.2f} minutes ...", flush=True)
        
        # Unzip results
        frame_indices = [r[0] for r in results]
        principal_axes_list = [r[1] for r in results]
        euler_angles = np.array([r[2] for r in results])
        deviation_angles = np.array([r[3] for r in results])

        # sort everything by frame indices
        sorted_indices = np.argsort(frame_indices)
        principal_axes_list = [principal_axes_list[i] for i in sorted_indices]
        euler_angles = euler_angles[sorted_indices]
        deviation_angles = deviation_angles[sorted_indices]

        # compute rmsd of arrestin
        rmsd = md.rmsd(traj, ref, atom_indices=arr_atoms) * 10

        # save data to text file
        data = np.column_stack((time, rmsd))
        np.savetxt(f"rmsd_arr_traj{i}.txt", data, header="Time_(ns)  RMSD_(Angstrom)", fmt='%.6f')
        data = np.column_stack((time, euler_angles))
        np.savetxt(f"euler_arr_traj{i}.txt", data, header="Time_(ns)  Roll_(deg)  Pitch_(deg)  Yaw_(deg)", fmt='%.6f')
        data = np.column_stack((time, deviation_angles))
        np.savetxt(f"deviation_arr_traj{i}.txt", data, header="Time_(ns)  Angle_PC1_(deg)  Angle_PC2_(deg)  Angle_PC3_(deg)", fmt='%.6f')

        # save trajectory with principal axes
        name = coordfile.split("/")[-1]
        name = ''.join(name.split(".")[:-1])
        name = f"{name}_aligned_stride{stride}_{i}_pca"
        write_traj_with_pc_axes(traj, principal_axes_list, arr_atoms, name)

        # plot distribution of euler angles in all trajectories
        plot_hist(euler_angles, f"angle_euler_{i}", xlabels=['Roll-PC1 (deg)', 'Pitch-PC2 (deg)', 'Yaw-PC3 (deg)'])
        # plot distribution of deviation angles in all trajectories
        plot_hist(deviation_angles, f"angle_deviation_{i}", xlabels=['Angle PC1 (deg)', 'Angle PC2 (deg', 'Angle PC3 (deg)'])

        # append to all angles
        try:
            deviation_angles_all = np.concatenate((deviation_angles_all, deviation_angles), axis=0)
            euler_angles_all = np.concatenate((euler_angles_all, euler_angles), axis=0)
            rmsd_arr_all = np.concatenate((rmsd_arr_all, rmsd), axis=0)
        except:
            deviation_angles_all = deviation_angles
            euler_angles_all = euler_angles
            rmsd_arr_all = rmsd

# plot distribution of euler angles in all trajectories
plot_hist(euler_angles_all, f"angle_euler_all", xlabels=['Roll-PC1 (deg)', 'Pitch-PC2 (deg)', 'Yaw-PC3 (deg)'])
# plot distribution of deviation angles in all trajectories
plot_hist(deviation_angles_all, f"angle_deviation_all", xlabels=['Angle PC1 (deg)', 'Angle PC2 (deg', 'Angle PC3 (deg)'])

ylabels = ['Roll-PC1 (deg)', 'Pitch-PC2 (deg)', 'Yaw-PC3 (deg)']
for i in range(3):
    plot_scatter(rmsd_arr_all, euler_angles_all[:,i], f'rmsd_euler{i}', xlabel="RMSD ($\AA$)", ylabel=ylabels[i])

print("... Done ...", flush=True)   