#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: My Nguyen
Email: mynguyen@usc.edu
Date: March 25, 2025
Description: Analyzes MD trajectories to calculate residue contacts, 
             hydrogen bonds, and generate heatmaps for visualization.

Dependencies:
    - numpy
    - matplotlib
    - seaborn
    - mdtraj
    - psutil

Usage:
    python mdtraj_contact.py

License: MIT License
"""

import gc
import psutil
from time import time
import multiprocessing as mp
import sys

import mdtraj as md
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rc('font', size=9)
matplotlib.rc('axes', titlesize=9)

# =============================================
#                    Input
# =============================================
coordfile = "step7_cat_868ns.xtc"
# 1 ns = 10 frames
topfile = "step5_input_Set_0_0_0.pdb"
warmup = 500 # number of frames to discard before stride
stride = 1

# receptor
rec_chainid = 0
rec_resid0 = 47 # index of the first residue of the receptor based on UnitProt sequence
# ligand
lig_chainid = 1
# other proteins (G protein/ Arrestin)
prot_chainid = [] # other protein chains if any, optional
prot_resid0 = [] # index of the first residue of proteins based on UnitProt sequence

contact_cutoff = 0.4  # nm, cutoff for contact calculation and salt-bridge
hbond_threshold = 0.1 # threshold for h-bond probability

# ===========================================================

def check_memory(nrows, ncols, dtype=np.float32, buffer=0.9):
    """
    Check if the matrix size exceeds the maximum memory limit.
    Return step size to reduce memory usage if necessary.
    
    nrows and ncols: dimensions of the matrix
    dtype: data type of the matrix (default is np.float32)
    buffer: fraction of memory to use 

    return:
    stride: int, step size to reduce memory usage if matrix size exceeds max_memory

    """
    max_memory = psutil.virtual_memory().available # in bytes
    element_size = np.dtype(dtype).itemsize
    matrix_size = nrows * ncols * element_size 
    if matrix_size  > max_memory * buffer:
        stride = int(np.ceil((nrows * ncols * element_size) / (max_memory * buffer)))
        print(f"Matrix size ({matrix_size/(1024**3):.2f} GB) exceeds {buffer*100:.0f}% of memory limit {max_memory/(1024**3):.2f} GB, using stride {stride} to reduce memory usage", flush=True)
    else:
        print(f"Matrix size ({matrix_size/(1024**3):.2f} GB) is within {buffer*100:.0f}% of memory limit {max_memory/(1024**3):.2f} GB", flush=True)
        stride = 1
    return stride

# =============================================
#               Load trajectory
# =============================================
time0 = time()
print("... Loading Trajectory ...", flush=True)
traj = md.load(coordfile,top=topfile,stride=stride)
traj = traj[int(warmup/stride):]
top = traj.topology
print("... Done Loading ...", flush=True)
print(f"Loaded {traj.n_frames} frames, {traj.n_atoms} atoms, {traj.n_residues} residues", flush=True)

# =============================================
#          Make list of residue pairs
# =============================================
rec_resid = [r.index for r in top.residues if r.chain.index == rec_chainid]
delta = rec_resid0 - rec_resid[0]  # adjust for the first residue index
rec_resid_correct = [r + delta for r in rec_resid]
lig_resid = [r.index for r in top.residues if r.chain.index == lig_chainid]
prot_resid = []
prot_resid_correct = []
delta = []
for ic,c in enumerate(prot_chainid):
    tmp = []
    for r in top.residues:
        if r.chain.index == c:
            prot_resid.append(r.index)
            tmp.append(r.index)
    delta.append(prot_resid0[ic] - tmp[0])  # adjust for the first residue index
    prot_resid_correct.extend([r + delta[ic] for r in tmp])  # correct residue index for each protein chain

rec_lig_pairs = [(l, r) for l in lig_resid for r in rec_resid]
rec_rec_pairs = [(r1, r2) for i, r1 in enumerate(rec_resid) for r2 in rec_resid[i+2:]]
rec_prot_pairs = [(p,r) for p in prot_resid for r in rec_resid]

rec_atoms = traj.top.select(f"chainid {rec_chainid}")  
lig_atoms = traj.top.select(f"chainid {lig_chainid}")
wat_atoms = traj.top.select("water")

# for estimating distance matrix when calculating contacts
rec_heavy_atoms = traj.top.select(f"chainid {rec_chainid} and not element H")  
lig_heavy_atoms = traj.top.select(f"chainid {lig_chainid} and not element H")
tmp = " ".join([str(c) for c in prot_chainid])
prot_heavy_atoms = traj.top.select(f"(chainid {tmp}) and not element H")

# ===============================================
# Calculate contacts between ligand and receptor
# ===============================================
print("\n... Calculating rec-lig contacts ...", flush=True)
stride_ = check_memory(traj.n_frames, len(rec_heavy_atoms) * len(lig_heavy_atoms))
d,_ = md.compute_contacts(traj[::stride_], contacts = rec_lig_pairs)

# calculate the probability of contact less than the cutoff, average over frames
contact_prob = (d < contact_cutoff).mean(axis=0)

# reconstruct into distance matrix for heatmap
contact_prob = contact_prob.reshape(len(lig_resid), len(rec_resid)).T
# (receptor residues as rows, ligand residues as columns)

# find contact pairs with at least one value > threshold
rows, cols = np.where(contact_prob > 0.7)
if len(rows):
    with open("contact_rec-lig.txt", "w") as f:
        f.write('# REC_resname REC_resid_index_correct REC_resid_mdtraj LIG_resname\n')
        f.write(f'# threshold: contact probability > 0.7, distance < {contact_cutoff}\n')
        for i, row in enumerate(rows):
                col = cols[i]
                rec = rec_resid[row]
                rec_correct = rec_resid_correct[rec_resid.index(rec)]
                lig = lig_resid[col]
                f.write(f"{top.residue(rec).name} {rec_correct} {rec} {top.residue(lig)}\n")

# plot on heatmap
plt.figure(figsize=(8, 20))
ax = sns.heatmap(contact_prob, cmap="YlGnBu", cbar_kws={'label': 'Contact Probability'},
                 vmin=0, vmax=1)
#ax.set_aspect('equal')
plt.ylabel("Receptor Residue Index", fontsize=10)
ax.set_yticks(np.arange(0, len(rec_resid), 5) + 0.5)
ax.set_yticklabels(rec_resid_correct[::5])
plt.xlabel("Ligand Residue Index", fontsize=10)
ax.set_xticks(np.arange(0, len(lig_resid), 1) + 0.5)
ax.set_xticklabels(np.arange(0, len(lig_resid)) + 1)
plt.savefig("contact_rec-lig.png",dpi=500, bbox_inches='tight')

del d, contact_prob
gc.collect()

# ===================================================
#       Calculate contacts receptor residues
# ===================================================
print("\n... Calculating rec-rec contacts ...", flush=True)
stride_ = check_memory(traj.n_frames, len(rec_heavy_atoms) * len(rec_heavy_atoms))
d,pairs = md.compute_contacts(traj[::stride_], contacts = rec_rec_pairs)

# calculate the probability of contact less than the cutoff, average over frames
contact_prob = (d < contact_cutoff).mean(axis=0)

# reconstruct into distance matrix for heatmap
tmp = np.zeros((len(rec_resid), len(rec_resid)))
for i, (r1, r2) in enumerate(pairs):
    tmp[rec_resid.index(r1),rec_resid.index(r2)] = contact_prob[i]
    
contact_prob = tmp

# find contact pairs with at least one value > threshold
rows, cols = np.where(contact_prob > 0.7)
if len(rows):
    with open("contact_rec-rec.txt", "w") as f:
        f.write('# REC_resname REC_resid_index_correct REC_resid_mdtraj REC_resname REC_resid_index_correct REC_resid_mdtraj\n')
        f.write(f'# threshold: contact probability > 0.7, distance < {contact_cutoff}\n')
        for i, row in enumerate(rows):
                col = cols[i]
                rec1 = rec_resid[row]
                rec1_correct = rec_resid_correct[rec_resid.index(rec1)]
                rec2 = rec_resid[col]
                rec2_correct = rec_resid_correct[rec_resid.index(rec2)]
                f.write(f"{top.residue(rec1).name} {rec1_correct} {rec1} ")
                f.write(f"{top.residue(rec2).name} {rec2_correct} {rec2} \n")

# plot on heatmap
plt.figure(figsize=(12,10))
ax = sns.heatmap(contact_prob, cmap="YlGnBu", cbar_kws={'label': 'Contact Probability'},
                 vmin=0, vmax=1)
ax.set_aspect('equal')
plt.ylabel("Receptor Residue Index", fontsize=10) # row
ax.set_yticks(np.arange(0, len(rec_resid), 5) + 0.5)
ax.set_yticklabels(rec_resid_correct[::5])
plt.xlabel("Receptor Residue Index", fontsize=10) # column
ax.set_xticks(np.arange(0, len(rec_resid), 5) + 0.5)
ax.set_xticklabels(rec_resid_correct[::5])
plt.savefig("contact_rec-rec.png",dpi=500, bbox_inches='tight')

del d, contact_prob, tmp
gc.collect()

# ========================================================
#  Calculate contacts between receptor and other proteins
# ========================================================
if len(rec_prot_pairs):
    print("\n... Calculating rec-prot contacts ...", flush=True)
    # stride traj so that number of frames < 2000
    stride_ = check_memory(traj.n_frames, len(rec_heavy_atoms) * len(prot_heavy_atoms))
    d,_ = md.compute_contacts(traj[::stride_], contacts = rec_prot_pairs)

    # calculate the probability of contact less than the cutoff, average over frames
    contact_prob = (d < contact_cutoff).mean(axis=0)

    # reconstruct into distance matrix for heatmap
    contact_prob = contact_prob.reshape(len(prot_resid), len(rec_resid)).T
    # (receptor residues as rows, protein residues as columns)

    # find contact pairs with at least one value > threshold
    rows, cols = np.where(contact_prob > 0.7)
    if len(rows):   
        with open("contact_rec-prot.txt", "w") as f:
            f.write('# REC_resname REC_resid_index_correct REC_resid_mdtraj PROT_resname PROT_resid_index_correct PROT_resid_mdtraj\n')
            f.write(f'# threshold: contact probability > 0.7, distance < {contact_cutoff}\n')
            for i, row in enumerate(rows):
                col = cols[i]
                rec = rec_resid[row]
                rec_correct = rec_resid_correct[rec_resid.index(rec)]
                prot = prot_resid[col]
                prot_correct = prot_resid_correct[prot_resid.index(prot)]
                f.write(f"{top.residue(rec).name} {rec_correct} {rec} ")
                f.write(f"{top.residue(prot).name} {prot_correct} {prot}\n")
                
    # plot on heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(contact_prob, cmap="YlGnBu", cbar_kws={'label': 'Contact Probability'},
                    vmin=0, vmax=1)
    ax.set_aspect('equal')
    plt.ylabel("Receptor Residue Index", fontsize=10)
    ax.set_yticks(np.arange(0, len(rec_resid), 5) + 0.5)
    ax.set_yticklabels(rec_resid_correct[::5])
    plt.xlabel("Protein Residue Index", fontsize=10)
    ax.set_xticks(np.arange(0, len(prot_resid), 5) + 0.5)
    ax.set_xticklabels(prot_resid_correct[::5])
    plt.savefig("contact_rec-prot.png",dpi=500, bbox_inches='tight')

    del d, contact_prob
    gc.collect()

# ==================
# Calculate H-bond
# ==================
# Find pocket atoms
rec_pocket_atoms = md.compute_neighbors(traj, cutoff=0.6, query_indices=lig_atoms, haystack_indices=rec_atoms)
rec_pocket_atoms = np.unique(np.concatenate(rec_pocket_atoms))

def process_frame(i):
    """Function to compute hydrogen bonds for a given frame index."""
    # Find water molecules within 0.35 nm of receptor pocket
    wat_pocket_atoms = md.compute_neighbors(traj[i], cutoff=0.35, query_indices=rec_pocket_atoms, haystack_indices=wat_atoms)

    pocket_atoms = np.concatenate([rec_pocket_atoms, lig_atoms, wat_pocket_atoms[0]])
    
    # Slice trajectory to only include water in pocket
    traj_pocket_i = traj[i].atom_slice(pocket_atoms)
    
    # Save first frame for inspection
    if i == 0:
        traj_pocket_i.save("step7_cat_pocket.pdb")
    
    # Map new atom index after slicing to original atom index
    map_aid = {a_new: a_old for a_new, a_old in enumerate(pocket_atoms)}  
    
    # Compute hydrogen bonds
    hbonds = md.baker_hubbard(traj_pocket_i, freq=0.1, exclude_water=False) 
    
    rec_wat_tmp, rec_lig_tmp = [], []
    for h in hbonds:
        h0, h2 = map_aid[h[0]], map_aid[h[2]]
        
        # Count hbonds between receptor and water
        if h0 in rec_pocket_atoms and h2 in wat_pocket_atoms[0]:
            rec_wat_tmp.append([h0, h2])
        elif h0 in wat_pocket_atoms[0] and h2 in rec_pocket_atoms:
            rec_wat_tmp.append([h2, h0])
        
        # Count hbonds between receptor and ligand
        if h0 in rec_pocket_atoms and h2 in lig_atoms:
            rec_lig_tmp.append([h0, h2])
        elif h0 in lig_atoms and h2 in rec_pocket_atoms:
            rec_lig_tmp.append([h2, h0])
    
    return np.array(rec_wat_tmp), np.array(rec_lig_tmp)

# Create multiprocessing pool
num_workers = mp.cpu_count()  # Use all available CPUs
print(f"\n... Calculating h-bonds using {num_workers} CPU(s)...", flush=True)
t0 = time()
with mp.Pool(num_workers) as pool:
    results = pool.map(process_frame, range(traj.n_frames))
t1 = time()
print(f"... Done calculating h-bonds in {(t1 - t0)/60:.2f} minutes ...", flush=True)

# Collect results
rec_wat_tmp = np.concatenate([r[0] for r in results if len(r[0]) > 0], axis=0)
rec_lig_tmp = np.concatenate([r[1] for r in results if len(r[1]) > 0], axis=0)

# ========================================================
#       Calculate H-bond between receptor and water
# ========================================================

# get unique atom pairs involved in h-bonds
rec_wat_hbonds_unique, rec_wat_hbonds_counts = np.unique(rec_wat_tmp, axis=0, return_counts=True)
rec_wat_hbonds_prob = rec_wat_hbonds_counts/ traj.n_frames

# only return bonds with probability > threshold 0.1
rec_wat_hbonds = rec_wat_hbonds_unique[rec_wat_hbonds_prob > hbond_threshold]
rec_wat_hbonds_prob = rec_wat_hbonds_prob[rec_wat_hbonds_prob > hbond_threshold]

print('\nFound %d H-bonds between receptor and pocket water' % len(rec_wat_hbonds), flush=True)
rec_aid = np.unique(rec_wat_hbonds[:,0]).tolist() # receptor atoms in h-bonds
wat_aid = np.unique(rec_wat_hbonds[:,1]).tolist() # water atoms in h-bonds
rec_name = [''] * len(rec_aid)
wat_name = [top.atom(a) for a in wat_aid]
hbond_prob = np.zeros((len(rec_aid), len(wat_aid))) # matrix, probabilities of h-bonds
print(f'involving {len(rec_aid)} unique receptor atoms')

if len(rec_wat_hbonds):
    with open("hbonds_rec-wat.txt", "w") as f:
        f.write('# REC_atom REC_resname REC_resid_Correct REC_resid_mdtraj WAT_atom Probability\n')
        f.write(f'# threshold: probability > {hbond_threshold}\n')
        print("Receptor \t\t-- \t\tWater \t\tProbability", flush=True)
        for i,(a,a_wat) in enumerate(rec_wat_hbonds):
            an = top.atom(a).name
            rn = top.atom(a).residue.name
            rid = top.atom(a).residue.index
            rid_correct = rec_resid_correct[rec_resid.index(rid)]
            prob = rec_wat_hbonds_prob[i]
            f.write(f"{an} {rn} {rid_correct} {rid} {top.atom(a_wat)} {prob:.2f}\n")
            print(f"{rn}{rid_correct}-{an} \t\t-- \t\t{top.atom(a_wat)} \t\t{prob:.2f}", flush=True)
        
            irow = rec_aid.index(a)
            icol = wat_aid.index(a_wat)
            hbond_prob[irow, icol] = prob  # store the probability of h-bond for each receptor atom and water atom
            if rec_name[irow] == '':
                rec_name[irow] = f"{rn}{rid_correct}-{an}"  
    
    # sum of probability for each receptor atom
    sum = hbond_prob.sum(axis=1, keepdims=True)  # sum over water atoms
    wat_name.append('Sum')  # add 'Sum' to the water atom labels
    # concatenate to the last column of hbond_prob
    hbond_prob = np.concatenate((hbond_prob, sum), axis=1)

    # plot on heatmap
    plt.figure(figsize=(15, 10))
    ax = sns.heatmap(hbond_prob, cmap="YlGnBu", cbar_kws={'label': 'H-bond Probability'},
                    vmin=0, vmax=1)
    ax.set_aspect('equal')
    plt.ylabel("Receptor Atom", fontsize=10)
    ax.set_yticks(np.arange(0, len(rec_name)) + 0.5)
    ax.set_yticklabels(rec_name, rotation=0)
    plt.xlabel("Water Atom", fontsize=10)
    ax.set_xticks(np.arange(0, len(wat_name)) + 0.5)
    ax.set_xticklabels(wat_name, rotation=90)
    plt.savefig("hbonds_rec-wat.png",dpi=500, bbox_inches='tight')

# ========================================================
#       Calculate H-bond between receptor and ligand
# ========================================================
# get unique atom pairs involved in h-bonds
rec_lig_hbonds_unique, rec_lig_hbonds_counts = np.unique(rec_lig_tmp, axis=0, return_counts=True)
rec_lig_hbonds_prob = rec_lig_hbonds_counts / traj.n_frames

# only return bonds with probability > threshold 0.1
rec_lig_hbonds = rec_lig_hbonds_unique[rec_lig_hbonds_prob > hbond_threshold]
rec_lig_hbonds_prob = rec_lig_hbonds_prob[rec_lig_hbonds_prob > hbond_threshold]

print('\nFound %d H-bonds between receptor and ligand' % len(rec_lig_hbonds), flush=True)
rec_aid = np.unique(rec_lig_hbonds[:,0]).tolist() # receptor atoms in h-bonds
lig_aid = np.unique(rec_lig_hbonds[:,1]).tolist() # water atoms in h-bonds
rec_name = [''] * len(rec_aid)
lig_name = [top.atom(a) for a in lig_aid]
hbond_prob = np.zeros((len(rec_aid), len(lig_aid))) # matrix, probabilities of h-bonds

if len(rec_lig_hbonds):
    # calculate the distance between acceptor and donor atoms for each hbond
    rec_lig_hbonds_dist = md.compute_distances(traj, rec_lig_hbonds_unique)
    rec_lig_hbonds_dist_mean = rec_lig_hbonds_dist.mean(axis=0)
    rec_lig_hbonds_dist_std = rec_lig_hbonds_dist.std(axis=0)

    with open("hbonds_rec-lig.txt", "w") as f:
        f.write('# REC_atom REC_resname REC_resid_Correct REC_resid_mdtraj LIG_atom LIG_resname Probability Distance_mean Distance_std\n')
        f.write(f'# threshold: probability > {hbond_threshold}\n')
        print("Receptor \t\t-- \t\tLigand \t\t\t\tProbability \tDistance", flush=True)
        for i,(a,a_lig) in enumerate(rec_lig_hbonds):
            an = top.atom(a).name
            rn = top.atom(a).residue.name
            rid = top.atom(a).residue.index
            rid_correct = rec_resid_correct[rec_resid.index(rid)]
            prob = rec_lig_hbonds_prob[i]
            d_mean = rec_lig_hbonds_dist_mean[i]
            d_std = rec_lig_hbonds_dist_std[i]

            f.write(f"{an} {rn} {rid_correct} {rid} {top.atom(a_lig).name} {top.atom(a_lig).residue} {prob:.2f} {d_mean:.2f} {d_std:.2f}\n")
            print(f"{rn}{rid_correct}-{an} \t\t-- \t\t{top.atom(a_lig)} \t\t\t\t{prob:.2f} \t{d_mean:.2f} +/- {d_std:.2f}", flush=True)
        
            irow = rec_aid.index(a)
            icol = lig_aid.index(a_lig)
            hbond_prob[irow, icol] = prob  # store the probability of h-bond for each receptor atom and water atom
            if rec_name[irow] == '':
                rec_name[irow] = f"{rn}{rid_correct}-{an}"  

    # plot on heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(hbond_prob, cmap="YlGnBu", cbar_kws={'label': 'H-bond Probability'},
                    vmin=0, vmax=1)
    ax.set_aspect('equal')
    plt.ylabel("Receptor Atom", fontsize=10)
    ax.set_yticks(np.arange(0, len(rec_name)) + 0.5)
    ax.set_yticklabels(rec_name, rotation=0)
    plt.xlabel("Ligand Atom", fontsize=10)
    ax.set_xticks(np.arange(0, len(lig_name)) + 0.5)
    ax.set_xticklabels(lig_name, rotation=90)
    plt.savefig("hbonds_rec-lig.png",dpi=500, bbox_inches='tight')

# ========================================================
#           Salt-bridge between receptor and ligand
# ========================================================
print(f"\n... Calculating salt-bridges between receptor and ligand ...", flush=True)
# Define acidic and basic residues for salt-bridge calculation
acidic_residues = ["ASP", "GLU"]
basic_residues = ["LYS", "ARG", "HIS"]

# Get name of acidic and basic atom names
acidic_atoms = []
basic_atoms = []
tmp = []
for r in top.residues:
    if not r.name in tmp:
        tmp.append(r.name)
        if r.name in acidic_residues:
            for a in r.atoms:
                if a.element.symbol in ["O", "N"] and a.is_sidechain:  # only consider oxygen and nitrogen atoms
                    acidic_atoms.append(a.name)
        elif r.name in basic_residues:
            for a in r.atoms:
                if a.element.symbol in ["O", "N"] and a.is_sidechain:  # only consider oxygen and nitrogen atoms
                    basic_atoms.append(a.name)

# Get indices of acidic and basic atoms
lig_acidic_atoms = [atom.index for atom in traj.topology.atoms if atom.residue.index in lig_resid and atom.residue.name in acidic_residues and atom.name in acidic_atoms]
lig_basic_atoms = [atom.index for atom in traj.topology.atoms if atom.residue.index in lig_resid and atom.residue.name in basic_residues and atom.name in basic_atoms]

rec_acidic_atoms = [atom.index for atom in traj.topology.atoms if atom.index in rec_pocket_atoms and atom.residue.name in acidic_residues and atom.name in acidic_atoms]
rec_basic_atoms = [atom.index for atom in traj.topology.atoms if atom.index in rec_pocket_atoms and atom.residue.name in basic_residues and atom.name in basic_atoms]

pairs1 = [[a, b] for a in lig_acidic_atoms for b in rec_basic_atoms]
pairs2 = [[a, b] for a in lig_basic_atoms for b in rec_acidic_atoms]
pairs = pairs1 + pairs2

d = md.compute_distances(traj, pairs)

# calculate the probability of contact less than the cutoff, average over frames
sb_prob = (d < contact_cutoff).mean(axis=0)

# make a list of actual atoms in pairs
lig_aid = []
rec_aid = []
if len(pairs1):
    lig_aid.extend(np.unique(np.array(pairs1)[:,0]).tolist())
    rec_aid.extend(np.unique(np.array(pairs1)[:,1]).tolist())

if len(pairs2):
    lig_aid.extend(np.unique(np.array(pairs2)[:,0]).tolist())
    rec_aid.extend(np.unique(np.array(pairs2)[:,1]).tolist())

# reconstruct into distance matrix for heatmap
tmp = np.zeros((len(rec_aid), len(lig_aid)))
# (rec atoms as rows, lig atoms as columns)
for i, (a_lig, a_rec) in enumerate(pairs):
    tmp[rec_aid.index(a_rec),lig_aid.index(a_lig)] = sb_prob[i]
sb_prob = tmp

rec_name = [''] * len(rec_aid)
lig_name = [top.atom(a) for a in lig_aid]

with open("saltbridge_rec-lig.txt", "w") as f:
    print("Receptor\t\t -- \tLigand \tProbability", flush=True)
    f.write('# LIG_atom LIG_resname REC_atom REC_resname REC_resid_index_correct REC_resid_mdtraj probability\n')
    f.write(f'# threshold: contact probability > 0.5, distance < {contact_cutoff}\n')        
    for i,(a_lig,a_rec) in enumerate(pairs):
        col = lig_aid.index(a_lig)
        row = rec_aid.index(a_rec)
        rec = top.atom(a_rec).residue.index
        rec_correct = rec_resid_correct[rec_resid.index(rec)]
        lig = top.atom(a_lig).residue.index
        prob = sb_prob[row,col]
        if prob > 0.5:
            f.write(f"{top.atom(a_lig)} {top.residue(lig)} {top.atom(a_rec).name} {top.residue(rec).name} {rec_correct} {rec} {prob}\n")
            print(f"{top.residue(rec).name}{rec_correct}-{top.atom(a_rec).name}\t\t -- \t {top.atom(a_lig)} \t{prob:.2f}", flush=True)
        if rec_name[row] == '':
            rec_name[row] = f"{top.residue(rec).name}{rec_correct}-{top.atom(a_rec).name}"

# plot on heatmap
plt.figure(figsize=(8, int(8*len(rec_name)/len(lig_name))))
ax = sns.heatmap(sb_prob, cmap="YlGnBu", cbar_kws={'label': 'Salt-bridge Probability'},
                 vmin=0, vmax=1)
ax.set_aspect('equal')
plt.ylabel("Receptor Atom", fontsize=10)
ax.set_yticks(np.arange(0, len(rec_name)) + 0.5)
ax.set_yticklabels(rec_name, rotation=0)
plt.xlabel("Ligand Atom", fontsize=10)
ax.set_xticks(np.arange(0, len(lig_name)) + 0.5)
ax.set_xticklabels(lig_name, rotation=90)
plt.savefig("saltbridge_rec-lig.png",dpi=500, bbox_inches='tight')


time1 = time()
print(f"\nTotal time: {(time1 - time0)/60:.2f} min", flush=True)
print("... Done ...", flush=True)