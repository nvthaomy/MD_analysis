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
import os
import gc
import psutil
from time import time
import multiprocessing as mp
import sys

import mdtraj as md
from mdtraj.core.topology import Topology

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
matplotlib.rc('font', size=10)
matplotlib.rc('axes', titlesize=10)

# =============================================
#                    Input
# =============================================
OUTDIR = 'contacts'
coordfiles = [
                # '../../../gromacs/step5_input.pdb',
                './step7_cat_1200ns_aligned_stride2_0.xtc',
                 './step7_cat_1200ns_aligned_stride2_1.xtc',
                 './step7_cat_1200ns_aligned_stride2_2.xtc',
                 './step7_cat_722ns_aligned_stride2_3.xtc',
                 './step7_cat_730ns_aligned_stride2_4.xtc',
                 './step7_cat_573ns_aligned_stride2_5.xtc',
]
# 1 ns = 5 frames
topfile = "../trajcat_Set_0_0_0/step7_cat.pdb"
warmup = 500 # number of frames to discard before stride
stride = 20

# receptor
rec_chainid = 0
rec_resid0 = 47 # index of the first residue of the receptor based on UnitProt sequence
# ligand
lig_chainid = 4
# other proteins (G protein/ Arrestin)
prot_chainid = [2] # other protein chains if any, optional
prot_resid0 = [14] # index of the first residue of proteins based on UnitProt sequence

contact_cutoff = 0.4  # nm, cutoff for contact calculation and salt-bridge
hbond_threshold = 0.1 # threshold for h-bond probability

# custom ligand residue definition
res_to_heavy_atom = {
    'ARG-1': ['N4', 'CA7', 'C28', 'O3', 'CB6', 'CG5', 'CD5', 'NE2', 'CZ3', 'NH4', 'NH5'],
    'ARG-2': ['N3', 'CA5', 'C27', 'O2', 'CB5', 'CG4', 'CD4', 'NE1', 'CZ2', 'NH2', 'NH3'],
    'PRO-3': ['N2', 'CA4', 'C26', 'O10', 'CB4', 'CG3', 'CD3'],
    'HYP-4': ['N1', 'CA3', 'C25', 'O9', 'CB2', 'CG2', 'CD2', 'OD'],
    'GLY-5': ['N10', 'CA2', 'C24', 'O8'],
    'THI-6': ['N9', 'C18', 'C17', 'O7', 'C19', 'C20', 'C21', 'C22', 'C23', 'S'],
    'SER-7': ['N8', 'CA1', 'C16', 'O6', 'CB1', 'OG'],
    'TIC-8': ['N7', 'C7', 'C6', 'O5', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15'],
    'OIC-9': ['N6', 'C31', 'C30', 'O1', 'C32', 'C33', 'C1', 'C2', 'C3', 'C4', 'C5'],
    'ARG-10': ['N5', 'CA6', 'C29', 'O4', 'OXT', 'CB3', 'CG1', 'CD1', 'NE3', 'CZ1', 'NH1', 'NH6']
}
lig_backbone = ['N4', 'CA7', 'C28', 'O3',
                   'N3', 'CA5', 'C27', 'O2',
                   'N2', 'CA4', 'C26', 'O10',
                   'N1', 'CA3', 'C25', 'O9',
                   'N10', 'CA2', 'C24', 'O8'
                   'N9', 'C18', 'C17', 'O7',
                   'N8', 'CA1', 'C16', 'O6',
                   'N7', 'C7', 'C6', 'O5',
                   'N6', 'C31', 'C30', 'O1',
                   'N5', 'CA6', 'C29', 'O4',
                   ]
                   
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

def draw_tm_domain(ax, axis='y'):
    """ Draw the TM domain on heatmap """
    domain_end_resid = np.array([v for i, v in enumerate(rec_domains.values()) if i != len(rec_domains.values()) - 1])
    domain_end_resid = domain_end_resid[:,1]
    domain_boundaries = [rec_resid[rec_resid_correct.index(end_resid)] for end_resid in domain_end_resid]
    # Draw horizontal lines at these boundaries (add 0.5 for heatmap alignment)
    if axis == 'y':
        for idx in domain_boundaries:
            ax.axhline(idx + 0.5, color='gray', linestyle='--', linewidth=0.8)
    else:
        for idx in domain_boundaries:
            ax.axvline(idx + 0.5, color='gray', linestyle='--', linewidth=0.8)
    # Annotate domain names on the right side of the plot
    ylims = ax.get_ylim()
    xlims = ax.get_xlim()
    for i, (domain_name, (start_resid, end_resid)) in enumerate(rec_domains.items()):
        # Find the indices in rec_resid_correct for start and end
        if start_resid in rec_resid_correct and end_resid in rec_resid_correct:
            start_idx = rec_resid_correct.index(start_resid)
            end_idx = rec_resid_correct.index(end_resid)
            # Compute the center position for the label
            center = (start_idx + end_idx) / 2 + 0.5
            # Place the annotation just outside the right edge
            if axis == 'y':
                ax.text(xlims[1] + 0.1, center, domain_name, va='center', ha='left', fontsize=8, rotation=0, color='black', clip_on=False)
            else:
                ax.text(center, ylims[1] + 0.1, domain_name, va='center', ha='center', fontsize=8, rotation=0, color='black', clip_on=False)

# =============================================
#               Load trajectory
# =============================================
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

time0 = time()
print("... Loading Trajectory ...", flush=True)
for i, coordfile in enumerate(coordfiles):
    if i == 0:
        traj = md.load(coordfile,top=topfile,stride=stride)
        traj = traj[int(warmup/stride):]
    else:
        traj_ = md.load(coordfile,top=topfile,stride=stride)
        traj_ = traj_[int(warmup/stride):]
        traj = traj + traj_
top = traj.topology
print("... Done Loading ...", flush=True)
print(f"Loaded {traj.n_frames} frames, {traj.n_atoms} atoms, {traj.n_residues} residues", flush=True)

# ===============================================================================================
#          Rebuild the topology if have customized definition for ligand residues
# ===============================================================================================
lig_sidechain_aid = []
if len(res_to_heavy_atom.keys()):
    print('... Found custom residues defined for ligand ...')
    lig_res_per_atom = []
    lig_atoms = traj.top.select(f"chainid {lig_chainid}")
    # Build a mapping from atom name to residue name in res_to_heavy_atom
    atomname_to_res = {}
    for res, atomnames in res_to_heavy_atom.items():
        for atom in atomnames:
            atomname_to_res[atom] = res

    cnt = 0
    for idx in lig_atoms:
        atom = traj.top.atom(idx)
        atom_name = atom.name
        if atom_name in atomname_to_res.keys():
            res_name = atomname_to_res[atom_name]
            cnt += 1
        else:
            res_name = 'H'

        lig_res_per_atom.append(res_name)

    print(f'Matched {cnt} atoms / {len(lig_atoms)} total atoms')
    print(f'{len(atomname_to_res.keys())} heavy atoms defined')

    print('\n... Rebuilding topology ...')
    # Get the original topology and ligand chain
    orig_top = traj.top
    lig_chain = list(orig_top.chains)[lig_chainid]

    # Create a new topology
    new_top = Topology()

    # Keep track of mapping from old atoms to new atoms (if needed)
    old_to_new_atom = {}

    for chain in orig_top.chains:
        if chain.index != lig_chainid:
            # Copy chain as is
            new_chain = new_top.add_chain(chain.index)
            for res in chain.residues:
                new_res = new_top.add_residue(res.name, new_chain, res.resSeq)
                for atom in res.atoms:
                    new_atom = new_top.add_atom(atom.name, atom.element, new_res, serial=atom.serial)
                    old_to_new_atom[atom] = new_atom
        else:
            # Rebuild ligand chain using lig_res_per_atom
            new_chain = new_top.add_chain(chain.index)
            # Build the order of residues: first those in res_to_heavy_atom (in order), then 'H' at the end if present
            ordered_resnames = list(res_to_heavy_atom.keys())
            ordered_resnames.append('H')
            # Remove any possible '-'-suffixes for matching
            ordered_resnames_clean = [''.join(r.split('-')[:-1]) if '-' in r else r for r in ordered_resnames]
            # Map from original resname (with possible -atom) to clean name
            resname_map = {r: (''.join(r.split('-')[:-1]) if '-' in r else r) for r in ordered_resnames}
            # Find if 'H' is present in lig_res_per_atom
            has_H = 'H' in lig_res_per_atom
            # Assign residue numbers starting from the original first residue's resSeq
            resSeq_counter = 1 #min([res.resSeq for res in chain.residues])
            resname_to_res = {}
            resname_to_resSeq = {}
            # First, add residues in the order of res_to_heavy_atom
            for resname in ordered_resnames:
                resname_clean = resname_map[resname]
                if resname not in resname_to_res:
                    new_res = new_top.add_residue(resname_clean, new_chain, resSeq_counter)
                    resname_to_res[resname] = new_res
                    resname_to_resSeq[resname] = resSeq_counter
                    resSeq_counter += 1
            # Now, add atoms to the new topology, assigning them to the correct residue
            for i, atom in enumerate(chain.atoms):
                resname = lig_res_per_atom[i]
                new_res = resname_to_res.get(resname)
                if not atom.name in lig_backbone and atom.element.symbol != 'H':
                    lig_sidechain_aid.append(atom.index)
                new_atom = new_top.add_atom(atom.name, atom.element, new_res, serial=atom.serial)
                old_to_new_atom[atom] = new_atom

    # Copy bonds from the original topology
    for bond in orig_top.bonds:
        a1 = bond[0]
        a2 = bond[1]
        if a1 in old_to_new_atom and a2 in old_to_new_atom:
            new_top.add_bond(old_to_new_atom[a1], old_to_new_atom[a2])

    # new_top now contains the modified topology
    # You can use new_top in place of traj.top for further analysis

    # Print unique residues in chain lig_chainid and number of atoms in each residue
    lig_chain = None
    for chain in new_top.chains:
        if chain.index == lig_chainid:
            lig_chain = chain
            break

    if lig_chain is not None:
        print(f"Residues in chain {lig_chainid}:")
        for res in lig_chain.residues:
            atom_count = len(list(res.atoms))
            print(f"  Residue: {res} (name={res.name}, resid={res.index}, resSeq={res.resSeq}), Number of atoms: {atom_count}")
    else:
        print(f"Chain with index {lig_chainid} not found in new_top.")
    
    print(f"Old topology: {orig_top.n_atoms} atoms, {orig_top.n_residues} residues, {orig_top.n_chains} chains")
    print(f"New topology: {new_top.n_atoms} atoms, {new_top.n_residues} residues, {new_top.n_chains} chains")

    # Update the trajectory to use the new topology
    top = new_top
    traj.top = top
    traj.topology = top

# =============================================
#          Make list of residue pairs
# =============================================
rec_resid = [r.index for r in top.residues if r.chain.index == rec_chainid]
delta = rec_resid0 - rec_resid[0]  # adjust for the first residue index
rec_resid_correct = [r + delta for r in rec_resid]
lig_resid = sorted([r.index for r in top.residues if r.chain.index == lig_chainid and r.name != 'H'])
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

rec_atoms = top.select(f"chainid {rec_chainid}")  
lig_atoms = top.select(f"chainid {lig_chainid}")
tmp = " ".join([str(c) for c in prot_chainid])
prot_atoms = top.select(f"(chainid {tmp})")
wat_atoms = top.select("water")

# for estimating distance matrix when calculating contacts
rec_heavy_atoms = top.select(f"chainid {rec_chainid} and not element H")  
lig_heavy_atoms = top.select(f"chainid {lig_chainid} and not element H")
tmp = " ".join([str(c) for c in prot_chainid])
prot_heavy_atoms = top.select(f"(chainid {tmp}) and not element H")

# ===============================================
# Calculate contacts between ligand and receptor
# ===============================================
print("\n... Calculating rec-lig contacts ...", flush=True)
stride_ = check_memory(traj.n_frames, len(rec_heavy_atoms) * len(lig_heavy_atoms))
d,_ = md.compute_contacts(traj[::stride_], contacts = rec_lig_pairs, ignore_nonprotein=False)

# calculate the probability of contact less than the cutoff, average over frames
contact_prob = (d < contact_cutoff).mean(axis=0)

# reconstruct into distance matrix for heatmap
contact_prob = contact_prob.reshape(len(lig_resid), len(rec_resid)).T
# (receptor residues as rows, ligand residues as columns)

# find contact pairs with at least one value > threshold
rows, cols = np.where(contact_prob > 0.7)
if len(rows):
    with open(os.path.join(OUTDIR, "contact_rec-lig.txt"), "w") as f:
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
gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.3)
ax = plt.subplot(gs[0])
cbar_ax = plt.subplot(gs[1])
ax = sns.heatmap(contact_prob, cmap="YlGnBu", cbar_kws={'label': 'Contact Probability'},
                 vmin=0, vmax=1, ax=ax, cbar_ax=cbar_ax)
ax.set_ylabel("Receptor Residue Index")
ax.set_yticks(np.arange(0, len(rec_resid), 5) + 0.5)
ax.set_yticklabels(rec_resid_correct[::5])
ax.set_xlabel("Ligand Residue Index")
ax.set_xticks(np.arange(0, len(lig_resid), 1) + 0.5)
ax.set_xticklabels(np.arange(0, len(lig_resid)) + 1)
draw_tm_domain(ax)
plt.savefig(os.path.join(OUTDIR, "contact_rec-lig.png"),dpi=500, bbox_inches='tight')

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
    with open(os.path.join(OUTDIR, "contact_rec-rec.txt"), "w") as f:
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
gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.3)
ax = plt.subplot(gs[0])
cbar_ax = plt.subplot(gs[1])
ax = sns.heatmap(contact_prob, cmap="YlGnBu", cbar_kws={'label': 'Contact Probability'},
                 vmin=0, vmax=1, ax=ax, cbar_ax=cbar_ax)
ax.set_aspect('equal')
ax.set_ylabel("Receptor Residue Index") # row
ax.set_yticks(np.arange(0, len(rec_resid), 5) + 0.5)
ax.set_yticklabels(rec_resid_correct[::5])
ax.set_xlabel("Receptor Residue Index") # column
ax.set_xticks(np.arange(0, len(rec_resid), 5) + 0.5)
ax.set_xticklabels(rec_resid_correct[::5])
draw_tm_domain(ax)
plt.savefig(os.path.join(OUTDIR, "contact_rec-rec.png"),dpi=500, bbox_inches='tight')

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
        with open(os.path.join(OUTDIR, "contact_rec-prot.txt"), "w") as f:
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
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.3)
    ax = plt.subplot(gs[0])
    cbar_ax = plt.subplot(gs[1])
    ax = sns.heatmap(contact_prob, cmap="YlGnBu", cbar_kws={'label': 'Contact Probability'},
                    vmin=0, vmax=1, ax=ax, cbar_ax=cbar_ax)
    ax.set_aspect('equal')
    ax.set_ylabel("Receptor Residue Index")
    ax.set_yticks(np.arange(0, len(rec_resid), 5) + 0.5)
    ax.set_yticklabels(rec_resid_correct[::5])
    ax.set_xlabel("Protein Residue Index")
    ax.set_xticks(np.arange(0, len(prot_resid), 5) + 0.5)
    ax.set_xticklabels(prot_resid_correct[::5])
    draw_tm_domain(ax)
    plt.savefig(os.path.join(OUTDIR, "contact_rec-prot.png"),dpi=500, bbox_inches='tight')

    del d, contact_prob
    gc.collect()

# ========================================================
#           Salt-bridge between receptor residues
# ========================================================
print(f"\n... Calculating salt-bridges between receptor residues ...", flush=True)

acidic_residues = ["ASP", "GLU"]
basic_residues = ["LYS", "ARG", "HIS"]
resid1 = [r for r in rec_resid if top.residue(r).name in acidic_residues]
resid2 = [r for r in rec_resid if top.residue(r).name in basic_residues]
ion_pairs = [(r1, r2) for r1 in resid1 for r2 in resid2]

d,pairs = md.compute_contacts(traj, contacts = ion_pairs, scheme='sidechain-heavy')
# get the resid actually in pairs
resid1 = np.unique(pairs[:,0]).tolist()
resid2 = np.unique(pairs[:,1]).tolist()

# calculate the probability of contact less than the cutoff, average over frames
contact_prob = (d < contact_cutoff).mean(axis=0)

# reconstruct into distance matrix for heatmap
tmp = np.zeros((len(resid1), len(resid2)))
resid1_name = [''] * len(resid1)
resid2_name = [''] * len(resid2)

for i, (r1, r2) in enumerate(pairs):
    row = resid1.index(r1)
    col = resid2.index(r2)
    tmp[row, col] = contact_prob[i]
    r1_correct = rec_resid_correct[rec_resid.index(r1)]
    r2_correct = rec_resid_correct[rec_resid.index(r2)]
    if resid1_name[row] == '':
        resid1_name[row] = f"{top.residue(r1).name}{r1_correct}"
    if resid2_name[col] == '':
        resid2_name[col] = f"{top.residue(r2).name}{r2_correct}"

contact_prob = tmp

# find contact pairs with at least one value > threshold
rows, cols = np.where(contact_prob > 0.5)
if len(rows):
    with open(os.path.join(OUTDIR, "saltbridge_rec-rec.txt"), "w") as f:
        f.write('# REC_resname REC_resid_index_correct REC_resid_mdtraj REC_resname REC_resid_index_correct REC_resid_mdtraj\n')
        f.write(f'# threshold: contact probability > 0.5, distance < {contact_cutoff}\n')
        for i, row in enumerate(rows):
            col = cols[i]
            r1 = resid1[row]
            r1_correct = rec_resid_correct[rec_resid.index(r1)]
            r2 = resid2[col]
            r2_correct = rec_resid_correct[rec_resid.index(r2)]
            f.write(f"{top.residue(r1).name} {r1_correct} {r1} ")
            f.write(f"{top.residue(r2).name} {r2_correct} {r2} \n")
            
# plot on heatmap
plt.figure(figsize=(8, int(8*len(resid1)/len(resid2))))
gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.3)
ax = plt.subplot(gs[0])
cbar_ax = plt.subplot(gs[1])
ax = sns.heatmap(contact_prob, cmap="YlGnBu", cbar_kws={'label': 'Contact Probability'},
                 vmin=0, vmax=1, ax=ax, cbar_ax=cbar_ax)
ax.set_aspect('equal')
ax.set_ylabel("Receptor Acidic Residue") # row
ax.set_yticks(np.arange(0, len(resid1_name)) + 0.5)
ax.set_yticklabels(resid1_name, rotation=0)
ax.set_xlabel("Receptor Basic Residue") # column
ax.set_xticks(np.arange(0, len(resid2_name)) + 0.5)
ax.set_xticklabels(resid2_name, rotation=90)
plt.savefig(os.path.join(OUTDIR, "saltbridge_rec-rec.png"),dpi=500, bbox_inches='tight')

del d, contact_prob, tmp
gc.collect()

# =====================================================================
#           Salt-bridge between receptor and protein residues
# =====================================================================
if len(prot_chainid):
    print(f"\n... Calculating salt-bridges between receptor and protein ...", flush=True)
    rec_interface_atoms = md.compute_neighbors(traj, cutoff=0.6, query_indices=prot_atoms, haystack_indices=rec_atoms)
    rec_interface_atoms = np.unique(np.concatenate(rec_interface_atoms))
    rec_interface_resid = [top.atom(aid).residue.index for aid in rec_interface_atoms]

    prot_interface_atoms = md.compute_neighbors(traj, cutoff=0.6, query_indices=rec_atoms, haystack_indices=prot_atoms)
    prot_interface_atoms = np.unique(np.concatenate(prot_interface_atoms))
    prot_interface_resid = [top.atom(aid).residue.index for aid in prot_interface_atoms]

    resid1_1 = [r for r in rec_resid if top.residue(r).name in acidic_residues and r in rec_interface_resid]
    resid2_1 = [r for r in prot_resid if top.residue(r).name in basic_residues and r in prot_interface_resid]
    pairs_1 = [(r1, r2) for r1 in resid1_1 for r2 in resid2_1]

    resid1_2 = [r for r in rec_resid if top.residue(r).name in basic_residues and r in rec_interface_resid]
    resid2_2 = [r for r in prot_resid if top.residue(r).name in acidic_residues and r in prot_interface_resid]
    pairs_2 = [(r1, r2) for r1 in resid1_2 for r2 in resid2_2]

    ion_pairs = pairs_1 + pairs_2

    if len(ion_pairs):
        d,pairs = md.compute_contacts(traj, contacts = ion_pairs, scheme='sidechain-heavy')

        # get the resid actually in pairs
        resid1 = np.unique(pairs[:,0]).tolist()
        resid2 = np.unique(pairs[:,1]).tolist()

        # calculate the probability of contact less than the cutoff, average over frames
        contact_prob = (d < contact_cutoff).mean(axis=0)

        # reconstruct into distance matrix for heatmap
        tmp = np.zeros((len(resid1), len(resid2)))
        resid1_name = [''] * len(resid1)
        resid2_name = [''] * len(resid2)

        for i, (r1, r2) in enumerate(pairs):
            row = resid1.index(r1)
            col = resid2.index(r2)
            tmp[row, col] = contact_prob[i]
            r1_correct = rec_resid_correct[rec_resid.index(r1)]
            r2_correct = prot_resid_correct[prot_resid.index(r2)]
            if resid1_name[row] == '':
                resid1_name[row] = f"{top.residue(r1).name}{r1_correct}"
            if resid2_name[col] == '':
                resid2_name[col] = f"{top.residue(r2).name}{r2_correct}"

        contact_prob = tmp

        # find contact pairs with at least one value > threshold
        rows, cols = np.where(contact_prob > 0.5)
        if len(rows):
            with open(os.path.join(OUTDIR, "saltbridge_rec-prot.txt"), "w") as f:
                f.write('# REC_resname REC_resid_index_correct REC_resid_mdtraj PROT_resname PROT_resid_index_correct PROT_resid_mdtraj\n')
                f.write(f'# threshold: contact probability > 0.5, distance < {contact_cutoff}\n')
                for i, row in enumerate(rows):
                    col = cols[i]
                    r1 = resid1[row]
                    r1_correct = rec_resid_correct[rec_resid.index(r1)]
                    r2 = resid2[col]
                    r2_correct = prot_resid_correct[prot_resid.index(r2)]
                    f.write(f"{top.residue(r1).name} {r1_correct} {r1} ")
                    f.write(f"{top.residue(r2).name} {r2_correct} {r2} \n")
                    
        # plot on heatmap
        plt.figure(figsize=(8, int(8*len(resid1)/len(resid2))))
        gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.3)
        ax = plt.subplot(gs[0])
        cbar_ax = plt.subplot(gs[1])
        ax = sns.heatmap(contact_prob, cmap="YlGnBu", cbar_kws={'label': 'Contact Probability'},
                        vmin=0, vmax=1, ax=ax, cbar_ax=cbar_ax)
        ax.set_aspect('equal')
        ax.set_ylabel("Receptor Residue") # row
        ax.set_yticks(np.arange(0, len(resid1_name)) + 0.5)
        ax.set_yticklabels(resid1_name, rotation=0)
        ax.set_xlabel("Protein Residue") # column
        ax.set_xticks(np.arange(0, len(resid2_name)) + 0.5)
        ax.set_xticklabels(resid2_name, rotation=90)
        plt.savefig(os.path.join(OUTDIR, "saltbridge_rec-prot.png"),dpi=500, bbox_inches='tight')

        del d, contact_prob, tmp
        gc.collect()
    else:
        print('No salt-bridge found')
# ========================================================
#           Salt-bridge between receptor and ligand
# ========================================================
print(f"\n... Calculating salt-bridges between receptor and ligand ...", flush=True)

# Find pocket atoms
rec_pocket_atoms = md.compute_neighbors(traj, cutoff=0.6, query_indices=lig_atoms, haystack_indices=rec_atoms)
rec_pocket_atoms = np.unique(np.concatenate(rec_pocket_atoms))
rec_pocket_resid = [top.atom(aid).residue.index for aid in rec_pocket_atoms]

resid1_1 = [r for r in rec_resid if top.residue(r).name in acidic_residues and r in rec_pocket_resid]
resid2_1 = [r for r in lig_resid if top.residue(r).name in basic_residues]
pairs_1 = [(r1, r2) for r1 in resid1_1 for r2 in resid2_1]

resid1_2 = [r for r in rec_resid if top.residue(r).name in basic_residues and r in rec_pocket_resid]
resid2_2 = [r for r in lig_resid if top.residue(r).name in acidic_residues]
pairs_2 = [(r1, r2) for r1 in resid1_2 for r2 in resid2_2]

ion_pairs = pairs_1 + pairs_2

if len(ion_pairs):
    d,pairs = md.compute_contacts(traj, contacts = ion_pairs, scheme='closest-heavy')

    # get the resid actually in pairs
    resid1 = np.unique(pairs[:,0]).tolist()
    resid2 = np.unique(pairs[:,1]).tolist()
    
    # calculate the probability of contact less than the cutoff, average over frames
    contact_prob = (d < contact_cutoff).mean(axis=0)

    # reconstruct into distance matrix for heatmap
    tmp = np.zeros((len(resid1), len(resid2)))
    resid1_name = [''] * len(resid1)
    resid2_name = [''] * len(resid2)

    for i, (r1, r2) in enumerate(pairs):
        row = resid1.index(r1)
        col = resid2.index(r2)
        tmp[row, col] = contact_prob[i]
        r1_correct = rec_resid_correct[rec_resid.index(r1)]
        r2_correct = r2
        if resid1_name[row] == '':
            resid1_name[row] = f"{top.residue(r1).name}{r1_correct}"
        if resid2_name[col] == '':
            for a in top.residue(r2).atoms:
                resname = str(a).split('-')[0]
                break
            resid2_name[col] = resname

    contact_prob = tmp

    # find contact pairs with at least one value > threshold
    rows, cols = np.where(contact_prob > 0.5)
    if len(rows):
        with open(os.path.join(OUTDIR, "saltbridge_rec-lig.txt"), "w") as f:
            f.write('# REC_resname REC_resid_index_correct REC_resid_mdtraj LIG_resname LIG_resid_mdtraj\n')
            f.write(f'# threshold: contact probability > 0.5, distance < {contact_cutoff}\n')
            for i, row in enumerate(rows):
                col = cols[i]
                r1 = resid1[row]
                r1_correct = rec_resid_correct[rec_resid.index(r1)]
                r2 = resid2[col]
                f.write(f"{top.residue(r1).name} {r1_correct} {r1} ")
                f.write(f"{resid2_name[col]} {r2} \n")
                
    # plot on heatmap
    plt.figure(figsize=(int(8 * max(0.5,len(resid2)/len(resid1))), 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.3)
    ax = plt.subplot(gs[0])
    cbar_ax = plt.subplot(gs[1])
    ax = sns.heatmap(contact_prob, cmap="YlGnBu", cbar_kws={'label': 'Contact Probability'},
                    vmin=0, vmax=1, ax=ax, cbar_ax=cbar_ax)
    ax.set_aspect('equal')
    ax.set_ylabel("Receptor Residue") # row
    ax.set_yticks(np.arange(0, len(resid1_name)) + 0.5)
    ax.set_yticklabels(resid1_name, rotation=0)
    ax.set_xlabel("Ligand Residue") # column
    ax.set_xticks(np.arange(0, len(resid2_name)) + 0.5)
    ax.set_xticklabels(resid2_name, rotation=90)
    plt.savefig(os.path.join(OUTDIR, "saltbridge_rec-lig.png"),dpi=500, bbox_inches='tight')

    del d, contact_prob, tmp
    gc.collect()
else:
    print('No salt-bridge found')

# ==================
# Calculate H-bond
# ==================

def process_frame(i):
    """Function to compute hydrogen bonds for a given frame index."""
    # Find water molecules within 0.35 nm of receptor pocket
    wat_pocket_atoms = md.compute_neighbors(traj[i], cutoff=0.35, query_indices=rec_pocket_atoms, haystack_indices=wat_atoms)

    pocket_atoms = np.concatenate([rec_pocket_atoms, lig_atoms, wat_pocket_atoms[0]])
    
    # Slice trajectory to only include water in pocket
    traj_pocket_i = traj[i].atom_slice(pocket_atoms)
    
    # Save first frame for inspection
    if i == 0:
        traj_pocket_i.save(os.path.join(OUTDIR, "step7_cat_pocket.pdb"))
    
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
rec_wat_tmp = [r[0] for r in results if len(r[0]) > 0]
if len(rec_wat_tmp):
    rec_wat_tmp = np.concatenate(rec_wat_tmp, axis=0)
rec_lig_tmp = [r[1] for r in results if len(r[1]) > 0]
if len(rec_lig_tmp):
    rec_lig_tmp = np.concatenate(rec_lig_tmp, axis=0)

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
    with open(os.path.join(OUTDIR, "hbonds_rec-wat.txt"), "w") as f:
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
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.3)
    ax = plt.subplot(gs[0])
    cbar_ax = plt.subplot(gs[1])
    ax = sns.heatmap(hbond_prob, cmap="YlGnBu", cbar_kws={'label': 'H-bond Probability'},
                    vmin=0, vmax=1, ax=ax, cbar_ax=cbar_ax)
    ax.set_aspect('equal')
    ax.set_ylabel("Receptor Atom")
    ax.set_yticks(np.arange(0, len(rec_name)) + 0.5)
    ax.set_yticklabels(rec_name, rotation=0)
    ax.set_xlabel("Water Atom")
    ax.set_xticks(np.arange(0, len(wat_name)) + 0.5)
    ax.set_xticklabels(wat_name, rotation=90)
    plt.savefig(os.path.join(OUTDIR, "hbonds_rec-wat.png"),dpi=500, bbox_inches='tight')

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

if len(rec_lig_hbonds):
    rec_aid = np.unique(rec_lig_hbonds[:,0]).tolist() # receptor atoms in h-bonds
    lig_aid = np.unique(rec_lig_hbonds[:,1]).tolist() # water atoms in h-bonds
    rec_name = [''] * len(rec_aid)
    lig_name = [top.atom(a) for a in lig_aid]
    hbond_prob = np.zeros((len(rec_aid), len(lig_aid))) # matrix, probabilities of h-bonds

    # calculate the distance between acceptor and donor atoms for each hbond
    rec_lig_hbonds_dist = md.compute_distances(traj, rec_lig_hbonds_unique)
    rec_lig_hbonds_dist_mean = rec_lig_hbonds_dist.mean(axis=0)
    rec_lig_hbonds_dist_std = rec_lig_hbonds_dist.std(axis=0)

    with open(os.path.join(OUTDIR, "hbonds_rec-lig.txt"), "w") as f:
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
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.3)
    ax = plt.subplot(gs[0])
    cbar_ax = plt.subplot(gs[1])
    ax = sns.heatmap(hbond_prob, cmap="YlGnBu", cbar_kws={'label': 'H-bond Probability'},
                    vmin=0, vmax=1, ax=ax, cbar_ax=cbar_ax)
    ax.set_aspect('equal')
    ax.set_ylabel("Receptor Atom")
    ax.set_yticks(np.arange(0, len(rec_name)) + 0.5)
    ax.set_yticklabels(rec_name, rotation=0)
    ax.set_xlabel("Ligand Atom")
    ax.set_xticks(np.arange(0, len(lig_name)) + 0.5)
    ax.set_xticklabels(lig_name, rotation=90)
    plt.savefig(os.path.join(OUTDIR, "hbonds_rec-lig.png"),dpi=500, bbox_inches='tight')

time1 = time()
print(f"\nTotal time: {(time1 - time0)/60:.2f} min", flush=True)
print("... Done ...", flush=True)
