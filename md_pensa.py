import os, sys
import numpy as np
import MDAnalysis as mda
import mdtraj as md
from time import time

sys.path.append('/scratch1/mynguyen/B2R/pensa_B2RHOE')
import pensa_utils as pu
from pensa_utils import *
pu.CWDIR = os.getcwd()

from pensa.preprocessing import load_selection, \
    extract_coordinates, extract_coordinates_combined, \
    extract_aligned_coordinates, extract_combined_grid

from pensa.comparison import *
from pensa.features import *
from pensa.statesinfo import *
from pensa.dimensionality import *
from pensa.clusters import *

warmup = 500 # number of frames to discard 
# 1 ns = 5 frames
preprocess = False

# Simulation A - Arrestin
root_dir_a = "/scratch1/mynguyen/B2R/B2R_Arr_HOE/charmm-gui-5191556142/MD/Trajectories/"
ref_file_a =  root_dir_a+'/../../gromacs/step5_input.psf'
pdb_file_a =  root_dir_a+'/trajcat_Set_0_0_0/step7_cat.pdb'
trj_file_a = [root_dir_a+'/analysis/step7_cat_1200ns_aligned_stride10_0.xtc',
              root_dir_a+'/analysis/step7_cat_1200ns_aligned_stride10_1.xtc',
              root_dir_a+'/analysis/step7_cat_1200ns_aligned_stride10_2.xtc',
              root_dir_a+'/analysis/step7_cat_1017ns_aligned_stride10_3.xtc',
              root_dir_a+'/analysis/step7_cat_1134ns_aligned_stride10_4.xtc',
              root_dir_a+'/analysis/step7_cat_749ns_aligned_stride10_5.xtc',
              ]
# Simulation B - Gq
root_dir_b = "/scratch1/mynguyen/B2R/B2R_Gq_HOE/charmm-gui-5130069506/MD/Trajectories/"
ref_file_b =  root_dir_b+'/../../gromacs/step5_input.psf'
pdb_file_b =  root_dir_b+'/trajcat_Set_0_0_0/step7_cat.pdb'
trj_file_b = [root_dir_b+'/analysis/step7_cat_1200ns_aligned_stride10_0.xtc',
              root_dir_b+'/analysis/step7_cat_1200ns_aligned_stride10_1.xtc',
              root_dir_b+'/analysis/step7_cat_956ns_aligned_stride10_2.xtc',
              root_dir_b+'/analysis/step7_cat_722ns_aligned_stride10_3.xtc',
              root_dir_b+'/analysis/step7_cat_730ns_aligned_stride10_4.xtc',
              root_dir_b+'/analysis/step7_cat_573ns_aligned_stride10_5.xtc'
              ]
# Base for the selection string for each simulation
sel_rec = "(segid PROA and resid 47:357) or (segid HETA)"
sel_base_a = f"(not name H*) and ({sel_rec})"
sel_base_b = f"(not name H*) and ({sel_rec})"
sel_base_ion_a = f"({sel_rec}) or segid IONS"
sel_base_ion_b = f"({sel_rec}) or segid IONS"
# Names of the output files
out_name_a = "traj/condition-a"
out_name_b = "traj/condition-b"
out_name_combined="traj/combined"

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
TM_domains = [val for key, val in rec_domains.items() if key.startswith('TM')]

# ==========================================================
#                       Preprocessing
# ==========================================================
def print_segid(filename):
    # Load the universe
    u = mda.Universe(filename)
    # Get list of segids in order 
    segids = []
    atom_names = []
    for atom in u.atoms:
        if atom.segid not in segids:
            segids.append(atom.segid)
        if atom.name not in atom_names:
            atom_names.append(atom.name)
    # Get the segid and number of atoms and residues
    segment_info = {}
    for seg in u.segments:
        num_atoms = len(seg.atoms)
        num_residues = len(seg.residues)
        segment_info[seg.segid] = (num_atoms, num_residues)
    # Print each segid
    print('segid\tatom count\tres count')
    for segid in segids:
        print(f"{segid}\t{segment_info[segid][0]}\t\t{segment_info[segid][1]}")
    print('Unique atom names:', atom_names)
    return segment_info


# Check the segid of the input files
print('\nUnique segid in ensemble a:')
seg_info_a = print_segid(ref_file_a)
print('\nUnique segid in ensemble b:')
seg_info_b = print_segid(ref_file_b)

# water selection
sel_base_water_a = f"{sel_rec} or (segid TIP3)"
sel_base_water_b = f"{sel_rec} or (segid TIP3)"


# TM domains selection
sel_base_tm_a = f"(not name H*) and (segid PROA and ("
for domain in TM_domains:
    sel_base_tm_a += f"resid {domain[0]}:{domain[1]} or "

    
sel_base_tm_a = sel_base_tm_a[:-4] + "))"
sel_base_tm_b = sel_base_tm_a

# Prepare subfolders
for subdir in ['traj', 'features', 'plots', 'vispdb', 'pca', 'clusters', 'results']:
    if not os.path.exists(subdir):
        os.makedirs(subdir)

if preprocess:
    print("\n... Extracting coordinates ...", flush=True)
    start_time = time()
    # Extract the coordinates of the receptor from the trajectory
    # receptor heavy atoms only
    extract_coordinates(ref_file_a, pdb_file_a, trj_file_a, out_name_a+"_receptor", sel_base_a, start_frame=warmup)
    extract_coordinates(ref_file_b, pdb_file_b, trj_file_b, out_name_b+"_receptor", sel_base_b, start_frame=warmup)
    # receptor heavy atoms in TM domains
    extract_coordinates(ref_file_a, pdb_file_a, trj_file_a, out_name_a+"_tm", sel_base_tm_a, start_frame=warmup)
    extract_coordinates(ref_file_b, pdb_file_b, trj_file_b, out_name_b+"_tm", sel_base_tm_b, start_frame=warmup)
    # receptor and water
    extract_coordinates(ref_file_a, pdb_file_a, trj_file_a, out_name_a+"_water", sel_base_water_a, start_frame=warmup)
    extract_coordinates(ref_file_b, pdb_file_b, trj_file_b, out_name_b+"_water", sel_base_water_b, start_frame=warmup)
    # receptor and ion
    # extract_coordinates(ref_file_a, pdb_file_a, trj_file_a, out_name_a+"_ion", sel_base_ion_a, start_frame=warmup)
    # extract_coordinates(ref_file_b, pdb_file_b, trj_file_b, out_name_b+"_ion", sel_base_ion_b, start_frame=warmup)

    print("\n... Aligning coordinates ...", flush=True)
    # Align the coordinates of the ensemble a/b to the average of ensemble b for comparing water feature later
    extract_aligned_coordinates(
        out_name_a+"_water.gro", out_name_a+"_water.xtc",
        out_name_b+"_water.gro", out_name_b+"_water.xtc",
        xtc_aligned = out_name_a+"_water_aligned.xtc",
        pdb_outname = out_name_b+"_water_average.pdb"
    )

    extract_aligned_coordinates(
        out_name_b+"_water.gro", out_name_b+"_water.xtc",
        out_name_b+"_water.gro", out_name_b+"_water.xtc",
        xtc_aligned = out_name_b+"_water_aligned.xtc",
        pdb_outname = out_name_b+"_water_average.pdb"
    )

    extract_aligned_coordinates(
        out_name_a+"_receptor.gro", out_name_a+"_receptor.xtc",
        out_name_b+"_receptor.gro", out_name_b+"_receptor.xtc",
        xtc_aligned = out_name_a+"_receptor_aligned.xtc",
        pdb_outname = out_name_b+"_receptor_average.pdb"
    )

    extract_aligned_coordinates(
        out_name_b+"_receptor.gro", out_name_b+"_receptor.xtc",
        out_name_b+"_receptor.gro", out_name_b+"_receptor.xtc",
        xtc_aligned = out_name_b+"_receptor_aligned.xtc",
        pdb_outname = out_name_b+"_receptor_average.pdb"
    )
    
    extract_combined_grid(
        out_name_a+"_water.gro",  out_name_a+"_water_aligned.xtc",
        out_name_b+"_water.gro",  out_name_b+"_water_aligned.xtc",
        atomgroup="OH2", write_grid_as="TIP3P",
        out_name="ab_grid_",
        use_memmap=True, memmap='traj/combined.mymemmap'
    )
    end_time = time()
    print(f"Time elapsed: {(end_time - start_time)/60:.2f} minutes")
else:
    print('\n... Skipping preprocessing trajectories ...', flush=True)
# ==========================================================
#                      Get features
# ==========================================================

print("\n... Loading structure features on receptor ...", flush=True)
# Load - backbone torsions: 'bb-torsions', - backbone C-alpha distances: 'bb-distances', and - sidechain torsions: 'sc-torsions'
sim_a_rec = read_structure_features(
    "traj/condition-a_receptor.gro",
    "traj/condition-a_receptor.xtc")
sim_a_rec_feat, sim_a_rec_data = sim_a_rec
for k in sim_a_rec_data.keys():
    print(k, sim_a_rec_data[k].shape)

sim_b_rec = read_structure_features(
    "traj/condition-b_receptor.gro",
    "traj/condition-b_receptor.xtc")
sim_b_rec_feat, sim_b_rec_data = sim_b_rec
for k in sim_b_rec_data.keys():
    print(k, sim_b_rec_data[k].shape)

print("\n... Loading structure features on receptor TM regions ...", flush=True)
# angles in cosine and sine
sim_a_tmr_feat, sim_a_tmr_data = read_structure_features(
    "traj/condition-a_tm.gro",
    "traj/condition-a_tm.xtc",
    cossin=True
)
sim_b_tmr_feat, sim_b_tmr_data = read_structure_features(
    "traj/condition-b_tm.gro",
    "traj/condition-b_tm.xtc",
    cossin=True
)
# ==========================================================
#                Relative Entropy on 3 features 
# ==========================================================
# JSD: Jensen–Shannon distance ranges from 0 to 1, where 0 is obtained
# for identical distributions and 1 is obtained for a pair of completely
# different distributions
# KLD: Kullback–Leibler divergence, upper is unbounded

calc_srel(sim_a_rec_feat, sim_a_rec_data, sim_b_rec_feat, sim_b_rec_data, feat='bb-torsions', output_prefix='receptor_bbtors')
calc_srel(sim_a_rec_feat, sim_a_rec_data, sim_b_rec_feat, sim_b_rec_data, feat='sc-torsions', output_prefix='receptor_sctors')
calc_srel(sim_a_rec_feat, sim_a_rec_data, sim_b_rec_feat, sim_b_rec_data, feat='bb-distances', output_prefix='receptor_bbdist')

# ==========================================================
#            State-Specific Information on features 
# ==========================================================
# SSI measure ISSI(xf) quantifies the degree to which con-
# formational state transitions of feature xf signal information about
# the ensembles i and j or the transitions between them.
# SSI: ranges from 0 bit to 1 bit, where 0 bit represents no shared information 
# and 1 bit represents maximal shared information between the ensemble
# (transitions) and the features.
_, _, bbtors_states, bbtors_res_feat_a, bbtors_res_data_a, bbtors_res_feat_b, bbtors_res_data_b = calc_ssi_feat_ensem(sim_a_rec_feat, sim_a_rec_data, sim_b_rec_feat, sim_b_rec_data, feat='bb-torsions', output_prefix='receptor_bbtors')
_, _, sctors_states, sctors_res_feat_a, sctors_res_data_a, sctors_res_feat_b, sctors_res_data_b = calc_ssi_feat_ensem(sim_a_rec_feat, sim_a_rec_data, sim_b_rec_feat, sim_b_rec_data, feat='sc-torsions', output_prefix='receptor_sctors')

# ==========================================================
#                PCA analysis + Clustering on TM domains
# ==========================================================
calc_pca_cluster(sim_a_tmr_feat, sim_a_tmr_data, sim_b_tmr_feat, sim_b_tmr_data, 
                feat='bb-torsions', output_prefix='combined_bbtors_tmr', 
                label_a='arrestin', label_b='Gq', num_clusters=5)

calc_pca_cluster(sim_a_tmr_feat, sim_a_tmr_data, sim_b_tmr_feat, sim_b_tmr_data, 
                feat='sc-torsions', output_prefix='combined_sctors_tmr',
                label_a='arrestin', label_b='Gq', num_clusters=5)
# this is slow
# calc_pca_cluster(sim_a_tmr_feat, sim_a_tmr_data, sim_b_tmr_feat, sim_b_tmr_data, 
#                 feat='bb-distances', output_prefix='combined_bbdist_tmr',
#                 label_a='arrestin', label_b='Gq', num_clusters=5)

# ==========================================================
#                     Water density
# ==========================================================
print("\n... Calculating water density ...", flush=True)
# don't use pre-generated grid
struc = "traj/condition-a_water.gro"
xtc = "traj/condition-a_water_aligned.xtc"
water_feat_a, water_data_a = read_water_features(
    structure_input = struc,
    xtc_input = xtc,
    top_waters = 10, 
    atomgroup = "OH2", 
    write_grid_as="TIP3P",
    out_name = "features/condition_a"
)

struc = "traj/condition-b_water.gro"
xtc = "traj/condition-b_water_aligned.xtc"
water_feat_b, water_data_b = read_water_features(
    structure_input = struc,
    xtc_input = xtc,
    top_waters = 10, 
    atomgroup = "OH2", 
    write_grid_as="TIP3P",
    out_name = "features/condition_b"
)

# in output PDB, B-factor column represent water occupancy frequency
# featurize sites common to both ensembles
grid = "ab_grid_OH2_density.dx"
struc = "traj/condition-a_water.gro"
xtc = "traj/condition-a_water_aligned.xtc"
water_feat_a, water_data_a = read_water_features(
    structure_input = struc,
    xtc_input = xtc,
    top_waters = 10, # featurize the top 10 most probable water sites
    atomgroup = "OH2", # water oxygen
    grid_input = grid,
    out_name = "features/condition_a_shared_grid"
)

struc = "traj/condition-b_water.gro"
xtc = "traj/condition-b_water_aligned.xtc"
water_feat_b, water_data_b = read_water_features(
    structure_input = struc,
    xtc_input = xtc,
    top_waters = 10, 
    atomgroup = "OH2", 
    grid_input = grid,
    out_name = "features/condition_b_shared_grid"
)

# ============ State-Specific Information on water density ==============
# get discrete states
water_states = get_discrete_states(
    water_data_a['WaterPocket_Distr'],
    water_data_b['WaterPocket_Distr'],
    discretize='gaussian', pbc=True
)

# define state boundaries: Water occupancy (is water in the pocket or not?) 
# is described as a binary feature with the values 0 or 1
water_occup_states = [[[-0.1, 0.5, 1.1]]] * len(water_states)

# comparing the occupancy between the two conditions
# pbc (bool, optional) – If true, the apply periodic bounary corrections on angular distribution inputs. The input for periodic correction must be radians. The default is True.
# h2o (bool, optional) – If true, the apply periodic bounary corrections for spherical angles with different periodicities. The default is False.

water_sites, water_ssi_by_occup,_,_,_,_,_ = calc_ssi_feat_ensem(water_feat_a, water_data_a, water_feat_b, water_data_b, states=water_occup_states, 
                    feat='WaterPocket_OccupDistr', verbose=True, h2o=False, pbc=False, output_prefix='water_occupancy')
water_sites, water_ssi_by_orient,_,_,_,_,_ = calc_ssi_feat_ensem(water_feat_a, water_data_a, water_feat_b, water_data_b, states=water_states, 
                    feat='WaterPocket_Distr', verbose=True, h2o=True, pbc=True, output_prefix='water_orientation')
# save to csv
data = np.array([water_sites, water_ssi_by_occup, water_ssi_by_orient]).T
out_filename = "water_sites"
np.savetxt(
    'results/'+out_filename+'_ssi.csv',
    data, fmt='%s', delimiter=',',
    header='Water_site, SSI_occupancy, SSI_orientation'
)

# write visualization PDB file
ref_filename = "features/condition_a_shared_grid_WaterSites.pdb"
out_filename = "vispdb/water_by_occupancy_ssi.pdb"
water_names = [str(n) for n in water_names_by_occup]
with open(ref_filename, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith('HETATM'):
            # get water name
            water_name = line[17:20].strip()
            # get the index of the water name in the list of water names
            wat_id = water_names.index(water_name)
            # get the occupancy value from the SSI value
            ssi_value = water_ssi_by_occup[wat_id]
            # write the ssi value to the B-factor column in the charactor 60-66 right aligned
            lines[i] = line[:60] + f"{ssi_value:6.2f}" + line[66:]
# write the modified lines to the output file
with open(out_filename, 'w') as f:
    f.writelines(lines)

ref_filename = "features/condition_a_shared_grid_WaterSites.pdb"
out_filename = "vispdb/water_by_orientation_ssi.pdb"
water_names = [str(n) for n in water_names_by_orient]
with open(ref_filename, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith('HETATM'):
            # get water name
            water_name = line[17:20].strip()
            # get the index of the water name in the list of water names
            wat_id = water_names.index(water_name)
            # get the occupancy value from the SSI value
            ssi_value = water_ssi_by_orient[wat_id]
            # write the ssi value to the B-factor column in the charactor 60-66 right aligned
            lines[i] = line[:60] + f"{ssi_value:6.2f}" + line[66:]
# write the modified lines to the output file
with open(out_filename, 'w') as f:
    f.writelines(lines)

# compute feature-feature communication between the water sites
calc_ssi_feat_feat(water_feat_a['WaterPocket_Distr'], water_data_a['WaterPocket_Distr'], 
                    water_feat_b['WaterPocket_Distr'], water_data_b['WaterPocket_Distr'], 
                    water_states, threshold=-100, output_prefix='water-pairs', h2o=True)

# ==========================================================
#                 Co-SSI between 2 features
# ==========================================================
all_feat_a = water_feat_a['WaterPocket_Distr'] + bbtors_res_feat_a
all_feat_b = water_feat_b['WaterPocket_Distr'] + bbtors_res_feat_b
all_data_a = np.array(list(water_data_a['WaterPocket_Distr']) + list(bbtors_res_data_a), dtype=object)
all_data_b = np.array(list(water_data_b['WaterPocket_Distr']) + list(bbtors_res_data_b), dtype=object)
all_states = water_states + bbtors_states

calc_ssi_feat_feat(all_feat_a, all_data_a, all_feat_b, all_data_b, all_states, output_prefix="water-bbtor-pairs")
calc_cossi(all_feat_a, all_data_a, all_feat_b, all_data_b, all_states, output_prefix="water-bbtor-pairs")

all_feat_a = bbdist_res_feat_a + bbtors_res_feat_a
all_feat_b = bbdist_res_feat_b + bbtors_res_feat_b
all_data_a = np.array(list(bbdist_res_data_a) + list(bbtors_res_data_a), dtype=object)
all_data_b = np.array(list(bbdist_res_data_b) + list(bbtors_res_data_b), dtype=object)
all_states = bbdist_states + bbtors_states

calc_ssi_feat_feat(all_feat_a, all_feat_b, all_data_a, all_data_b, all_states, output_prefix="bbdist-bbtor-pairs")
calc_cossi(all_feat_a, all_data_a, all_feat_b, all_data_b, all_states, output_prefix="bbdist-bbtor-pairs")

# ==========================================================
#                     Ions density
# ==========================================================
# print("\n... Calculating ion density ...", flush=True)
# struc = "traj/condition-a_ion.gro"
# xtc = "traj/condition-a_ion.xtc"
# atom_feat, atom_data = read_atom_features(
#     structure_input = struc,
#     xtc_input = xtc,
#     top_atoms = 10,
#     atomgroup = "SOD",
#     element = "Na",
#     out_name = "features/condition_a_Na",
#     write = True
# )

# struc = "traj/condition-b_ion.gro"
# xtc = "traj/condition-b_ion.xtc"
# atom_feat, atom_data = read_atom_features(
#     structure_input = struc,
#     xtc_input = xtc,
#     top_atoms = 10,
#     atomgroup = "SOD",
#     element = "Na",
#     out_name = "features/condition_b_Na",
#     write = True
# )
print('\n... Done ...', flush=True)
