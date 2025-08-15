import os, sys
import numpy as np
import MDAnalysis as mda
import mdtraj as md

from pensa.preprocessing import load_selection, \
    extract_coordinates, extract_coordinates_combined, \
    extract_aligned_coordinates, extract_combined_grid

from pensa.comparison import *
from pensa.features import *
from pensa.statesinfo import *
from pensa.dimensionality import *
from pensa.clusters import *

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

from time import time
CWDIR = '.'

def calc_srel(sim_a_feat, sim_a_data, sim_b_feat, sim_b_data, 
              feat='bb-torsions', output_prefix='receptor_bbtors', ref_filename="traj/condition-a_receptor.gro"):
    """
    Calculate the relative entropy (Jensen-Shannon distance and Kullback-Leibler divergence)
    between the distributions of a given feature (default: 'bb-torsions') for two simulation ensembles.

    JSD: Jensen–Shannon distance ranges from 0 to 1, where 0 is obtained
    for identical distributions and 1 is obtained for a pair of completely different distributions
    KLD: Kullback–Leibler divergence, upper is unbounded

    This function performs the following:
    - Computes the relative entropy metrics (JSD and KLD) for the specified feature between two ensembles.
    - Prints the most different features based on JSD.
    - Plots and saves the maximum JSD per residue for visualization.
    - Saves the results to CSV files.

    Parameters
    ----------
    sim_a_feat : dict
        Dictionary containing features for simulation A (from read_structure_features). Keys are feature names, values are feature arrays.
    sim_a_data : dict
        Dictionary containing data arrays for simulation A (from read_structure_features). Keys are feature names, values are data arrays.
    sim_b_feat : dict
        Dictionary containing features for simulation B (from read_structure_features). Keys are feature names, values are feature arrays.
    sim_b_data : dict
        Dictionary containing data arrays for simulation B (from read_structure_features). Keys are feature names, values are data arrays.
    feat : str
        The feature to analyze, options are 'bb-torsions', 'bb-distances', 'sc-torsions'
    output_prefix: str
        Prefix / name of output files
    ref_filename: str
        Path of structure file to visualize srel 

    Returns
    -------
    feat_names : list
        Names of the features analyzed.
    jsd : np.ndarray
        Jensen-Shannon distances for each feature.
    kld_ab : np.ndarray
        Kullback-Leibler divergence from A to B for each feature.
    kld_ba : np.ndarray
        Kullback-Leibler divergence from B to A for each feature.
    """
    start_time = time()
    print(f"\n... Calculating relative entropy of {feat} ...", flush=True)
    relen = relative_entropy_analysis(
        sim_a_feat[feat], sim_b_feat[feat],
        sim_a_data[feat], sim_b_data[feat],
        bin_num=10, verbose=False
    )
    feat_names, jsd, kld_ab, kld_ba = relen

    # print out the most different features
    sf = sort_features(feat_names, jsd)
    print('Feature\t\t\t JS distance')
    for f in sf[:10]: print(f"{f[0]}\t\t\t {f[1]}")

    # plot the maximum deviation of the features related to a certain residue
    # The following function also writes the maximum Jensen-Shannon distance per residue 
    # in the “B factor” field of a PDB file

    if feat == 'bb-distances':
        vis = distances_visualization(
            feat_names, jsd, os.path.join(CWDIR, f"plots/{output_prefix}_jsd.pdf"),
            vmin = 0.0, vmax = 1.0, cbar_label='JSD')
    else:
        vis = residue_visualization(
            feat_names, jsd, os.path.join(CWDIR, ref_filename),
            os.path.join(CWDIR, f"plots/{output_prefix}_jsd.pdf"),
            os.path.join(CWDIR, f"vispdb/{output_prefix}_jsd.pdb"),
            y_label=f'max. JS dist. of {feat}'
        )
    
    np.savetxt(
        os.path.join(CWDIR, f"results/{output_prefix}_relen.csv"),
        np.array(relen).T, fmt='%s', delimiter=',',
        header='Name, JSD(A,B), KLD(A,B), KLD(B,A)'
    )
    np.savetxt(
        os.path.join(CWDIR, f"results/{output_prefix}_jsd.csv"),
        np.array(vis).T, fmt='%s', delimiter=',',
        header='Residue, max. JSD(A,B)'
    )
    end_time = time()
    print(f"Time elapsed: {(end_time - start_time)/60:.2f} minutes")
    return feat_names, jsd, kld_ab, kld_ba

def calc_ssi_feat_ensem(sim_a_feat, sim_a_data, sim_b_feat, sim_b_data, states=None, h2o=False, pbc=True, verbose=False,
            feat='bb-torsions', output_prefix='receptor_bbtors', ref_filename="traj/condition-a_receptor.gro"):

    """
    Calculate state-specific information (SSI) for a given feature between two simulation ensembles.

    This function computes the SSI for a specified feature (e.g., backbone or sidechain torsions)
    by combining all time series data from the same residue into a multivariate feature, determining
    discrete conformational states, and quantifying the degree to which state transitions signal
    information about the ensemble identity.

    Parameters
    ----------
    sim_a_feat : dict
        Feature metadata for simulation A.
    sim_a_data : dict
        Feature data for simulation A.
    sim_b_feat : dict
        Feature metadata for simulation B.
    sim_b_data : dict
        Feature data for simulation B.
    feat : str, optional
        The feature to analyze (default is 'bb-torsions').
    output_prefix : str, optional
        Prefix for output files (default is 'receptor_bbtors').
    ref_filename : str, optional
        Reference structure filename for visualization (default is "traj/condition-a_receptor.gro").

    Returns
    -------
    resnames : list
        List of residue names or identifiers.
    ssi : np.ndarray
        State-specific information values for each residue.
    """
    start_time = time()
    if feat == 'bb-distances':
        print(f'SSI is not implemented for {feat}')
        return None
        
    print(f"\n... Calculating state-specific information between ensemble and {feat} ...", flush=True)
    if feat in ['WaterPocket_OccupDistr', 'WaterPocket_Distr']:
        if states is None:
            raise Exception('Need to provide states if calculate SSI between ensemble and water')
        res_feat_a, res_feat_b, res_data_a, res_data_b = sim_a_feat[feat], sim_b_feat[feat], sim_a_data[feat], sim_b_data[feat]
    else:
        # combine all time series data from the same residue to one multivariate feature
        res_feat_a, res_data_a = get_multivar_res(
            sim_a_feat[feat], sim_a_data[feat]
        )
        res_feat_b, res_data_b = get_multivar_res(
            sim_b_feat[feat], sim_b_data[feat]
        )
        # determin state boundaries: distributions are decomposed into the individual Gaussians 
        # which fit the distribution, and conformational microstates are determined based on 
        # the Gaussian intersects.
        states = get_discrete_states(res_data_a, res_data_b)

    # calculate SSI
    # SSI measure ISSI(xf) quantifies the degree to which con-
    # formational state transitions of feature xf signal information about
    # the ensembles i and j or the transitions between them.
    # SSI: ranges from 0 bit to 1 bit, where 0 bit represents no shared information 
    # and 1 bit represents maximal shared information between the ensemble
    # (transitions) and the features.
    if verbose:
        print('Feature \t SSI',flush=True)
    resnames, ssi = ssi_ensemble_analysis(
        res_feat_a, res_feat_b,
        res_data_a, res_data_b,
        states, verbose=verbose, h2o=h2o, pbc=pbc
    )
    if not feat in ['WaterPocket_OccupDistr', 'WaterPocket_Distr']:
        vis = residue_visualization(
            resnames, ssi, ref_filename,
            os.path.join(CWDIR, f"plots/{output_prefix}_ssi.pdf"),
            os.path.join(CWDIR, f"vispdb/{output_prefix}_ssi.pdb"),
            y_label=f'max. SSI of {feat}'
        )
    end_time = time()
    print(f"Time elapsed: {(end_time - start_time)/60:.2f} minutes")
    return resnames, ssi, states, res_feat_a, res_data_a, res_feat_b, res_data_b

def find_centroid(ref_filename, traj_prefix, cluster_idx, n_jobs = -1):
    """
    Find and save the centroid structure for each cluster in a set of trajectory files.

    For each unique cluster index in `cluster_idx`, this function loads the corresponding trajectory
    (with filename pattern "{traj_prefix}_c{k}.xtc"), computes the pairwise RMSD matrix (excluding hydrogens),
    and identifies the frame that is most similar (in an exponential sense) to all other frames as the centroid.
    The centroid structure is then saved as a PDB file with the filename pattern "{traj_prefix}_c{k}_centroid.pdb".

    Parameters
    ----------
    ref_filename : str
        Path to the reference topology file (e.g., .gro or .pdb) used to load the trajectory.
    traj_prefix : str
        Prefix for the trajectory files. The function expects files named "{traj_prefix}_c{k}.xtc" for each cluster k.
    cluster_idx : array-like
        Array of cluster indices for the frames or just unique cluster indices present in traj. The function will process each unique cluster index.

    Returns
    -------
    None
        The function saves the centroid structure for each cluster as a PDB file.
    """
    start_time = time()
    from joblib import Parallel, delayed, cpu_count
    print(f"\n... Find cluster centroid for {traj_prefix} ...", flush=True)
    actual_n_jobs = cpu_count() if n_jobs == -1 else min(n_jobs,cpu_count())
    print(f"    Using {actual_n_jobs} parallel jobs for RMSD calculation", flush=True)
    uniq_cidx = np.unique(cluster_idx)
    for k in uniq_cidx :
        print(f"    cluster {k}")
        traj = md.load(traj_prefix+f"_c{k}.xtc", top=ref_filename)
        atom_indices = [a.index for a in traj.topology.atoms if a.element.symbol != 'H']
        # Parallelize the RMSD calculation using joblib for speedup
        def compute_rmsd_row(i):
            # Returns the RMSD of frame i to all frames, in order
            return md.rmsd(traj, traj, i, atom_indices=atom_indices)
        # Each row i corresponds to frame i, so indices are preserved
        distances_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(compute_rmsd_row)(i) for i in range(traj.n_frames)
        )
        distances = np.stack(distances_list, axis=0)
        # centroid is conformation that has highest similarity scores to all other conformations
        beta = 1
        index = np.exp(-beta*distances / distances.std()).sum(axis=1).argmax()
        centroid = traj[index]
        outname = traj_prefix+f"_c{k}_centroid.pdb"
        centroid.save(outname)

    end_time = time()
    print(f"Time elapsed: {(end_time - start_time)/60:.2f} minutes")


def calc_pca_cluster(sim_a_feat, sim_a_data, sim_b_feat, sim_b_data, 
            feat='bb-torsions', output_prefix='combined_bbtors_tmr', 
            traj_a="traj/condition-a_receptor_aligned.xtc",
            traj_b="traj/condition-b_receptor_aligned.xtc",
            top_a="traj/condition-a_receptor.gro",
            top_b="traj/condition-b_receptor.gro",
            label_a='arrestin', label_b='Gq',
            num_clusters=5, n_pca=3):

    """
    Perform PCA analysis and clustering on the specified structural feature for two simulation ensembles.

    This function combines the feature data from two simulation conditions, performs principal component analysis (PCA) on the combined data,
    and generates plots for eigenvalues, feature correlations, and projections along principal components. It also supports clustering
    of the projected data and can save representative centroid structures for each cluster.

    Parameters
    ----------
    sim_a_feat : dict
        Dictionary containing features for simulation A (from read_structure_features). Keys are feature names, values are feature arrays.
    sim_a_data : dict
        Dictionary containing data arrays for simulation A (from read_structure_features). Keys are feature names, values are data arrays.
    sim_b_feat : dict
        Dictionary containing features for simulation B (from read_structure_features). Keys are feature names, values are feature arrays.
    sim_b_data : dict
        Dictionary containing data arrays for simulation B (from read_structure_features). Keys are feature names, values are data arrays.
    feat : str, optional
        The feature to analyze, options are 'bb-torsions', 'bb-distances', 'sc-torsions'. Default is 'bb-torsions'.
    output_prefix : str, optional
        Prefix for output files. Default is 'combined_bbtors_tmr'.
    traj_a : str, optional
        Path to trajectory file for simulation A. Default is "traj/condition-a_receptor_aligned.xtc".
    traj_b : str, optional
        Path to trajectory file for simulation B. Default is "traj/condition-b_receptor_aligned.xtc".
    top_a : str, optional
        Path to topology file for simulation A. Default is "traj/condition-a_receptor.gro".
    top_b : str, optional
        Path to topology file for simulation B. Default is "traj/condition-b_receptor.gro".
    label_a : str, optional
        Label for simulation A (used in plots). Default is 'arrestin'.
    label_b : str, optional
        Label for simulation B (used in plots). Default is 'Gq'.
    num_clusters : int, optional
        Number of clusters to use for clustering analysis. Default is 5.

    Returns
    -------
    None
        This function generates plots and files as side effects but does not return a value.
    """
    start_time = time()
    print(f"\n... PCA analysis on {feat} ...", flush=True)

    # combine the data of the two simulation conditions.
    combined_data = np.concatenate([sim_a_data[feat], sim_b_data[feat]], 0)
    # sim_a_data[feat] = n_frames_a x n_torsions

    # get PCA of the combined data, returns a scikit-learn PCA object
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    # dir(pca_combined) to list all the attributes
    pca_combined = calculate_pca(combined_data)
    idx, ev = pca_eigenvalues_plot(
        pca_combined, num=12,
        plot_file=os.path.join(CWDIR, f'plots/{output_prefix}_eigenvalues.pdf')
    )
    # find the most relevant features of each PC
    _ = pca_feature_correlation(
        sim_a_feat[feat], sim_a_data[feat],
        pca=pca_combined, num=5, threshold=0.4, 
        plot_file = os.path.join(CWDIR, f'plots/{output_prefix}_pca_features.pdf'), 
        add_labels=False
    )
    # compare how the frames of each ensemble are distributed along the PCA
    _ = compare_projections(
        sim_a_data[feat],
        sim_b_data[feat],
        pca_combined,
        num = 5,
        label_a=label_a, label_b=label_b,
        saveas=os.path.join(CWDIR, f'plots/{output_prefix}_projections.pdf') 
    )
    # sort trajectories by PCs
    # frames from both original trajectories 
    # are ordered by their value along the respective components.
    # PCA transformation: Z = XW, where X is the data matrix, W is the PCA matrix (pca_combined.components_)
    # Z (n_frames x n_components): PC score matrix
    # sorting traj by PC1 = sort the first column of Z
    # traj is sorted from negative to positive value of PC1 = describe traj from one extreme of motion to the other
    _ = sort_trajs_along_common_pc(
        sim_a_data[feat], sim_b_data[feat],
        os.path.join(CWDIR,top_a), os.path.join(CWDIR,top_b), 
        os.path.join(CWDIR,traj_a), os.path.join(CWDIR,traj_b),
        os.path.join(CWDIR, f'pca/receptor_by_{output_prefix}'), num_pc=5, start_frame=0
    )

    print(f"\n... Clustering on {n_pca} highest PCs of {feat} into {num_clusters} clusters ...", flush=True)
    # clustering in the space of the three highest principal components, pc_a_data = [n_frames, n_pca]
    pc_a_name, pc_a_data = get_components_pca(sim_a_data[feat], n_pca, pca_combined)
    pc_b_name, pc_b_data = get_components_pca(sim_b_data[feat], n_pca, pca_combined)



    # Create a figure with a main scatter plot and marginal KDEs
    fig = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[7,2], height_ratios=[2,7],
                        wspace=0.05, hspace=0.05)

    # Main scatter plot
    ax_scatter = plt.subplot(gs[1, 0])
    ax_scatter.scatter(pc_a_data[:, 0], pc_a_data[:, 1], c='tab:blue', label=label_a, alpha=0.2, s=18, edgecolors='none')
    ax_scatter.scatter(pc_b_data[:, 0], pc_b_data[:, 1], c='tab:orange', label=label_b, alpha=0.2, s=18, edgecolors='none')
    ax_scatter.set_xlabel('PC1')
    ax_scatter.set_ylabel('PC2')
    plt.suptitle(f'Scatter plot of first 2 PCs on {output_prefix}')# , y=1.02)

    # Top KDE (PC1)
    ax_top = plt.subplot(gs[0, 0], sharex=ax_scatter)
    sns.kdeplot(pc_a_data[:, 0], ax=ax_top, color='tab:blue', fill=True, alpha=0.3, linewidth=1)
    sns.kdeplot(pc_b_data[:, 0], ax=ax_top, color='tab:orange', fill=True, alpha=0.3, linewidth=1)
    ax_top.axis('off')

    # Right KDE (PC2)
    ax_right = plt.subplot(gs[1, 1], sharey=ax_scatter)
    sns.kdeplot(pc_a_data[:, 1], ax=ax_right, color='tab:blue', fill=True, alpha=0.3, linewidth=1, vertical=True)
    sns.kdeplot(pc_b_data[:, 1], ax=ax_right, color='tab:orange', fill=True, alpha=0.3, linewidth=1, vertical=True)
    ax_right.axis('off')

    # Adjust limits for marginal plots to match scatter
    ax_top.set_xlim(ax_scatter.get_xlim())
    ax_right.set_ylim(ax_scatter.get_ylim())

    # Move legend to scatter axis
    ax_scatter.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CWDIR, f'plots/{output_prefix}_pc1_pc2.pdf'))
    plt.close()

    cc = obtain_combined_clusters(
        pc_a_data, pc_b_data, label_a=label_a, label_b=label_b, start_frame=0,
        algorithm='kmeans', max_iter=2000, num_clusters=num_clusters, min_dist=12,
        saveas=os.path.join(CWDIR, f'plots/{output_prefix}.pdf')
    )
    cidx, cond, oidx, wss, centroids = cc
    np.savetxt(
        os.path.join(CWDIR, f'results/{output_prefix}_indices.csv'),
        np.array([cidx, cond, oidx], dtype=int).T,
        delimiter=',', fmt='%i',
        header='Cluster, Condition, Index within condition'
    )

    # sort the frames from each ensemble into these clusters and write traj
    # can compare the clusters between the two ensembles to see how they are distributed
    # in the PCA space
    traj_prefix_a = os.path.join(CWDIR, f"clusters/{output_prefix}_condition-a")
    traj_prefix_b = os.path.join(CWDIR, f"clusters/{output_prefix}_condition-b")
    _ = write_cluster_traj(
        cidx[cond==0], os.path.join(CWDIR,top_a), os.path.join(CWDIR,traj_a),
        traj_prefix_a, start_frame=0
    )
    _ = write_cluster_traj(
        cidx[cond==1], os.path.join(CWDIR,top_b), os.path.join(CWDIR,traj_b),
        traj_prefix_b, start_frame=0
    )

    # find cluster centroids and write to file
    find_centroid(os.path.join(CWDIR,top_a), traj_prefix_a, cidx[cond==0])
    find_centroid(os.path.join(CWDIR,top_b), traj_prefix_b, cidx[cond==1])

    # check for number of optimal clusters
    # A common method to obtain the optimal number of clusters is the elbow plot. 
    # We plot the within-sum-of-squares (WSS) for a few repetitions for 
    # an increasing number of clusters. Then we look for the “elbow” in the resulting plot. 
    # Unfortunately, sometimes there is no clear result though.
    wss_avg, wss_std = wss_over_number_of_combined_clusters(
        pc_a_data, pc_b_data, label_a=label_a, label_b=label_b,
        start_frame=0, algorithm='kmeans',
        max_iter=100, num_repeats = 5, max_num_clusters = 12,
        plot_file = os.path.join(CWDIR, f'plots/{output_prefix}_wss.pdf')
    )
    end_time = time()
    print(f"Time elapsed: {(end_time - start_time)/60:.2f} minutes")


def calc_ssi_feat_feat(sim_a_feat, sim_a_data, sim_b_feat, sim_b_data, states, threshold=0.5,
                        output_prefix="water-bbtor-pairs", h2o=False):
    """This function computes the SSI, which quantifies the degree of communication between
    pairs of features across two ensembles"""
    start_time = time()
    print(f"\n... Calculating state-specific information for {output_prefix} ...", flush=True)
    # get SSI for combined features
    # quantify the amount of feature-feature communication between features
    pairs_names, pairs_ssi = ssi_feature_analysis(
        sim_a_feat, sim_b_feat,
        sim_a_data, sim_b_data,
        states, verbose=False, h2o=h2o
    )
    # filter pairs with SSI > threshold, excluding self-SSI
    relevant = np.abs(pairs_ssi) > threshold
    not_self = np.array([name.split(' & ')[0] != name.split(' & ')[1] for name in pairs_names])
    relevant *= not_self
    argrelev = np.argwhere(relevant).flatten()
    all_relevant_pairs_names = [pairs_names[i] for i in argrelev]
    all_relevant_pairs_ssi = pairs_ssi[relevant]
    # sort by SSI
    _ = sort_features(all_relevant_pairs_names, all_relevant_pairs_ssi)
    all_relevant_pairs_names = _[:,0]
    all_relevant_pairs_ssi = _[:,1]
    # save to csv
    f1 = [pair.split(' & ')[0] for pair in all_relevant_pairs_names]
    f2 = [pair.split(' & ')[1] for pair in all_relevant_pairs_names]
    data = np.array([f1, f2, all_relevant_pairs_ssi]).T
    np.savetxt(
         os.path.join(CWDIR, f'results/{output_prefix}_ssi.csv'),
        data, fmt='%s', delimiter=',',
        header='Feature_i, Feature_j, SSI'
    )
    # plot the top 10
    pair_features_heatmap(
        all_relevant_pairs_names[:10], all_relevant_pairs_ssi[:10],
         os.path.join(CWDIR, f"plots/{output_prefix}_ssi.pdf"),
        #vmin=0.0, vmax=1.0, 
        cbar_label='SSI',
        separator=' & '
    )
    end_time = time()
    print(f"Time elapsed: {(end_time - start_time)/60:.2f} minutes")


def calc_cossi(sim_a_feat, sim_a_data, sim_b_feat, sim_b_data, states,
                threshold=0.5, output_prefix="water-bbtor-pairs"):
    """
    Calculate Co-SSI (ICoSSI) statistics between pairs of features from two ensembles.
    Co-SSImeasures how the switch between ensembles affects the communication between two features.

    The function performs the following steps:
        1. Computes SSI for all feature pairs and filters for relevant pairs (SSI > 0.5, excluding self-pairs).
        2. Sorts and saves the relevant SSI pairs to a CSV file and generates a heatmap plot.
        3. Computes Co-SSI for all feature pairs, filters for relevant pairs (|Co-SSI| > 0.5, excluding self-pairs),
           sorts, saves to a CSV file, and generates a heatmap plot.

    Args:
        sim_a_feat (list): List of feature names from ensemble A.
        sim_b_feat (list): List of feature names from ensemble B.
        sim_a_data (np.ndarray): Data array for features from ensemble A.
        sim_b_data (np.ndarray): Data array for features from ensemble B.
        states : list or np.ndarray
            Discrete state assignments for each feature, used for SSI/Co-SSI calculation.
        output_prefix (str, optional): Prefix for output files. Default is "water-bbtor-pairs".

    Returns:
        None. Results are saved to CSV files and plots are generated.
    """
    start_time = time()
    print(f"\n... Calculating State-specific co-information for {output_prefix} ...", flush=True)
    # State Specific Information Co-SSI statistic between two features and the ensembles
    # ICoSSI can be positive or negative, indicating whether the switch between ensem-
    # bles increases (ICoSSI > 0), decreases (ICoSSI < 0), or does not affect
    # (ICoSSI = 0) the communication between two features x1 and x2
    pairs_names, pairs_ssi, pairs_cossi = cossi_featens_analysis(
        sim_a_feat, sim_b_feat, sim_a_feat, sim_b_feat,
        sim_a_data, sim_b_data, sim_a_data, sim_b_data,
        states, states, verbose=False
    )
    # filter pairs with absolute Co-SSI > threshold, excluding self-SSI
    relevant = np.abs(pairs_cossi) > threshold
    not_self = np.array([name.split(' & ')[0] != name.split(' & ')[1] for name in pairs_names])
    relevant *= not_self
    argrelev = np.argwhere(relevant).flatten()
    all_relevant_pairs_names = [pairs_names[i] for i in argrelev]
    all_relevant_pairs_cossi = pairs_cossi[relevant]
    # sort by abs(Co-SSI)
    _ = sort_features(all_relevant_pairs_names, np.abs(all_relevant_pairs_cossi))
    all_relevant_pairs_names = _[:,0]
    all_relevant_pairs_cossi = _[:,1]

    # save to csv
    f1 = [pair.split(' & ')[0] for pair in all_relevant_pairs_names]
    f2 = [pair.split(' & ')[1] for pair in all_relevant_pairs_names]
    data = np.array([f1, f2, all_relevant_pairs_cossi]).T

    np.savetxt(
        os.path.join(CWDIR, f'results/{output_prefix}_cossi.csv'),
        data, fmt='%s', delimiter=',',
        header='Feature_i, Feature_j, Co-SSI'
    )
    # plot the top 10
    pair_features_heatmap(
        all_relevant_pairs_names[:10], all_relevant_pairs_cossi[:10],
        os.path.join(CWDIR, f'plots/{output_prefix}_cossi.pdf'),
        #vmin=0.0, vmax=1.0, 
        cbar_label='Co-SSI',
        separator=' & '
    )
    end_time = time()
    print(f"Time elapsed: {(end_time - start_time)/60:.2f} minutes")