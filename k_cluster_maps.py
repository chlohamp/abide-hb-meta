import argparse
import os
import os.path as op
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting

parser = argparse.ArgumentParser(description="Extract and plot mean cluster maps from participant vectors.")
parser.add_argument('--results_dir', required=True, help='Directory with clustering results and data matrix')
parser.add_argument('--k_optimal', type=int, required=True, help='Optimal k value (number of clusters)')
parser.add_argument('--threshold', type=float, default=None, help='Threshold value for plotting (default: None, auto-threshold)')
parser.add_argument('--use_pval', action='store_true', help='Use p-value thresholding instead of raw intensity')
parser.add_argument('--pval_threshold', type=float, default=0.05, help='Voxel-level p-value threshold (default: 0.05)')
parser.add_argument('--cluster_threshold', type=int, default=10, help='Cluster size threshold in voxels (default: 10)')
args = parser.parse_args()

results_dir = args.results_dir
k_optimal = args.k_optimal
threshold = args.threshold
use_pval = args.use_pval
pval_threshold = args.pval_threshold
cluster_threshold = args.cluster_threshold
figures_dir = op.join(results_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

# Load group assignments from k-specific subfolder (e.g., results_dir/k_3/...)
k_dir = op.join(results_dir, f"k_{k_optimal}")
assignments_csv = op.join(k_dir, f"hierarchical_connectivity_groups_k{k_optimal}.csv")
if not op.exists(assignments_csv):
    raise FileNotFoundError(f"Group assignment file not found: {assignments_csv}")
assignments = pd.read_csv(assignments_csv)

# Load data matrix and masker
data_matrix_file = op.join(results_dir, "clustering_data_matrix.npy")
masker_file = op.join(results_dir, "masker.pkl")
if not op.exists(data_matrix_file):
    raise FileNotFoundError(f"Data matrix file not found: {data_matrix_file}")
data_matrix = np.load(data_matrix_file)

# Load subject IDs and map paths from metadata (needed for masker fitting)
metadata_file = op.join(results_dir, "clustering_metadata.json")
if not op.exists(metadata_file):
    raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
import json
with open(metadata_file, "r") as f:
    metadata = json.load(f)
subject_ids = metadata["participant_ids"]
map_paths = metadata["map_paths"]
subject_id_to_index = {sid: i for i, sid in enumerate(subject_ids)}

# Try to load masker from pickle, else create using whole-brain MNI template
masker = None
if op.exists(masker_file):
    import joblib
    masker = joblib.load(masker_file)
    print(f"Loaded masker from: {masker_file}")
else:
    # Create masker using whole-brain MNI template (same as used in clustering)
    print("Creating NiftiMasker with whole-brain MNI template...")
    masker = NiftiMasker(
        mask_strategy="whole-brain-template",
        standardize=False,  # Data is already standardized in the matrix
        memory_level=1,
        verbose=1,
    )
    # Fit masker using one of the original connectivity maps
    print(f"Fitting masker with reference image: {map_paths[0]}")
    masker.fit(map_paths[0])
    # Save masker for future use
    import joblib
    joblib.dump(masker, masker_file)
    print(f"Saved masker to: {masker_file}")

# For each cluster, extract participant vectors, compute mean, and save/plot
from scipy import stats
from scipy.ndimage import label

for group in sorted(assignments['group'].unique()):
    group_subjects = assignments[assignments['group'] == group]['subject_id']
    indices = [subject_id_to_index[sid] for sid in group_subjects]
    group_vectors = data_matrix[indices]
    mean_vector = group_vectors.mean(axis=0)
    
    # Compute p-values using one-sample t-test against zero
    if use_pval:
        t_stats, p_values = stats.ttest_1samp(group_vectors, 0, axis=0)
        # Create stat map based on t-statistics (signed by mean direction)
        stat_vector = t_stats.copy()
        
        # Apply voxel-level p-value threshold
        stat_vector[p_values > pval_threshold] = 0
        
        # Apply cluster-level thresholding
        # Convert to 3D image for cluster analysis
        stat_img = masker.inverse_transform(stat_vector)
        stat_data = stat_img.get_fdata()
        
        # Find connected components (clusters) separately for positive and negative
        pos_mask = stat_data > 0
        neg_mask = stat_data < 0
        
        # Label clusters
        pos_labeled, pos_n_clusters = label(pos_mask)
        neg_labeled, neg_n_clusters = label(neg_mask)
        
        # Filter by cluster size
        for cluster_id in range(1, pos_n_clusters + 1):
            cluster_mask = pos_labeled == cluster_id
            if np.sum(cluster_mask) < cluster_threshold:
                stat_data[cluster_mask] = 0
                
        for cluster_id in range(1, neg_n_clusters + 1):
            cluster_mask = neg_labeled == cluster_id
            if np.sum(cluster_mask) < cluster_threshold:
                stat_data[cluster_mask] = 0
        
        # Update stat image
        stat_img = nib.Nifti1Image(stat_data, stat_img.affine, stat_img.header)
        plot_img = stat_img
        plot_vector = masker.transform(stat_img).ravel()
        plot_title = f"Cluster {group} T-stat Map (k={k_optimal}, p<{pval_threshold}, cluster>{cluster_threshold})"
        cbar_label = "T-statistic"
    else:
        plot_vector = mean_vector
        plot_img = masker.inverse_transform(mean_vector)
        plot_title = f"Cluster {group} Mean Map (k={k_optimal})"
        cbar_label = "Mean Connectivity"
    
    # Inverse transform to 3D Nifti
    mean_img = masker.inverse_transform(mean_vector)
    
    # Save mean map
    nii_path = op.join(figures_dir, f"k_cluster_maps_k{k_optimal}_group{group}.nii.gz")
    mean_img.to_filename(nii_path)
    
    # Save stat map if using p-values
    if use_pval:
        stat_path = op.join(figures_dir, f"k_cluster_maps_k{k_optimal}_group{group}_tstat_clustcorr.nii.gz")
        plot_img.to_filename(stat_path)
    
    # Plot and save
    # For stat maps, use a lower threshold or None to show surviving voxels
    plot_threshold = None if use_pval else threshold
    
    # Load MNI template for background
    from nilearn import datasets
    mni_template = datasets.load_mni152_template(resolution=2)
    
    display = plotting.plot_stat_map(
        plot_img, 
        title=plot_title, 
        display_mode='ortho', 
        threshold=plot_threshold,
        colorbar=True,
        cmap='cold_hot',
        black_bg=False,
        draw_cross=True
    )
    suffix = "_tstat_clustcorr" if use_pval else ""
    fig_path = op.join(figures_dir, f"k_cluster_maps_k{k_optimal}_group{group}{suffix}.png")
    display.savefig(fig_path, dpi=300)
    display.close()
    
    # Print statistics about surviving voxels
    if use_pval:
        n_surviving = np.sum(plot_vector != 0)
        print(f"Group {group}: {n_surviving} voxels survived correction (p<{pval_threshold}, cluster>{cluster_threshold})")
    print(f"Saved: {nii_path} and {fig_path}")
