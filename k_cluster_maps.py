#!/usr/bin/env python3
"""
Extract and plot mean cluster maps from participant connectivity vectors.

This script takes hierarchical clustering results and creates mean connectivity maps
for each cluster, with optional p-value thresholding and cluster correction.

Usage:
    python k_cluster_maps.py --results_dir derivatives/hierarchical_clustering --k_optimal 2
    python k_cluster_maps.py --results_dir derivatives/hierarchical_clustering --k_optimal 3 --use_pval --pval_threshold 0.05 --cluster_threshold 10
"""

import argparse
import os
import os.path as op
import json
import joblib
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import label
from scipy.stats import false_discovery_control
from nilearn import plotting, datasets
from nilearn.maskers import NiftiMasker


def get_parser():
    """Command line argument parser"""
    p = argparse.ArgumentParser(
        description="Extract and plot mean cluster maps from participant vectors."
    )
    p.add_argument(
        "--results_dir",
        required=True,
        help="Directory with clustering results and data matrix",
    )
    p.add_argument(
        "--k_optimal",
        type=int,
        required=True,
        help="Optimal k value (number of clusters)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold value for plotting (default: None, auto-threshold)",
    )
    p.add_argument(
        "--use_pval",
        action="store_true",
        help="Use p-value thresholding instead of raw intensity",
    )
    p.add_argument(
        "--pval_threshold",
        type=float,
        default=0.05,
        help="Voxel-level p-value threshold (default: 0.05)",
    )
    p.add_argument(
        "--cluster_threshold",
        type=int,
        default=10,
        help="Cluster size threshold in voxels (default: 10)",
    )
    p.add_argument(
        "--correction",
        type=str,
        choices=["none", "fwe", "fdr", "all"],
        default="none",
        help="Multiple comparison correction method: none, fwe, fdr, or all (default: none)",
    )
    p.add_argument(
        "--fwe_threshold",
        type=float,
        default=0.05,
        help="FWE-corrected alpha level (default: 0.05)",
    )
    p.add_argument(
        "--fdr_threshold",
        type=float,
        default=0.05,
        help="FDR-corrected q-value threshold (default: 0.05)",
    )
    return p


def load_masker(masker_file, map_paths):
    """Load or create NiftiMasker for transforming data"""
    masker = None
    if op.exists(masker_file):
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
        joblib.dump(masker, masker_file)
        print(f"Saved masker to: {masker_file}")
    return masker


def apply_pval_thresholding(
    group_vectors, masker, pval_threshold, cluster_threshold
):
    """Apply p-value and cluster-size thresholding to group vectors"""
    # Compute p-values using one-sample t-test against zero
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
    plot_vector = masker.transform(stat_img).ravel()

    return stat_img, plot_vector


def apply_fwe_correction(group_vectors, masker, fwe_threshold, cluster_threshold):
    """Apply Family-Wise Error (FWE) correction using Bonferroni method"""
    # Compute p-values using one-sample t-test against zero
    t_stats, p_values = stats.ttest_1samp(group_vectors, 0, axis=0)
    
    # Apply Bonferroni correction: divide alpha by number of tests
    n_voxels = len(p_values)
    bonferroni_threshold = fwe_threshold / n_voxels
    
    print(f"  FWE correction: {n_voxels} voxels, Bonferroni threshold = {bonferroni_threshold:.2e}")
    
    # Create stat map based on t-statistics
    stat_vector = t_stats.copy()
    
    # Apply FWE-corrected threshold
    stat_vector[p_values > bonferroni_threshold] = 0
    
    # Apply cluster-level thresholding
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
    plot_vector = masker.transform(stat_img).ravel()
    
    return stat_img, plot_vector


def apply_fdr_correction(group_vectors, masker, fdr_threshold, cluster_threshold):
    """Apply False Discovery Rate (FDR) correction using Benjamini-Hochberg method"""
    # Compute p-values using one-sample t-test against zero
    t_stats, p_values = stats.ttest_1samp(group_vectors, 0, axis=0)
    
    # Apply FDR correction using Benjamini-Hochberg
    try:
        # Try using scipy's false_discovery_control (available in scipy >= 1.7.0)
        reject = false_discovery_control(p_values, method='bh', axis=None) <= fdr_threshold
        fdr_mask = reject
    except:
        # Fallback: manual Benjamini-Hochberg procedure
        n_voxels = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_pvals = p_values[sorted_indices]
        
        # Find largest i such that P(i) <= (i/m)*q
        thresholds = (np.arange(1, n_voxels + 1) / n_voxels) * fdr_threshold
        below_threshold = sorted_pvals <= thresholds
        
        if np.any(below_threshold):
            max_idx = np.where(below_threshold)[0][-1]
            fdr_pval_threshold = sorted_pvals[max_idx]
            fdr_mask = p_values <= fdr_pval_threshold
        else:
            fdr_mask = np.zeros_like(p_values, dtype=bool)
    
    n_surviving = np.sum(fdr_mask)
    print(f"  FDR correction: {n_surviving} voxels survived at q={fdr_threshold}")
    
    # Create stat map based on t-statistics
    stat_vector = t_stats.copy()
    
    # Apply FDR mask
    stat_vector[~fdr_mask] = 0
    
    # Apply cluster-level thresholding
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
    plot_vector = masker.transform(stat_img).ravel()
    
    return stat_img, plot_vector


def apply_intensity_thresholding(mean_img, threshold):
    """Apply intensity thresholding to mean image"""
    if threshold is not None:
        mean_data = mean_img.get_fdata()
        mean_data[np.abs(mean_data) < threshold] = 0
        mean_img = nib.Nifti1Image(mean_data, mean_img.affine, mean_img.header)
    return mean_img


def generate_threshold_suffix(use_pval, pval_threshold, cluster_threshold, threshold):
    """Generate filename suffix based on thresholding parameters"""
    if use_pval:
        # Extract all digits after decimal of p-value (e.g., 0.05 -> "05", 0.001 -> "001")
        pval_str = str(pval_threshold).split(".")[-1]
        vox_str = f"{cluster_threshold}"
    else:
        pval_str = "NA"
        vox_str = f"{threshold}" if threshold is not None else "NA"
    return f"_clust_p-{pval_str}_vox-{vox_str}"


def save_and_plot_cluster_map(
    group,
    k_optimal,
    mean_img,
    plot_img,
    plot_vector,
    plot_title,
    nifti_dir,
    figures_dir,
    threshold_suffix,
    use_pval,
    pval_threshold,
    cluster_threshold,
    correction_type="none",
):
    """Save nifti files and create plots for a cluster map"""
    # Add correction type to filename if applicable
    if correction_type != "none":
        corr_suffix = f"_{correction_type}"
    else:
        corr_suffix = ""
    
    # Save mean map (always save unthresholded mean)
    nii_path = op.join(
        nifti_dir,
        f"k_cluster_maps_k{k_optimal}_group{group}{threshold_suffix}.nii.gz",
    )
    mean_img.to_filename(nii_path)

    # Save stat map if using p-values
    if use_pval:
        stat_path = op.join(
            nifti_dir,
            f"k_cluster_maps_k{k_optimal}_group{group}_tstat{corr_suffix}{threshold_suffix}.nii.gz",
        )
        plot_img.to_filename(stat_path)

    # Load MNI template for background
    mni_template = datasets.load_mni152_template(resolution=2)

    # Plot and save without additional thresholding since data is already thresholded
    display = plotting.plot_stat_map(
        plot_img,
        title=plot_title,
        display_mode="ortho",
        threshold=0,  # Threshold is 0 because nifti files are already thresholded
        colorbar=True,
        cmap="cold_hot",
        black_bg=False,
        draw_cross=False,
    )

    # Make figure filename consistent with nifti naming
    if use_pval:
        fig_path = op.join(
            figures_dir,
            f"k_cluster_maps_k{k_optimal}_group{group}_tstat{corr_suffix}{threshold_suffix}.png",
        )
    else:
        fig_path = op.join(
            figures_dir,
            f"k_cluster_maps_k{k_optimal}_group{group}{threshold_suffix}.png",
        )

    display.savefig(fig_path, dpi=300)
    display.close()

    # Print statistics about surviving voxels
    if use_pval:
        n_surviving = np.sum(plot_vector != 0)
        if correction_type != "none":
            print(
                f"Group {group} ({correction_type.upper()}): {n_surviving} voxels survived correction"
            )
        else:
            print(
                f"Group {group}: {n_surviving} voxels survived correction (p<{pval_threshold}, cluster>{cluster_threshold})"
            )
    print(f"Saved: {nii_path if not use_pval else stat_path} and {fig_path}")
    
    return n_surviving if use_pval else None


def process_cluster_maps(
    results_dir,
    k_optimal,
    threshold,
    use_pval,
    pval_threshold,
    cluster_threshold,
    correction="none",
    fwe_threshold=0.05,
    fdr_threshold=0.05,
):
    """Main processing function to extract and plot cluster maps"""
    # Setup directories
    k_dir = op.join(results_dir, f"k_{k_optimal}")
    figures_dir = op.join(k_dir, "figures")
    nifti_dir = op.join(k_dir, "nifti")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(nifti_dir, exist_ok=True)

    # Load group assignments from k-specific subfolder
    assignments_csv = op.join(
        k_dir, f"hierarchical_connectivity_groups_k{k_optimal}.csv"
    )
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
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    subject_ids = metadata["participant_ids"]
    map_paths = metadata["map_paths"]
    subject_id_to_index = {sid: i for i, sid in enumerate(subject_ids)}

    # Load or create masker
    masker = load_masker(masker_file, map_paths)

    # Determine which corrections to apply
    if correction == "all":
        corrections_to_apply = ["none", "fwe", "fdr"]
    else:
        corrections_to_apply = [correction]

    # For each cluster, extract participant vectors, compute mean, and save/plot
    for group in sorted(assignments["group"].unique()):
        print(f"\nProcessing Group {group}...")
        group_subjects = assignments[assignments["group"] == group]["subject_id"]
        indices = [subject_id_to_index[sid] for sid in group_subjects]
        group_vectors = data_matrix[indices]
        mean_vector = group_vectors.mean(axis=0)

        # Create mean image (always save unthresholded mean)
        mean_img = masker.inverse_transform(mean_vector)
        if not use_pval:
            mean_img = apply_intensity_thresholding(mean_img, threshold)

        # Process each correction method
        for corr_type in corrections_to_apply:
            print(f"  Applying {corr_type.upper()} correction...")
            
            # Generate threshold suffix for filenames
            if corr_type == "none":
                threshold_suffix = generate_threshold_suffix(
                    use_pval, pval_threshold, cluster_threshold, threshold
                )
            elif corr_type == "fwe":
                threshold_suffix = generate_threshold_suffix(
                    use_pval, fwe_threshold, cluster_threshold, threshold
                )
            elif corr_type == "fdr":
                threshold_suffix = generate_threshold_suffix(
                    use_pval, fdr_threshold, cluster_threshold, threshold
                )

            # Apply thresholding based on method
            if use_pval:
                if corr_type == "none":
                    plot_img, plot_vector = apply_pval_thresholding(
                        group_vectors, masker, pval_threshold, cluster_threshold
                    )
                    plot_title = f"Cluster {group} T-stat Map (k={k_optimal}, p<{pval_threshold}, cluster>{cluster_threshold})"
                elif corr_type == "fwe":
                    plot_img, plot_vector = apply_fwe_correction(
                        group_vectors, masker, fwe_threshold, cluster_threshold
                    )
                    plot_title = f"Cluster {group} T-stat Map (k={k_optimal}, FWE p<{fwe_threshold}, cluster>{cluster_threshold})"
                elif corr_type == "fdr":
                    plot_img, plot_vector = apply_fdr_correction(
                        group_vectors, masker, fdr_threshold, cluster_threshold
                    )
                    plot_title = f"Cluster {group} T-stat Map (k={k_optimal}, FDR q<{fdr_threshold}, cluster>{cluster_threshold})"
            else:
                plot_vector = mean_vector
                plot_img = masker.inverse_transform(mean_vector)
                plot_title = f"Cluster {group} Mean Map (k={k_optimal})"

            # Save and plot
            save_and_plot_cluster_map(
                group,
                k_optimal,
                mean_img,
                plot_img,
                plot_vector,
                plot_title,
                nifti_dir,
                figures_dir,
                threshold_suffix,
                use_pval,
                pval_threshold if corr_type == "none" else (fwe_threshold if corr_type == "fwe" else fdr_threshold),
                cluster_threshold,
                correction_type=corr_type,
            )


def main():
    """Main function"""
    args = get_parser().parse_args()

    print("Cluster Map Extraction and Plotting Tool")
    print("=" * 40)
    print(f"Results directory: {args.results_dir}")
    print(f"K value: {args.k_optimal}")
    print(f"Use p-value thresholding: {args.use_pval}")
    if args.use_pval:
        print(f"Correction method: {args.correction}")
        if args.correction in ["none", "all"]:
            print(f"  Uncorrected p-value threshold: {args.pval_threshold}")
        if args.correction in ["fwe", "all"]:
            print(f"  FWE alpha level: {args.fwe_threshold}")
        if args.correction in ["fdr", "all"]:
            print(f"  FDR q-value threshold: {args.fdr_threshold}")
        print(f"Cluster size threshold: {args.cluster_threshold} voxels")
    else:
        print(f"Intensity threshold: {args.threshold}")

    try:
        process_cluster_maps(
            args.results_dir,
            args.k_optimal,
            args.threshold,
            args.use_pval,
            args.pval_threshold,
            args.cluster_threshold,
            args.correction,
            args.fwe_threshold,
            args.fdr_threshold,
        )

        print("\nSUCCESS!")
        print(f"Cluster maps saved to: {op.join(args.results_dir, f'k_{args.k_optimal}')}")

        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

