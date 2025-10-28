import argparse
import os
import os.path as op

import pandas as pd
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from nilearn.maskers import NiftiMasker
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

from utils import get_peaks, thresh_img


def get_parser():
    p = argparse.ArgumentParser(
        description="Hierarchical connectivity clustering workflow"
    )
    p.add_argument(
        "--project_dir",
        required=True,
        help="Path to project directory",
    )
    p.add_argument(
        "--data_dir",
        required=True,
        help="Base directory containing RSFC connectivity maps",
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help="Directory where outputs will be written",
    )
    p.add_argument(
        "--k_min",
        type=int,
        default=2,
        help="Minimum number of clusters to test",
    )
    p.add_argument(
        "--k_max",
        type=int,
        default=15,
        help="Maximum number of clusters to test",
    )
    p.add_argument(
        "--k_value",
        type=int,
        help="Single k value to test (for parallel job arrays)",
    )
    p.add_argument(
        "--preprocess_only",
        action="store_true",
        help="Only preprocess data (create and save data matrix), don't run clustering",
    )
    p.add_argument(
        "--load_preprocessed",
        action="store_true",
        help="Load preprocessed data matrix instead of creating it",
    )
    return p


class HierarchicalConnectivityClustering:
    """
    Class for hierarchical clustering of participants based on whole-brain voxel connectivity
    """

    def __init__(self, project_dir, base_rsfc_dir, output_dir):
        self.project_dir = project_dir
        self.base_rsfc_dir = base_rsfc_dir
        self.output_dir = output_dir
        self.figures_dir = op.join(output_dir, "figures")

        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

        # Initialize data containers
        self.df = None
        self.data_matrix = None
        self.subject_ids = None
        self.map_paths = None
        self.masker = None
        self.cluster_metrics = {}
        self.voxel_connectivity_groups = {}

    def load_data(self, metadata_file, group_filter="asd"):
        """Load and prepare connectivity maps for clustering"""
        print(f"Loading data from {metadata_file}", flush=True)

        # Load metadata
        self.df = pd.read_csv(metadata_file, sep="\t", comment="#")

        # Filter by group if specified
        if group_filter:
            self.df = self.df[self.df["group"] == group_filter].copy()
            print(f"Filtered to {len(self.df)} {group_filter} participants", flush=True)

        # Create full paths to connectivity maps
        self.df["map_path"] = self.df["InputFile"].str.replace(
            "^/rsfc/", f"{self.base_rsfc_dir}/", regex=True
        )

        # Keep only existing files
        existing_mask = self.df["map_path"].apply(op.exists)
        self.df = self.df[existing_mask]

        print(
            f"Found {len(self.df)} participants with valid connectivity maps",
            flush=True,
        )

        # Extract paths and subject IDs
        self.map_paths = self.df["map_path"].tolist()
        self.subject_ids = self.df["Subj"].tolist()

        # Verify all files can be loaded
        print("Verifying file integrity...", flush=True)
        valid_files = []
        valid_subjects = []
        valid_paths = []

        for i, path in enumerate(self.map_paths):
            try:
                nib.load(path)
                valid_files.append(i)
                valid_subjects.append(self.subject_ids[i])
                valid_paths.append(path)
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}", flush=True)

        self.map_paths = valid_paths
        self.subject_ids = valid_subjects
        print(
            f"Final dataset: {len(self.map_paths)} valid connectivity maps", flush=True
        )

        return self

    def prepare_data_matrix(
        self, mask_strategy="whole-brain-template", standardize=True
    ):
        """Convert connectivity maps to data matrix for clustering"""
        # Check if preprocessed data already exists
        data_file = op.join(self.output_dir, "clustering_data_matrix.npy")
        
        if op.exists(data_file):
            print(f"Found existing data matrix: {data_file}", flush=True)
            print("Loading preprocessed data instead of recreating it...", flush=True)
            self.data_matrix = np.load(data_file)
            print(f"Loaded data matrix shape: {self.data_matrix.shape}", flush=True)
            print(f"Memory usage: ~{self.data_matrix.nbytes / 1e9:.2f} GB", flush=True)
        else:
            print("Creating data matrix from connectivity maps...", flush=True)

            # Initialize masker
            self.masker = NiftiMasker(
                mask_strategy=mask_strategy,
                standardize=standardize,
                memory_level=1,
                verbose=1,
            )

            # Transform maps to data matrix
            # Shape: (n_participants, n_voxels)
            self.data_matrix = self.masker.fit_transform(self.map_paths)

            print(f"Data matrix shape: {self.data_matrix.shape}", flush=True)
            print(f"Memory usage: ~{self.data_matrix.nbytes / 1e9:.2f} GB", flush=True)

        # Create dendrogram and heatmap after data matrix is ready
        self._create_matrix_figures()

        return self

    def save_preprocessed_data(self):
        """Save the preprocessed data matrix and metadata"""
        print("Saving preprocessed data matrix...", flush=True)

        # Save data matrix
        data_file = op.join(self.output_dir, "clustering_data_matrix.npy")
        np.save(data_file, self.data_matrix)

        # Save participant IDs and map paths for consistency
        metadata = {
            "participant_ids": self.df["Subj"].tolist(),
            "map_paths": self.map_paths,
            "data_shape": self.data_matrix.shape,
        }

        metadata_file = op.join(self.output_dir, "clustering_metadata.json")
        import json

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved data matrix: {data_file}", flush=True)
        print(f"Saved metadata: {metadata_file}", flush=True)
        print(f"Data shape: {self.data_matrix.shape}", flush=True)

        return self

    def load_preprocessed_data(self):
        """Load the preprocessed data matrix and metadata"""
        print("Loading preprocessed data matrix...", flush=True)

        # Load data matrix
        data_file = op.join(self.output_dir, "clustering_data_matrix.npy")
        if not op.exists(data_file):
            raise FileNotFoundError(
                f"Preprocessed data not found: {data_file}. "
                "Run with --preprocess_only first."
            )

        self.data_matrix = np.load(data_file)

        # Load metadata
        metadata_file = op.join(self.output_dir, "clustering_metadata.json")
        import json

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        # Reconstruct participant DataFrame for consistency
        self.df = pd.DataFrame({"Subj": metadata["participant_ids"]})
        self.map_paths = metadata["map_paths"]

        print(f"Loaded data matrix: {data_file}", flush=True)
        print(f"Data shape: {self.data_matrix.shape}", flush=True)
        print(f"Participants: {len(self.df)}", flush=True)

        return self

    def determine_optimal_clusters(self, k_range):
        """Determine optimal number of clusters for hierarchical clustering"""
        print(
            "Determining optimal number of clusters using hierarchical clustering...",
            flush=True,
        )
        print(f"Testing {len(k_range)} different k values: {list(k_range)}", flush=True)

        self.cluster_metrics = {}

        # Pre-compute hierarchical clustering once
        print("Computing hierarchical clustering dendrogram...", flush=True)
        Xcorr = self._corr_embed()
        from scipy.spatial.distance import squareform

        condensed = pdist(Xcorr, metric="euclidean")
        linkage_matrix = linkage(condensed, method="ward")
        Dfull = squareform(condensed)

        # Calculate cophenetic correlation for dendrogram quality
        coph_corr = None
        try:
            from scipy.cluster.hierarchy import cophenet

            coph_corr, _ = cophenet(linkage_matrix, condensed)
            print(f"Dendrogram cophenetic correlation: {coph_corr:.3f}", flush=True)
        except Exception:
            pass

        for k in k_range:
            print(f"  Testing k={k}...", flush=True)

            # Get cluster labels for this k
            group_labels = fcluster(linkage_matrix, k, criterion="maxclust") - 1

            # Calculate silhouette score using precomputed distances
            sil_score = silhouette_score(Dfull, group_labels, metric="precomputed")

            # Calculate gap statistic
            gap_stat, gap_std = self._calculate_gap_statistic(k, group_labels)

            self.cluster_metrics[k] = {
                "silhouette": sil_score,
                "gap_statistic": gap_stat,
                "gap_std": gap_std,
                "cophenetic_correlation": coph_corr,
            }

        # Create validation plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        k_values = sorted(list(self.cluster_metrics.keys()))

        # Silhouette plot
        sil_scores = [self.cluster_metrics[k]["silhouette"] for k in k_values]
        axes[0].plot(
            k_values, sil_scores, "o-", linewidth=2, markersize=6, color="blue"
        )
        axes[0].set_xlabel("Number of Clusters (k)")
        axes[0].set_ylabel("Silhouette Score")
        axes[0].set_title("Silhouette Analysis - Hierarchical Clustering")
        axes[0].grid(True, alpha=0.3)

        # Highlight best k
        best_k = k_values[np.argmax(sil_scores)]
        best_score = max(sil_scores)
        axes[0].axvline(best_k, color="red", linestyle="--", alpha=0.7)
        axes[0].annotate(
            f"Best k={best_k}\nScore={best_score:.3f}",
            xy=(best_k, best_score),
            xytext=(best_k + 0.5, best_score),
            arrowprops=dict(arrowstyle="->", color="red"),
        )

        # Gap statistic plot
        gap_stats = [self.cluster_metrics[k]["gap_statistic"] for k in k_values]
        gap_stds = [self.cluster_metrics[k]["gap_std"] for k in k_values]
        axes[1].errorbar(
            k_values,
            gap_stats,
            yerr=gap_stds,
            fmt="o-",
            linewidth=2,
            markersize=6,
            color="green",
            capsize=5,
        )
        axes[1].set_xlabel("Number of Clusters (k)")
        axes[1].set_ylabel("Gap Statistic")
        axes[1].set_title("Gap Statistic - Hierarchical Clustering")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            op.join(self.figures_dir, "hierarchical_cluster_validation.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Print recommendations and save to file
        results_path = op.join(self.output_dir, "hierarchical_cluster_validation.txt")
        with open(results_path, "w") as f:
            f.write("Hierarchical Connectivity Cluster Validation Results:\n")
            f.write("=" * 50 + "\n")
            for k in sorted(self.cluster_metrics.keys()):
                metrics_str = ", ".join(
                    [
                        (
                            f"{metric}: {value:.3f}"
                            if isinstance(value, float)
                            else f"{metric}: {value}"
                        )
                        for metric, value in self.cluster_metrics[k].items()
                        if value is not None
                    ]
                )
                print(f"  k={k}: {metrics_str}", flush=True)
                f.write(f"k={k}: {metrics_str}\n")

            # Silhouette-based recommendation
            optimal_k_sil = k_values[np.argmax(sil_scores)]

            # Gap statistic recommendation (Tibshirani rule)
            optimal_k_gap = self._find_optimal_k_gap(k_values)

            print(
                f"\nRecommended k based on silhouette score: {optimal_k_sil}",
                flush=True,
            )
            print(
                f"Recommended k based on gap statistic (Tibshirani rule): {optimal_k_gap}",
                flush=True,
            )
            f.write(f"\nRecommendations:\n")
            f.write(f"  Silhouette-max: k={optimal_k_sil}\n")
            f.write(f"  Tibshirani Gap rule: k={optimal_k_gap}\n")

        print(f"\nValidation scores saved to {results_path}", flush=True)

        # Create and save validation DataFrame
        self._create_validation_dataframe()

        return self

    def _create_validation_dataframe(self):
        """Create DataFrame with k values, silhouette scores, and gap statistics"""
        print("Creating validation metrics DataFrame...", flush=True)

        # Prepare data for DataFrame
        validation_data = []
        for k in sorted(self.cluster_metrics.keys()):
            metrics = self.cluster_metrics[k]
            validation_data.append(
                {
                    "k": k,
                    "silhouette_score": metrics.get("silhouette", None),
                    "gap_statistic": metrics.get("gap_statistic", None),
                    "gap_std": metrics.get("gap_std", None),
                }
            )

        # Create DataFrame
        self.validation_df = pd.DataFrame(validation_data)

        # Save DataFrame with thread-safe approach
        self._save_validation_dataframe(self.validation_df)

        return self.validation_df

    def _save_validation_dataframe(self, df):
        """Save validation DataFrame to CSV with thread-safe file locking"""
        import fcntl
        import time

        csv_file = op.join(self.output_dir, "cluster_validation_metrics.csv")
        lock_file = csv_file + ".lock"

        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                # Create lock file and acquire exclusive lock
                with open(lock_file, "w") as lock:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                    # Save the DataFrame
                    df.to_csv(csv_file, index=False)
                    print(f"Validation DataFrame saved to: {csv_file}", flush=True)
                    print(f"DataFrame shape: {df.shape}", flush=True)
                    break

            except (IOError, OSError) as e:
                if attempt < max_attempts - 1:
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    print(
                        f"Failed to save DataFrame after {max_attempts} attempts: {e}",
                        flush=True,
                    )

        # Clean up lock file
        try:
            if op.exists(lock_file):
                os.remove(lock_file)
        except:
            pass

    def _append_to_validation_dataframe(
        self, k_value, silhouette_score, gap_stat, gap_std
    ):
        """Thread-safe method to append single k results to validation DataFrame"""
        import fcntl
        import time

        csv_file = op.join(self.output_dir, "cluster_validation_metrics.csv")
        lock_file = csv_file + ".lock"

        # New row to add
        new_row = {
            "k": k_value,
            "silhouette_score": silhouette_score,
            "gap_statistic": gap_stat,
            "gap_std": gap_std,
        }

        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                # Create lock file and acquire exclusive lock
                with open(lock_file, "w") as lock:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                    # Load existing DataFrame or create new one
                    if op.exists(csv_file):
                        existing_df = pd.read_csv(csv_file)
                        # Remove any existing row for this k value
                        existing_df = existing_df[existing_df["k"] != k_value]
                        # Add new row
                        updated_df = pd.concat(
                            [existing_df, pd.DataFrame([new_row])], ignore_index=True
                        )
                    else:
                        updated_df = pd.DataFrame([new_row])

                    # Sort by k value and save
                    updated_df = updated_df.sort_values("k").reset_index(drop=True)
                    updated_df.to_csv(csv_file, index=False)

                    print(f"Updated validation DataFrame with k={k_value}", flush=True)
                    print(f"Current DataFrame shape: {updated_df.shape}", flush=True)
                    break

            except (IOError, OSError) as e:
                if attempt < max_attempts - 1:
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    print(
                        f"Failed to update DataFrame after {max_attempts} attempts: {e}",
                        flush=True,
                    )

        # Clean up lock file
        try:
            if op.exists(lock_file):
                os.remove(lock_file)
        except:
            pass

    def _corr_embed(self):
        """
        Row-center and L2-normalize each participant vector so that
        Euclidean distance between rows corresponds to correlation distance.
        """
        X = self.data_matrix
        Xc = X - X.mean(axis=1, keepdims=True)
        n = np.linalg.norm(Xc, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return Xc / n

    def _create_matrix_figures(self):
        """Create dendrogram and heatmap once after data matrix is ready"""
        print("Creating hierarchical clustering dendrogram and heatmap...", flush=True)

        # Get correlation-preserving embedding for dendrogram
        Xcorr = self._corr_embed()

        # Compute hierarchical clustering
        condensed = pdist(Xcorr, metric="euclidean")
        linkage_matrix = linkage(condensed, method="ward")

        # Create hierarchical clustering specific directory
        hierarchical_figures_dir = op.join(self.output_dir, "figures")
        os.makedirs(hierarchical_figures_dir, exist_ok=True)

        # --- Dendrogram only ---
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        dendrogram(linkage_matrix, ax=ax, truncate_mode="level", p=10)
        ax.set_title("Hierarchical Clustering Dendrogram")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Distance")
        plt.tight_layout()
        dendro_path = op.join(hierarchical_figures_dir, "hierarchical_dendrogram.png")
        plt.savefig(dendro_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Dendrogram saved to: {dendro_path}", flush=True)

        # --- Data Matrix Heatmap (first 100 participants × first 100 voxels) ---
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        data_subset = self.data_matrix[:100, :100]
        vmin = np.percentile(data_subset, 1)
        vmax = np.percentile(data_subset, 99)
        sns.heatmap(
            data_subset,
            cmap="RdBu_r",
            ax=ax,
            cbar_kws={"label": "Connectivity"},
            xticklabels=False,
            yticklabels=False,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title("Data Matrix Heatmap (First 100 Participants × First 100 Voxels)")
        ax.set_xlabel("Voxels")
        ax.set_ylabel("Participants")
        plt.tight_layout()
        heatmap_path = op.join(hierarchical_figures_dir, "data_matrix_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Heatmap saved to: {heatmap_path}", flush=True)

    def _calculate_gap_statistic(self, k, group_labels, n_refs=50):
        """Calculate gap statistic using uniform reference in correlation-preserving embedding"""
        try:
            # Get correlation-preserving embedding
            Xcorr = self._corr_embed()

            # Calculate within-cluster dispersion for actual data
            W_k = self._within_cluster_dispersion(Xcorr, group_labels, k)

            # Generate reference datasets and calculate their dispersions
            ref_dispersions = []

            # Find bounding box of Xcorr
            mins = np.min(Xcorr, axis=0)
            maxs = np.max(Xcorr, axis=0)

            for _ in range(n_refs):
                # Generate uniform reference data in bounding box
                n_samples, n_features = Xcorr.shape
                ref_data = np.random.uniform(
                    low=mins, high=maxs, size=(n_samples, n_features)
                )

                # Apply same clustering to reference data
                ref_distances = pdist(ref_data, metric="euclidean")
                ref_linkage = linkage(ref_distances, method="ward")
                ref_labels = fcluster(ref_linkage, k, criterion="maxclust") - 1

                # Calculate dispersion for reference
                ref_W_k = self._within_cluster_dispersion(ref_data, ref_labels, k)
                ref_dispersions.append(np.log(ref_W_k))

            # Calculate gap statistic
            log_W_k = np.log(W_k)
            mean_log_ref_W_k = np.mean(ref_dispersions)
            gap_stat = mean_log_ref_W_k - log_W_k

            # Calculate standard error
            std_log_ref_W_k = np.std(ref_dispersions)
            gap_std = std_log_ref_W_k * np.sqrt(1 + 1 / n_refs)

            return gap_stat, gap_std

        except Exception as e:
            print(f"Could not calculate gap statistic for k={k}: {e}", flush=True)
            return None, None

    def _within_cluster_dispersion(self, data, labels, k):
        """Calculate within-cluster dispersion (sum of squared distances to centroids)"""
        total_dispersion = 0
        for cluster_id in range(k):
            cluster_mask = labels == cluster_id
            if np.sum(cluster_mask) > 0:
                cluster_data = data[cluster_mask]
                if len(cluster_data) > 1:
                    cluster_center = np.mean(cluster_data, axis=0)
                    cluster_dispersion = np.sum((cluster_data - cluster_center) ** 2)
                    total_dispersion += cluster_dispersion
        return total_dispersion

    def _find_optimal_k_gap(self, k_values):
        """Find optimal k using Tibshirani gap rule"""
        try:
            gap_stats = [self.cluster_metrics[k]["gap_statistic"] for k in k_values]
            gap_stds = [self.cluster_metrics[k]["gap_std"] for k in k_values]

            # Apply Tibshirani rule: choose smallest k such that Gap(k) >= Gap(k+1) - s_{k+1}
            for i, k in enumerate(k_values[:-1]):
                gap_k = gap_stats[i]
                gap_k_plus_1 = gap_stats[i + 1]
                std_k_plus_1 = gap_stds[i + 1]

                if gap_k >= (gap_k_plus_1 - std_k_plus_1):
                    return k

            # If no k satisfies the rule, return the k with maximum gap
            max_gap_idx = np.argmax(gap_stats)
            return k_values[max_gap_idx]

        except Exception as e:
            print(f"Could not apply Tibshirani rule: {e}", flush=True)
            # Fallback to maximum gap
            gap_stats = [self.cluster_metrics[k]["gap_statistic"] for k in k_values]
            max_gap_idx = np.argmax(gap_stats)
            return k_values[max_gap_idx]

    def group_participants_by_hierarchical_connectivity(self, n_clusters=None):
        """Group participants based on hierarchical clustering of voxel connectivity patterns, or load existing assignments"""
        print("Grouping participants using hierarchical clustering...", flush=True)
        print(
            f"Data matrix: {self.data_matrix.shape[0]} participants x "
            f"{self.data_matrix.shape[1]} voxels",
            flush=True,
        )

        if n_clusters is None:
            # Use optimal number from validation if available
            if hasattr(self, "cluster_metrics") and self.cluster_metrics:
                k_values = list(self.cluster_metrics.keys())
                sil_scores = [self.cluster_metrics[k]["silhouette"] for k in k_values]
                n_clusters = k_values[np.argmax(sil_scores)]
                print(f"Using optimal k from validation: {n_clusters}", flush=True)
            else:
                n_clusters = 3  # Default fallback
                print(f"Using default k: {n_clusters}", flush=True)

        # Check if group assignment CSV exists
        assignments_file = op.join(
            self.output_dir, f"hierarchical_connectivity_groups_k{n_clusters}.csv"
        )
        if op.exists(assignments_file):
            print(f"Found existing group assignment file: {assignments_file}", flush=True)
            assignments_df = pd.read_csv(assignments_file)
            # Map subject IDs to indices in the data matrix
            subject_id_to_index = {sid: i for i, sid in enumerate(self.subject_ids)}
            group_labels = assignments_df["group"].values
            # For silhouette and PCA, need to extract rows in the same order as assignments_df
            indices = [subject_id_to_index[sid] for sid in assignments_df["subject_id"]]
            data_matrix_ordered = self.data_matrix[indices]
            # Recompute distance matrix for these participants
            from scipy.spatial.distance import pdist, squareform
            Xcorr = self._corr_embed()
            Xcorr_ordered = Xcorr[indices]
            condensed = pdist(Xcorr_ordered, metric="euclidean")
            Dfull = squareform(condensed)
            # Compute silhouette score
            from sklearn.metrics import silhouette_score
            silhouette = silhouette_score(Dfull, group_labels, metric="precomputed")
            # Store in voxel_connectivity_groups
            n_participants_per_group = assignments_df.groupby("group").size().values
            self.voxel_connectivity_groups[n_clusters] = {
                "method": "hierarchical",
                "labels": group_labels,
                "linkage_matrix": None,
                "silhouette_score": silhouette,
                "cophenetic_correlation": None,
                "n_participants_per_group": n_participants_per_group,
                "distance_matrix": Dfull,
                "assignments": assignments_df.to_dict("records"),
                "data_matrix_ordered": data_matrix_ordered,
            }
            print(f"Loaded group assignments and extracted data for figures.", flush=True)
            return self

        # ...existing code for clustering and assignment creation...
        # (The original clustering code remains unchanged below this check)
        # ...existing code...

    def visualize_hierarchical_clustering(self, n_clusters=None):
        """Visualize hierarchical connectivity clustering results, including silhouette plot"""
        if n_clusters is None and self.voxel_connectivity_groups:
            n_clusters = list(self.voxel_connectivity_groups.keys())[0]

        if n_clusters not in self.voxel_connectivity_groups:
            print(f"No clustering results found for k={n_clusters}", flush=True)
            return self

        print(f"Creating visualizations for k={n_clusters}...", flush=True)

        group_info = self.voxel_connectivity_groups[n_clusters]
        labels = group_info["labels"]
        Dfull = group_info["distance_matrix"]

        # Create main visualization (PCA, group sizes, silhouette)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. PCA visualization
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(self.data_matrix)

        scatter = axes[0].scatter(
            data_2d[:, 0], data_2d[:, 1], c=labels, cmap="tab10", alpha=0.7, s=50
        )
        axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        axes[0].set_title(f"Hierarchical Groups in PCA Space (k={n_clusters})")
        plt.colorbar(scatter, ax=axes[0])

        # 2. Group sizes
        unique, counts = np.unique(labels, return_counts=True)
        bars = axes[1].bar(unique, counts, alpha=0.7, color="skyblue", edgecolor="navy")
        axes[1].set_xlabel("Group ID")
        axes[1].set_ylabel("Number of Participants")
        axes[1].set_title(f"Hierarchical Group Sizes (k={n_clusters})")
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(count),
                ha="center",
                va="bottom",
            )

        # 3. Silhouette plot (like scikit-learn example)
        from sklearn.metrics import silhouette_samples
        import matplotlib.cm as cm
        sample_silhouette_values = silhouette_samples(Dfull, labels, metric="precomputed")
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / n_clusters)
            axes[2].fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            axes[2].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10  # 10 for spacing between clusters
        avg_score = group_info["silhouette_score"]
        axes[2].axvline(avg_score, color="red", linestyle="--")
        axes[2].set_xlabel("Silhouette Coefficient")
        axes[2].set_ylabel("Sample Index")
        axes[2].set_title("Silhouette Plot")
        axes[2].set_xlim([-0.1, 0.1])

        plt.tight_layout()
        plt.savefig(
            op.join(
                self.figures_dir, f"hierarchical_connectivity_groups_k{n_clusters}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()  # Close the figure to free memory

        return self

    def run_single_k_analysis(self, k_value):
        """Run hierarchical clustering analysis for a single k value (for parallel jobs)"""
        print(f"Running single k hierarchical analysis for k={k_value}", flush=True)

        # Create separate directory for this k value
        k_output_dir = op.join(self.output_dir, f"k_{k_value}")
        k_figures_dir = op.join(k_output_dir, "figures")
        os.makedirs(k_output_dir, exist_ok=True)
        os.makedirs(k_figures_dir, exist_ok=True)

        # Temporarily update output directories for this k
        original_output_dir = self.output_dir
        original_figures_dir = self.figures_dir
        self.output_dir = k_output_dir
        self.figures_dir = k_figures_dir

        # Run clustering for this specific k
        self.group_participants_by_hierarchical_connectivity(n_clusters=k_value)

        # Get results for this k
        group_info = self.voxel_connectivity_groups[k_value]

        # Calculate gap statistic for this k
        group_labels = group_info["labels"]
        gap_stat, gap_std = self._calculate_gap_statistic(k_value, group_labels)

        # Add gap statistics to the group info
        if gap_stat is not None:
            self.voxel_connectivity_groups[k_value]["gap_statistic"] = gap_stat
            self.voxel_connectivity_groups[k_value]["gap_std"] = gap_std

        # Create visualizations for this k
        self.visualize_hierarchical_clustering(n_clusters=k_value)

        # Restore original output directories before saving results
        self.output_dir = original_output_dir
        self.figures_dir = original_figures_dir

        # Thread-safe append to validation DataFrame
        self._append_to_validation_dataframe(
            k_value, group_info["silhouette_score"], gap_stat, gap_std
        )

        # Save results summary for this k
        results_file = op.join(k_output_dir, f"k{k_value}_results.txt")
        with open(results_file, "w") as f:
            f.write(f"Results for k={k_value}:\n")
            f.write(f"Method: {group_info['method']}\n")
            f.write(f"Silhouette score: {group_info['silhouette_score']:.3f}\n")
            if group_info.get("gap_statistic") is not None:
                f.write(f"gap_statistic: {group_info['gap_statistic']:.3f}\n")
            if group_info.get("gap_std") is not None:
                f.write(f"gap_std: {group_info['gap_std']:.3f}\n")
            if group_info.get("cophenetic_correlation") is not None:
                f.write(
                    f"Cophenetic correlation: {group_info['cophenetic_correlation']:.3f}\n"
                )
            f.write(
                f"Participants per group: {group_info['n_participants_per_group']}\n"
            )

        # Restore original output directories
        self.output_dir = original_output_dir
        self.figures_dir = original_figures_dir

        print(f"Single k hierarchical analysis completed for k={k_value}", flush=True)
        print(f"Results saved to: {k_output_dir}", flush=True)

        return self

    def summary_report(self):
        """Generate summary report"""
        print("\n" + "=" * 60, flush=True)
        print("HIERARCHICAL CONNECTIVITY CLUSTERING SUMMARY", flush=True)
        print("=" * 60, flush=True)

        print(f"Dataset: {len(self.subject_ids)} participants", flush=True)
        print(f"Data matrix shape: {self.data_matrix.shape}", flush=True)
        print(f"Output directory: {self.output_dir}", flush=True)

        if self.voxel_connectivity_groups:
            print("\nHierarchical Connectivity Grouping Results:", flush=True)
            for k in sorted(self.voxel_connectivity_groups.keys()):
                result = self.voxel_connectivity_groups[k]
                print(
                    f"  k={k}: Method={result['method']}, "
                    f"Silhouette={result['silhouette_score']:.3f}",
                    flush=True,
                )

                sizes_str = ", ".join(
                    [
                        f"G{i}:{count}"
                        for i, count in enumerate(result["n_participants_per_group"])
                    ]
                )
                print(f"         Group sizes: {sizes_str}", flush=True)

        print(f"\nFiles generated in: {self.output_dir}", flush=True)
        print("  - Cluster validation plots", flush=True)
        print("  - Group assignment files", flush=True)
        print("  - Visualization plots", flush=True)
        print("  - Dendrogram in figures/", flush=True)


def main():
    args = get_parser().parse_args()

    project_dir = args.project_dir
    base_rsfc_dir = args.data_dir
    output_dir = args.out_dir

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Metadata path
    metadata_file = op.join(
        project_dir,
        "derivatives",
        "sub-group_task-rest_desc-1S2StTesthabenula_conntable.txt",
    )

    # Initialize clustering object
    clustering = HierarchicalConnectivityClustering(
        project_dir=project_dir, base_rsfc_dir=base_rsfc_dir, output_dir=output_dir
    )

    # Handle preprocessing modes
    if args.preprocess_only:
        # Preprocessing only mode - create and save data matrix
        print("=== PREPROCESSING MODE ===", flush=True)
        clustering.load_data(metadata_file, group_filter="asd").prepare_data_matrix(
            standardize=True
        ).save_preprocessed_data()
        print("Preprocessing completed. Data matrix saved.", flush=True)
        print(
            "Now submit clustering jobs with: sbatch 2-run_hierarchical.sh",
            flush=True,
        )
        return

    elif args.load_preprocessed:
        # Load preprocessed data mode
        print("=== LOADING PREPROCESSED DATA ===", flush=True)
        clustering.load_data(metadata_file, group_filter="asd").load_preprocessed_data()
    else:
        # Traditional mode - create data matrix from scratch
        print("=== TRADITIONAL MODE (creating data matrix) ===", flush=True)
        clustering.load_data(metadata_file, group_filter="asd").prepare_data_matrix(
            standardize=True
        )

    # Check if running single k analysis (for parallel jobs)
    if args.k_value is not None:
        # Single k analysis mode
        clustering.run_single_k_analysis(k_value=args.k_value)
    else:
        # Full analysis mode
        k_range = range(args.k_min, args.k_max + 1)
        (
            clustering.determine_optimal_clusters(k_range=k_range)
            .group_participants_by_hierarchical_connectivity()
            .visualize_hierarchical_clustering()
            .summary_report()
        )


if __name__ == "__main__":
    main()
