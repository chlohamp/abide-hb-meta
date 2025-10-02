# abide-hb-meta

Clustering autistic participants based on **whole-brain habenula functional connectivity** (rsFC).  
This project builds participant × participant similarity matrices from rsFC maps and applies K-means clustering across a range of *k*, saving cluster assignments, centroid maps, and QC figures.

---

## Repository structure

```
.
├── kmeans-workflow.py        # Main pipeline (load maps, build similarity, run K-means, generate outputs)
├── run_kmeans.sh             # HPC launcher for the workflow (SLURM job script)
├── utils.py                  # Helper functions (e.g., thresholding)
├── derivatives/              # Output directory (results and figures saved here)
└── log/kmeans_workflow/      # Optional logs
```

---

## Pipeline overview

1. **Load metadata** (`.tsv`) and construct paths to participant rsFC maps.
2. **Mask & vectorize** maps into a participant × voxel data matrix.
3. **Compute similarity** across participants (Pearson correlation).
4. **Determine optimal k** (silhouette and elbow methods on PCA-reduced distance matrix).
5. **Perform clustering** (K-means for k in a user-specified range).
6. **Generate outputs**:
   - Cluster **centroid beta** maps and **z-maps** (+ thresholded versions).
   - **Cluster assignments** (CSV per k and TXT lists per cluster).
   - **Figures**: similarity heatmaps, dendrogram, clustermaps, k-validation curves, PCA scatter plots, etc.

---

## Inputs

- **Metadata file** (TSV/TXT) with at least:
  - `Subj` → subject ID  
  - `group` → group label (e.g., `asd`)  
  - `InputFile` → path beginning with `/rsfc/...` that points to each participant’s rsFC NIfTI (rewritten to your `base_rsfc_dir`)
- **rsFC maps directory** (`base_rsfc_dir`): root folder containing the participant rsFC NIfTI maps.

> By default, the workflow filters to `group == "asd"`. This can be modified or disabled.

---

## Outputs

All results are saved under `derivatives/k_clustering_matrix/`:

```
derivatives/k_clustering_matrix/
├── figures/
│   ├── similarity_analysis.png
│   ├── similarity_clustered.png
│   ├── cluster_validation.png
│   └── clusters_k_{k}_visualization.png
├── cluster_validation_scores.txt
└── k_{k}_clusters/
    ├── cluster_{c}_centroid_beta.nii.gz
    ├── cluster_{c}_beta_thresh-0.2.nii.gz
    ├── cluster_{c}_centroid_zmap.nii.gz
    ├── cluster_{c}_z_thresh-3.09.nii.gz
    ├── cluster_{c}_participants.txt
    └── cluster_assignments_k_{k}.csv
```

---

## Software requirements

- Python ≥ 3.9  
- Packages:
  - `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`
  - `nibabel`, `nilearn`
  - `fastcluster` *(optional, currently imported but not required)*

**Conda example:**
```bash
conda create -n hbmeta python=3.10 -y
conda activate hbmeta
pip install pandas numpy scipy scikit-learn matplotlib seaborn nibabel nilearn fastcluster
```

---

## Usage

### Local run
Edit the `main()` section of `kmeans-workflow.py`:
```python
project_dir = "/path/to/project"
metadata_file = op.join(project_dir, "derivatives/your_metadata_file.txt")
base_rsfc_dir = "/path/to/rsfc_maps"
```

Then run:
```bash
python kmeans-workflow.py
```

### HPC run
Submit the SLURM job script:
```bash
sbatch run_kmeans.sh
```

This script wraps `kmeans-workflow.py` for execution on the cluster, producing the same outputs under `derivatives/k_clustering_matrix/`.

---

## License
[MIT License](LICENSE)
