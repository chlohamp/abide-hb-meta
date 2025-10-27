#!/bin/bash
#SBATCH --job-name=k_cluster_maps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2gb
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=IB_44C_512G
#SBATCH --time=00:30:00
# Outputs ----------------------------------
#SBATCH --output=/home/data/nbc/misc-projects/meta-analyses/abide-hb-meta/log/%x/%x_%j.out
#SBATCH --error=/home/data/nbc/misc-projects/meta-analyses/abide-hb-meta/log/%x/%x_%j.err
# ------------------------------------------
# Extract and plot cluster mean maps
# Submit with: sbatch 4-run_k_cluster_maps.sh
# Monitor with: squeue -u $USER

pwd; hostname; date
set -e

#==============Shell script==============#

PROJECT_DIR="/home/data/nbc/misc-projects/meta-analyses/abide-hb-meta"
OUTPUT_DIR=${PROJECT_DIR}/derivatives/hierarchical_clustering
K_OPTIMAL=2           # Set optimal k value here
THRESHOLD=None        # Threshold for plotting (use 'None' for auto-threshold)
USE_PVAL=true         # Set to 'true' to use p-value thresholding
PVAL_THRESHOLD=0.05   # Voxel-level p-value threshold (only used if USE_PVAL=true)
CLUSTER_THRESHOLD=10  # Cluster size threshold in voxels (only used if USE_PVAL=true)

# Load environment
module load miniconda3-4.5.11-gcc-8.2.0-oqs2mbg
source activate /home/champ007/kmeans_env

# Run the extraction/plotting script
cmd="python -u ${PROJECT_DIR}/k_cluster_maps.py \
    --results_dir ${OUTPUT_DIR} \
    --k_optimal ${K_OPTIMAL}"

# Add threshold if specified
if [[ "${THRESHOLD}" != "None" ]]; then
    cmd="${cmd} --threshold ${THRESHOLD}"
fi

# Add p-value thresholding if enabled
if [[ "${USE_PVAL}" == "true" ]]; then
    cmd="${cmd} --use_pval --pval_threshold ${PVAL_THRESHOLD} --cluster_threshold ${CLUSTER_THRESHOLD}"
fi

echo "Commandline: $cmd"
eval $cmd

if [[ $? -eq 0 ]]; then
    echo "==================================================="
    echo "CLUSTER MAP EXTRACTION COMPLETED SUCCESSFULLY"
    echo "==================================================="
    echo "Maps saved to: ${OUTPUT_DIR}/figures/"
    echo "Files generated: k_cluster_maps_k${K_OPTIMAL}_*.nii.gz"
    echo "==================================================="
else
    echo "==================================================="
    echo "ERROR: Extraction/plotting failed!"
    echo "==================================================="
    exit 1
fi

exit 0

date
