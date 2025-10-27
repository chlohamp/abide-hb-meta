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

# K values to process - can be a single value or space-separated list
K_VALUES="3 4"          # Examples: "2" or "2 3 4 5" or "2 3 4 5 6 7 8 9"

THRESHOLD=None        # Threshold for plotting (use 'None' for auto-threshold)
USE_PVAL=true         # Set to 'true' to use p-value thresholding
PVAL_THRESHOLD=0.05   # Voxel-level p-value threshold (only used if USE_PVAL=true)
CLUSTER_THRESHOLD=20  # Cluster size threshold in voxels (only used if USE_PVAL=true)
CORRECTION=all        # Correction method: none, fwe, fdr, or all (default: none)
FWE_THRESHOLD=0.05    # FWE-corrected alpha level (only used if CORRECTION=fwe or all)
FDR_THRESHOLD=0.05    # FDR-corrected q-value threshold (only used if CORRECTION=fdr or all)

# Load environment
module load miniconda3-4.5.11-gcc-8.2.0-oqs2mbg
source activate /home/champ007/kmeans_env

echo "=========================================="
echo "Processing K values: ${K_VALUES}"
echo "=========================================="

# Loop through each k value
for K_VALUE in ${K_VALUES}; do
    echo ""
    echo "=========================================="
    echo "Processing K = ${K_VALUE}"
    echo "=========================================="
    
    # Run the extraction/plotting script
    cmd="python -u ${PROJECT_DIR}/k_cluster_maps.py \
        --results_dir ${OUTPUT_DIR} \
        --k_optimal ${K_VALUE}"

    # Add threshold if specified
    if [[ "${THRESHOLD}" != "None" ]]; then
        cmd="${cmd} --threshold ${THRESHOLD}"
    fi

    # Add p-value thresholding if enabled
    if [[ "${USE_PVAL}" == "true" ]]; then
        cmd="${cmd} --use_pval --pval_threshold ${PVAL_THRESHOLD} --cluster_threshold ${CLUSTER_THRESHOLD}"
        cmd="${cmd} --correction ${CORRECTION}"
        
        # Add FWE and FDR thresholds if using those corrections
        if [[ "${CORRECTION}" == "fwe" ]] || [[ "${CORRECTION}" == "all" ]]; then
            cmd="${cmd} --fwe_threshold ${FWE_THRESHOLD}"
        fi
        if [[ "${CORRECTION}" == "fdr" ]] || [[ "${CORRECTION}" == "all" ]]; then
            cmd="${cmd} --fdr_threshold ${FDR_THRESHOLD}"
        fi
    fi

    echo "Commandline: $cmd"
    eval $cmd

    if [[ $? -eq 0 ]]; then
        echo "==================================================="
        echo "K=${K_VALUE}: COMPLETED SUCCESSFULLY"
        echo "==================================================="
    else
        echo "==================================================="
        echo "ERROR: K=${K_VALUE} failed!"
        echo "==================================================="
        exit 1
    fi
done

echo ""
echo "==================================================="
echo "ALL CLUSTER MAP EXTRACTIONS COMPLETED SUCCESSFULLY"
echo "==================================================="
echo "Maps saved to: ${OUTPUT_DIR}/k_*/nifti/ and ${OUTPUT_DIR}/k_*/figures/"
echo "Processed K values: ${K_VALUES}"
echo "==================================================="

exit 0

date
