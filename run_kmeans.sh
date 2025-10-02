#!/bin/bash
#SBATCH --job-name=kmeans_workflow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4gb
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=default-alternate
# Outputs ----------------------------------
#SBATCH --output=/home/data/nbc/misc-projects/meta-analyses/abide-hb-meta-lib/log/%x/%x_%A-%a.out
#SBATCH --error=/home/data/nbc/misc-projects/meta-analyses/abide-hb-meta-lib/log/%x/%x_%A-%a.err
# ------------------------------------------
# Max # CPUs = 360
# modified from ABIDE workflow

# IB_44C_512G, IB_40C_512G, IB_16C_96G, for running workflow
# Liberty use default-alternate partition

# investor, for testing
pwd; hostname; date
set -e

#==============Shell script==============#

PROJECT_DIR="/home/data/nbc/misc-projects/meta-analyses/abide-hb-meta-lib"
OUTPUT_DIR=${PROJECT_DIR}/derivatives/k_clustering
DATA_DIR="/home/data/nbc/Laird_ABIDE/dset/derivatives/rsfc-habenula"


# Load evironment
module load miniconda3-4.5.11-gcc-8.2.0-oqs2mbg
source activate ${PROJECT_DIR}/kmeans_env 

# Setup done, run the command
cmd="python ${PROJECT_DIR}/kmeans-workflow.py \
    --project_dir ${PROJECT_DIR} \
    --data_dir ${DATA_DIR} \
    --out_dir ${OUTPUT_DIR}"

echo Commandline: $cmd
eval $cmd


#echo Finished tasks with exit code $exitcode
exit $exitcode

date