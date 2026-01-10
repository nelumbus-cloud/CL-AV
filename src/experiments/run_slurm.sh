#!/bin/bash
#SBATCH -p research-gpu
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 560 
#SBATCH --job-name=cl_av_exp
#SBATCH --output=experiment_output_%j.log
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sb2ek@mtmail.mtsu.edu

cd $SLURM_SUBMIT_DIR

export PATH=/opt/ohpc/pub/apps/miniconda/bin:$PATH
eval "$(conda shell.bash hook)"
conda activate cl-av

EXPERIMENTAL_MODE="random"
# Define a temporary directory for checkpoints
TMP_CHECKPOINT_DIR="/tmp/$SLURM_JOB_ID/checkpoints"
mkdir -p $TMP_CHECKPOINT_DIR

python src/experiments/run_experiment.py \
    --version v1.0-mini \
    --epochs 20 \
    --curriculum_mode $EXPERIMENTAL_MODE \
    --batch_size 4 \
    --lr 0.005 \
    --resume \
    --checkpoint_dir $TMP_CHECKPOINT_DIR
