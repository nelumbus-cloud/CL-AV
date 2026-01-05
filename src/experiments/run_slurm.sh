#!/bin/bash
#SBATCH -p a100
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 60
#SBATCH --job-name=cl_av_exp
#SBATCH --output=experiment_output_%j.log
#SBATCH --mail-type=END
#SBATCH --mail-user=sb2ek@mtmail.mtsu.edu

cd $SLURM_SUBMIT_DIR

export PATH=/opt/ohpc/pub/apps/miniconda/bin:$PATH
eval "$(conda shell.bash hook)"
conda activate cl-av

EXPERIMENT_MODE="random"

python src/experiments/run_experiment.py \
    --version v1.0-mini \
    --epochs 20 \
    --curriculum_mode $EXPERIMENT_MODE \
    --batch_size 4 \
    --lr 0.005 \
    --resume
