#!/bin/bash
#SBATCH -p research-gpu
#SBATCH --gres=gpu:A5000:1
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

EXPERIMENTAL_MODE="clear_only"
# Define a persistent directory for checkpoints (relative to submit dir)
CHECKPOINT_DIR="checkpoints"
mkdir -p $CHECKPOINT_DIR

python src/experiments/run_experiment.py \
    --version v1.0-mini \
    --epochs 20 \
    --curriculum_mode $EXPERIMENTAL_MODE \
    --batch_size 4 \
    --lr 0.005 \
    --resume \
    --checkpoint_dir $CHECKPOINT_DIR

# Generate Plots
python src/experiments/plot_loss.py --log_file log_${EXPERIMENTAL_MODE}.csv --output_file loss_plot_${EXPERIMENTAL_MODE}.png
