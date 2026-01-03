#!/bin/bash
#SBATCH -p research-gpu
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 120
#SBATCH --job-name=cl_av_exp
#SBATCH --output=experiment_output_%j.log

# 1. Environment Setup
echo "Setting up environment..."

# Go to the directory where sbatch was called (The project root)
cd $SLURM_SUBMIT_DIR
echo "Working Directory: $(pwd)"

# Init Conda
# Try sourcing the system conda profile directly if typically available
# Or rely on user's .bashrc but ensure it runs despite non-interactive
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

# Explicitly try to load modules or init conda
module load miniconda || echo "Module load miniconda failed or ignored"
module load cuda/12.1 || echo "Module load cuda failed or ignored"

# If conda is not a function, try to find it
if ! command -v conda &> /dev/null; then
    echo "Conda not found in PATH. Trying common locations..."
    export PATH=/opt/ohpc/pub/apps/miniconda/bin:$PATH
fi

# Activate Environment
echo "Activating 'cl-av'..."
source activate cl-av || conda activate cl-av

# Debug checks
which python
python --version
nvidia-smi

# 2. Run Experiments

# A. Baseline: Random Augmentation
echo "--- Starting Baseline (Random) Experiment ---"
python src/experiments/run_experiment.py \
    --version v1.0-trainval \
    --epochs 12 \
    --curriculum_mode random \
    --batch_size 4 \
    --lr 0.005

# B. Proposed: Linear Curriculum
echo "--- Starting Proposed (Curriculum) Experiment ---"
python src/experiments/run_experiment.py \
    --version v1.0-trainval \
    --epochs 12 \
    --curriculum_mode linear \
    --batch_size 4 \
    --lr 0.005

# 3. Robustness Evaluation on Both
echo "--- Evaluating Baseline ---"
# Find latest random checkpoint
CKPT_RAND=$(ls -t checkpoint_random_epoch_*.pth | head -1)
python src/experiments/evaluate.py --checkpoint $CKPT_RAND --version v1.0-trainval

echo "--- Evaluating Curriculum ---"
# Find latest linear checkpoint
CKPT_LIN=$(ls -t checkpoint_linear_epoch_*.pth | head -1)
python src/experiments/evaluate.py --checkpoint $CKPT_LIN --version v1.0-trainval

# 4. Push Results
echo "--- Pushing Results to GitHub ---"
git add *.csv *.pth experiment_output_*.log
git commit -m "Automated experiment results from Slurm Job $SLURM_JOB_ID"
git push
