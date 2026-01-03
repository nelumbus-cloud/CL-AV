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
# Robust initialization for Slurm
export PATH=/opt/ohpc/pub/apps/miniconda/bin:$PATH

# Initialize Conda for bash
eval "$(conda shell.bash hook)"

# Activate Environment
echo "Activating 'cl-av'..."
conda activate cl-av

# Fallback: If python is still not found, try adding the env path directly
# Assuming standard location ~/.conda/envs/cl-av or similar.
# We check if python works.
if ! command -v python &> /dev/null; then
    echo "Conda activate failed to put python in path. Trying to guess env path..."
    # Try finding the env path
    ENV_PATH=$(conda info --envs | grep cl-av | awk '{print $NF}')
    if [ -n "$ENV_PATH" ]; then
        echo "Found env at $ENV_PATH, adding to PATH"
        export PATH=$ENV_PATH/bin:$PATH
    fi
fi

# Debug checks
echo "Python location: $(which python)"
python --version
nvidia-smi

# 2. Run Experiments

# A. Baseline: Random Augmentation
echo "--- Starting Baseline (Random) Experiment ---"
python src/experiments/run_experiment.py \
    --version v1.0-mini \
    --epochs 20 \
    --curriculum_mode random \
    --batch_size 4 \
    --lr 0.005

# B. Proposed: Linear Curriculum
echo "--- Starting Proposed (Curriculum) Experiment ---"
python src/experiments/run_experiment.py \
    --version v1.0-mini \
    --epochs 20 \
    --curriculum_mode linear \
    --batch_size 4 \
    --lr 0.005

# 3. Robustness Evaluation on Both
echo "--- Evaluating Baseline ---"
# Find latest random checkpoint
CKPT_RAND=$(ls -t checkpoint_random_epoch_*.pth | head -1)
python src/experiments/evaluate.py --checkpoint $CKPT_RAND --version v1.0-mini

echo "--- Evaluating Curriculum ---"
# Find latest linear checkpoint
CKPT_LIN=$(ls -t checkpoint_linear_epoch_*.pth | head -1)
python src/experiments/evaluate.py --checkpoint $CKPT_LIN --version v1.0-mini

# 4. Push Results
echo "--- Pushing Results to GitHub ---"
git add *.csv *.pth experiment_output_*.log
git commit -m "Automated experiment results from Slurm Job $SLURM_JOB_ID"
git push || echo "WARNING: Git push failed. Please push manually from the login node."
