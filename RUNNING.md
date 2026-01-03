
## Running Instructions

### 1. Verify Pipeline (Quick Test)
Use the `mini` dataset to verify that the pipeline runs correctly without errors. This forces 2 epochs and a small batch size.

```bash
python src/experiments/run_experiment.py --verify
```

### 2. Full Training Experiment
Run the full experiment on the complete dataset (e.g., `v1.0-trainval`).

**Proposed Method (Curriculum Learning):**
```bash
python src/experiments/run_experiment.py --version v1.0-trainval --curriculum_mode linear --epochs 20
```

**Baseline (Random Augmentation):**
```bash
python src/experiments/run_experiment.py --version v1.0-trainval --curriculum_mode random --epochs 20
```

### 3. Downloading Data
Use the helper script to download the dataset if needed:
```bash
python src/data/download_nuscenes.py
```
