
You are now ready to run the research experiments.

### Verification (Mini Dataset)
To ensure everything is working correctly on the `mini` dataset (which you should ensure is present in `/projects/sb2ek/datasets/nuscenes`):

```bash
python src/experiments/run_experiment.py --verify
```
*Expected Output:* Training should run for 2 epochs on the mini dataset and complete without errors.

### Full Experiment (TrainVal Dataset)
Once verified, run the full curriculum learning experiment:

```bash
python src/experiments/run_experiment.py --version v1.0-trainval --curriculum_mode linear --epochs 20
```

### Evaluation
To evaluate the models:
```bash
python src/experiments/evaluate.py --version v1.0-trainval --data_root /projects/sb2ek/datasets/nuscenes
```
