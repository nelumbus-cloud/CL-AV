# Research Experiment: Curriculum Learning for Robust Perception

This repository contains the implementation of the research experiment described in "Curriculum Learning for Robust Perception in Extreme Weather".

## Project Structure

- `src/simulation/`: Physics-based weather models (Rain, Fog).
- `src/data/`: Data loading and processing (NuScenes + Depth Projection).
- `src/training/`: Curriculum learning logic and training loops.
- `src/experiments/`: Scripts to run experiments and evaluation.
- `src/utils/`: Helper functions.

## Prerequisites

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. **Data**: You must download the [NuScenes Dataset](https://www.nuscenes.org/download).
   - Expected structure:
     ```
     /data/sets/nuscenes/
         v1.0-trainval/
         samples/
         sweeps/
         maps/
     ```

## Running the Experiment

To run the curriculum learning training experiment:

```bash
python src/experiments/run_experiment.py --data_root /path/to/nuscenes --epochs 20
```

## Running Evaluation

To evaluate the model against the categorical difficulty splits (Validation on Real Weather data):

```bash
python src/experiments/evaluate.py --data_root /path/to/nuscenes
```
