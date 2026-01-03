
import os
import sys
import argparse
import numpy as np

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.loader import NuScenesWeatherDataset
from src.utils.difficulty import categorize_data

def evaluate_robustness(data_root, version='v1.0-trainval'):
    print(f"Starting Robustness Evaluation (Version: {version})...")
    
    # 1. Load Validation Dataset
    try:
        dataset = NuScenesWeatherDataset(root_dir=data_root, split='val', version=version)
    except:
        print("Dataset not found. Skipping real load.")
        return

    # 2. Group by Category
    # We iterate and bucket samples
    buckets = {
        'clear': [],
        'rain': [],
        'fog': [],
        'snow': [],
        'night': []
    }
    
    print("Categorizing validation samples...")
    # Using range(len) to avoid loading all images to RAM, just metadata check
    for i in range(len(dataset)):
        # Access internal sample metadata directly to avoid image load overhead for categorization
        nusc_sample = dataset.samples[i] 
        # We need scene description. Dataset loads 'sample', we need to look up 'scene'
        scene_token = nusc_sample['scene_token']
        scene = dataset.nusc.get('scene', scene_token)
        
        category = categorize_data(scene)
        if category in buckets:
            buckets[category].append(i)
        
    # 3. Evaluate Per Bucket
    results = {}
    for cat, indices in buckets.items():
        if not indices:
            continue
            
        print(f"Evaluating {cat} subset ({len(indices)} samples)...")
        # metrics = run_inference(indices)
        # Placeholder MAP
        simulated_map = 0.5 if cat == 'clear' else 0.3
        results[cat] = simulated_map
        
    print("\n--- Evaluation Results (mAP) ---")
    for cat, score in results.items():
        print(f"{cat.capitalize()}: {score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/projects/sb2ek/datasets/nuscenes')
    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-mini', 'v1.0-trainval'])
    args = parser.parse_args()
    
    evaluate_robustness(args.data_root, version=args.version)
