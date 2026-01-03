
import os
import argparse
import numpy as np
import cv2
import sys
from tqdm import tqdm

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.loader import NuScenesWeatherDataset
from src.training.curriculum import WeatherAugmentor

def visualize_augmentation(data_root, version='v1.0-mini', output_dir='viz_results'):
    print(f"Generating visualizations from {version}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Dataset
    dataset = NuScenesWeatherDataset(root_dir=data_root, version=version, split='train')
    augmentor = WeatherAugmentor()
    
    # Select a few random samples to visualize (e.g. 3 distinct scenes)
    indices = np.random.choice(len(dataset), 3, replace=False)
    
    # Define difficulties to visualize
    difficulties = [0.0, 0.3, 0.6, 0.9]
    weather_types = ['rain', 'fog', 'snow']
    
    for idx_count, idx in enumerate(indices):
        print(f"Processing sample {idx}...")
        sample = dataset.get_raw_sample(idx) # Returns dict with 'image' (numpy), 'depth'
        
        orig_img = sample['image'] # (H, W, 3)
        depth = sample['depth']
        
        # Create a grid for this sample
        # Rows: Weather Types, Cols: Difficulty
        
        for w_type in weather_types:
            row_images = []
            for diff in difficulties:
                # Apply Augmentation
                # We need to manually call augmentor because dataset.__getitem__ usually handles it inside 
                # but our dataset structure separates the augmentor call in training loop
                # ACTUALLY: The dataset.__getitem__ has 'weather_generator' param but run_experiment 
                # handles it via 'set_weather_severity'.
                # We will use the augmentor instance directly here.
                
                aug_img = augmentor.apply(orig_img.copy(), depth, difficulty_lambda=diff, weather_type=w_type)
                
                # Add text
                aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR) # OpenCV uses BGR
                cv2.putText(aug_img, f"{w_type} p={diff}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                row_images.append(aug_img)
            
            # Concatenate row
            row_viz = np.hstack(row_images)
            
            # Save row
            out_name = f"sample_{idx_count}_{w_type}.jpg"
            cv2.imwrite(os.path.join(output_dir, out_name), row_viz)
            print(f"Saved {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/projects/sb2ek/datasets/nuscenes')
    parser.add_argument('--version', type=str, default='v1.0-mini')
    parser.add_argument('--output', type=str, default='viz_results')
    args = parser.parse_args()
    
    visualize_augmentation(args.data_root, args.version, args.output)
