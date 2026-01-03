
import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.loader import NuScenesWeatherDataset, collate_fn
from src.training.model import get_object_detection_model

def evaluate_robustness(data_root, checkpoint_path, version='v1.0-trainval', batch_size=4):
    print(f"Starting Robustness Evaluation (Loss-based)...")
    print(f"Checkpoint: {checkpoint_path}")
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")

    # 1. Load Model
    num_classes = 11
    model = get_object_detection_model(num_classes)
    
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")
    else:
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return

    model.to(device)
    # Note: To calculate loss, model must be in training mode? 
    # FasterRCNN returns loss dict only in train() mode. In eval() mode it returns detections.
    # For "Validation Loss", we usually keep model.train() but separate gradient calc (torch.no_grad).
    # However, model.train() also enables Dropout/BatchNorm updates.
    # Standard PyTorch FasterRCNN hack: keep .train() to get losses, but use no_grad.
    model.train() 

    # 2. Setup Dataset (Validation Split)
    dataset = NuScenesWeatherDataset(root_dir=data_root, split='val', version=version)
    
    # 3. Define Test Conditions (Synthetic Curriculums)
    # We test on Fog at different severities to see if Curriculum learned robustness.
    weather_conditions = [
        ('clear', 0.0),
        ('fog_light', 0.3),
        ('fog_med', 0.6),
        ('fog_heavy', 0.9),
        ('rain_heavy', 0.9),
        ('snow_heavy', 0.9)
    ]
    
    results = {}

    for name, severity in weather_conditions:
        print(f"\nEvaluating Condition: {name} (Severity={severity})...")
        
        # Configure Dataset
        if 'fog' in name:
            dataset.set_weather_severity(severity, mode='fog')
        elif 'rain' in name:
            dataset.set_weather_severity(severity, mode='rain')
        elif 'snow' in name:
            dataset.set_weather_severity(severity, mode='snow')
        else:
            dataset.set_weather_severity(0.0) # Clear

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn
        )

        total_loss = 0
        batches = 0
        
        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc=name):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
                batches += 1
                
                # Limit validation size for speed if on mini
                if version == 'v1.0-mini' and batches > 20:
                    break

        avg_loss = total_loss / batches if batches > 0 else 0
        results[name] = avg_loss
        print(f"Result {name}: Loss = {avg_loss:.4f}")

    print("\n--- Final Robustness Profile (Validation Loss) ---")
    print(f"{'Condition':<15} | {'Loss':<10}")
    print("-" * 30)
    for name, loss in results.items():
        print(f"{name:<15} | {loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/projects/sb2ek/datasets/nuscenes')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-mini', 'v1.0-trainval'])
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    
    evaluate_robustness(args.data_root, args.checkpoint, version=args.version, batch_size=args.batch_size)
