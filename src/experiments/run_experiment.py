
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.loader import NuScenesWeatherDataset, collate_fn
from src.training.curriculum import CurriculumSampler, WeatherAugmentor
from src.training.model import get_object_detection_model

def run_training_experiment(data_root, epochs=10, curriculum_mode='linear', batch_size=4, lr=0.005):
    print(f"Starting REAL Research Experiment: Curriculum={curriculum_mode}, Epochs={epochs}")
    
    # 1. Setup Data & Model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Dataset
    # We need a custom collate_fn because images/boxes are variable size
    print(f"Loading dataset from {data_root}...")
    dataset = NuScenesWeatherDataset(root_dir=data_root, split='train')
    
    # DataLoader
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=collate_fn 
    )

    # Model: Faster R-CNN
    num_classes = 11 # 10 NuScenes classes + background
    model = get_object_detection_model(num_classes)
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Curriculum Strategy
    augmentor = WeatherAugmentor()
    sampler = CurriculumSampler(mode=curriculum_mode, total_epochs=epochs)

    # 2. Training Loop
    model.train()
    
    for epoch in range(epochs):
        # Determine Current Difficulty for this Epoch
        current_lambda = sampler.get_difficulty_lambda(epoch)
        print(f"\n--- Epoch {epoch+1}/{epochs} | Difficulty Lambda: {current_lambda:.2f} ---")
        
        # Inject difficulty into dataset for this epoch
        # (Since we are doing per-epoch curriculum, we update the dataset's global weather param)
        dataset.set_weather_severity(current_lambda)
        
        total_loss = 0
        iteration = 0
        
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Clear Gradients
            optimizer.zero_grad()
            
            # Forward Pass (Loss is calculated internally by Faster R-CNN in train mode)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward Pass
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
            
            if iteration % 10 == 0:
                print(f"Epoch: {epoch+1}, Iter: {iteration}, Loss: {losses.item():.4f}")
            iteration += 1

        print(f"Epoch {epoch+1} Average Loss: {total_loss/len(data_loader)}")
        
        # Save Checkpoint
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")


    print("Experiment Completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path configuration
    parser.add_argument('--data_root', type=str, default='/projects/sb2ek/datasets/nuscenes', help='Path to NuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-mini', choices=['v1.0-mini', 'v1.0-trainval'], help='NuScenes version to use')
    
    # Experiment configuration
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--curriculum_mode', type=str, default='linear', choices=['linear', 'step', 'random', 'clear_only'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.005)
    
    # Mode flags
    parser.add_argument('--verify', action='store_true', help='Run a quick verification on mini dataset with few epochs')
    
    args = parser.parse_args()
    
    # Validation / Overrides for verification mode
    if args.verify:
        print("--- VERIFICATION MODE ---")
        args.version = 'v1.0-mini'
        args.epochs = 2
        args.batch_size = 2
        print(f"Forcing version={args.version}, epochs={args.epochs}, batch_size={args.batch_size} for verification.")

    # Call main training function
    run_training_experiment(
        data_root=args.data_root, 
        epochs=args.epochs,
        curriculum_mode=args.curriculum_mode,
        batch_size=args.batch_size,
        lr=args.lr,
        version=args.version
    )
