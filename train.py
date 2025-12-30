"""
Park Surveillance - YOLO Training Script
Train YOLOv11 model to detect authorized and unauthorized activities in parks
"""

import os
from ultralytics import YOLO
import torch
import argparse
from pathlib import Path

def train_model(
    data_yaml='dataset/data.yaml',
    model_size='n',
    epochs=100,
    batch_size=16,
    img_size=512,
    device='',
    project='runs/detect',
    name='train'
):
    """
    Train YOLO model for park surveillance
    
    Args:
        data_yaml: Path to data.yaml file
        model_size: Model size (n/s/m/l/x) - n is fastest, x is most accurate
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Image size for training
        device: Device to use ('' for auto, 'cpu', '0' for GPU 0)
        project: Project directory for saving results
        name: Name of the training run
    """
    
    # Check if CUDA is available
    if device == '':
        device = '0' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Training YOLOv11{model_size} model")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {img_size}")
    
    # Initialize YOLO model
    model = YOLO(f'yolo11{model_size}.pt')
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        patience=50,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        plots=True,  # Generate plots
        verbose=True,
        val=True,  # Validate during training
        pretrained=True,
        optimizer='auto',
        lr0=0.01,  # Initial learning rate
        lrf=0.01,  # Final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        box=7.5,  # Box loss gain
        cls=0.5,  # Class loss gain
        dfl=1.5,  # Distribution focal loss gain
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,  # HSV-Saturation augmentation
        hsv_v=0.4,  # HSV-Value augmentation
        degrees=0.0,  # Rotation augmentation
        translate=0.1,  # Translation augmentation
        scale=0.5,  # Scaling augmentation
        shear=0.0,  # Shear augmentation
        perspective=0.0,  # Perspective augmentation
        flipud=0.0,  # Vertical flip augmentation
        fliplr=0.5,  # Horizontal flip augmentation
        mosaic=1.0,  # Mosaic augmentation
        mixup=0.0,  # Mixup augmentation
    )
    
    # Print training results
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    
    # Get the best model path
    best_model_path = Path(project) / name / 'weights' / 'best.pt'
    last_model_path = Path(project) / name / 'weights' / 'last.pt'
    
    print(f"\nBest model saved to: {best_model_path}")
    print(f"Last model saved to: {last_model_path}")
    
    # Validate the best model
    print("\nValidating best model...")
    best_model = YOLO(str(best_model_path))
    metrics = best_model.val()
    
    print("\n" + "="*50)
    print("Validation Metrics:")
    print("="*50)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    return best_model_path, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLO model for park surveillance')
    parser.add_argument('--data', type=str, default='dataset/data.yaml', 
                        help='Path to data.yaml file')
    parser.add_argument('--model', type=str, default='n', 
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, 
                        help='Batch size')
    parser.add_argument('--img-size', type=int, default=512, 
                        help='Image size for training')
    parser.add_argument('--device', type=str, default='', 
                        help='Device to use (empty for auto, cpu, or 0 for GPU)')
    parser.add_argument('--project', type=str, default='runs/detect', 
                        help='Project directory')
    parser.add_argument('--name', type=str, default='train', 
                        help='Name of the training run')
    
    args = parser.parse_args()
    
    # Train the model
    train_model(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device,
        project=args.project,
        name=args.name
    )
