"""
Park Surveillance - Simple Inference Runner
Interactive script to run inference on images or videos
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO

def run_inference():
    """Interactive inference runner"""
    
    # Model path
    model_path = 'runs/detect/train/weights/best.pt'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print("Please train the model first using: py train.py")
        return
    
    print("="*60)
    print("PARK SURVEILLANCE - INFERENCE")
    print("="*60)
    
    # Get input from user
    print("\nEnter the path to your image or video file:")
    print("Example: dataset/test/images/image1.jpg")
    print("Example: path/to/video.mp4")
    source = input("\nPath: ").strip()
    
    if not source:
        print("❌ Error: No path provided!")
        return
    
    # Check if source exists
    if not os.path.exists(source):
        print(f"❌ Error: File not found: {source}")
        return
    
    # Get confidence threshold
    print("\nEnter confidence threshold (0.0 to 1.0, default 0.25):")
    conf_input = input("Confidence: ").strip()
    conf = float(conf_input) if conf_input else 0.25
    
    # Determine if it's image or video
    source_path = Path(source)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    is_video = source_path.suffix.lower() in video_extensions
    
    # Set output directory
    if is_video:
        output_dir = 'inference_results/videos'
    else:
        output_dir = 'inference_results/images'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("RUNNING INFERENCE...")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Source: {source}")
    print(f"Type: {'Video' if is_video else 'Image'}")
    print(f"Confidence: {conf}")
    print(f"Output: {output_dir}")
    print("="*60 + "\n")
    
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(
        source=source,
        conf=conf,
        save=True,
        project=output_dir,
        name='detect',
        exist_ok=True
    )
    
    print("\n" + "="*60)
    print("✅ INFERENCE COMPLETED!")
    print("="*60)
    print(f"Results saved to: {output_dir}/detect")
    print("="*60)

if __name__ == '__main__':
    try:
        run_inference()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
