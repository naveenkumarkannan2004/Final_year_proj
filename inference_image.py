"""
Park Surveillance - Image Inference Script
Perform object detection on images using trained YOLO model
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime

def run_inference_on_images(
    model_path='runs/detect/train/weights/best.pt',
    source=None,
    output_dir='inference_results/images',
    conf_threshold=0.25,
    iou_threshold=0.45,
    save_results=True,
    show_labels=True,
    show_conf=True
):
    """
    Run inference on images
    
    Args:
        model_path: Path to trained YOLO model (default: runs/detect/train/weights/best.pt)
        source: Path to image file or directory (required)
        output_dir: Directory to save results (default: inference_results/images)
        conf_threshold: Confidence threshold for detections
        iou_threshold: IOU threshold for NMS
        save_results: Whether to save annotated images
        show_labels: Whether to show class labels
        show_conf: Whether to show confidence scores
    """
    
    # Validate source
    if source is None:
        raise ValueError("Source path is required! Please provide --source argument.")
    
    # Load the model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Create output directory
    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {output_path}")
    
    # Get source path
    source_path = Path(source)
    
    # Get list of images
    if source_path.is_file():
        image_files = [source_path]
    elif source_path.is_dir():
        # Get all image files in directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(source_path.glob(f'*{ext}'))
            image_files.extend(source_path.glob(f'*{ext.upper()}'))
    else:
        raise ValueError(f"Source path does not exist: {source}")
    
    print(f"\nFound {len(image_files)} images to process")
    
    # Process each image
    results_summary = []
    
    for idx, image_file in enumerate(image_files, 1):
        print(f"\nProcessing [{idx}/{len(image_files)}]: {image_file.name}")
        
        # Run inference
        results = model.predict(
            source=str(image_file),
            conf=conf_threshold,
            iou=iou_threshold,
            save=False,
            verbose=False
        )
        
        # Get the result
        result = results[0]
        
        # Count detections by class
        detections = {}
        if len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                if class_name not in detections:
                    detections[class_name] = []
                detections[class_name].append(confidence)
        
        # Print detection summary
        if detections:
            print(f"  Detections:")
            for class_name, confidences in detections.items():
                avg_conf = np.mean(confidences)
                print(f"    - {class_name}: {len(confidences)} objects (avg conf: {avg_conf:.2f})")
        else:
            print(f"  No detections")
        
        # Save annotated image
        if save_results:
            # Get annotated image
            annotated_img = result.plot(
                conf=show_conf,
                labels=show_labels,
                line_width=2,
                font_size=12
            )
            
            # Save image
            output_file = output_path / f"{image_file.stem}_detected{image_file.suffix}"
            cv2.imwrite(str(output_file), annotated_img)
            print(f"  Saved to: {output_file}")
        
        # Store summary
        results_summary.append({
            'image': image_file.name,
            'detections': detections,
            'total_objects': sum(len(v) for v in detections.values())
        })
    
    # Print overall summary
    print("\n" + "="*60)
    print("INFERENCE SUMMARY")
    print("="*60)
    print(f"Total images processed: {len(image_files)}")
    print(f"Images with detections: {sum(1 for r in results_summary if r['total_objects'] > 0)}")
    print(f"Total objects detected: {sum(r['total_objects'] for r in results_summary)}")
    
    # Count by class
    class_counts = {}
    for r in results_summary:
        for class_name, confidences in r['detections'].items():
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += len(confidences)
    
    if class_counts:
        print("\nDetections by class:")
        for class_name, count in class_counts.items():
            print(f"  - {class_name}: {count}")
    
    if save_results:
        print(f"\nAll results saved to: {output_path}")
    
    return results_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run YOLO inference on images')
    parser.add_argument('--model', type=str, default='runs/detect/train/weights/best.pt',
                        help='Path to trained YOLO model (.pt file) - default: runs/detect/train/weights/best.pt')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image file or directory (required)')
    parser.add_argument('--output', type=str, default='inference_results/images',
                        help='Output directory for results - default: inference_results/images')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (0-1)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IOU threshold for NMS (0-1)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save annotated images')
    parser.add_argument('--no-labels', action='store_true',
                        help='Do not show class labels')
    parser.add_argument('--no-conf', action='store_true',
                        help='Do not show confidence scores')
    
    args = parser.parse_args()
    
    # Run inference
    run_inference_on_images(
        model_path=args.model,
        source=args.source,
        output_dir=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        save_results=not args.no_save,
        show_labels=not args.no_labels,
        show_conf=not args.no_conf
    )
