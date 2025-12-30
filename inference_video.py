"""
Park Surveillance - Video Inference Script
Perform object detection on videos using trained YOLO model
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import time
from datetime import datetime

def run_inference_on_video(
    model_path='runs/detect/train/weights/best.pt',
    source=None,
    output_dir='inference_results/videos',
    conf_threshold=0.25,
    iou_threshold=0.45,
    save_video=True,
    show_labels=True,
    show_conf=True,
    display=False
):
    """
    Run inference on video
    
    Args:
        model_path: Path to trained YOLO model (default: runs/detect/train/weights/best.pt)
        source: Path to video file (required)
        output_dir: Directory to save results (default: inference_results/videos)
        conf_threshold: Confidence threshold for detections
        iou_threshold: IOU threshold for NMS
        save_video: Whether to save output video
        show_labels: Whether to show class labels
        show_conf: Whether to show confidence scores
        display: Whether to display video while processing
    """
    
    # Validate source
    if source is None:
        raise ValueError("Source path is required! Please provide --source argument.")
    
    # Load the model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Open video
    source_path = Path(source)
    if not source_path.exists():
        raise ValueError(f"Video file does not exist: {source}")
    
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {source}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    
    # Create output directory and video writer
    video_writer = None
    if save_video:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / f"{source_path.stem}_detected.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_file),
            fourcc,
            fps,
            (width, height)
        )
        print(f"\nOutput will be saved to: {output_file}")
    
    # Process video
    print("\nProcessing video...")
    frame_count = 0
    detection_stats = {
        'total_frames': 0,
        'frames_with_detections': 0,
        'total_detections': 0,
        'class_counts': {}
    }
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run inference
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        result = results[0]
        
        # Update statistics
        detection_stats['total_frames'] += 1
        if len(result.boxes) > 0:
            detection_stats['frames_with_detections'] += 1
            detection_stats['total_detections'] += len(result.boxes)
            
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                if class_name not in detection_stats['class_counts']:
                    detection_stats['class_counts'][class_name] = 0
                detection_stats['class_counts'][class_name] += 1
        
        # Get annotated frame
        annotated_frame = result.plot(
            conf=show_conf,
            labels=show_labels,
            line_width=2,
            font_size=12
        )
        
        # Add frame counter and FPS
        elapsed_time = time.time() - start_time
        processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        cv2.putText(
            annotated_frame,
            f"Frame: {frame_count}/{total_frames} | FPS: {processing_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Save frame
        if video_writer is not None:
            video_writer.write(annotated_frame)
        
        # Display frame
        if display:
            cv2.imshow('Park Surveillance', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopped by user")
                break
        
        # Print progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames} frames) | FPS: {processing_fps:.1f}")
    
    # Cleanup
    cap.release()
    if video_writer is not None:
        video_writer.release()
    if display:
        cv2.destroyAllWindows()
    
    # Print summary
    elapsed_time = time.time() - start_time
    print("\n" + "="*60)
    print("VIDEO INFERENCE SUMMARY")
    print("="*60)
    print(f"Total frames processed: {detection_stats['total_frames']}")
    print(f"Frames with detections: {detection_stats['frames_with_detections']}")
    print(f"Total detections: {detection_stats['total_detections']}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Average FPS: {frame_count/elapsed_time:.2f}")
    
    if detection_stats['class_counts']:
        print("\nDetections by class:")
        for class_name, count in detection_stats['class_counts'].items():
            print(f"  - {class_name}: {count}")
    
    if save_video:
        print(f"\nOutput video saved to: {output_file}")
    
    return detection_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run YOLO inference on video')
    parser.add_argument('--model', type=str, default='runs/detect/train/weights/best.pt',
                        help='Path to trained YOLO model (.pt file) - default: runs/detect/train/weights/best.pt')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to video file (required)')
    parser.add_argument('--output', type=str, default='inference_results/videos',
                        help='Output directory for results - default: inference_results/videos')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (0-1)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IOU threshold for NMS (0-1)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save output video')
    parser.add_argument('--no-labels', action='store_true',
                        help='Do not show class labels')
    parser.add_argument('--no-conf', action='store_true',
                        help='Do not show confidence scores')
    parser.add_argument('--display', action='store_true',
                        help='Display video while processing')
    
    args = parser.parse_args()
    
    # Run inference
    run_inference_on_video(
        model_path=args.model,
        source=args.source,
        output_dir=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        save_video=not args.no_save,
        show_labels=not args.no_labels,
        show_conf=not args.no_conf,
        display=args.display
    )
