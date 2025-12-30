"""
Park Surveillance - Real-time Webcam Inference Script
Perform real-time object detection using webcam
"""

import argparse
from ultralytics import YOLO
import cv2
import time
import numpy as np
from collections import deque

def run_webcam_inference(
    model_path,
    camera_id=0,
    conf_threshold=0.25,
    iou_threshold=0.45,
    show_labels=True,
    show_conf=True,
    alert_on_unauthorized=True,
    save_detections=False,
    output_dir='runs/inference/webcam'
):
    """
    Run real-time inference on webcam feed
    
    Args:
        model_path: Path to trained YOLO model
        camera_id: Camera device ID (0 for default webcam)
        conf_threshold: Confidence threshold for detections
        iou_threshold: IOU threshold for NMS
        show_labels: Whether to show class labels
        show_conf: Whether to show confidence scores
        alert_on_unauthorized: Show alert when unauthorized activity detected
        save_detections: Save frames with detections
        output_dir: Directory to save detection frames
    """
    
    # Load the model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Open webcam
    print(f"Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open camera {camera_id}")
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get actual properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Camera opened successfully")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    
    # Create output directory if saving detections
    if save_detections:
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"\nDetection frames will be saved to: {output_path}")
    
    # FPS calculation
    fps_queue = deque(maxlen=30)
    
    # Detection statistics
    stats = {
        'total_frames': 0,
        'frames_with_detections': 0,
        'unauthorized_count': 0,
        'authorized_count': 0
    }
    
    print("\n" + "="*60)
    print("REAL-TIME PARK SURVEILLANCE")
    print("="*60)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("  'r' - Reset statistics")
    print("="*60 + "\n")
    
    frame_count = 0
    last_alert_time = 0
    alert_cooldown = 2.0  # seconds between alerts
    
    try:
        while True:
            start_time = time.time()
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break
            
            frame_count += 1
            stats['total_frames'] += 1
            
            # Run inference
            results = model.predict(
                source=frame,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            result = results[0]
            
            # Process detections
            has_detections = len(result.boxes) > 0
            has_unauthorized = False
            
            if has_detections:
                stats['frames_with_detections'] += 1
                
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    if class_name == 'Unauthorised':
                        stats['unauthorized_count'] += 1
                        has_unauthorized = True
                    elif class_name == 'authorised':
                        stats['authorized_count'] += 1
            
            # Get annotated frame
            annotated_frame = result.plot(
                conf=show_conf,
                labels=show_labels,
                line_width=2,
                font_size=12
            )
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps_queue.append(1.0 / elapsed if elapsed > 0 else 0)
            current_fps = np.mean(fps_queue)
            
            # Add overlay information
            overlay_y = 30
            
            # FPS counter
            cv2.putText(
                annotated_frame,
                f"FPS: {current_fps:.1f}",
                (10, overlay_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            overlay_y += 30
            
            # Detection count
            if has_detections:
                cv2.putText(
                    annotated_frame,
                    f"Detections: {len(result.boxes)}",
                    (10, overlay_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                overlay_y += 30
            
            # Alert for unauthorized activity
            if alert_on_unauthorized and has_unauthorized:
                current_time = time.time()
                if current_time - last_alert_time > alert_cooldown:
                    print(f"⚠️  ALERT: Unauthorized activity detected at frame {frame_count}")
                    last_alert_time = current_time
                
                # Add visual alert
                alert_text = "⚠️ UNAUTHORIZED ACTIVITY DETECTED!"
                text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
                text_x = (width - text_size[0]) // 2
                
                # Background rectangle
                cv2.rectangle(
                    annotated_frame,
                    (text_x - 10, height - 60),
                    (text_x + text_size[0] + 10, height - 20),
                    (0, 0, 255),
                    -1
                )
                
                # Alert text
                cv2.putText(
                    annotated_frame,
                    alert_text,
                    (text_x, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    3
                )
                
                # Save frame if enabled
                if save_detections:
                    from pathlib import Path
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = Path(output_dir) / f"unauthorized_{timestamp}_frame{frame_count}.jpg"
                    cv2.imwrite(str(save_path), annotated_frame)
            
            # Display frame
            cv2.imshow('Park Surveillance - Real-time Detection', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                from pathlib import Path
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = Path(output_dir) / f"manual_save_{timestamp}.jpg"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path), annotated_frame)
                print(f"Frame saved to: {save_path}")
            elif key == ord('r'):
                stats = {
                    'total_frames': 0,
                    'frames_with_detections': 0,
                    'unauthorized_count': 0,
                    'authorized_count': 0
                }
                print("Statistics reset")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        print(f"Total frames processed: {stats['total_frames']}")
        print(f"Frames with detections: {stats['frames_with_detections']}")
        print(f"Unauthorized detections: {stats['unauthorized_count']}")
        print(f"Authorized detections: {stats['authorized_count']}")
        
        if stats['total_frames'] > 0:
            detection_rate = (stats['frames_with_detections'] / stats['total_frames']) * 100
            print(f"Detection rate: {detection_rate:.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run real-time YOLO inference on webcam')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID (default: 0)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (0-1)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IOU threshold for NMS (0-1)')
    parser.add_argument('--no-labels', action='store_true',
                        help='Do not show class labels')
    parser.add_argument('--no-conf', action='store_true',
                        help='Do not show confidence scores')
    parser.add_argument('--no-alert', action='store_true',
                        help='Disable alerts for unauthorized activity')
    parser.add_argument('--save-detections', action='store_true',
                        help='Save frames with unauthorized detections')
    parser.add_argument('--output', type=str, default='runs/inference/webcam',
                        help='Output directory for saved frames')
    
    args = parser.parse_args()
    
    # Run webcam inference
    run_webcam_inference(
        model_path=args.model,
        camera_id=args.camera,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        show_labels=not args.no_labels,
        show_conf=not args.no_conf,
        alert_on_unauthorized=not args.no_alert,
        save_detections=args.save_detections,
        output_dir=args.output
    )
