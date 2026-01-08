"""
YOLO Detection Service for Park Activity Monitoring
"""
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import config
from typing import List, Tuple, Dict
import os

class YOLODetector:
    """YOLO-based detector for authorized/unauthorized activities"""
    
    def __init__(self, model_path: str = config.MODEL_PATH):
        """Initialize YOLO detector with the trained model"""
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def detect_image(self, image_path: str, conf_threshold: float = config.CONFIDENCE_THRESHOLD):
        """
        Detect activities in an image
        
        Args:
            image_path: Path to the image file
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Tuple of (annotated_image, detections_dict)
        """
        try:
            # Run inference
            results = self.model(image_path, conf=conf_threshold)
            
            # Get annotated image
            annotated_img = results[0].plot()
            
            # Extract detection information
            detections = self._parse_detections(results[0])
            
            return annotated_img, detections
        
        except Exception as e:
            print(f"Error during image detection: {e}")
            return None, {}
    
    def detect_video(self, video_path: str, conf_threshold: float = config.CONFIDENCE_THRESHOLD, skip_frames: int = 1):
        """
        Detect activities in a video frame by frame
        
        Args:
            video_path: Path to the video file
            conf_threshold: Confidence threshold for detections
            skip_frames: Process every Nth frame (1 = all frames, 5 = every 5th frame for 5x speed)
            
        Yields:
            Tuple of (frame_number, annotated_frame, detections_dict)
        """
        cap = cv2.VideoCapture(video_path)
        frame_number = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames for faster processing
                if frame_number % skip_frames != 0:
                    frame_number += 1
                    continue
                
                # Run inference on frame
                results = self.model(frame, conf=conf_threshold, verbose=False)
                
                # Get annotated frame
                annotated_frame = results[0].plot()
                
                # Extract detection information
                detections = self._parse_detections(results[0])
                
                yield frame_number, annotated_frame, detections
                frame_number += 1
        
        finally:
            cap.release()
    
    def extract_unauthorized_frames(self, video_path: str, conf_threshold: float = config.CONFIDENCE_THRESHOLD, skip_frames: int = 1):
        """
        Extract frames containing unauthorized activities from video
        
        Args:
            video_path: Path to the video file
            conf_threshold: Confidence threshold for detections
            skip_frames: Process every Nth frame for faster processing
            
        Returns:
            List of tuples (frame_number, frame_image, detections)
        """
        unauthorized_frames = []
        
        for frame_num, annotated_frame, detections in self.detect_video(video_path, conf_threshold, skip_frames):
            # Check if frame contains unauthorized activity
            if self._has_unauthorized_activity(detections):
                unauthorized_frames.append((frame_num, annotated_frame, detections))
        
        return unauthorized_frames
    
    def _parse_detections(self, result) -> Dict:
        """
        Parse YOLO detection results into a structured dictionary
        
        Args:
            result: YOLO result object
            
        Returns:
            Dictionary with detection information
        """
        detections = {
            'total_count': 0,
            'authorized_count': 0,
            'unauthorized_count': 0,
            'boxes': [],
            'classes': [],
            'confidences': []
        }
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            detections['total_count'] = len(boxes)
            detections['boxes'] = boxes.tolist()
            detections['classes'] = classes.tolist()
            detections['confidences'] = confidences.tolist()
            
            # Count authorized vs unauthorized
            # Check actual class names from the model to handle any class ID order
            for cls in classes:
                cls_id = int(cls)
                cls_name = self.model.names[cls_id].lower()
                
                # Check if the class name contains 'unauthorized' or 'unauthorised'
                if 'unauthor' in cls_name:  # Handles both spellings
                    detections['unauthorized_count'] += 1
                else:
                    detections['authorized_count'] += 1
        
        return detections
    
    
    def _has_unauthorized_activity(self, detections: Dict) -> bool:
        """Check if detections contain unauthorized activity"""
        return detections.get('unauthorized_count', 0) > 0
    
    def is_class_unauthorized(self, class_id: int) -> bool:
        """
        Check if a class ID represents unauthorized activity
        
        Args:
            class_id: The class ID to check
            
        Returns:
            True if the class is unauthorized, False otherwise
        """
        cls_name = self.model.names[int(class_id)].lower()
        return 'unauthor' in cls_name
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name from class ID"""
        return self.model.names[int(class_id)]
    
    def save_annotated_image(self, annotated_img: np.ndarray, output_path: str):
        """Save annotated image to file"""
        cv2.imwrite(output_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
    
    def save_annotated_video(self, video_path: str, output_path: str, conf_threshold: float = config.CONFIDENCE_THRESHOLD, skip_frames: int = 1):
        """
        Process video and save with annotations
        
        Args:
            video_path: Input video path
            output_path: Output video path
            conf_threshold: Confidence threshold
            skip_frames: Process every Nth frame for faster processing
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Adjust FPS if skipping frames
        output_fps = fps // skip_frames if skip_frames > 1 else fps
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        
        try:
            for _, annotated_frame, _ in self.detect_video(video_path, conf_threshold, skip_frames):
                # Convert RGB to BGR for video writer
                frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
        
        finally:
            cap.release()
            out.release()
