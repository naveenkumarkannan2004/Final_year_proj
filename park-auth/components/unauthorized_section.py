"""
Unauthorized activity section component
"""
import streamlit as st
import numpy as np
from typing import List, Tuple
import cv2
import os
import config

def display_unauthorized_frames(unauthorized_frames: List[Tuple[int, np.ndarray, dict]]):
    """
    Display extracted unauthorized activity frames in a grid
    
    Args:
        unauthorized_frames: List of tuples (frame_number, frame_image, detections)
    """
    st.markdown("---")
    st.markdown("## 🚨 Unauthorized Activity Frames")
    
    if not unauthorized_frames:
        st.info("ℹ️ No unauthorized activities detected in the video.")
        return
    
    st.warning(f"⚠️ **{len(unauthorized_frames)} frame(s) with unauthorized activities detected!**")
    
    # Display frames in a grid (3 columns)
    cols_per_row = 3
    
    for i in range(0, len(unauthorized_frames), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j in range(cols_per_row):
            idx = i + j
            if idx < len(unauthorized_frames):
                frame_num, frame_img, detections = unauthorized_frames[idx]
                
                with cols[j]:
                    # Display frame
                    st.image(frame_img, channels="RGB", use_container_width=True)
                    
                    # Frame information
                    st.caption(f"**Frame:** {frame_num}")
                    st.caption(f"**Unauthorized Count:** {detections.get('unauthorized_count', 0)}")
                    
                    # Show confidence scores for unauthorized detections
                    if detections.get('classes') and detections.get('confidences'):
                        from services.yolo_detector import YOLODetector
                        detector = YOLODetector()
                        
                        for cls, conf in zip(detections['classes'], detections['confidences']):
                            if detector.is_class_unauthorized(int(cls)):
                                st.caption(f"⚠️ Confidence: {conf*100:.1f}%")
                    
                    # Download button for individual frame
                    frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
                    success, buffer = cv2.imencode('.jpg', frame_bgr)
                    if success:
                        st.download_button(
                            label="💾 Download",
                            data=buffer.tobytes(),
                            file_name=f"unauthorized_frame_{frame_num}.jpg",
                            mime="image/jpeg",
                            key=f"download_frame_{idx}",
                            use_container_width=True
                        )
    
    # Option to download all frames as a zip
    st.markdown("---")
    if st.button("📦 Download All Unauthorized Frames", use_container_width=True):
        st.info("💡 Tip: Use the individual download buttons above to save specific frames.")

def display_unauthorized_section_for_image(detections: dict, annotated_image: np.ndarray):
    """
    Display unauthorized activity section for a single image
    
    Args:
        detections: Detection dictionary
        annotated_image: Annotated image
    """
    st.markdown("---")
    st.markdown("## 🚨 Unauthorized Activity Detection")
    
    unauthorized_count = detections.get('unauthorized_count', 0)
    
    if unauthorized_count == 0:
        st.success("✅ No unauthorized activities detected in this image!")
        return
    
    st.error(f"⚠️ **{unauthorized_count} unauthorized activity(ies) detected!**")
    
    # Display the annotated image again with focus on unauthorized
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(annotated_image, channels="RGB", use_container_width=True)
    
    with col2:
        st.markdown("### Detection Details")
        
        if detections.get('classes') and detections.get('confidences'):
            from services.yolo_detector import YOLODetector
            detector = YOLODetector()
            
            unauthorized_detections = []
            for idx, (cls, conf) in enumerate(zip(detections['classes'], detections['confidences'])):
                if detector.is_class_unauthorized(int(cls)):
                    unauthorized_detections.append((idx + 1, conf))
            
            for det_num, conf in unauthorized_detections:
                st.markdown(f"**Detection {det_num}**")
                st.markdown(f"- Class: Unauthorized")
                st.markdown(f"- Confidence: {conf*100:.1f}%")
                st.markdown("---")
        
        # Download button
        frame_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode('.jpg', frame_bgr)
        if success:
            st.download_button(
                label="💾 Download Annotated Image",
                data=buffer.tobytes(),
                file_name="unauthorized_detection.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
