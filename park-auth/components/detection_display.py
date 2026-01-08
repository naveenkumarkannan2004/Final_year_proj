"""
Detection display component
"""
import streamlit as st
import numpy as np
from PIL import Image

def display_detection_results(annotated_image: np.ndarray, detections: dict, file_type: str = "image"):
    """
    Display detection results with statistics
    
    Args:
        annotated_image: Annotated image/frame with bounding boxes
        detections: Dictionary containing detection information
        file_type: Type of file ("image" or "video")
    """
    st.markdown("### 📊 Detection Results")
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Detections",
            value=detections.get('total_count', 0),
            delta=None
        )
    
    with col2:
        authorized_count = detections.get('authorized_count', 0)
        st.metric(
            label="✅ Authorized",
            value=authorized_count,
            delta=None,
            delta_color="normal"
        )
    
    with col3:
        unauthorized_count = detections.get('unauthorized_count', 0)
        st.metric(
            label="⚠️ Unauthorized",
            value=unauthorized_count,
            delta=None,
            delta_color="inverse"
        )
    
    # Alert for unauthorized activities
    if unauthorized_count > 0:
        st.error(f"🚨 **Alert:** {unauthorized_count} unauthorized activity(ies) detected!")
    elif detections.get('total_count', 0) > 0:
        st.success("✅ All detected activities are authorized!")
    else:
        st.info("ℹ️ No activities detected in this " + file_type)
    
    # Display annotated image
    st.markdown("### 🖼️ Annotated " + file_type.capitalize())
    st.image(annotated_image, use_container_width=True, channels="RGB")
    
    # Display detailed detection information
    if detections.get('total_count', 0) > 0:
        with st.expander("📋 Detailed Detection Information"):
            # Import detector to check class names
            from services.yolo_detector import YOLODetector
            detector = YOLODetector()
            
            for idx, (cls, conf) in enumerate(zip(
                detections.get('classes', []),
                detections.get('confidences', [])
            )):
                # Determine class name by checking actual model class names
                cls_name = detector.get_class_name(int(cls))
                is_unauthorized = detector.is_class_unauthorized(int(cls))
                confidence_pct = conf * 100
                
                # Color code based on class
                if is_unauthorized:
                    st.markdown(f"**Detection {idx + 1}:** ⚠️ {cls_name} (Confidence: {confidence_pct:.1f}%)")
                else:
                    st.markdown(f"**Detection {idx + 1}:** ✅ {cls_name} (Confidence: {confidence_pct:.1f}%)")

def display_video_summary(total_frames: int, unauthorized_frame_count: int):
    """
    Display video processing summary
    
    Args:
        total_frames: Total number of frames processed
        unauthorized_frame_count: Number of frames with unauthorized activities
    """
    st.markdown("### 📹 Video Analysis Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Frames", total_frames)
    
    with col2:
        st.metric("⚠️ Frames with Unauthorized Activity", unauthorized_frame_count)
    
    with col3:
        if total_frames > 0:
            percentage = (unauthorized_frame_count / total_frames) * 100
            st.metric("Percentage", f"{percentage:.1f}%")
    
    if unauthorized_frame_count > 0:
        st.error(f"🚨 **Alert:** Unauthorized activities detected in {unauthorized_frame_count} frame(s)!")
    else:
        st.success("✅ No unauthorized activities detected in the video!")
