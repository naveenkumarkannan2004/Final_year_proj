"""
Main application page with detection functionality
"""
import streamlit as st
from services.auth_service import AuthService
from services.yolo_detector import YOLODetector
from utils.file_handler import FileHandler
from components.chat_widget import render_chat_widget
from components.detection_display import display_detection_results, display_video_summary
from components.unauthorized_section import display_unauthorized_frames, display_unauthorized_section_for_image
import config
import os

def render_main_page():
    """Render the main application page"""
    
    # Check authentication
    if not AuthService.is_authenticated():
        st.error("❌ Please login to access this page")
        st.stop()
    
    # Page header
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown(f"# {config.APP_ICON} {config.APP_TITLE}")
        st.markdown(f"**Welcome, {AuthService.get_current_user()}!**")
    with col2:
        if st.button("🚪 Logout", use_container_width=True):
            AuthService.logout()
            st.rerun()
    
    st.markdown("---")
    
    # Render chat widget in sidebar
    render_chat_widget()
    
    # Main content area
    st.markdown("## 📤 Upload Media for Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image or video file",
        type=config.SUPPORTED_IMAGE_FORMATS + config.SUPPORTED_VIDEO_FORMATS,
        help=f"Supported formats: Images ({', '.join(config.SUPPORTED_IMAGE_FORMATS)}), Videos ({', '.join(config.SUPPORTED_VIDEO_FORMATS)})"
    )
    
    if uploaded_file is not None:
        # Display file info
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.info(f"📁 **File:** {uploaded_file.name} ({file_size_mb:.2f} MB)")
        
        # Check file size
        if file_size_mb > config.MAX_FILE_SIZE_MB:
            st.error(f"❌ File size exceeds maximum limit of {config.MAX_FILE_SIZE_MB} MB")
            return
        
        # Save uploaded file
        file_path = FileHandler.save_uploaded_file(uploaded_file)
        
        # Determine file type
        is_image = FileHandler.is_image(uploaded_file.name)
        is_video = FileHandler.is_video(uploaded_file.name)
        
        if is_image:
            process_image(file_path)
        elif is_video:
            process_video(file_path)
        else:
            st.error("❌ Unsupported file format")
    
    else:
        # Show instructions when no file is uploaded
        st.markdown("""
        ### 📋 Instructions
        
        1. **Upload** an image or video file using the uploader above
        2. **Wait** for the AI to analyze the content
        3. **Review** the detection results and any alerts
        4. **Check** the unauthorized activity section for flagged content
        5. **Ask** the AI assistant (in sidebar) if you have questions
        
        ### 🎯 What We Detect
        
        - ✅ **Authorized Activities:** Normal, permitted activities in the park
        - ⚠️ **Unauthorized Activities:** Prohibited or suspicious activities
        
        ### 💡 Tips
        
        - Use clear, well-lit images/videos for best results
        - Higher confidence scores indicate more reliable detections
        - Check the chat assistant for help understanding results
        """)

def process_image(image_path: str):
    """Process and display results for an image"""
    st.markdown("---")
    st.markdown("## 🖼️ Image Analysis")
    
    # Initialize session state for detection results
    if 'image_detection_done' not in st.session_state:
        st.session_state.image_detection_done = False
    if 'image_detections' not in st.session_state:
        st.session_state.image_detections = None
    if 'image_annotated' not in st.session_state:
        st.session_state.image_annotated = None
    if 'current_image_path' not in st.session_state:
        st.session_state.current_image_path = None
    
    # Reset detection if new image is uploaded
    if st.session_state.current_image_path != image_path:
        st.session_state.image_detection_done = False
        st.session_state.image_detections = None
        st.session_state.image_annotated = None
        st.session_state.current_image_path = image_path
    
    # Display image preview
    st.image(image_path, caption="Uploaded Image", use_container_width=True)
    
    # Detect button
    if st.button("🔍 Detect Activities", use_container_width=True, type="primary"):
        # Initialize detector
        with st.spinner("Loading YOLO model..."):
            detector = YOLODetector()
        
        # Run detection
        with st.spinner("Analyzing image..."):
            annotated_img, detections = detector.detect_image(image_path)
        
        if annotated_img is not None:
            st.session_state.image_annotated = annotated_img
            st.session_state.image_detections = detections
            st.session_state.image_detection_done = True
            st.rerun()
        else:
            st.error("❌ Error processing image")
    
    # Display results if detection has been done
    if st.session_state.image_detection_done and st.session_state.image_annotated is not None:
        # Display results
        display_detection_results(
            st.session_state.image_annotated, 
            st.session_state.image_detections, 
            "image"
        )
        
        # Display unauthorized activity section
        display_unauthorized_section_for_image(
            st.session_state.image_detections, 
            st.session_state.image_annotated
        )
        
        # Generate Report button
        st.markdown("---")
        if st.button("📄 Generate Report", use_container_width=True, type="secondary"):
            from services.report_generator import ReportGenerator
            
            with st.spinner("Generating PDF report..."):
                report_gen = ReportGenerator()
                report_path = report_gen.generate_image_report(
                    filename=os.path.basename(image_path),
                    detections=st.session_state.image_detections,
                    annotated_image=st.session_state.image_annotated
                )
            
            st.success("✅ Report generated successfully!")
            
            # Provide download link
            with open(report_path, "rb") as f:
                st.download_button(
                    label="💾 Download PDF Report",
                    data=f,
                    file_name=os.path.basename(report_path),
                    mime="application/pdf",
                    use_container_width=True
                )

def process_video(video_path: str):
    """Process and display results for a video"""
    st.markdown("---")
    st.markdown("## 🎬 Video Analysis")
    
    # Initialize session state for video detection results
    if 'video_detection_done' not in st.session_state:
        st.session_state.video_detection_done = False
    if 'video_unauthorized_frames' not in st.session_state:
        st.session_state.video_unauthorized_frames = []
    if 'video_total_frames' not in st.session_state:
        st.session_state.video_total_frames = 0
    if 'current_video_path' not in st.session_state:
        st.session_state.current_video_path = None
    
    # Reset detection if new video is uploaded
    if st.session_state.current_video_path != video_path:
        st.session_state.video_detection_done = False
        st.session_state.video_unauthorized_frames = []
        st.session_state.video_total_frames = 0
        st.session_state.current_video_path = video_path
    
    # Display video info
    import cv2
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    
    st.info(f"📹 **Video Info:** {total_frames} frames, {fps} FPS, Duration: {duration:.1f}s")
    
    # Processing speed selector
    st.markdown("### ⚡ Processing Speed")
    speed_option = st.select_slider(
        "Choose processing speed",
        options=["🐢 Accurate (All Frames)", "⚖️ Balanced (5 FPS)", "🚀 Fast (2 FPS)"],
        value="⚖️ Balanced (5 FPS)",
        help="Accurate: Process all frames | Balanced: Process at 5 FPS | Fast: Process at 2 FPS"
    )
    
    # Calculate skip_frames based on target FPS
    if speed_option == "🐢 Accurate (All Frames)":
        skip_frames = 1
        target_fps = fps
    elif speed_option == "⚖️ Balanced (5 FPS)":
        # Process at 5 FPS
        skip_frames = max(1, fps // 5)
        target_fps = 5
    else:  # Fast (2 FPS)
        # Process at 2 FPS
        skip_frames = max(1, fps // 2)
        target_fps = 2
    
    # Show estimated processing time
    if skip_frames == 1:
        st.caption(f"⏱️ Processing all {fps} frames per second - most accurate but slowest")
    else:
        speedup = fps / target_fps
        st.caption(f"⏱️ Processing at ~{target_fps} FPS (every {skip_frames} frame(s)) - {speedup:.0f}x faster")
    
    # Detect button
    if st.button("🔍 Detect Activities", use_container_width=True, type="primary"):
        # Initialize detector
        with st.spinner("Loading YOLO model..."):
            detector = YOLODetector()
        
        # Calculate frames to process
        frames_to_process = total_frames // skip_frames
        
        # Extract unauthorized frames
        with st.spinner(f"Analyzing video frames... Processing ~{frames_to_process} of {total_frames} frames"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process video and extract unauthorized frames
            unauthorized_frames = []
            frame_count = 0
            
            for frame_num, annotated_frame, detections in detector.detect_video(video_path, skip_frames=skip_frames):
                frame_count += 1
                
                # Update progress (based on actual frames processed)
                progress = min((frame_num + 1) / total_frames, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_num + 1}/{total_frames} (analyzing every {skip_frames} frame(s))")
                
                # Check for unauthorized activity
                if detections.get('unauthorized_count', 0) > 0:
                    unauthorized_frames.append((frame_num, annotated_frame, detections))
            
            progress_bar.empty()
            status_text.empty()
        
        st.session_state.video_unauthorized_frames = unauthorized_frames
        st.session_state.video_total_frames = total_frames
        st.session_state.video_detection_done = True
        st.success(f"✅ Video processing complete! Analyzed {frame_count} frames (every {skip_frames} frame(s)).")
        st.rerun()
    
    # Display results if detection has been done
    if st.session_state.video_detection_done:
        st.success(f"✅ Video processing complete! Analyzed {st.session_state.video_total_frames} frames.")
        
        # Display summary
        display_video_summary(
            st.session_state.video_total_frames, 
            len(st.session_state.video_unauthorized_frames)
        )
        
        # Display unauthorized frames
        display_unauthorized_frames(st.session_state.video_unauthorized_frames)
        
        # Generate Report button
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Generate Report", use_container_width=True, type="secondary"):
                from services.report_generator import ReportGenerator
                
                with st.spinner("Generating PDF report..."):
                    report_gen = ReportGenerator()
                    report_path = report_gen.generate_video_report(
                        filename=os.path.basename(video_path),
                        total_frames=st.session_state.video_total_frames,
                        unauthorized_frames=st.session_state.video_unauthorized_frames
                    )
                
                st.success("✅ Report generated successfully!")
                
                # Provide download link
                with open(report_path, "rb") as f:
                    st.download_button(
                        label="💾 Download PDF Report",
                        data=f,
                        file_name=os.path.basename(report_path),
                        mime="application/pdf",
                        use_container_width=True,
                        key="download_video_report"
                    )
        
        with col2:
            # Option to process and download annotated video
            if st.button("🎥 Generate Annotated Video", use_container_width=True):
                output_path = os.path.join(config.OUTPUT_DIR, "annotated_video.mp4")
                
                with st.spinner("Generating annotated video... This may take a while."):
                    detector = YOLODetector()
                    detector.save_annotated_video(video_path, output_path)
                
                st.success("✅ Annotated video generated!")
                
                # Provide download link
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="💾 Download Annotated Video",
                        data=f,
                        file_name="annotated_video.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                        key="download_annotated_video"
                    )
