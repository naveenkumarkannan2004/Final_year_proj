# 🏞️ Park Activity Monitoring System

A comprehensive Streamlit-based web application for monitoring park activities using AI-powered YOLO object detection. The system identifies and classifies activities as authorized or unauthorized, with integrated chatbot assistance.

## ✨ Features

- 🔐 **Secure Authentication** - Login system with bcrypt password hashing
- 🤖 **YOLO Detection** - AI-powered activity detection using custom-trained model
- 📸 **Image Analysis** - Upload and analyze images for activity detection
- 🎬 **Video Processing** - Frame-by-frame video analysis with unauthorized activity extraction
- 🚨 **Alert System** - Automatic flagging of unauthorized activities
- 💬 **AI Chatbot** - Interactive assistant for park guidance and system help
- 📊 **Detailed Reports** - Comprehensive detection statistics and visualizations
- 💾 **Export Functionality** - Download annotated images/videos and extracted frames

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd d:\Antigravity\park_auth
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Access the application:**
   - Open your browser and navigate to `http://localhost:8501`

## 🔑 Demo Credentials

For demonstration purposes, use these credentials:

**Admin Account:**
- Username: `admin`
- Password: `admin123`

**User Account:**
- Username: `user`
- Password: `user123`

## 📁 Project Structure

```
park_auth/
├── app.py                          # Main application entry point
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── best.pt                         # YOLO model weights
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── pages/
│   ├── __init__.py
│   ├── login_page.py              # Authentication page
│   └── main_page.py               # Main application page
├── services/
│   ├── __init__.py
│   ├── auth_service.py            # Authentication service
│   ├── yolo_detector.py           # YOLO detection service
│   └── chatbot.py                 # Chatbot service
├── components/
│   ├── __init__.py
│   ├── chat_widget.py             # Chat interface component
│   ├── detection_display.py      # Detection results display
│   └── unauthorized_section.py   # Unauthorized activity section
├── utils/
│   ├── __init__.py
│   └── file_handler.py            # File handling utilities
├── temp/                          # Temporary files (auto-created)
└── outputs/                       # Output files (auto-created)
```

## 🎯 How to Use

### 1. Login
- Navigate to the application URL
- Enter your credentials (see Demo Credentials above)
- Click "Login"

### 2. Upload Media
- Click the file uploader on the main page
- Select an image or video file
- Supported formats:
  - Images: JPG, JPEG, PNG, BMP
  - Videos: MP4, AVI, MOV, MKV

### 3. View Results
- **Image Analysis:** Instant detection results with annotated image
- **Video Analysis:** Frame-by-frame processing with progress indicator
- **Unauthorized Activity Section:** Dedicated area showing flagged frames

### 4. Use the Chatbot
- Access the chatbot in the sidebar
- Ask questions about:
  - Authorized/unauthorized activities
  - Park rules and regulations
  - How to use the system
  - Detection accuracy and confidence scores

### 5. Download Results
- Download annotated images/videos
- Export individual unauthorized activity frames
- Save detection evidence for reporting

## 🔧 Configuration

Edit `config.py` to customize:

- **Model Settings:**
  - `MODEL_PATH`: Path to YOLO model
  - `CONFIDENCE_THRESHOLD`: Detection confidence threshold (default: 0.5)
  
- **File Upload:**
  - `MAX_FILE_SIZE_MB`: Maximum upload size (default: 200 MB)
  - `SUPPORTED_IMAGE_FORMATS`: Allowed image formats
  - `SUPPORTED_VIDEO_FORMATS`: Allowed video formats

- **Authentication:**
  - `SESSION_TIMEOUT_MINUTES`: Session timeout duration (default: 30 minutes)

## 🤖 YOLO Model

The system uses a custom-trained YOLO model (`best.pt`) with two classes:
- **Class 0:** Authorized activities
- **Class 1:** Unauthorized activities

The model analyzes uploaded media and classifies detected activities accordingly.

## 💬 Chatbot Capabilities

The integrated chatbot can help with:
- Understanding authorized vs unauthorized activities
- Park rules and regulations
- System usage instructions
- Detection result interpretation
- Confidence score explanations
- Video processing information

## 🛡️ Security Notes

**For Study/Demo Purposes Only:**
- Uses hardcoded credentials with bcrypt hashing
- Session-based authentication with timeout
- Not suitable for production use without proper database integration

## 📊 Detection Results

The system provides:
- **Total Detections:** Count of all detected activities
- **Authorized Count:** Number of authorized activities
- **Unauthorized Count:** Number of unauthorized activities
- **Confidence Scores:** Reliability metric for each detection
- **Bounding Boxes:** Visual indicators on annotated media
- **Frame Extraction:** Automatic extraction of problematic frames from videos

## 🎨 UI Features

- **Dark Theme:** Modern, eye-friendly interface
- **Responsive Design:** Works on various screen sizes
- **Real-time Progress:** Live updates during video processing
- **Interactive Chat:** Conversational AI assistant
- **Visual Alerts:** Color-coded warnings and notifications
- **Grid Display:** Organized view of multiple detections

## 📝 Tips for Best Results

1. **Image Quality:** Use clear, well-lit images/videos
2. **Camera Angle:** Front-facing views work best
3. **Resolution:** Higher resolution provides better detection
4. **File Size:** Compress large videos for faster processing
5. **Lighting:** Avoid extreme shadows or overexposure

## 🐛 Troubleshooting

**Model Loading Error:**
- Ensure `best.pt` is in the project root directory
- Check that ultralytics is properly installed

**Upload Failed:**
- Verify file format is supported
- Check file size is under the limit
- Ensure sufficient disk space

**Slow Video Processing:**
- Normal for long videos
- Consider reducing video resolution
- Process shorter clips for faster results

## 📄 License

This project is for educational and study purposes.

## 🙏 Acknowledgments

- **Ultralytics YOLO** - Object detection framework
- **Streamlit** - Web application framework
- **OpenCV** - Computer vision library

## 📧 Support

For questions or issues:
1. Check the chatbot for common questions
2. Review this README
3. Consult the configuration settings

---

**Built with ❤️ for Park Safety and Monitoring**
