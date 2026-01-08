"""
Configuration file for Park Activity Monitoring Application
"""
import os

# Model Configuration
# Use absolute path to ensure model is found on Streamlit Cloud
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Class names from the YOLO model
CLASS_NAMES = {
    0: "authorized",
    1: "unauthorized"
}

# Unauthorized activity detection
UNAUTHORIZED_CLASS = "unauthorized"
UNAUTHORIZED_CLASS_ID = 1

# File Upload Configuration
MAX_FILE_SIZE_MB = 200
SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "bmp"]
SUPPORTED_VIDEO_FORMATS = ["mp4", "avi", "mov", "mkv"]

# Authentication Configuration
# For study purposes - simple hardcoded credentials
# Password: "admin123" (hashed with bcrypt)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD_HASH = "$2b$12$G.yyF8TUKl1bnSURqjbVbOHpJmtkNXzYixCBJNsqKJ/3JMZabrun6"  # admin123

# Sample user credentials
# Password: "user123" (hashed with bcrypt)
USER_USERNAME = "user"
USER_PASSWORD_HASH = "$2b$12$W2J.pso89jMkbhEEI49muemztKKl9yWHf.5LXxtqMcsmffhap6cnm"  # user123

# Session Configuration
SESSION_TIMEOUT_MINUTES = 30

# Application Settings
APP_TITLE = "Park Activity Monitoring System"
APP_ICON = "🏞️"

# Paths
# Use absolute paths to work correctly on Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Create necessary directories
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
