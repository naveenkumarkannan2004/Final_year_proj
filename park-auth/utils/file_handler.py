"""
File handling utilities
"""
import os
import tempfile
import shutil
from pathlib import Path
import config

class FileHandler:
    """Utility class for file operations"""
    
    @staticmethod
    def save_uploaded_file(uploaded_file, directory: str = config.TEMP_DIR) -> str:
        """
        Save an uploaded Streamlit file to disk
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            directory: Directory to save the file
            
        Returns:
            Path to the saved file
        """
        os.makedirs(directory, exist_ok=True)
        
        file_path = os.path.join(directory, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    
    @staticmethod
    def validate_file_type(filename: str, allowed_extensions: list) -> bool:
        """
        Validate file extension
        
        Args:
            filename: Name of the file
            allowed_extensions: List of allowed extensions (without dot)
            
        Returns:
            True if valid, False otherwise
        """
        extension = Path(filename).suffix.lower().lstrip('.')
        return extension in allowed_extensions
    
    @staticmethod
    def is_image(filename: str) -> bool:
        """Check if file is an image"""
        return FileHandler.validate_file_type(filename, config.SUPPORTED_IMAGE_FORMATS)
    
    @staticmethod
    def is_video(filename: str) -> bool:
        """Check if file is a video"""
        return FileHandler.validate_file_type(filename, config.SUPPORTED_VIDEO_FORMATS)
    
    @staticmethod
    def cleanup_temp_files():
        """Remove all temporary files"""
        if os.path.exists(config.TEMP_DIR):
            for file in os.listdir(config.TEMP_DIR):
                file_path = os.path.join(config.TEMP_DIR, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
    
    @staticmethod
    def get_file_size_mb(file_path: str) -> float:
        """Get file size in MB"""
        return os.path.getsize(file_path) / (1024 * 1024)
    
    @staticmethod
    def create_temp_file(suffix: str = "") -> str:
        """Create a temporary file and return its path"""
        fd, path = tempfile.mkstemp(suffix=suffix, dir=config.TEMP_DIR)
        os.close(fd)
        return path
