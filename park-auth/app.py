"""
Park Activity Monitoring System - Main Application
"""
import streamlit as st
from services.auth_service import AuthService
from pages.login_page import render_login_page
from pages.main_page import render_main_page
import config

# Page configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        border: 2px dashed #2c5aa0;
        border-radius: 10px;
        padding: 2rem;
        background-color: rgba(44, 90, 160, 0.05);
    }
    
    /* Chat message styling */
    [data-testid="stChatMessage"] {
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Image styling */
    img {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Alert boxes */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(30, 33, 39, 0.95);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #2c5aa0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'login_time' not in st.session_state:
        st.session_state.login_time = None

def main():
    """Main application entry point"""
    # Initialize session state
    initialize_session_state()
    
    # Route to appropriate page based on authentication status
    if AuthService.is_authenticated():
        render_main_page()
    else:
        render_login_page()

if __name__ == "__main__":
    main()
