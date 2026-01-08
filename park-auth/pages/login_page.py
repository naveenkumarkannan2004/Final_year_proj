"""
Login page with authentication
"""
import streamlit as st
from services.auth_service import AuthService
import config

def render_login_page():
    """Render the login page"""
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"# {config.APP_ICON} {config.APP_TITLE}")
        st.markdown("### 🔐 Login")
        st.markdown("---")
        
        # Login form
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            submit_button = st.form_submit_button("Login", use_container_width=True)
            
            if submit_button:
                if not username or not password:
                    st.error("❌ Please enter both username and password")
                else:
                    # Attempt login
                    if AuthService.login(username, password):
                        st.success("✅ Login successful! Redirecting...")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
        
        # Demo credentials info
        st.markdown("---")
        with st.expander("ℹ️ Demo Credentials"):
            st.markdown("""
            **For demonstration purposes:**
            
            **Admin Account:**
            - Username: `admin`
            - Password: `admin123`
            
            **User Account:**
            - Username: `user`
            - Password: `user123`
            """)
        
        # Additional info
        st.markdown("---")
        st.info("""
        🏞️ **Park Activity Monitoring System**
        
        This system uses AI to detect and classify activities in park areas as authorized or unauthorized.
        
        Features:
        - 🔍 Image & Video Analysis
        - 🤖 YOLO-based Detection
        - 🚨 Unauthorized Activity Alerts
        - 💬 AI Assistant for Guidance
        """)
