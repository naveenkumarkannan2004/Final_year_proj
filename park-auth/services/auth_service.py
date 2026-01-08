"""
Authentication service using bcrypt for password hashing
"""
import bcrypt
import streamlit as st
from datetime import datetime, timedelta
import config

class AuthService:
    """Simple authentication service for study purposes"""
    
    # Hardcoded user database (for study purposes only)
    USERS = {
        config.ADMIN_USERNAME: config.ADMIN_PASSWORD_HASH,
        config.USER_USERNAME: config.USER_PASSWORD_HASH
    }
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        try:
            return bcrypt.checkpw(
                password.encode('utf-8'),
                hashed_password.encode('utf-8')
            )
        except Exception as e:
            print(f"Password verification error: {e}")
            return False
    
    @staticmethod
    def authenticate(username: str, password: str) -> bool:
        """Authenticate a user with username and password"""
        if username not in AuthService.USERS:
            return False
        
        stored_hash = AuthService.USERS[username]
        return AuthService.verify_password(password, stored_hash)
    
    @staticmethod
    def login(username: str, password: str) -> bool:
        """Login a user and set session state"""
        if AuthService.authenticate(username, password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.login_time = datetime.now()
            return True
        return False
    
    @staticmethod
    def logout():
        """Logout the current user"""
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.login_time = None
    
    @staticmethod
    def is_authenticated() -> bool:
        """Check if user is authenticated"""
        if not st.session_state.get('authenticated', False):
            return False
        
        # Check session timeout
        login_time = st.session_state.get('login_time')
        if login_time:
            elapsed = datetime.now() - login_time
            if elapsed > timedelta(minutes=config.SESSION_TIMEOUT_MINUTES):
                AuthService.logout()
                return False
        
        return True
    
    @staticmethod
    def get_current_user() -> str:
        """Get the current logged-in username"""
        return st.session_state.get('username', 'Guest')


# Helper function to generate password hashes (for setup)
def generate_hash(password: str):
    """Utility function to generate password hash"""
    print(f"Hash for '{password}': {AuthService.hash_password(password)}")


if __name__ == "__main__":
    # Generate hashes for setup
    generate_hash("admin123")
    generate_hash("user123")
