"""
Chat widget component for the chatbot
"""
import streamlit as st
from services.chatbot import ParkChatbot

def initialize_chat_history():
    """Initialize chat history in session state"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.chatbot = ParkChatbot()
        # Add welcome message
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': "👋 Welcome to the Park Activity Monitoring Assistant! I can help you understand authorized activities, park rules, and how to use this system. How can I help you today?"
        })

def render_chat_widget():
    """Render the chat widget in the sidebar"""
    with st.sidebar:
        st.markdown("### 💬 Park Assistant")
        st.markdown("---")
        
        # Initialize chat history
        initialize_chat_history()
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message['role']):
                    st.write(message['content'])
        
        # Quick reply buttons
        st.markdown("**Quick Questions:**")
        col1, col2 = st.columns(2)
        
        quick_replies = st.session_state.chatbot.get_quick_replies()
        
        with col1:
            if st.button(quick_replies[0], key="quick1", use_container_width=True):
                handle_user_message(quick_replies[0])
            if st.button(quick_replies[2], key="quick3", use_container_width=True):
                handle_user_message(quick_replies[2])
        
        with col2:
            if st.button(quick_replies[1], key="quick2", use_container_width=True):
                handle_user_message(quick_replies[1])
            if st.button(quick_replies[3], key="quick4", use_container_width=True):
                handle_user_message(quick_replies[3])
        
        # Chat input
        user_input = st.chat_input("Ask me anything about park activities...")
        
        if user_input:
            handle_user_message(user_input)

def handle_user_message(user_message: str):
    """Handle user message and get bot response"""
    # Add user message to history
    st.session_state.chat_history.append({
        'role': 'user',
        'content': user_message
    })
    
    # Get bot response
    bot_response = st.session_state.chatbot.get_response(user_message)
    
    # Add bot response to history
    st.session_state.chat_history.append({
        'role': 'assistant',
        'content': bot_response
    })
    
    # Rerun to update UI
    st.rerun()
