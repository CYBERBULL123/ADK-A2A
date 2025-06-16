"""
Core configuration for the Streamlit application.
"""

import streamlit as st

def configure_app():
    """Configure the main Streamlit application settings."""
    st.set_page_config(
        page_title="ADK & A2A Learning Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo/help',
            'Report a Bug': 'https://github.com/your-repo/issues',
            'About': '''
            # ADK & A2A Learning Dashboard
            
            This is a comprehensive learning platform for Google's Agent Development Kit (ADK) 
            and Agent-to-Agent (A2A) protocols.
            
            Built with ❤️ using Streamlit and Python.
            '''
        }
    )
