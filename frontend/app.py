"""
Main Streamlit Application Entry Point.

This is the main entry point for the modular ADK & A2A Learning Dashboard.
It uses Streamlit's multipage app architecture for better organization.
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from frontend.core.config import configure_app
from frontend.core.navigation import setup_navigation

def main():
    """Main application entry point."""
    # Configure the Streamlit app
    configure_app()
    
    # Setup navigation
    setup_navigation()

if __name__ == "__main__":
    main()
