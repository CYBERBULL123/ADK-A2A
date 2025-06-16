"""
Core navigation system for the modular frontend.
"""

import streamlit as st
from typing import Dict, Any, List
from pathlib import Path

def setup_navigation():
    """Setup the main navigation system."""
    # Initialize session state
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("# 🤖 ADK Dashboard")
        st.markdown("---")
        
        # Main navigation sections
        navigation_sections = {
            "🏠 Overview": [
                ("Home", "pages/home.py"),
                ("Getting Started", "pages/getting_started.py"),
                ("Documentation", "pages/documentation.py")
            ],
            "🤖 Agents": [
                ("Basic Agents", "pages/agents/basic_agents.py"),
                ("Multi-Agent Systems", "pages/agents/multi_agent.py"),
                ("A2A Protocol", "pages/agents/a2a_agents.py")
            ],
            "🛠️ Tools": [
                ("Tool Library", "pages/tools/tool_library.py"),
                ("Custom Tools", "pages/tools/custom_tools.py"),
                ("Tool Testing", "pages/tools/tool_testing.py")
            ],
            "📊 Analytics": [
                ("Performance Monitor", "pages/analytics/performance.py"),
                ("Agent Metrics", "pages/analytics/metrics.py"),
                ("System Status", "pages/analytics/system_status.py")
            ],
            "⚙️ Settings": [
                ("Configuration", "pages/settings/configuration.py"),
                ("Environment", "pages/settings/environment.py")
            ]
        }
        
        for section, pages in navigation_sections.items():
            st.markdown(f"### {section}")
            for page_name, page_file in pages:
                if st.button(page_name, key=f"nav_{page_name}", use_container_width=True):
                    st.session_state.current_page = page_name
                    st.rerun()
            st.markdown("---")
    
    # Load the appropriate page based on current selection
    load_page(st.session_state.current_page)

def load_page(page_name: str):
    """Load the appropriate page based on the current selection."""
    # Map page names to their corresponding modules
    page_mapping = {
        "Home": "frontend.pages.home",
        "Getting Started": "frontend.pages.getting_started", 
        "Documentation": "frontend.pages.documentation",
        "Basic Agents": "frontend.pages.agents.basic_agents",
        "Multi-Agent Systems": "frontend.pages.agents.multi_agent",
        "A2A Protocol": "frontend.pages.agents.a2a_agents",
        "Tool Library": "frontend.pages.tools.tool_library",
        "Custom Tools": "frontend.pages.tools.custom_tools",
        "Tool Testing": "frontend.pages.tools.tool_testing",
        "Performance Monitor": "frontend.pages.analytics.performance",
        "Agent Metrics": "frontend.pages.analytics.metrics",
        "System Status": "frontend.pages.analytics.system_status",
        "Configuration": "frontend.pages.settings.configuration",
        "Environment": "frontend.pages.settings.environment"
    }
    
    try:
        if page_name in page_mapping:
            module_path = page_mapping[page_name]
            module = __import__(module_path, fromlist=[''])
            if hasattr(module, 'render'):
                module.render()
            else:
                st.error(f"Page {page_name} does not have a render function")
        else:
            st.error(f"Page {page_name} not found")
    except ImportError as e:
        st.error(f"Error loading page {page_name}: {e}")
        st.info("This page is still under development.")
