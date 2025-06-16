"""
Home page for the ADK & A2A Learning Dashboard.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Dict, Any

from frontend.utils.ui_components import ui, charts

def render():
    """Render the home page."""
    # Page header
    ui.header(
        "ADK & A2A Learning Dashboard",
        "Comprehensive platform for Agent Development Kit and Agent-to-Agent protocols",
        "🤖"
    )
    
    # Quick stats section
    render_quick_stats()
    
    # Feature overview
    render_feature_overview()
    
    # Recent activity
    render_recent_activity()
    
    # System status
    render_system_status()

def render_quick_stats():
    """Render quick statistics."""
    st.subheader("📊 Quick Overview")
    
    # Mock data for demonstration
    metrics = [
        {
            "label": "Active Agents",
            "value": "12",
            "delta": "+3 from yesterday",
            "delta_color": "normal"
        },
        {
            "label": "Total Tools",
            "value": "45",
            "delta": "+2 new tools",
            "delta_color": "normal"
        },
        {
            "label": "A2A Connections", 
            "value": "8",
            "delta": "2 active sessions",
            "delta_color": "normal"
        },
        {
            "label": "System Uptime",
            "value": "99.9%",
            "delta": "Excellent",
            "delta_color": "normal"
        }
    ]
    
    ui.metric_grid(metrics, cols=4)

def render_feature_overview():
    """Render feature overview cards."""
    st.subheader("🚀 Platform Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if ui.feature_card(
            "Basic Agents",
            "Create and experiment with simple agents using Google ADK. Perfect for learning the fundamentals.",
            "🤖",
            "Explore Agents",
            "explore_basic"
        ):
            st.session_state.current_page = "Basic Agents"
            st.rerun()
    
    with col2:
        if ui.feature_card(
            "Multi-Agent Systems",
            "Build complex workflows with multiple coordinated agents working together.",
            "🔗",
            "Build Workflows",
            "explore_multi"
        ):
            st.session_state.current_page = "Multi-Agent Systems"
            st.rerun()
    
    with col3:
        if ui.feature_card(
            "A2A Protocol",
            "Implement Agent-to-Agent communication protocols for distributed systems.",
            "📡",
            "Learn A2A",
            "explore_a2a"
        ):
            st.session_state.current_page = "A2A Protocol"
            st.rerun()

def render_recent_activity():
    """Render recent activity section."""
    st.subheader("📈 Recent Activity")
    
    # Generate mock activity data
    dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='D')
    activity_data = pd.DataFrame({
        'Date': dates,
        'Agent Creations': [random.randint(2, 8) for _ in dates],
        'Tool Usage': [random.randint(10, 30) for _ in dates],
        'A2A Sessions': [random.randint(1, 5) for _ in dates]
    })
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📊 Activity Chart", "📋 Activity Table"])
    
    with tab1:
        charts.line_chart(
            activity_data,
            x='Date',
            y='Agent Creations',
            title="Daily Agent Creation Activity"
        )
    
    with tab2:
        ui.data_table(activity_data, "Recent Activity Log")

def render_system_status():
    """Render system status section."""
    st.subheader("⚡ System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔧 Services")
        
        services = [
            ("ADK Core", "online"),
            ("Agent Registry", "online"), 
            ("Tool Manager", "online"),
            ("A2A Gateway", "warning"),
            ("Analytics Engine", "online")
        ]
        
        for service, status in services:
            ui.status_indicator(status, f"{service} - {status.title()}")
            st.write("")
    
    with col2:
        st.markdown("### 📊 Performance")
        
        # Mock performance data
        perf_data = pd.DataFrame({
            'Metric': ['CPU Usage', 'Memory Usage', 'Network I/O', 'Disk Usage'],
            'Value': [45, 62, 23, 78],
            'Threshold': [80, 85, 90, 85]
        })
        
        for _, row in perf_data.iterrows():
            progress_color = "normal" if row['Value'] < row['Threshold'] else "inverse"
            ui.progress_bar(
                row['Value'], 
                f"{row['Metric']}: {row['Value']}%",
                100
            )

def render_quick_actions():
    """Render quick action buttons."""
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🤖 Create Agent", use_container_width=True):
            st.session_state.current_page = "Basic Agents"
            st.rerun()
    
    with col2:
        if st.button("🛠️ Browse Tools", use_container_width=True):
            st.session_state.current_page = "Tool Library"
            st.rerun()
    
    with col3:
        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state.current_page = "Performance Monitor"
            st.rerun()

if __name__ == "__main__":
    render()
