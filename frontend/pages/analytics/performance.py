"""
Performance Monitor page.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from frontend.utils.ui_components import ui

def render():
    """Render the performance monitor page."""
    ui.header(
        "Performance Monitor",
        "Monitor system and agent performance",
        "📊"
    )
    
    # Performance overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ui.metric_card(
            "System CPU",
            "67%",
            "+2.3%",
            icon="🖥️"
        )
    
    with col2:
        ui.metric_card(
            "Memory Usage",
            "4.2 GB",
            "+0.5 GB",
            icon="🧠"
        )
    
    with col3:
        ui.metric_card(
            "Active Agents",
            "12",
            "+3",
            icon="🤖"
        )
    
    with col4:
        ui.metric_card(
            "Avg Response",
            "245ms",
            "-15ms",
            icon="⚡"
        )
    
    st.markdown("---")
    
    # Real-time performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("� CPU Usage Over Time")
        
        # Generate sample CPU data
        times = pd.date_range(
            start=datetime.now() - timedelta(hours=1),
            end=datetime.now(),
            freq='1min'
        )
        cpu_data = np.random.normal(65, 10, len(times))
        cpu_data = np.clip(cpu_data, 0, 100)
        
        fig_cpu = go.Figure()
        fig_cpu.add_trace(go.Scatter(
            x=times,
            y=cpu_data,
            mode='lines',
            name='CPU %',
            line=dict(color='#1f77b4', width=2)
        ))
        fig_cpu.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            xaxis_title="Time",
            yaxis_title="CPU %"
        )
        st.plotly_chart(fig_cpu, use_container_width=True)
    
    with col2:
        st.subheader("🧠 Memory Usage")
        
        # Generate sample memory data
        memory_data = np.random.normal(4.2, 0.5, len(times))
        memory_data = np.clip(memory_data, 0, 8)
        
        fig_mem = go.Figure()
        fig_mem.add_trace(go.Scatter(
            x=times,
            y=memory_data,
            mode='lines',
            name='Memory GB',
            line=dict(color='#ff7f0e', width=2)
        ))
        fig_mem.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            xaxis_title="Time",
            yaxis_title="Memory (GB)"
        )
        st.plotly_chart(fig_mem, use_container_width=True)
    
    st.markdown("---")
    
    # Agent performance breakdown
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🤖 Agent Performance Breakdown")
        
        # Sample agent performance data
        agent_data = pd.DataFrame({
            'Agent': ['Research Agent', 'Analysis Agent', 'Communication Agent', 'Planning Agent', 'Tool Agent'],
            'Avg Response Time (ms)': [320, 180, 145, 290, 210],
            'Success Rate (%)': [94, 98, 96, 89, 92],
            'Memory Usage (MB)': [245, 180, 120, 200, 165]
        })
        
        fig_agents = px.scatter(
            agent_data, 
            x='Avg Response Time (ms)', 
            y='Success Rate (%)',
            size='Memory Usage (MB)',
            hover_data=['Agent'],
            title="Agent Performance (Response Time vs Success Rate)"
        )
        fig_agents.update_layout(height=400)
        st.plotly_chart(fig_agents, use_container_width=True)
    
    with col2:
        st.subheader("⚡ Response Time Distribution")
        
        # Response time histogram
        response_times = np.random.gamma(2, 50, 1000)
        
        fig_hist = px.histogram(
            x=response_times,
            nbins=30,
            title="Response Time Distribution"
        )
        fig_hist.update_layout(
            height=400,
            xaxis_title="Response Time (ms)",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    
    # Performance alerts and recommendations
    st.subheader("🚨 Performance Alerts & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.warning("⚠️ **High CPU Usage Detected**\nSystem CPU usage has been consistently above 70% for the past 15 minutes.")
        st.info("💡 **Recommendation**: Consider scaling horizontally or optimizing agent workloads.")
    
    with col2:
        st.success("✅ **All Agents Responding**\nAll 12 active agents are responding within acceptable limits.")
        st.info("💡 **Tip**: Current configuration is performing well. Monitor for any changes in load patterns.")
    
    # Auto-refresh toggle
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        auto_refresh = st.checkbox("🔄 Auto-refresh", value=False)
    
    with col2:
        refresh_interval = st.selectbox(
            "Refresh interval", 
            ["5 seconds", "10 seconds", "30 seconds", "1 minute"],
            index=1
        )
    
    if auto_refresh:
        interval_map = {
            "5 seconds": 5,
            "10 seconds": 10,
            "30 seconds": 30,
            "1 minute": 60
        }
        time.sleep(interval_map[refresh_interval])
        st.rerun()

if __name__ == "__main__":
    render()
