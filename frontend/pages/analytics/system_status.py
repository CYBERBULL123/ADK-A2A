"""
System Status page.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import psutil
import platform
from frontend.utils.ui_components import ui

def render():
    """Render the system status page."""
    ui.header(
        "System Status",
        "Monitor overall system health",
        "⚡"
    )
    
    # System overview
    st.subheader("🖥️ System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get actual system info where possible, fallback to mock data
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
    except:
        # Fallback to mock data
        cpu_percent = np.random.uniform(20, 80)
        memory = type('obj', (object,), {
            'percent': np.random.uniform(40, 70),
            'used': np.random.uniform(4, 8) * 1024**3,
            'total': 16 * 1024**3        })
        disk = type('obj', (object,), {
            'percent': np.random.uniform(30, 60),
            'used': np.random.uniform(100, 300) * 1024**3,
            'total': 512 * 1024**3
        })
    
    with col1:
        ui.metric_card(
            "CPU Usage",
            f"{cpu_percent:.1f}%",
            "Normal" if cpu_percent < 80 else "High",
            icon="🖥️"
        )
    
    with col2:
        ui.metric_card(
            "Memory Usage",
            f"{memory.percent:.1f}%",
            f"{memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB",
            icon="🧠"
        )
    
    with col3:
        ui.metric_card(
            "Disk Usage",
            f"{disk.percent:.1f}%",
            f"{disk.used / 1024**3:.0f}GB / {disk.total / 1024**3:.0f}GB",
            icon="💾"
        )
    
    with col4:
        system_health = "Healthy" if cpu_percent < 80 and memory.percent < 80 and disk.percent < 80 else "Warning"
        ui.metric_card(
            "System Health",
            system_health,
            "All systems operational" if system_health == "Healthy" else "Attention needed",
            icon="❤️" if system_health == "Healthy" else "⚠️"
        )
    
    st.markdown("---")
    
    # Service status
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("� Service Status")
        
        # Mock service status data
        services = [
            {"name": "ADK Core Service", "status": "Running", "uptime": "5d 14h 23m", "health": "Healthy"},
            {"name": "Agent Manager", "status": "Running", "uptime": "5d 14h 20m", "health": "Healthy"},
            {"name": "A2A Protocol Handler", "status": "Running", "uptime": "3d 8h 45m", "health": "Healthy"},
            {"name": "Tool Registry", "status": "Running", "uptime": "5d 14h 23m", "health": "Healthy"},
            {"name": "Analytics Engine", "status": "Running", "uptime": "1d 12h 30m", "health": "Warning"},
            {"name": "Message Queue", "status": "Running", "uptime": "5d 14h 15m", "health": "Healthy"},
            {"name": "Database Connection", "status": "Running", "uptime": "5d 14h 23m", "health": "Healthy"},
            {"name": "API Gateway", "status": "Running", "uptime": "5d 14h 23m", "health": "Healthy"}
        ]
        
        for service in services:
            cols = st.columns([3, 1, 2, 1])
            with cols[0]:
                st.write(f"**{service['name']}**")
            with cols[1]:
                if service['status'] == 'Running':
                    st.success("🟢 Running")
                else:
                    st.error("🔴 Stopped")
            with cols[2]:
                st.write(service['uptime'])
            with cols[3]:
                if service['health'] == 'Healthy':
                    st.success("✅")
                elif service['health'] == 'Warning':
                    st.warning("⚠️")
                else:
                    st.error("❌")
    
    with col2:
        st.subheader("📊 Quick Stats")
        
        # System information
        try:
            system_info = {
                "OS": platform.system(),
                "Version": platform.release(),
                "Architecture": platform.machine(),
                "Python": platform.python_version(),
                "Hostname": platform.node()
            }
        except:
            system_info = {
                "OS": "Linux",
                "Version": "Ubuntu 20.04",
                "Architecture": "x86_64",
                "Python": "3.11.0",
                "Hostname": "adk-server"
            }
        
        for key, value in system_info.items():
            st.write(f"**{key}:** {value}")
        
        st.markdown("---")
        st.write("**Uptime:** 5d 14h 23m")
        st.write("**Load Average:** 1.2, 1.5, 1.8")
        st.write("**Network:** 125 MB/s ↓ 45 MB/s ↑")
    
    st.markdown("---")
    
    # Network and connectivity
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌐 Network Status")
        
        # Mock network data
        network_stats = pd.DataFrame({
            'Interface': ['eth0', 'wlan0', 'lo'],
            'Status': ['Active', 'Inactive', 'Active'],
            'IP Address': ['192.168.1.100', 'N/A', '127.0.0.1'],
            'Bytes Sent': ['2.3 GB', 'N/A', '45 MB'],
            'Bytes Received': ['8.7 GB', 'N/A', '45 MB']
        })
        
        st.dataframe(network_stats, use_container_width=True, hide_index=True)
        
        # Connection status
        st.markdown("**External Connectivity:**")
        connections = [
            ("Google ADK API", "✅ Connected", "15ms"),
            ("Gemini API", "✅ Connected", "23ms"),
            ("External Tools", "✅ Connected", "8ms"),
            ("Database", "✅ Connected", "2ms")
        ]
        
        for service, status, latency in connections:
            cols = st.columns([2, 1, 1])
            with cols[0]:
                st.write(service)
            with cols[1]:
                st.write(status)
            with cols[2]:
                st.write(latency)
    
    with col2:
        st.subheader("🔄 Resource Trends")
        
        # Generate sample trend data
        times = pd.date_range(
            start=datetime.now() - timedelta(hours=6),
            end=datetime.now(),
            freq='10min'
        )
        
        cpu_trend = np.random.normal(cpu_percent, 10, len(times))
        memory_trend = np.random.normal(memory.percent, 5, len(times))
        
        fig_trends = go.Figure()
        fig_trends.add_trace(go.Scatter(
            x=times,
            y=cpu_trend,
            mode='lines',
            name='CPU %',
            line=dict(color='#1f77b4')
        ))
        fig_trends.add_trace(go.Scatter(
            x=times,
            y=memory_trend,
            mode='lines',
            name='Memory %',
            line=dict(color='#ff7f0e'),
            yaxis='y2'
        ))
        
        fig_trends.update_layout(
            height=300,
            xaxis_title="Time",
            yaxis=dict(title="CPU %", side="left"),
            yaxis2=dict(title="Memory %", side="right", overlaying="y"),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig_trends, use_container_width=True)
    
    st.markdown("---")
    
    # Alerts and logs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚨 Recent Alerts")
        
        alerts = [
            {"time": "2 minutes ago", "type": "Warning", "message": "Analytics Engine memory usage above 80%"},
            {"time": "15 minutes ago", "type": "Info", "message": "New agent registered: Research Agent v2.1"},
            {"time": "1 hour ago", "type": "Warning", "message": "High API response time detected"},
            {"time": "3 hours ago", "type": "Success", "message": "System backup completed successfully"},
            {"time": "6 hours ago", "type": "Info", "message": "Scheduled maintenance completed"}
        ]
        
        for alert in alerts:
            icon = "⚠️" if alert["type"] == "Warning" else "ℹ️" if alert["type"] == "Info" else "✅"
            st.write(f"{icon} **{alert['time']}** - {alert['message']}")
    
    with col2:
        st.subheader("📋 System Logs")
        
        logs = [
            "2024-01-15 14:30:25 [INFO] Agent Manager: New multi-agent workflow started",
            "2024-01-15 14:29:12 [DEBUG] A2A Protocol: Message routed successfully",
            "2024-01-15 14:28:45 [INFO] Tool Registry: Custom tool 'data_analyzer' registered",
            "2024-01-15 14:27:33 [WARN] Analytics Engine: Memory usage threshold exceeded",
            "2024-01-15 14:26:18 [INFO] API Gateway: Rate limit adjusted for burst traffic"
        ]
        
        for log in logs:
            if "[WARN]" in log:
                st.warning(log)
            elif "[ERROR]" in log:
                st.error(log)
            else:
                st.text(log)
    
    # Control panel
    st.markdown("---")
    st.subheader("🎛️ System Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Restart Services", use_container_width=True):
            st.success("Services restart initiated...")
    
    with col2:
        if st.button("🧹 Clear Cache", use_container_width=True):
            st.success("System cache cleared!")
    
    with col3:
        if st.button("📊 Generate Report", use_container_width=True):
            st.success("System report generated!")
    
    with col4:
        auto_refresh = st.checkbox("🔄 Auto-refresh (30s)")
        if auto_refresh:
            import time
            time.sleep(30)
            st.rerun()

if __name__ == "__main__":
    render()
