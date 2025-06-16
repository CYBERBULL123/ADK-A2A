"""
UI Utilities for Modern Streamlit Dashboard.

This module provides reusable UI components and formatting utilities
for creating an engaging and professional user interface.
"""

import streamlit as st
import markdown
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
import base64


def load_css():
    """Load custom CSS styles for the dashboard."""
    try:
        with open("frontend/styles.css", "r") as f:
            css = f"<style>{f.read()}</style>"
            st.markdown(css, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Custom CSS file not found. Using default styling.")


def create_header(title: str, subtitle: str = "", icon: str = "🤖"):
    """Create a modern header with gradient background."""
    st.markdown(f"""
    <div class="main-header fade-in">
        <h1>{icon} {title}</h1>
        {f'<p>{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def create_feature_card(title: str, description: str, icon: str, action_text: str = None, action_key: str = None):
    """Create a feature card with hover effects."""
    card_html = f"""
    <div class="feature-card fade-in">
        <div class="icon">{icon}</div>
        <h3>{title}</h3>
        <p>{description}</p>
    </div>
    """
    
    col = st.container()
    with col:
        st.markdown(card_html, unsafe_allow_html=True)
        if action_text and action_key:
            return st.button(action_text, key=action_key, use_container_width=True)
    return False


def create_status_card(title: str, content: str, status: str = "info", icon: str = "ℹ️"):
    """Create a status card with appropriate styling."""
    status_class = f"status-{status}"
    
    st.markdown(f"""
    <div class="custom-card fade-in">
        <div class="status-indicator {status_class}">
            {icon} <strong>{title}</strong>
        </div>
        <div style="margin-top: 1rem;">
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_metric_card(value: str, label: str, change: str = None, change_type: str = "positive"):
    """Create a metric card with value, label, and optional change indicator."""
    change_html = ""
    if change:
        change_class = "positive" if change_type == "positive" else "negative"
        change_symbol = "↗️" if change_type == "positive" else "↘️"
        change_html = f'<div class="metric-change {change_class}">{change_symbol} {change}</div>'
    
    st.markdown(f"""
    <div class="metric-card fade-in">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {change_html}
    </div>
    """, unsafe_allow_html=True)


def format_agent_response(response: str) -> str:
    """Format agent response with proper markdown rendering and styling."""
    if not response:
        return "No response available."
    
    # Convert markdown to HTML
    html_content = markdown.markdown(response, extensions=['tables', 'fenced_code'])
    
    # Add custom styling
    formatted_html = f"""
    <div class="response-container fade-in">
        {html_content}
    </div>
    """
    
    return formatted_html


def display_agent_response(response: str, title: str = "Agent Response"):
    """Display formatted agent response in a styled container."""
    if not response:
        st.warning("No response available.")
        return
    
    st.markdown(f"**{title}:**")
    formatted_response = format_agent_response(response)
    st.markdown(formatted_response, unsafe_allow_html=True)


def create_chat_interface():
    """Create a chat-like interface for agent interactions."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for i, message in enumerate(st.session_state.chat_history):
            message_type = "user" if message["type"] == "user" else "agent"
            timestamp = message.get("timestamp", "")
            
            st.markdown(f"""
            <div class="chat-message {message_type}">
                <div style="font-size: 0.8rem; opacity: 0.7; margin-bottom: 0.5rem;">
                    {message["sender"]} • {timestamp}
                </div>
                <div>{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return chat_container


def add_chat_message(sender: str, content: str, message_type: str = "agent"):
    """Add a message to the chat history."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    message = {
        "sender": sender,
        "content": content,
        "type": message_type,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }
    
    st.session_state.chat_history.append(message)


def clear_chat_history():
    """Clear the chat history."""
    if 'chat_history' in st.session_state:
        st.session_state.chat_history = []


def create_progress_indicator(current_step: int, total_steps: int, step_names: List[str]):
    """Create a progress indicator showing current step in a process."""
    progress_html = '<div class="progress-container">'
    progress_html += f'<h4>Progress: Step {current_step} of {total_steps}</h4>'
    progress_html += '<div style="display: flex; align-items: center; margin: 1rem 0;">'
    
    for i, step_name in enumerate(step_names, 1):
        if i < current_step:
            status_class = "status-success"
            icon = "✅"
        elif i == current_step:
            status_class = "status-info pulse"
            icon = "🔄"
        else:
            status_class = "status-warning"
            icon = "⏳"
        
        progress_html += f'''
        <div class="status-indicator {status_class}" style="margin: 0.25rem;">
            {icon} {step_name}
        </div>
        '''
        
        if i < len(step_names):
            progress_html += '<div style="flex: 1; height: 2px; background: #e9ecef; margin: 0 0.5rem;"></div>'
    
    progress_html += '</div></div>'
    
    st.markdown(progress_html, unsafe_allow_html=True)


def create_agent_network_visualization(agents: List[Dict[str, Any]]):
    """Create a visual representation of agent network."""
    network_html = '<div class="custom-card">'
    network_html += '<h3>🌐 Agent Network</h3>'
    network_html += '<div style="display: flex; align-items: center; justify-content: center; flex-wrap: wrap; padding: 2rem;">'
    
    for i, agent in enumerate(agents):
        network_html += f'''
        <div class="network-node pulse">
            {agent.get("icon", "🤖")}
        </div>
        '''
        
        if i < len(agents) - 1:
            network_html += '<div class="network-connection"></div>'
    
    network_html += '</div>'
    
    # Add agent details
    network_html += '<div style="margin-top: 1rem;">'
    for agent in agents:
        status_icon = "🟢" if agent.get("status") == "active" else "🔴"
        network_html += f'''
        <div style="display: flex; align-items: center; margin: 0.5rem 0;">
            {status_icon} <strong>{agent.get("name", "Unknown")}</strong> - {agent.get("type", "Generic")}
            <span style="margin-left: auto; font-size: 0.9rem; color: #6c757d;">
                Port: {agent.get("port", "N/A")}
            </span>
        </div>
        '''
    
    network_html += '</div></div>'
    
    st.markdown(network_html, unsafe_allow_html=True)


def create_network_visualization(agent_names):
    """Create a simple network visualization of agents."""
    st.markdown(f"""
    <div class="custom-card">
        <h4 style="color: #2E86AB; text-align: center; margin-bottom: 1rem;">🕸️ Agent Network</h4>
        <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 1rem;">
            {' '.join([f'<div style="background: linear-gradient(135deg, #2E86AB, #A23B72); color: white; padding: 0.5rem 1rem; border-radius: 20px; text-align: center; min-width: 100px;">🤖 {name}</div>' for name in agent_names])}
        </div>
        <div style="text-align: center; margin-top: 1rem; color: #6c757d;">
            <small>Interconnected agent network with {len(agent_names)} active nodes</small>
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_code_block(code: str, language: str = "python", title: str = None):
    """Create a styled code block with syntax highlighting."""
    if title:
        st.markdown(f"**{title}:**")
    
    st.code(code, language=language)


def create_expandable_section(title: str, content: str, icon: str = "📋"):
    """Create an expandable section with custom styling."""
    with st.expander(f"{icon} {title}"):
        if content.startswith("<"):
            st.markdown(content, unsafe_allow_html=True)
        else:
            st.markdown(content)


def create_data_table(data: List[Dict], title: str = None):
    """Create a styled data table."""
    if title:
        st.markdown(f"**{title}:**")
    
    if not data:
        st.info("No data available.")
        return
    
    # Convert to HTML table for better styling
    table_html = '<div class="custom-card">'
    table_html += '<table style="width: 100%; border-collapse: collapse;">'
    
    # Header
    if data:
        table_html += '<thead><tr style="background: #f8f9fa;">'
        for key in data[0].keys():
            table_html += f'<th style="padding: 0.75rem; text-align: left; border-bottom: 2px solid #dee2e6;">{key}</th>'
        table_html += '</tr></thead>'
        
        # Body
        table_html += '<tbody>'
        for row in data:
            table_html += '<tr style="border-bottom: 1px solid #dee2e6;">'
            for value in row.values():
                table_html += f'<td style="padding: 0.75rem;">{value}</td>'
            table_html += '</tr>'
        table_html += '</tbody>'
    
    table_html += '</table></div>'
    
    st.markdown(table_html, unsafe_allow_html=True)


def create_notification(message: str, notification_type: str = "info", auto_dismiss: bool = True):
    """Create a notification with different types."""
    icons = {
        "success": "✅",
        "warning": "⚠️", 
        "error": "❌",
        "info": "ℹ️"
    }
    
    icon = icons.get(notification_type, "ℹ️")
    
    if notification_type == "success":
        st.success(f"{icon} {message}")
    elif notification_type == "warning":
        st.warning(f"{icon} {message}")
    elif notification_type == "error":
        st.error(f"{icon} {message}")
    else:
        st.info(f"{icon} {message}")


def create_loading_spinner(text: str = "Processing..."):
    """Create a loading spinner with custom text."""
    return st.spinner(text)


def format_timestamp(timestamp = None) -> str:
    """Format timestamp for display. Accepts datetime objects or ISO format strings."""
    if timestamp is None:
        timestamp = datetime.now()
    elif isinstance(timestamp, str):
        try:
            # Parse ISO format string to datetime
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            # If parsing fails, return the string as-is
            return timestamp
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def create_sidebar_info(title: str, items: List[Dict[str, str]]):
    """Create an information section in the sidebar."""
    with st.sidebar:
        st.markdown(f"### {title}")
        for item in items:
            st.markdown(f"**{item['label']}:** {item['value']}")


def download_button(data: str, filename: str, mime_type: str = "text/plain"):
    """Create a download button for data."""
    b64_data = base64.b64encode(data.encode()).decode()
    href = f'data:{mime_type};base64,{b64_data}'
    
    st.markdown(f"""
    <a href="{href}" download="{filename}" class="custom-button">
        📥 Download {filename}
    </a>
    """, unsafe_allow_html=True)


def create_tabs_with_icons(tab_configs: List[Dict[str, str]]):
    """Create tabs with icons and return tab objects."""
    tab_names = [f"{config['icon']} {config['name']}" for config in tab_configs]
    return st.tabs(tab_names)


def highlight_text(text: str, highlight_terms: List[str]) -> str:
    """Highlight specific terms in text."""
    for term in highlight_terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        text = pattern.sub(f'<span class="highlight">{term}</span>', text)
    return text


def create_collapsible_json(data: Dict[str, Any], title: str = "Data"):
    """Create a collapsible JSON viewer."""
    with st.expander(f"📄 {title}"):
        st.json(data)


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def create_tooltip(text: str, tooltip_text: str) -> str:
    """Create text with tooltip on hover."""
    return f'<span class="tooltip" data-tooltip="{tooltip_text}">{text}</span>'
