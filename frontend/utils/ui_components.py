"""
Reusable UI components for the modular frontend.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path

def load_global_css():
    """Load global CSS styles."""
    css_path = Path(__file__).parent.parent / "assets" / "styles.css"
    if css_path.exists():
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback CSS
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
        }
        
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
            border-left: 4px solid #667eea;
        }
        
        .feature-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
            transition: transform 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-online { background-color: #4CAF50; }
        .status-offline { background-color: #f44336; }
        .status-warning { background-color: #ff9800; }
        </style>
        """, unsafe_allow_html=True)

class UIComponents:
    """Collection of reusable UI components."""
    
    @staticmethod
    def header(title: str, subtitle: str = "", icon: str = "🤖"):
        """Create a styled header."""
        st.markdown(f"""        <div class="main-header">
            <h1>{icon} {title}</h1>
            {f'<p style="font-size: 1.2em; margin-top: 1rem;">{subtitle}</p>' if subtitle else ''}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def metric_card(title: str, value: Union[str, int, float], delta: str = None, 
                   delta_color: str = "normal", icon: str = None):
        """Create a metric card with optional delta and icon."""
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Add icon to title if provided
            display_title = f"{icon} {title}" if icon else title
            st.metric(
                label=display_title,
                value=value,
                delta=delta,
                delta_color=delta_color
            )
    
    @staticmethod
    def feature_card(title: str, description: str, icon: str = "📋", 
                    button_text: str = None, button_key: str = None):
        """Create a feature card with optional action button."""
        with st.container():
            st.markdown(f"""
            <div class="feature-card">
                <h3>{icon} {title}</h3>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if button_text and button_key:
                return st.button(button_text, key=button_key, use_container_width=True)
        return False
    
    @staticmethod
    def status_indicator(status: str, label: str = ""):
        """Create a status indicator."""
        status_class = f"status-{status.lower()}"
        st.markdown(f"""
        <span class="status-indicator {status_class}"></span>
        <span>{label or status.title()}</span>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def progress_bar(value: float, label: str = "", max_value: float = 100):
        """Create a styled progress bar."""
        if label:
            st.write(label)
        st.progress(value / max_value)
    
    @staticmethod
    def code_block(code: str, language: str = "python", title: str = None):
        """Create a styled code block."""
        if title:
            st.subheader(title)
        st.code(code, language=language)
    
    @staticmethod
    def info_box(title: str, content: str, box_type: str = "info"):
        """Create an info box with different types."""
        if box_type == "info":
            st.info(f"**{title}**\n\n{content}")
        elif box_type == "warning":
            st.warning(f"**{title}**\n\n{content}")
        elif box_type == "error":
            st.error(f"**{title}**\n\n{content}")
        elif box_type == "success":
            st.success(f"**{title}**\n\n{content}")
    
    @staticmethod
    def data_table(data: pd.DataFrame, title: str = None, height: int = None):
        """Create a styled data table."""
        if title:
            st.subheader(title)
        
        if height:
            st.dataframe(data, height=height, use_container_width=True)
        else:
            st.dataframe(data, use_container_width=True)
    
    @staticmethod
    def tabs_container(tab_data: Dict[str, Any]):
        """Create a tabs container with content."""
        tab_names = list(tab_data.keys())
        tabs = st.tabs(tab_names)
        
        for i, (tab_name, content) in enumerate(tab_data.items()):
            with tabs[i]:
                if callable(content):
                    content()
                else:
                    st.write(content)
    
    @staticmethod
    def metric_grid(metrics: List[Dict[str, Any]], cols: int = 4):
        """Create a grid of metrics."""
        columns = st.columns(cols)
        
        for i, metric in enumerate(metrics):
            col_idx = i % cols
            with columns[col_idx]:
                st.metric(
                    label=metric.get("label", ""),
                    value=metric.get("value", ""),
                    delta=metric.get("delta"),
                    delta_color=metric.get("delta_color", "normal")
                )

class ChartComponents:
    """Collection of chart components."""
    
    @staticmethod
    def line_chart(data: pd.DataFrame, x: str, y: str, title: str = None,
                  color: str = None):
        """Create a line chart."""
        fig = px.line(data, x=x, y=y, color=color, title=title)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def bar_chart(data: pd.DataFrame, x: str, y: str, title: str = None,
                 color: str = None):
        """Create a bar chart."""
        fig = px.bar(data, x=x, y=y, color=color, title=title)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def pie_chart(data: pd.DataFrame, values: str, names: str, title: str = None):
        """Create a pie chart."""
        fig = px.pie(data, values=values, names=names, title=title)
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def scatter_plot(data: pd.DataFrame, x: str, y: str, title: str = None,
                    color: str = None, size: str = None):
        """Create a scatter plot."""
        fig = px.scatter(data, x=x, y=y, color=color, size=size, title=title)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)

# Convenience instances
ui = UIComponents()
charts = ChartComponents()
