"""
Tool Library page for browsing and managing tools.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from frontend.utils.ui_components import ui, charts

def render():
    """Render the tool library page."""
    ui.header(
        "Tool Library",
        "Discover and manage tools for your agents",
        "🛠️"
    )
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Browse Tools",
        "📦 Tool Categories", 
        "⭐ Featured Tools",
        "📊 Usage Analytics"
    ])
    
    with tab1:
        render_tool_browser()
    
    with tab2:
        render_tool_categories()
    
    with tab3:
        render_featured_tools()
    
    with tab4:
        render_usage_analytics()

def render_tool_browser():
    """Render tool browser interface."""
    st.subheader("🔍 Browse Available Tools")
    
    # Search and filters
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_query = st.text_input(
            "Search Tools",
            placeholder="Search by name, description, or category...",
            key="tool_search"
        )
    
    with col2:
        category_filter = st.selectbox(
            "Category",
            ["All", "Web", "Data", "AI/ML", "Utility", "Custom"],
            key="category_filter"
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort By",
            ["Name", "Popularity", "Recent", "Rating"],
            key="sort_by"
        )
      # Tool grid
    tools = get_available_tools()
    filtered_tools = filter_tools(tools, search_query, category_filter, sort_by)
    
    # Display tools in a grid
    cols_per_row = 3
    for i in range(0, len(filtered_tools), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(filtered_tools):
                tool = filtered_tools[i + j]
                with col:
                    render_tool_card(tool)
    
    # Check for tool details to display
    render_tool_details_section(filtered_tools)

def render_tool_card(tool: Dict[str, Any]):
    """Render a single tool card."""
    with st.container():
        # Tool header
        st.markdown(f"### {tool['icon']} {tool['name']}")
        
        # Tool info
        st.write(f"**Category:** {tool['category']}")
        st.write(f"**Version:** {tool['version']}")
        
        # Rating
        rating_stars = "⭐" * int(tool['rating'])
        st.write(f"**Rating:** {rating_stars} ({tool['rating']}/5)")
        
        # Description
        st.write(tool['description'])
        
        # Usage stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Downloads", tool['downloads'])
        with col2:
            st.metric("Users", tool['users'])
          # Actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📖 Details", key=f"details_{tool['id']}", use_container_width=True):
                st.session_state[f"show_details_{tool['id']}"] = True
                st.rerun()
        with col2:
            if st.button("🔧 Use Tool", key=f"use_{tool['id']}", use_container_width=True):
                add_tool_to_agent(tool)
        
        st.markdown("---")

def get_available_tools() -> List[Dict[str, Any]]:
    """Get list of available tools."""
    # Mock data - replace with actual tool discovery
    tools = [
        {
            "id": "web_search",
            "name": "Web Search",
            "category": "Web",
            "version": "1.2.0",
            "icon": "🔍",
            "description": "Search the web for real-time information",
            "rating": 4.8,
            "downloads": "10K+",
            "users": "2.5K",
            "parameters": ["query", "max_results"],
            "author": "ADK Team"
        },
        {
            "id": "data_analyzer",
            "name": "Data Analyzer",
            "category": "Data",
            "version": "2.1.0",
            "icon": "📊",
            "description": "Analyze datasets and generate insights",
            "rating": 4.6,
            "downloads": "8K+",
            "users": "1.8K",
            "parameters": ["data_file", "analysis_type"],
            "author": "DataCorp"
        },
        {
            "id": "image_generator",
            "name": "Image Generator",
            "category": "AI/ML",
            "version": "1.0.5",
            "icon": "🎨",
            "description": "Generate images from text descriptions",
            "rating": 4.9,
            "downloads": "15K+",
            "users": "3.2K",
            "parameters": ["prompt", "style", "size"],
            "author": "AI Artists"
        },
        {
            "id": "file_manager",
            "name": "File Manager",
            "category": "Utility",
            "version": "1.5.2",
            "icon": "📁",
            "description": "Manage files and directories",
            "rating": 4.3,
            "downloads": "5K+",
            "users": "1.2K",
            "parameters": ["operation", "path"],
            "author": "Utils Inc"
        },
        {
            "id": "weather_api",
            "name": "Weather API",
            "category": "Web",
            "version": "1.3.0",
            "icon": "🌤️",
            "description": "Get current weather and forecasts",
            "rating": 4.7,
            "downloads": "12K+",
            "users": "2.8K",
            "parameters": ["location", "units"],
            "author": "Weather Co"
        },
        {
            "id": "text_summarizer",
            "name": "Text Summarizer",
            "category": "AI/ML",
            "version": "2.0.1",
            "icon": "📝",
            "description": "Summarize long texts automatically",
            "rating": 4.5,
            "downloads": "9K+",
            "users": "2.1K",
            "parameters": ["text", "max_length"],
            "author": "NLP Labs"
        }
    ]
    return tools

def filter_tools(tools: List[Dict[str, Any]], search: str, category: str, sort: str) -> List[Dict[str, Any]]:
    """Filter and sort tools based on criteria."""
    filtered = tools.copy()
    
    # Apply search filter
    if search:
        filtered = [
            tool for tool in filtered
            if search.lower() in tool['name'].lower() or 
               search.lower() in tool['description'].lower() or
               search.lower() in tool['category'].lower()
        ]
    
    # Apply category filter
    if category != "All":
        filtered = [tool for tool in filtered if tool['category'] == category]
    
    # Apply sorting
    if sort == "Name":
        filtered.sort(key=lambda x: x['name'])
    elif sort == "Popularity":
        filtered.sort(key=lambda x: float(x['downloads'].replace('K+', '000').replace('+', '')), reverse=True)
    elif sort == "Rating":
        filtered.sort(key=lambda x: x['rating'], reverse=True)
    
    return filtered

def render_tool_categories():
    """Render tool categories."""
    st.subheader("📦 Tool Categories")
    
    categories = {
        "🌐 Web Tools": {
            "count": 15,
            "description": "Tools for web scraping, API calls, and online data retrieval",
            "popular": ["Web Search", "Weather API", "News Fetcher"]
        },
        "📊 Data Tools": {
            "count": 12,
            "description": "Tools for data analysis, processing, and visualization",
            "popular": ["Data Analyzer", "CSV Parser", "Chart Generator"]
        },
        "🤖 AI/ML Tools": {
            "count": 20,
            "description": "AI and machine learning tools for various tasks",
            "popular": ["Image Generator", "Text Summarizer", "Language Translator"]
        },
        "🔧 Utility Tools": {
            "count": 8,
            "description": "General utility tools for file management and system operations",
            "popular": ["File Manager", "PDF Reader", "Calculator"]
        },
        "🎨 Creative Tools": {
            "count": 6,
            "description": "Tools for creative tasks and content generation",
            "popular": ["Image Editor", "Music Generator", "Story Writer"]
        },
        "🛠️ Custom Tools": {
            "count": 3,
            "description": "User-created custom tools",
            "popular": ["My Custom Tool", "Team Workflow", "Special Function"]
        }
    }
    
    cols = st.columns(2)
    for i, (category, info) in enumerate(categories.items()):
        with cols[i % 2]:
            with st.container():
                st.markdown(f"### {category}")
                st.write(f"**Tools Available:** {info['count']}")
                st.write(info['description'])
                
                st.markdown("**Popular Tools:**")
                for tool in info['popular']:
                    st.write(f"• {tool}")
                
                if st.button(f"Browse {category.split()[1]}", key=f"browse_{i}", use_container_width=True):
                    st.info(f"Browsing {category} tools...")
                
                st.markdown("---")

def render_featured_tools():
    """Render featured tools section."""
    st.subheader("⭐ Featured Tools")
    
    featured_tools = [
        {
            "name": "Advanced Web Scraper",
            "description": "Powerful web scraping with JavaScript rendering support",
            "why_featured": "Most downloaded tool this month",
            "badge": "🔥 Trending"
        },
        {
            "name": "Smart Data Insights",
            "description": "AI-powered data analysis with automated insights generation",
            "why_featured": "Highest rated by users",
            "badge": "⭐ Top Rated"
        },
        {
            "name": "Multi-Modal AI Assistant",
            "description": "Handles text, images, and audio processing in one tool",
            "why_featured": "Editor's choice for versatility",
            "badge": "✨ Editor's Pick"
        }
    ]
    
    for tool in featured_tools:
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"#### {tool['name']}")
                st.write(tool['description'])
                st.caption(f"Featured because: {tool['why_featured']}")
            
            with col2:
                st.markdown(f"**{tool['badge']}**")
                if st.button("Try Now", key=f"featured_{tool['name']}", use_container_width=True):
                    st.info(f"Loading {tool['name']}...")
            
            st.markdown("---")

def render_usage_analytics():
    """Render tool usage analytics."""
    st.subheader("📊 Tool Usage Analytics")
    
    # Mock analytics data
    import random
    from datetime import datetime, timedelta
    
    # Usage metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ui.metric_card("Total Tools", "64", "+5 this week")
    with col2:
        ui.metric_card("Active Users", "1,247", "+89 today")
    with col3:
        ui.metric_card("Daily Usage", "3,456", "+12% from yesterday")
    with col4:
        ui.metric_card("Avg Rating", "4.6", "+0.1 this month")
    
    # Usage trends
    st.markdown("### 📈 Usage Trends")
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    usage_data = pd.DataFrame({
        'Date': dates,
        'Tool Usage': [random.randint(100, 500) for _ in dates],
        'New Tools': [random.randint(0, 3) for _ in dates],
        'Active Users': [random.randint(50, 200) for _ in dates]
    })
    
    # Create tabs for different chart views
    chart_tab1, chart_tab2 = st.tabs(["📊 Usage Volume", "🔧 Tool Performance"])
    
    with chart_tab1:
        charts.line_chart(
            usage_data,
            x='Date',
            y='Tool Usage',
            title="Daily Tool Usage"
        )
    
    with chart_tab2:
        # Top tools by usage
        top_tools_data = pd.DataFrame({
            'Tool': ['Web Search', 'Data Analyzer', 'Image Generator', 'Weather API', 'Text Summarizer'],
            'Usage Count': [1250, 890, 756, 634, 523],
            'Success Rate': [98.5, 94.2, 96.8, 99.1, 92.3]
        })
        
        charts.bar_chart(
            top_tools_data,
            x='Tool',
            y='Usage Count',
            title="Top Tools by Usage"        )

def add_tool_to_agent(tool: Dict[str, Any]):
    """Add a tool to an agent."""
    # Initialize tool selections in session state
    if "selected_tools" not in st.session_state:
        st.session_state.selected_tools = []
    
    if tool['id'] not in [t['id'] for t in st.session_state.selected_tools]:
        st.session_state.selected_tools.append(tool)
        st.success(f"✅ {tool['name']} added to your toolkit!")
    else:
        st.warning(f"⚠️ {tool['name']} is already in your toolkit!")

def render_tool_details_section(tools: List[Dict[str, Any]]):
    """Render tool details section if any tool details are requested."""
    for tool in tools:
        if st.session_state.get(f"show_details_{tool['id']}", False):
            st.markdown("---")
            st.subheader(f"📖 {tool['name']} Details")
            
            # Close button
            if st.button("❌ Close Details", key=f"close_details_{tool['id']}"):
                st.session_state[f"show_details_{tool['id']}"] = False
                st.rerun()
            
            # Tool details (no nested columns)
            st.markdown("### Basic Information")
            st.write(f"**Name:** {tool['name']}")
            st.write(f"**Category:** {tool['category']}")
            st.write(f"**Version:** {tool['version']}")
            st.write(f"**Author:** {tool['author']}")
            st.write(f"**Rating:** {'⭐' * int(tool['rating'])} ({tool['rating']}/5)")
            
            st.markdown("### Usage Statistics")
            st.write(f"**Downloads:** {tool['downloads']}")
            st.write(f"**Active Users:** {tool['users']}")
            st.write(f"**Parameters:** {', '.join(tool['parameters'])}")
            
            st.markdown("### Description")
            st.write(tool['description'])
            
            st.markdown("### Example Usage")
            example_code = f"""
# Example usage of {tool['name']}
tool = get_tool('{tool['id']}')
result = tool.execute(
    {tool['parameters'][0]}="example_value"
)
print(result)
"""
            ui.code_block(example_code, "python", "Python Example")
