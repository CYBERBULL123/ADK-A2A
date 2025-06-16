"""
Getting Started page for new users.
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from frontend.utils.ui_components import ui

def render():
    """Render the getting started page."""
    ui.header(
        "Getting Started",
        "Your journey into Agent Development Kit and A2A protocols begins here",
        "🚀"
    )
    
    # Progress tracking
    render_progress_tracker()
    
    # Quick start guide
    render_quick_start()
    
    # Learning path
    render_learning_path()
    
    # Resources section
    render_resources()

def render_progress_tracker():
    """Render progress tracking for new users."""
    st.subheader("📈 Your Learning Progress")
    
    # Initialize progress in session state
    if "learning_progress" not in st.session_state:
        st.session_state.learning_progress = {
            "environment_setup": False,
            "first_agent": False,
            "tool_integration": False,
            "multi_agent": False,
            "a2a_protocol": False
        }
    
    progress = st.session_state.learning_progress
    completed = sum(progress.values())
    total = len(progress)
    
    # Progress bar
    progress_percentage = (completed / total) * 100
    ui.progress_bar(progress_percentage, f"Overall Progress: {completed}/{total} steps completed")
    
    # Progress items
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Completed Steps")
        for step, completed in progress.items():
            if completed:
                step_name = step.replace("_", " ").title()
                st.write(f"✅ {step_name}")
    
    with col2:
        st.markdown("### 📋 Remaining Steps")
        for step, completed in progress.items():
            if not completed:
                step_name = step.replace("_", " ").title()
                st.write(f"⏳ {step_name}")

def render_quick_start():
    """Render quick start guide."""
    st.subheader("⚡ Quick Start Guide")
    
    # Step-by-step guide
    steps = [
        {
            "title": "1. Environment Setup",
            "description": "Verify your environment and dependencies",
            "action": "Check Environment",
            "page": "Environment"
        },
        {
            "title": "2. Create Your First Agent", 
            "description": "Build a simple agent to understand the basics",
            "action": "Create Agent",
            "page": "Basic Agents"
        },
        {
            "title": "3. Explore Tools",
            "description": "Discover available tools and integrate them",
            "action": "Browse Tools",
            "page": "Tool Library"
        },
        {
            "title": "4. Multi-Agent Systems",
            "description": "Learn to coordinate multiple agents",
            "action": "Build Workflow",
            "page": "Multi-Agent Systems"
        },
        {
            "title": "5. A2A Protocol",
            "description": "Implement agent-to-agent communication",
            "action": "Learn A2A",
            "page": "A2A Protocol"
        }
    ]
    
    for i, step in enumerate(steps):
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.markdown(f"### {step['title']}")
                st.write(step['description'])
            
            with col2:
                progress_key = list(st.session_state.learning_progress.keys())[i]
                if st.session_state.learning_progress[progress_key]:
                    st.success("✅ Completed")
                else:
                    if st.button(step['action'], key=f"quick_start_{i}"):
                        st.session_state.current_page = step['page']
                        st.rerun()
            
            with col3:
                # Mark as complete button
                if not st.session_state.learning_progress[progress_key]:
                    if st.button("Mark Complete", key=f"complete_{i}"):
                        st.session_state.learning_progress[progress_key] = True
                        st.rerun()
        
        if i < len(steps) - 1:
            st.markdown("---")

def render_learning_path():
    """Render detailed learning path."""
    st.subheader("🎯 Learning Path")
    
    learning_modules = {
        "🟢 Beginner": [
            "Understanding ADK Architecture",
            "Creating Simple Agents", 
            "Basic Tool Integration",
            "Agent Configuration",
            "Testing and Debugging"
        ],
        "🟡 Intermediate": [
            "Multi-Agent Coordination",
            "Custom Tool Development",
            "State Management",
            "Error Handling",
            "Performance Optimization"
        ],
        "🔴 Advanced": [
            "A2A Protocol Implementation",
            "Distributed Agent Systems",
            "Security Considerations",
            "Production Deployment",
            "Monitoring and Analytics"
        ]
    }
    
    ui.tabs_container({
        level: lambda modules=modules: render_module_list(modules)
        for level, modules in learning_modules.items()
    })

def render_module_list(modules):
    """Render a list of learning modules."""
    for module in modules:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"📚 {module}")
        with col2:
            if st.button("Start", key=f"module_{module}", use_container_width=True):
                st.info(f"Module '{module}' is coming soon!")

def render_resources():
    """Render additional resources."""
    st.subheader("📚 Additional Resources")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📖 Documentation")
        docs = [
            "Agent Development Guide",
            "A2A Protocol Specification",
            "Multi-Agent Best Practices",
            "Tool Development Tutorial",
            "Deployment Guidelines"
        ]
        
        for doc in docs:
            if st.button(doc, key=f"doc_{doc}", use_container_width=True):
                st.session_state.current_page = "Documentation"
                st.rerun()
    
    with col2:
        st.markdown("### 🎥 Video Tutorials")
        videos = [
            "ADK Overview (10 min)",
            "Creating Your First Agent (15 min)",
            "Multi-Agent Workflows (20 min)",
            "A2A Implementation (25 min)",
            "Production Deployment (30 min)"
        ]
        
        for video in videos:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.write(f"🎬 {video}")
            with col_b:
                if st.button("▶️", key=f"video_{video}"):
                    st.info("Video tutorials coming soon!")

def render_helpful_tips():
    """Render helpful tips for beginners."""
    st.subheader("💡 Helpful Tips")
    
    tips = [
        {
            "title": "Start Simple",
            "content": "Begin with basic agents before moving to complex multi-agent systems.",
            "type": "info"
        },
        {
            "title": "Use the Playground",
            "content": "Experiment with different configurations in the agent playground.",
            "type": "info"
        },
        {
            "title": "Read the Logs",
            "content": "Monitor agent logs to understand behavior and debug issues.",
            "type": "warning"
        },
        {
            "title": "Join the Community",
            "content": "Connect with other developers in our community forums.",
            "type": "success"
        }
    ]
    
    for tip in tips:
        ui.info_box(tip["title"], tip["content"], tip["type"])

if __name__ == "__main__":
    render()
