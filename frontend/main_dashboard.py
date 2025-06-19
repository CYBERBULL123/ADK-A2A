"""
Main Streamlit Dashboard for ADK & A2A Learning Project.

This interactive dashboard provides hands-on experience with:
- Agent creation and testing
- Multi-agent coordination
- A2A protocol demonstration
- Tool integration and testing
- Performance monitoring and evaluation
"""

import streamlit as st
import asyncio
import json
import uuid
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, List
import sys
import random
import uuid
import json
from pathlib import Path
import os
import markdown

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config import config
from utils import validate_environment, display_welcome, console
from agents.basic import create_agent, SimpleAgent, SearchAgent, ToolAgent, StatefulAgent
from agents.multi_agent import CoordinatorAgent, WorkflowOrchestrator, EXAMPLE_WORKFLOWS
from agents.a2a import SmartA2AAgent, A2AOrchestrator

# Import tools with error handling
try:
    from tools import CUSTOM_TOOLS, get_tool_info, list_user_tools
except ImportError as e:
    console.print(f"[yellow]Warning: Could not import some tools functions: {e}[/yellow]")
    from tools import CUSTOM_TOOLS, get_tool_info
    
    # Define a fallback for list_user_tools if it's not available
    def list_user_tools():
        return []
from frontend.ui_utils import (
    load_css, create_header, create_feature_card, create_status_card,
    create_metric_card, display_agent_response, create_chat_interface,
    add_chat_message, clear_chat_history, create_progress_indicator,
    create_agent_network_visualization, create_code_block, create_expandable_section,
    create_data_table, create_notification, format_timestamp, create_tabs_with_icons,
    format_agent_response, display_agent_response, create_network_visualization
)

# Import the new tools page
from frontend.tools_page import show_tools_dashboard as show_tools_page


# Configure Streamlit page
st.set_page_config(
    page_title="ADK & A2A Learning Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_css()


def load_markdown_file(filename: str) -> str:
    """Load markdown file from docs folder."""
    try:
        docs_path = Path(__file__).parent.parent / "docs" / filename
        if docs_path.exists():
            with open(docs_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return f"# Documentation Not Found\n\nThe file `{filename}` could not be located in the docs folder."
    except Exception as e:
        return f"# Error Loading Documentation\n\nError reading `{filename}`: {str(e)}"


def render_markdown_content(content: str) -> None:
    """Render markdown content with proper formatting."""
    try:
        # Remove the main title if it exists (we'll handle it in the UI)
        lines = content.split('\n')
        if lines and lines[0].startswith('# '):
            content = '\n'.join(lines[1:])
        
        st.markdown(content)
    except Exception as e:
        st.error(f"Error rendering markdown: {e}")


def main():
    """Main dashboard function."""
    # Sidebar navigation with modern styling
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%); 
                padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">
        <h2 style="color: white; text-align: center; margin: 0;">🎯 Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Choose a learning module:",
        [
            "🏠 Overview",
            "🤖 Basic Agents", 
            "🔗 Multi-Agent Systems",
            "🌐 A2A Protocol",
            "🛠️ Custom Tools",
            "📊 Performance Analytics",
            "🎯 Evaluation Framework",
            "📚 Documentation"
        ]
    )      # Environment status with modern styling
    env_status = validate_environment()
    
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <h4 style="margin: 0 0 0.5rem 0; color: #2E86AB;">🔧 Environment Status</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if env_status:
            create_status_card("Environment", "All systems operational", "success", "✅")
        else:
            create_status_card("Environment", "Configuration needed", "warning", "⚠️")
            st.warning("Some features may be limited without proper API keys.")
            
    # Show warning if environment is not properly configured
    if not env_status:
        st.warning("⚠️ Please configure your API keys in the .env file for full functionality.")
      # Route to appropriate page
    if page == "🏠 Overview":
        show_overview()
    elif page == "🤖 Basic Agents":
        show_basic_agents()
    elif page == "🔗 Multi-Agent Systems":
        show_multi_agent_systems()
    elif page == "🌐 A2A Protocol":
        show_a2a_protocol()
    elif page == "🛠️ Custom Tools":
        show_tools_page()
    elif page == "📊 Performance Analytics":
        show_performance_analytics()
    elif page == "🎯 Evaluation Framework":
        show_evaluation_framework()
    elif page == "📚 Documentation":
        show_documentation()


def show_overview():
    """Show project overview and learning objectives."""
    
    # Modern header - only on overview page
    create_header(
        "ADK & A2A Learning Dashboard", 
        "Master Agent Development Kit and Agent-to-Agent protocols through interactive learning",
        "🤖"
    )
    
    # Learning objectives section with modern cards
    st.markdown("## 🎯 Learning Objectives")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        create_feature_card(
            "Agent Development Kit",
            "Master ADK fundamentals: agent architecture, tool integration, model connectivity, and code-first development patterns.",
            "🤖"
        )
    
    with col2:
        create_feature_card(
            "Multi-Agent Systems", 
            "Build coordinated agent teams with hierarchies, workflows, task delegation, and communication protocols.",
            "🔗"
        )
    
    with col3:
        create_feature_card(
            "A2A Protocol",
            "Implement distributed agent networks with remote communication, protocol design, and real-time coordination.",
            "🌐"
        )
    
    # Learning path visualization
    st.markdown("---")
    st.markdown("## 📚 Progressive Learning Path")
    
    learning_phases = [
        {"phase": "Phase 1", "title": "Foundations", "icon": "🏗️"},
        {"phase": "Phase 2", "title": "Coordination", "icon": "🔗"},
        {"phase": "Phase 3", "title": "Distribution", "icon": "🌐"},
        {"phase": "Phase 4", "title": "Production", "icon": "🚀"}
    ]
    
    cols = st.columns(4)
    for i, (col, phase) in enumerate(zip(cols, learning_phases)):
        with col:
            st.markdown(f"""
            <div class="custom-card" style="text-align: center; min-height: 200px;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">{phase['icon']}</div>
                <h3 style="color: #2E86AB; margin-bottom: 0.5rem;">{phase['phase']}</h3>
                <h4>{phase['title']}</h4>
                <div style="margin-top: 1rem; font-size: 0.9rem; color: #6c757d;">
                    {'Completed' if i < 2 else 'In Progress' if i == 2 else 'Upcoming'}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Quick start section
    st.markdown("---")
    st.markdown("## 🚀 Quick Start Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🤖 Try Basic Agent", use_container_width=True, type="primary"):
            try:
                st.session_state.demo_agent = SimpleAgent("QuickDemo")
                response = st.session_state.demo_agent.chat("Hello! Tell me about ADK in one paragraph.")
                display_agent_response(response, "Quick Demo Response")
            except Exception as e:
                create_notification(f"Error creating demo agent: {e}", "error")
    
    with col2:
        if st.button("🔧 Check Environment", use_container_width=True):
            validation = config.validate_config()
            if validation["valid"]:
                create_notification("Environment is properly configured!", "success")
                
                # Show detailed status
                with st.expander("📋 Detailed Environment Status"):
                    status_data = [
                        {"Component": "Google API", "Status": "✅ Connected", "Details": "ADK ready"},
                        {"Component": "Streamlit", "Status": "✅ Running", "Details": f"Version {st.__version__}"},
                        {"Component": "Python", "Status": "✅ Compatible", "Details": f"Version {sys.version.split()[0]}"},
                    ]
                    create_data_table(status_data, "System Status")
            else:
                create_notification("Please configure your API keys in .env file", "warning")
    
    with col3:
        if st.button("📖 View Documentation", use_container_width=True):
            st.info("📚 Navigate to the Documentation tab for comprehensive guides and examples!")
    
    # Recent activity simulation
    st.markdown("---")
    st.markdown("## 📈 Learning Progress")
    
    # Mock progress data
    progress_data = {
        "Module": ["Basic Agents", "Multi-Agent", "A2A Protocol", "Tools", "Analytics"],
        "Completion": [100, 75, 45, 60, 30],
        "Exercises": [5, 8, 3, 7, 2],
        "Status": ["✅ Complete", "🔄 In Progress", "🔄 In Progress", "🔄 In Progress", "⏳ Pending"]
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        df = pd.DataFrame(progress_data)
        fig = px.bar(df, x="Module", y="Completion", title="Learning Module Progress (%)",
                    color="Completion", color_continuous_scale="viridis")
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Overall statistics
        total_exercises = sum(progress_data["Exercises"])
        avg_completion = sum(progress_data["Completion"]) / len(progress_data["Completion"])
        
        create_metric_card(f"{avg_completion:.0f}%", "Overall Progress", "+15%", "positive")
        create_metric_card(f"{total_exercises}", "Total Exercises", "+3", "positive")
        create_metric_card("4/5", "Modules Started", "+1", "positive")


def show_basic_agents():
    """Show basic agent examples and testing interface."""
    st.markdown("## 🤖 Basic Agents Playground")
    st.markdown("Explore different types of agents and understand their capabilities through hands-on interaction.")
    
    # Agent type selection with modern cards
    st.markdown("### Select Agent Type")
    
    agent_configs = [
        {"type": "Simple Agent", "icon": "💬", "desc": "Basic conversational agent for general interactions"},
        {"type": "Search Agent", "icon": "🔍", "desc": "Web-enabled agent for research and fact-finding"},
        {"type": "Tool Agent", "icon": "🛠️", "desc": "Multi-tool agent with weather, calculator, and more"},
        {"type": "Stateful Agent", "icon": "🧠", "desc": "Memory-enabled agent for personalized conversations"}
    ]
    
    cols = st.columns(4)
    selected_agent = None
    
    for i, (col, config) in enumerate(zip(cols, agent_configs)):
        with col:
            if st.button(f"{config['icon']}\n{config['type']}", 
                        use_container_width=True, 
                        key=f"agent_select_{i}",
                        help=config['desc']):
                selected_agent = config['type']
                st.session_state.selected_agent_type = config['type']
    
    # Use session state to remember selection
    if 'selected_agent_type' not in st.session_state:
        st.session_state.selected_agent_type = "Simple Agent"
    
    agent_type = st.session_state.selected_agent_type
    
    # Initialize session state for agents
    if 'current_agent' not in st.session_state:
        st.session_state.current_agent = None
    
    # Agent creation and testing
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🔧 Agent Configuration")
        
        # Agent info card
        agent_info = next(config for config in agent_configs if config['type'] == agent_type)
        st.markdown(f"""
        <div class="custom-card">
            <div style="text-align: center; margin-bottom: 1rem;">
                <div style="font-size: 3rem;">{agent_info['icon']}</div>
                <h3 style="color: #2E86AB; margin: 0.5rem 0;">{agent_info['type']}</h3>
                <p style="color: #6c757d; margin: 0;">{agent_info['desc']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        agent_name = st.text_input("Agent Name", value=f"Demo_{agent_type.replace(' ', '_')}")
        
        if st.button("🚀 Create Agent", use_container_width=True, type="primary"):
            try:
                with st.spinner("Creating agent..."):
                    if agent_type == "Simple Agent":
                        st.session_state.current_agent = create_agent("simple", agent_name)
                    elif agent_type == "Search Agent":
                        st.session_state.current_agent = create_agent("search", agent_name)
                    elif agent_type == "Tool Agent":
                        st.session_state.current_agent = create_agent("tool", agent_name)
                    elif agent_type == "Stateful Agent":
                        st.session_state.current_agent = create_agent("stateful", agent_name)
                
                create_notification(f"Successfully created {agent_type}: {agent_name}", "success")
                clear_chat_history()  # Clear previous conversations
                
            except Exception as e:
                create_notification(f"Error creating agent: {e}", "error")
          
        # Agent status display
        if st.session_state.current_agent:
            agent_obj_type = type(st.session_state.current_agent).__name__
            
            st.markdown("### 📊 Agent Status")
            create_status_card(
                f"🤖 {st.session_state.current_agent.name}",
                f"Type: {agent_obj_type}<br>Status: Active<br>Created: {format_timestamp()}",
                "success",
                "🟢"
            )
              # Agent capabilities
            capabilities = {
                "SimpleAgent": ["💬 Natural conversation", "🧠 Basic reasoning"],
                "SearchAgent": ["💬 Natural conversation", "🔍 Web search", "📊 Information retrieval"],
                "ToolAgent": ["💬 Natural conversation", "🌤️ Weather queries", "🧮 Calculations", "📊 Multi-tool access"],
                "StatefulAgent": ["💬 Natural conversation", "🧠 Memory retention", "📚 Context awareness", "📈 Learning"]
            }
            
            if agent_obj_type in capabilities:
                with st.expander("🎯 Agent Capabilities"):
                    for capability in capabilities[agent_obj_type]:
                        st.markdown(f"• {capability}")
    
    with col2:
        st.markdown("### 💬 Interactive Chat")
        
        if st.session_state.current_agent:            # Chat interface with history
            chat_container = create_chat_interface()
            
            # Clear input trigger
            if 'clear_input' not in st.session_state:
                st.session_state.clear_input = False
            
            # Input area without columns - full width
            input_key = f"chat_input_{st.session_state.get('input_counter', 0)}"
            user_message = st.text_input(
                "💬 Enter your message:",
                placeholder="Ask the agent something...",
                key=input_key
            )
            
            send_button = st.button("📤 Send Message", use_container_width=True, type="primary")
            
            if send_button and user_message:
                # Add user message to chat
                add_chat_message("You", user_message, "user")
                
                # Increment counter to clear input
                st.session_state.input_counter = st.session_state.get('input_counter', 0) + 1
                try:
                    with st.spinner("Agent is thinking..."):
                        # Get response based on agent type
                        if hasattr(st.session_state.current_agent, 'chat'):
                            response = st.session_state.current_agent.chat(user_message)
                        elif hasattr(st.session_state.current_agent, 'search_and_respond'):
                            response = st.session_state.current_agent.search_and_respond(user_message)
                        elif hasattr(st.session_state.current_agent, 'process_request'):
                            response = st.session_state.current_agent.process_request(user_message)
                        elif hasattr(st.session_state.current_agent, 'chat_with_memory'):
                            response = st.session_state.current_agent.chat_with_memory(user_message)
                        else:
                            response = "Error: Unknown agent method"
                    
                    # Add agent response to chat
                    add_chat_message(st.session_state.current_agent.name, response, "agent")
                    
                    # Show conversation summary for stateful agents
                    if hasattr(st.session_state.current_agent, 'get_conversation_summary'):
                        summary = st.session_state.current_agent.get_conversation_summary()
                        st.info(f"💭 Conversation Summary: {summary}")
                    
                    st.rerun()
                
                except Exception as e:
                    create_notification(f"Error: {e}", "error")
                    add_chat_message("System", f"Error occurred: {e}", "agent")
            
            # Chat controls
            st.markdown("---")
            col_clear, col_export = st.columns(2)
            
            with col_clear:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    clear_chat_history()
                    if hasattr(st.session_state.current_agent, 'clear_history'):
                        st.session_state.current_agent.clear_history()
                    create_notification("Chat history cleared", "info")
                    st.rerun()
            
            with col_export:
                if st.button("📥 Export Chat", use_container_width=True):
                    if 'chat_history' in st.session_state and st.session_state.chat_history:
                        chat_data = "\n".join([
                            f"[{msg['timestamp']}] {msg['sender']}: {msg['content']}"
                            for msg in st.session_state.chat_history
                        ])
                        st.download_button(
                            "💾 Download Chat History",
                            chat_data,
                            f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            "text/plain",
                            use_container_width=True
                        )
            
            # Tips based on agent type
            agent_obj_type = type(st.session_state.current_agent).__name__
            tips = {
                "SimpleAgent": "💡 Try asking general questions, requesting explanations, or having casual conversations.",
                "SearchAgent": "💡 Ask about current events, recent news, or request factual information that requires web search.",
                "ToolAgent": "💡 Try asking for weather information ('weather in Paris') or math calculations ('calculate 15% of 250').",
                "StatefulAgent": "💡 Have a multi-turn conversation. The agent will remember context from previous messages."
            }
            
            if agent_obj_type in tips:
                st.info(tips[agent_obj_type])
        else:
            st.info("👆 Create an agent first to start chatting!")
            
            # Show sample conversations
            st.markdown("### 📖 Example Conversations")
            
            examples = {
                "Simple Agent": [
                    "What is the Agent Development Kit?",
                    "Explain the benefits of multi-agent systems",
                    "How do I get started with ADK?"
                ],
                "Search Agent": [
                    "What are the latest developments in AI agents?",
                    "Search for recent news about Google's AI research",
                    "Find information about agent frameworks"
                ],
                "Tool Agent": [
                    "What's the weather like in New York?",
                    "Calculate the tip for a $45.50 bill at 18%",
                    "What tools do you have available?"
                ],
                "Stateful Agent": [
                    "Hi, I'm working on a Python project",
                    "Can you help me understand decorators?",
                    "What were we just discussing?"
                ]
            }
            
            if agent_type in examples:
                st.markdown(f"**Try these with {agent_type}:**")
                for example in examples[agent_type]:
                    st.markdown(f"• {example}")
    
    # Agent comparison section
    st.markdown("---")
    st.markdown("### 📊 Agent Comparison Matrix")
    
    comparison_data = [
        {
            "Agent Type": "Simple Agent", 
            "Capabilities": "Basic conversation",
            "Use Cases": "General assistance, Q&A",
            "Complexity": "Low",
            "Memory": "None",
            "Tools": "None"
        },
        {
            "Agent Type": "Search Agent",
            "Capabilities": "Web search + conversation", 
            "Use Cases": "Research, fact-checking",
            "Complexity": "Medium",
            "Memory": "Session only",
            "Tools": "Web search"
        },
        {
            "Agent Type": "Tool Agent",
            "Capabilities": "Multiple tools + conversation",
            "Use Cases": "Task automation, calculations",
            "Complexity": "Medium", 
            "Memory": "Session only",
            "Tools": "Weather, Calculator, etc."
        },
        {
            "Agent Type": "Stateful Agent",
            "Capabilities": "Memory + conversation",
            "Use Cases": "Personalized assistance",
            "Complexity": "High",
            "Memory": "Persistent",
            "Tools": "Basic set"
        }
    ]
    
    create_data_table(comparison_data, "Agent Types Comparison")


def show_multi_agent_systems():
    """Show multi-agent system examples and orchestration."""
    create_header("🔗 Multi-Agent Systems", "Build sophisticated agent teams that coordinate to solve complex problems")
    
    # System type selection with modern cards
    st.markdown("### 🎯 Select System Architecture")
    
    system_configs = [
        {"type": "Coordinator System", "icon": "🎯", "desc": "Hierarchical system with specialized agents"},
        {"type": "Workflow Orchestrator", "icon": "⚙️", "desc": "Sequential workflow automation"},
        {"type": "Custom Hierarchy", "icon": "🏗️", "desc": "Build your own agent organization"}
    ]
    
    cols = st.columns(3)
    selected_system = None
    
    for i, (col, config) in enumerate(zip(cols, system_configs)):
        with col:
            if st.button(f"{config['icon']}\n{config['type']}", 
                        use_container_width=True, 
                        key=f"system_select_{i}",
                        help=config['desc']):
                selected_system = config['type']
                st.session_state.selected_system_type = config['type']
    
    # Use session state to remember selection
    if 'selected_system_type' not in st.session_state:
        st.session_state.selected_system_type = "Coordinator System"
    
    system_type = st.session_state.selected_system_type
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🔧 System Configuration")
        
        # System info card
        system_info = next(config for config in system_configs if config['type'] == system_type)
        st.markdown(f"""
        <div class="custom-card">
            <div style="text-align: center; margin-bottom: 1rem;">
                <div style="font-size: 3rem;">{system_info['icon']}</div>
                <h3 style="color: #2E86AB; margin: 0.5rem 0;">{system_info['type']}</h3>
                <p style="color: #6c757d; margin: 0;">{system_info['desc']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if system_type == "Coordinator System":
            st.markdown("""
            **🤖 Agent Team Composition:**
            - 🔍 **Research Specialist** - Gathers information and data
            - 📊 **Analysis Specialist** - Processes and analyzes findings  
            - ✍️ **Writing Specialist** - Creates final outputs
            - 🎯 **Main Coordinator** - Orchestrates the workflow
            """)
            
            project_description = st.text_area(
                "📝 Project Description:",
                placeholder="Describe the complex project you want the agent team to work on...",
                height=120,
                help="Be specific about your requirements and desired outcomes"
            )            
            if st.button("🚀 Execute Project", use_container_width=True, type="primary"):
                if project_description:
                    with st.spinner("🤖 Agent team is working on your project..."):
                        try:
                            coordinator = CoordinatorAgent("Project_Coordinator")
                            result = coordinator.execute_complex_project(project_description)
                            
                            st.session_state.last_project_result = result
                            st.session_state.last_coordinator = coordinator
                            st.session_state.last_project_description = project_description
                            
                            if result["success"]:
                                create_notification("Project completed successfully! 🎉", "success")
                            else:
                                create_notification("Project execution encountered issues", "warning")
                        
                        except Exception as e:
                            create_notification(f"Error executing project: {e}", "error")
                else:
                    create_notification("Please enter a project description", "warning")
              # Continue Workflow section
            if hasattr(st.session_state, 'last_project_result') and st.session_state.last_project_result:
                st.markdown("---")
                create_expandable_section(
                    "Continue Incomplete Workflow",
                    """
                    <p>If the initial execution didn't fully complete or you need additional analysis, 
                    you can continue the workflow with more specific instructions.</p>
                    """,
                    "🔄"
                )
                
                additional_context = st.text_area(
                    "💡 Additional Context/Instructions:",
                    placeholder="Provide specific guidance, additional requirements, or areas to focus on...",
                    height=100,
                    key="continue_context"
                )
                
                col_continue1, col_continue2 = st.columns(2)
                
                with col_continue1:
                    if st.button("🔄 Continue Workflow", use_container_width=True, type="secondary"):
                        if hasattr(st.session_state, 'last_coordinator'):
                            with st.spinner("🔄 Continuing workflow with enhanced context..."):
                                try:
                                    enhanced_context = (
                                        f"Previous result was incomplete. "
                                        f"Additional instructions: {additional_context if additional_context else 'Complete the full analysis'}"
                                    )
                                    
                                    result = st.session_state.last_coordinator.continue_workflow(
                                        st.session_state.last_project_description,
                                        enhanced_context
                                    )
                                    
                                    st.session_state.last_project_result = result
                                    
                                    if result["success"]:
                                        create_notification("Workflow continued successfully! ✨", "success")
                                    else:
                                        create_notification("Workflow continuation failed", "error")
                                
                                except Exception as e:
                                    create_notification(f"Error continuing workflow: {e}", "error")
                
                with col_continue2:
                    if st.button("🎯 Force Complete Research", use_container_width=True, type="secondary"):
                        if hasattr(st.session_state, 'last_coordinator'):
                            with st.spinner("🎯 Forcing comprehensive research completion..."):
                                try:
                                    research_result = st.session_state.last_coordinator.force_complete_research(
                                        st.session_state.last_project_description
                                    )
                                    
                                    if research_result.success:
                                        create_notification("Research completed successfully! 🔍", "success")
                                        st.session_state.force_research_result = research_result
                                    else:
                                        create_notification("Research completion failed", "error")
                                
                                except Exception as e:
                                    create_notification(f"Error forcing research: {e}", "error")
        elif system_type == "Workflow Orchestrator":
            st.markdown("**🔧 Available Workflows:**")
            
            workflow_name = st.selectbox(
                "Select workflow to execute:",
                list(EXAMPLE_WORKFLOWS.keys()),
                help="Choose from predefined automation workflows"
            )
            
            workflow = EXAMPLE_WORKFLOWS[workflow_name]
            
            # Workflow info card
            st.markdown(f"""
            <div class="custom-card">
                <h4 style="color: #2E86AB; margin-bottom: 0.5rem;">📋 {workflow_name}</h4>
                <p style="margin-bottom: 0.5rem;"><strong>Description:</strong> {workflow['description']}</p>
                <p style="margin-bottom: 0;"><strong>Required Inputs:</strong> {', '.join(workflow['required_inputs'])}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Input collection
            inputs = {}
            st.markdown("**📝 Workflow Inputs:**")
            for input_field in workflow['required_inputs']:
                inputs[input_field] = st.text_input(
                    f"{input_field.replace('_', ' ').title()}:",
                    help=f"Enter the {input_field.replace('_', ' ')} for the workflow"
                )
            
            if st.button("⚙️ Execute Workflow", use_container_width=True, type="primary"):
                if all(inputs.values()):
                    with st.spinner("🔄 Executing workflow..."):
                        try:
                            orchestrator = WorkflowOrchestrator()
                            for name, wf in EXAMPLE_WORKFLOWS.items():
                                orchestrator.register_workflow(name, wf)
                            
                            # Note: This would need to be run in an async context
                            create_notification("Workflow execution simulated (requires async runtime)", "info")
                            st.session_state.workflow_inputs = inputs
                            st.session_state.executed_workflow = workflow_name
                        
                        except Exception as e:
                            create_notification(f"Error executing workflow: {e}", "error")
                else:
                    create_notification("Please fill in all required inputs", "warning")
        
        elif system_type == "Custom Hierarchy":
            st.markdown("**🏗️ Build Custom Agent Hierarchy:**")
            
            st.info("🚧 Custom hierarchy builder coming soon! This will allow you to design your own agent team structure.")
            
            # Placeholder for custom hierarchy builder
            num_agents = st.slider("Number of agents:", 2, 8, 3)
            
            for i in range(num_agents):
                with st.expander(f"🤖 Agent {i+1} Configuration"):
                    agent_name = st.text_input(f"Agent {i+1} Name:", value=f"Agent_{i+1}")
                    agent_role = st.selectbox(f"Role:", ["Coordinator", "Specialist", "Worker"], key=f"role_{i}")
                    agent_capabilities = st.multiselect(
                        f"Capabilities:", 
                        ["Research", "Analysis", "Writing", "Calculation", "Web Search"],
                        key=f"caps_{i}"
                    )
    with col2:
        st.markdown("### 📊 Execution Results")
        
        # Show results if available
        if 'last_project_result' in st.session_state:
            result = st.session_state.last_project_result
            
            # Success/failure indicator with modern styling
            if result["success"]:
                create_status_card("✅ Project Completed", "All phases executed successfully", "success", "🎉")
            else:
                create_status_card("❌ Project Failed", "Some phases encountered errors", "error", "⚠️")
            
            # Task breakdown with enhanced display
            st.markdown("**📋 Task Execution Timeline:**")
            for i, task in enumerate(result["tasks"], 1):
                status_icon = "✅" if task.success else "❌"
                status_color = "#28a745" if task.success else "#dc3545"
                
                st.markdown(f"""
                <div style="border-left: 4px solid {status_color}; padding: 0.5rem 1rem; margin: 0.5rem 0; background: #f8f9fa; border-radius: 4px;">
                    <strong>{status_icon} Phase {i}:</strong> {task.task}
                    {f'<br><span style="font-size: 0.9em;">Error: {task.result}</span>' if not task.success else ''}
                </div>
                """, unsafe_allow_html=True)
            
            # Final result with better formatting
            if result["success"]:
                st.markdown("**📋 Final Result:**")
                display_agent_response(result["final_result"], "Project Output")
            
            # Show forced research result if available
            if 'force_research_result' in st.session_state:
                st.markdown("---")
                st.markdown("**🎯 Enhanced Research Result:**")
                research_result = st.session_state.force_research_result
                if research_result.success:
                    create_status_card("🔍 Research Completed", "Enhanced research phase successful", "success", "✨")
                    display_agent_response(research_result.result, "Research Output")
                else:
                    create_status_card("❌ Research Failed", f"Research error: {research_result.result}", "error", "⚠️")
        
        elif 'executed_workflow' in st.session_state:
            workflow_name = st.session_state.executed_workflow
            create_status_card(f"⚙️ Workflow: {workflow_name}", "Execution simulated successfully", "info", "🔄")
            
            st.markdown("**📝 Workflow Inputs:**")
            if 'workflow_inputs' in st.session_state:
                for key, value in st.session_state.workflow_inputs.items():
                    st.markdown(f"• **{key.replace('_', ' ').title()}:** {value}")
        
        else:
            st.info("🚀 Execute a project or workflow to see detailed results here")
            
            # Show example outputs
            with st.expander("💡 Example Output Preview"):
                st.markdown("""
                **Sample Multi-Agent Output:**
                
                🔍 **Research Phase:** Gathered 15 sources on market trends  
                📊 **Analysis Phase:** Identified 3 key opportunities  
                ✍️ **Writing Phase:** Generated comprehensive 2,500-word report  
                🎯 **Coordination:** Successfully delivered complete analysis
                """)


def show_a2a_protocol():
    """Show A2A protocol examples and testing."""
    create_header("🌐 A2A Protocol", "Enable distributed agent communication across networks for scalable architectures")
    
    st.markdown("""
    **Agent-to-Agent (A2A) Protocol** enables seamless communication between agents 
    across different processes, servers, or geographic locations, creating truly distributed AI systems.
    """)
    
    # A2A Network Configuration Section - Full Width
    st.markdown("## 🛠️ A2A Network Configuration")
    
    # Agent configuration with enhanced UI
    num_agents = st.slider("Number of A2A Agents:", 2, 5, 3, help="Configure the size of your agent network")
    
    agent_configs = []
    cols = st.columns(min(num_agents, 3))  # Create columns for agent configs
    
    for i in range(num_agents):
        with cols[i % 3]:  # Distribute agents across columns
            with st.container():
                st.markdown(f"**🤖 Agent {i+1}:**")
                name = st.text_input(f"Name:", value=f"A2A_Agent_{i+1}", key=f"agent_name_{i}")
                port = st.number_input(f"Port:", value=8500+i, min_value=8500, max_value=9000, key=f"agent_port_{i}")
                agent_type = st.selectbox(f"Type:", 
                                        ["Research Agent", "Analysis Agent", "Communication Hub", "Task Executor"],
                                        key=f"agent_type_{i}")
                agent_configs.append({"name": name, "port": port, "type": agent_type})
    
    if st.button("🚀 Start A2A Network", use_container_width=True, type="primary"):
        st.session_state.a2a_agents = agent_configs
        create_notification(f"✅ Configured {num_agents} A2A agents successfully!", "success")
        
        # Show network status
        st.markdown("**🌐 Network Status:**")
        status_cols = st.columns(min(len(agent_configs), 4))
        for i, agent in enumerate(agent_configs):
            with status_cols[i % 4]:
                create_status_card(
                    f"🤖 {agent['name']}", 
                    f"Type: {agent['type']}<br>Port: {agent['port']}<br>Status: Online",
                    "success",
                    "🟢"
                )    
    # Network Communication Section - Full Width
    st.markdown("---")
    st.markdown("## 💬 Network Communication")
    
    if 'a2a_agents' in st.session_state:
        # Agent selection with improved UI
        agent_names = [agent["name"] for agent in st.session_state.a2a_agents]
        
        col_sender, col_receiver = st.columns(2)
        
        with col_sender:
            sender = st.selectbox("📤 Sender Agent:", agent_names, help="Choose the agent that will send the message")
        
        with col_receiver:
            receiver = st.selectbox("📥 Receiver Agent:", 
                                  [name for name in agent_names if name != sender],
                                  help="Choose the target agent for the message")
        
        # Message configuration with enhanced options
        st.markdown("**📋 Message Configuration:**")
        
        col_action, col_priority = st.columns([2, 1])
        
        with col_action:
            action = st.selectbox("Action Type:", 
                                ["chat", "analyze", "collaborate", "ping", "data_transfer", "task_request"],
                                help="Select the type of action to perform")
        
        with col_priority:
            priority = st.selectbox("Priority:", ["Low", "Medium", "High", "Critical"])
        
        message_data = st.text_area("💌 Message Content:", 
                                  placeholder="Enter the message content or data to send...",
                                  height=100)
          # Advanced options
        with st.expander("⚙️ Advanced Options"):
            correlation_id = st.text_input("Correlation ID:", placeholder="Optional - for request tracking")
            timeout = st.number_input("Timeout (seconds):", value=30, min_value=5, max_value=300)
            retry_count = st.number_input("Retry Attempts:", value=3, min_value=0, max_value=10)

        if st.button("📨 Send A2A Message", use_container_width=True, type="primary"):
            if message_data:
                with st.spinner(f"📡 Sending {action} from {sender} to {receiver}..."):
                    try:
                        # Create actual A2A communication
                        from agents.a2a import SmartA2AAgent
                        
                        # Initialize agents if not already done
                        if 'a2a_active_agents' not in st.session_state:
                            st.session_state.a2a_active_agents = {}
                        
                        # Create sender agent if not exists
                        if sender not in st.session_state.a2a_active_agents:
                            st.session_state.a2a_active_agents[sender] = SmartA2AAgent(
                                agent_id=f"agent_{sender.lower().replace(' ', '_')}",
                                name=sender
                            )                        
                        # Create receiver agent if not exists  
                        if receiver not in st.session_state.a2a_active_agents:
                            st.session_state.a2a_active_agents[receiver] = SmartA2AAgent(
                                agent_id=f"agent_{receiver.lower().replace(' ', '_')}",
                                name=receiver
                            )
                        
                        sender_agent = st.session_state.a2a_active_agents[sender]
                        receiver_agent = st.session_state.a2a_active_agents[receiver]                        
                        # Create the message
                        message_obj = {
                            "id": f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
                            "type": "request",
                            "action": action,
                            "content": message_data,
                            "sender": sender,
                            "receiver": receiver,
                            "priority": priority,
                            "timestamp": datetime.now().isoformat(),
                            "correlation_id": correlation_id if correlation_id else f"corr_{uuid.uuid4().hex[:8]}",
                            "timeout": timeout,
                            "retry_count": retry_count
                        }
                        
                        # Process the message and get response
                        response_obj = receiver_agent.process_a2a_message(message_obj)
                        
                        # Create response message
                        response_message = {
                            "id": f"resp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
                            "type": "response", 
                            "original_message_id": message_obj["id"],
                            "sender": receiver,
                            "receiver": sender,
                            "correlation_id": message_obj["correlation_id"],
                            "timestamp": datetime.now().isoformat(),
                            "response_data": response_obj,
                            "processing_time": "~1.2s",
                            "success": True
                        }
                        
                        create_notification(f"✅ Message exchange completed between {sender} and {receiver}!", "success")
                        
                        # Display both request and response
                        col_req, col_resp = st.columns(2)
                        
                        with col_req:
                            st.markdown("**📤 Sent Message:**")
                            display_agent_response(json.dumps(message_obj, indent=2), f"Request from {sender}")
                        
                        with col_resp:
                            st.markdown("**📥 Received Response:**")
                            display_agent_response(json.dumps(response_message, indent=2), f"Response from {receiver}")
                        
                        # Show actual agent responses
                        st.markdown("---")
                        st.markdown("**🤖 Agent Processing Results:**")
                        
                        col_sender_resp, col_receiver_resp = st.columns(2)
                        
                        with col_sender_resp:
                            st.markdown(f"**📤 {sender} (Sender):**")
                            sender_status = f"✅ Successfully sent '{action}' message to {receiver}"
                            display_agent_response(sender_status, "Sender Status")
                        
                        with col_receiver_resp:
                            st.markdown(f"**📥 {receiver} (Receiver):**")
                            display_agent_response(response_obj.get("response", "No response content"), "Agent Response")
                        
                        # Store conversation in session for history
                        if 'a2a_conversations' not in st.session_state:
                            st.session_state.a2a_conversations = []
                        
                        conversation = {
                            "conversation_id": message_obj["correlation_id"],
                            "timestamp": datetime.now().isoformat(),
                            "participants": [sender, receiver],
                            "request": message_obj,
                            "response": response_message,
                            "agent_responses": {
                                "sender_status": sender_status,
                                "receiver_response": response_obj
                            }
                        }
                        st.session_state.a2a_conversations.append(conversation)
                        
                    except Exception as e:
                        create_notification(f"❌ A2A Communication Error: {str(e)}", "error")
                        st.error(f"Failed to establish A2A communication: {e}")
            else:
                create_notification("Please enter message content", "warning")        
        # Conversation history
        if 'a2a_conversations' in st.session_state and st.session_state.a2a_conversations:
            st.markdown("---")
            st.markdown("**💬 A2A Conversation History:**")
            
            # Show recent conversations (last 3)
            recent_conversations = list(reversed(st.session_state.a2a_conversations[-3:]))
            
            for i, conv in enumerate(recent_conversations):
                participants = " ↔ ".join(conv['participants'])
                timestamp = datetime.fromisoformat(conv['timestamp']).strftime("%H:%M:%S")
                
                with st.expander(f"🗣️ {participants} | {timestamp}", expanded=(i == 0)):
                    col_req, col_resp = st.columns(2)
                    
                    with col_req:
                        st.markdown("**📤 Request:**")
                        st.code(f"""
                                Action: {conv['request']['action']}
                                From: {conv['request']['sender']}
                                To: {conv['request']['receiver']}
                                Content: {conv['request']['content'][:100]}...
                                """)
                    
                    with col_resp:
                        st.markdown("**📥 Response:**")
                        resp_content = conv['agent_responses']['receiver_response'].get('response', 'No content')
                        st.code(f"""
                                Status: {conv['response']['success']}
                                Processing: {conv['response']['processing_time']}
                                Response: {resp_content[:100]}...
                                """)
                    
                    # Show full conversation details
                    if st.button(f"🔍 View Full Details", key=f"details_{conv['conversation_id']}"):
                        st.markdown("**📋 Complete Conversation Log:**")
                        
                        st.markdown("**🔹 Request Message:**")
                        st.json(conv['request'])                        
                        st.markdown("**🔹 Response Message:**")
                        st.json(conv['response'])
                        
                        st.markdown("**🔹 Agent Processing:**")
                        st.json(conv['agent_responses'])
            
            # Clear conversation history button
            if st.button("🗑️ Clear Conversation History", type="secondary"):
                st.session_state.a2a_conversations = []
                create_notification("Conversation history cleared", "info")
                st.rerun()
        
        # Show active agents status
        if 'a2a_active_agents' in st.session_state and st.session_state.a2a_active_agents:
            st.markdown("---")
            st.markdown("**🤖 Active A2A Agents Status:**")
            
            agent_status_cols = st.columns(min(len(st.session_state.a2a_active_agents), 3))
            
            for i, (agent_name, agent_obj) in enumerate(st.session_state.a2a_active_agents.items()):
                with agent_status_cols[i % 3]:
                    create_status_card(
                        f"🤖 {agent_name}",
                        f"Type: SmartA2AAgent<br>Status: Active<br>ID: {agent_obj.agent_id}",
                        "success",
                        "🟢"
                    )
            
            # Quick test buttons
            st.markdown("**⚡ Quick Agent Tests:**")
            test_cols = st.columns(3)
            
            with test_cols[0]:
                if st.button("🏓 Ping All Agents", use_container_width=True):
                    ping_results = []
                    for agent_name, agent_obj in st.session_state.a2a_active_agents.items():
                        try:
                            result = agent_obj.process_a2a_message({
                                "action": "ping",
                                "content": "Health check",
                                "sender": "System"
                            })
                            ping_results.append(f"✅ {agent_name}: {result['response']}")
                        except Exception as e:
                            ping_results.append(f"❌ {agent_name}: Error - {e}")
                    
                    for result in ping_results:
                        st.write(result)            
            with test_cols[1]:
                if st.button("🧠 Test AI Capabilities", use_container_width=True):
                    test_question = "What is your role in this A2A network?"
                    for agent_name, agent_obj in st.session_state.a2a_active_agents.items():
                        try:
                            result = agent_obj.process_a2a_message({
                                "action": "chat",
                                "content": test_question,
                                "sender": "Tester"
                            })
                            st.write(f"🤖 **{agent_name}:** {result['response'][:100]}...")
                        except Exception as e:
                            st.write(f"❌ {agent_name}: Error - {e}")
            
            with test_cols[2]:
                if st.button("🔄 Agent Round-Robin", use_container_width=True):
                    agents = list(st.session_state.a2a_active_agents.keys())
                    if len(agents) >= 2:
                        # Send message from first agent to second, then second to third, etc.
                        message = "Hello colleague! How are you doing?"
                        chain_results = []
                        
                        for i in range(len(agents)):
                            sender = agents[i]
                            receiver = agents[(i + 1) % len(agents)]
                            
                            try:
                                receiver_agent = st.session_state.a2a_active_agents[receiver]
                                result = receiver_agent.process_a2a_message({
                                    "action": "chat",
                                    "content": message,
                                    "sender": sender
                                })
                                chain_results.append(f"{sender} → {receiver}: {result['response'][:50]}...")
                                message = result['response']  # Use response as next message
                            except Exception as e:
                                chain_results.append(f"{sender} → {receiver}: Error - {e}")
                        
                        for result in chain_results:
                            st.write(result)
    else:
        st.info("🔧 Configure and start A2A network first to enable communication")
        
        # Show network architecture diagram
        st.markdown("**🏗️ Expected Network Architecture:**")
        create_feature_card(
            "Distributed Agent Network",
            "Agents communicate across different servers and locations",
            "🌐"
        )
    
    # Protocol details with enhanced tabs
    st.markdown("---")
    create_header("📚 A2A Protocol Specifications", "Technical details and implementation examples")
    
    protocol_tabs = create_tabs_with_icons([
        {"name": "Message Format", "icon": "📋"},
        {"name": "Communication Flow", "icon": "🔄"},
        {"name": "Security & Auth", "icon": "🔒"},
        {"name": "Implementation", "icon": "💻"},
        {"name": "Testing Tools", "icon": "🧪"}
    ])
    
    with protocol_tabs[0]:  # Message Format
        st.markdown("**📋 A2A Message Structure:**")
        
        example_message = {
            "id": "msg_20250612_103000_abc123",
            "version": "1.0",
            "type": "request",
            "sender": {
                "id": "agent_research_001",
                "name": "Research Specialist",
                "address": "http://192.168.1.100:8501"
            },
            "receiver": {
                "id": "agent_analysis_002", 
                "name": "Analysis Specialist",
                "address": "http://192.168.1.101:8502"
            },
            "payload": {
                "action": "analyze_data",
                "parameters": {
                    "data_source": "market_research.json",
                    "analysis_type": "trend_analysis",
                    "priority": "high"
                },
                "metadata": {
                    "expected_duration": "5-10 minutes",
                    "result_format": "json"
                }
            },
            "timestamp": "2025-06-12T10:30:00Z",
            "correlation_id": "proj_market_analysis_001",
            "timeout": 600,
            "retry_policy": {
                "max_attempts": 3,
                "backoff_strategy": "exponential"
            }
        }
        
        create_code_block(json.dumps(example_message, indent=2), "json", "Complete A2A Message Example")
        
        # Message field descriptions
        field_descriptions = [
            {"Field": "id", "Type": "string", "Description": "Unique message identifier"},
            {"Field": "version", "Type": "string", "Description": "A2A protocol version"},
            {"Field": "type", "Type": "enum", "Description": "request | response | notification | error"},
            {"Field": "sender", "Type": "object", "Description": "Source agent information"},
            {"Field": "receiver", "Type": "object", "Description": "Target agent information"},
            {"Field": "payload", "Type": "object", "Description": "Message content and parameters"},
            {"Field": "timestamp", "Type": "ISO 8601", "Description": "Message creation time"},
            {"Field": "correlation_id", "Type": "string", "Description": "Links related messages"},
            {"Field": "timeout", "Type": "integer", "Description": "Request timeout in seconds"},
            {"Field": "retry_policy", "Type": "object", "Description": "Retry configuration"}
        ]
        
        create_data_table(field_descriptions, "Message Field Reference")
    
    with protocol_tabs[1]:  # Communication Flow
        st.markdown("**🔄 A2A Communication Flow:**")
        
        flow_steps = [
            {"Step": "1", "Phase": "Connection", "Description": "Agent discovers and connects to target agent"},
            {"Step": "2", "Phase": "Authentication", "Description": "Mutual authentication and authorization"},
            {"Step": "3", "Phase": "Request", "Description": "Sender constructs and sends A2A message"},
            {"Step": "4", "Phase": "Processing", "Description": "Receiver processes request and executes action"},
            {"Step": "5", "Phase": "Response", "Description": "Receiver sends back result or acknowledgment"},
            {"Step": "6", "Phase": "Correlation", "Description": "Messages linked via correlation ID"},
            {"Step": "7", "Phase": "Error Handling", "Description": "Timeout, retry, and error management"}        ]
        
        create_data_table(flow_steps, "Communication Flow Steps")
        
        st.markdown("**📊 Flow Diagram:**")
        create_progress_indicator(5, 7, ["Discovery", "Auth", "Request", "Process", "Response", "Correlation", "Error Handling"])
    
    with protocol_tabs[2]:  # Security & Auth
        st.markdown("**🔒 Security & Authentication:**")
        
        security_features = [
            {"Feature": "🔐 Mutual TLS", "Status": "Implemented", "Description": "Certificate-based authentication"},
            {"Feature": "🎫 JWT Tokens", "Status": "Supported", "Description": "Token-based authorization"},
            {"Feature": "🛡️ Message Encryption", "Status": "Required", "Description": "End-to-end encryption"},
            {"Feature": "📝 Message Signing", "Status": "Optional", "Description": "Digital signatures for integrity"},
            {"Feature": "🚪 Access Control", "Status": "Configurable", "Description": "Role-based permissions"},
            {"Feature": "📊 Audit Logging", "Status": "Enabled", "Description": "Complete communication logs"}
        ]
        
        create_data_table(security_features, "Security Features")
        
        create_code_block("""
                                # Security configuration example
                                a2a_config = {
                                    "security": {
                                        "tls_enabled": True,
                                        "certificate_path": "/path/to/cert.pem",
                                        "private_key_path": "/path/to/key.pem",
                                        "jwt_secret": "your-jwt-secret",
                                        "encryption_algorithm": "AES-256-GCM",
                                        "require_message_signing": True
                                    },
                                    "access_control": {
                                        "allow_list": ["agent_1", "agent_2"],
                                        "rate_limiting": {
                                            "requests_per_minute": 100,
                                            "burst_capacity": 10
                                        }
                                    }
                                }
                                        """, "python", "Security Configuration")
    
    with protocol_tabs[3]:  # Implementation
        st.markdown("**💻 Implementation Example:**")
        
        create_code_block("""
                            from agents.a2a import SmartA2AAgent
                            import asyncio

                            async def setup_a2a_network():    # Create A2A enabled agents
                                research_agent = SmartA2AAgent(
                                    agent_id="research_001",
                                    name="Research Specialist", 
                                    port=8501
                                )
                                
                                analysis_agent = SmartA2AAgent(
                                    agent_id="analysis_002", 
                                    name="Analysis Specialist",
                                    port=8502
                                )
                                
                                # Start network servers
                                await research_agent.start_server()
                                await analysis_agent.start_server()
                                
                                # Register handlers
                                research_agent.register_handler("search", handle_search_request)
                                analysis_agent.register_handler("analyze", handle_analysis_request)
                                
                                # Send cross-network request
                                response = await research_agent.send_request(
                                    target_url="http://localhost:8502",
                                    action="analyze",
                                    data={
                                        "dataset": "market_trends.csv",
                                        "analysis_type": "correlation",
                                        "parameters": {"confidence_level": 0.95}
                                    },
                                    timeout=120
                                )
                                
                                print(f"Analysis result: {response.result}")
                                return response

                            # Run the network
                            if __name__ == "__main__":
                                result = asyncio.run(setup_a2a_network())
                                    """, "python", "Complete A2A Implementation")
    
    with protocol_tabs[4]:  # Testing Tools
        st.markdown("**🧪 A2A Testing & Debugging Tools:**")
        
        # Message validator
        st.markdown("**📋 Message Validator:**")
        
        test_message = st.text_area(
            "Enter A2A message JSON to validate:",
            value=json.dumps(example_message, indent=2),
            height=200
        )
        
        col_validate, col_format = st.columns(2)
        
        with col_validate:
            if st.button("✅ Validate Message", use_container_width=True):
                try:
                    parsed = json.loads(test_message)
                    required_fields = ["id", "type", "sender", "receiver", "payload", "timestamp"]
                    
                    missing_fields = [field for field in required_fields if field not in parsed]
                    
                    if missing_fields:
                        create_notification(f"❌ Missing required fields: {', '.join(missing_fields)}", "error")
                    else:
                        create_notification("✅ Message format is valid!", "success")
                        
                except json.JSONDecodeError as e:
                    create_notification(f"❌ Invalid JSON: {e}", "error")
        
        with col_format:
            if st.button("🎨 Format Message", use_container_width=True):
                try:
                    parsed = json.loads(test_message)
                    formatted = json.dumps(parsed, indent=2, sort_keys=True)
                    st.session_state.formatted_message = formatted
                    create_notification("✅ Message formatted successfully!", "success")
                except json.JSONDecodeError as e:
                    create_notification(f"❌ Cannot format invalid JSON: {e}", "error")
        
        if 'formatted_message' in st.session_state:
            st.text_area("Formatted Message:", value=st.session_state.formatted_message, height=200)
        
        # Network diagnostics
        st.markdown("---")
        st.markdown("**🔍 Network Diagnostics:**")
        
        diagnostic_cols = st.columns(3)
        
        with diagnostic_cols[0]:
            if st.button("🏥 Health Check", use_container_width=True):
                create_notification("All agents responding normally", "success")
        
        with diagnostic_cols[1]:
            if st.button("📊 Performance Test", use_container_width=True):
                create_notification("Average latency: 45ms", "info")
        
        with diagnostic_cols[2]:
            if st.button("🛡️ Security Scan", use_container_width=True):
                create_notification("No security issues detected", "success")


def show_performance_analytics():
    """Show performance analytics and monitoring."""
    create_header("📊 Performance Analytics", "Monitor, analyze, and optimize your agent systems")
    
    # Generate enhanced mock performance data
    dates = pd.date_range(start='2025-06-01', end='2025-06-12', freq='D')
    
    performance_data = {
        'Date': dates,
        'Requests': [50 + i*5 + (i%3)*10 + random.randint(-5, 15) for i in range(len(dates))],
        'Response_Time': [1.2 + (i%4)*0.3 + random.uniform(-0.2, 0.5) for i in range(len(dates))],
        'Success_Rate': [95 + (i%2)*3 + random.uniform(-2, 3) for i in range(len(dates))],
        'Agent_Count': [3 + (i//3) for i in range(len(dates))],
        'CPU_Usage': [15 + (i%5)*5 + random.randint(-3, 8) for i in range(len(dates))],
        'Memory_Usage': [250 + i*10 + random.randint(-20, 30) for i in range(len(dates))]
    }
    
    df = pd.DataFrame(performance_data)
    
    # Real-time metrics overview
    st.markdown("### ⚡ Real-Time System Metrics")
    
    # Current metrics with enhanced cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_requests = df['Requests'].iloc[-1]
        prev_requests = df['Requests'].iloc[-2]
        change_requests = ((current_requests - prev_requests) / prev_requests) * 100
        create_metric_card(
            f"{current_requests:,}", 
            "Daily Requests", 
            f"{change_requests:+.1f}%", 
            "positive" if change_requests > 0 else "negative"
        )
    
    with col2:
        current_response = df['Response_Time'].iloc[-1]
        prev_response = df['Response_Time'].iloc[-2]
        change_response = ((current_response - prev_response) / prev_response) * 100
        create_metric_card(
            f"{current_response:.2f}s", 
            "Avg Response Time", 
            f"{change_response:+.1f}%", 
            "negative" if change_response > 0 else "positive"
        )
    
    with col3:
        current_success = df['Success_Rate'].iloc[-1]
        prev_success = df['Success_Rate'].iloc[-2]
        change_success = current_success - prev_success
        create_metric_card(
            f"{current_success:.1f}%", 
            "Success Rate", 
            f"{change_success:+.1f}%", 
            "positive" if change_success > 0 else "negative"
        )
    
    with col4:
        current_agents = df['Agent_Count'].iloc[-1]
        prev_agents = df['Agent_Count'].iloc[-2]
        change_agents = current_agents - prev_agents
        create_metric_card(
            f"{current_agents}", 
            "Active Agents", 
            f"{change_agents:+.0f}" if change_agents != 0 else "→", 
            "positive" if change_agents > 0 else "neutral"
        )
    
    # Performance charts with enhanced layout
    st.markdown("---")
    st.markdown("### 📈 Performance Trends")
    
    # Chart selection tabs
    chart_tabs = create_tabs_with_icons([
        {"name": "Request Volume", "icon": "📊"},
        {"name": "Response Times", "icon": "⚡"},
        {"name": "Success Rates", "icon": "✅"},
        {"name": "Resource Usage", "icon": "💻"},
        {"name": "Comparative", "icon": "📈"}
    ])
    
    with chart_tabs[0]:  # Request Volume
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.line(df, x='Date', y='Requests', title='📊 Daily Request Volume',
                         markers=True, line_shape="spline")
            fig.update_layout(height=400)
            fig.add_hline(y=df['Requests'].mean(), line_dash="dash", 
                         annotation_text=f"Average: {df['Requests'].mean():.0f}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Request distribution
            hourly_requests = [random.randint(2, 8) for _ in range(24)]
            hours = list(range(24))
            
            fig_hourly = px.bar(x=hours, y=hourly_requests, title="⏰ Hourly Distribution")
            fig_hourly.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        # Request analytics
        st.markdown("**📋 Request Analytics:**")
        
        analytics_data = [
            {"Metric": "Total Requests", "Value": f"{df['Requests'].sum():,}", "Period": "12 days"},
            {"Metric": "Peak Day", "Value": f"{df['Requests'].max()}", "Period": df[df['Requests'] == df['Requests'].max()]['Date'].dt.strftime('%Y-%m-%d').iloc[0]},
            {"Metric": "Average Daily", "Value": f"{df['Requests'].mean():.0f}", "Period": "requests/day"},
            {"Metric": "Growth Rate", "Value": f"{((df['Requests'].iloc[-1] - df['Requests'].iloc[0]) / df['Requests'].iloc[0] * 100):+.1f}%", "Period": "vs start"}
        ]
        
        create_data_table(analytics_data, "Request Statistics")
    
    with chart_tabs[1]:  # Response Times
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.line(df, x='Date', y='Response_Time', title='⚡ Response Time Trends',
                         markers=True, line_shape="spline")
            fig.update_layout(height=400)
            fig.add_hline(y=2.0, line_dash="dash", line_color="red",
                         annotation_text="SLA Threshold: 2.0s")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Response time distribution
            response_percentiles = {
                "P50": df['Response_Time'].quantile(0.5),
                "P90": df['Response_Time'].quantile(0.9),
                "P95": df['Response_Time'].quantile(0.95),
                "P99": df['Response_Time'].quantile(0.99)
            }
            
            fig_perc = px.bar(x=list(response_percentiles.keys()), 
                             y=list(response_percentiles.values()),
                             title="📊 Response Time Percentiles")
            fig_perc.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_perc, use_container_width=True)
        
        # Performance status
        avg_response = df['Response_Time'].mean()
        if avg_response < 1.5:
            status = "Excellent"
            color = "success"
        elif avg_response < 2.0:
            status = "Good" 
            color = "info"
        else:
            status = "Needs Attention"
            color = "warning"
        
        create_status_card(
            f"⚡ Performance: {status}",
            f"Average response time: {avg_response:.2f}s<br>SLA compliance: {(df['Response_Time'] < 2.0).mean()*100:.1f}%",
            color,
            "🎯"
        )
    
    with chart_tabs[2]:  # Success Rates
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.line(df, x='Date', y='Success_Rate', title='✅ Success Rate Trends',
                         markers=True, line_shape="spline")
            fig.update_layout(height=400, yaxis_range=[90, 100])
            fig.add_hline(y=95.0, line_dash="dash", line_color="orange",
                         annotation_text="Target: 95%")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Error breakdown (simulated)
            error_types = ["Timeout", "API Error", "Network", "Invalid Input", "Unknown"]
            error_counts = [5, 3, 2, 8, 1]
            
            fig_errors = px.pie(values=error_counts, names=error_types, 
                               title="🔍 Error Type Distribution")
            fig_errors.update_layout(height=400)
            st.plotly_chart(fig_errors, use_container_width=True)
        
        # Success rate insights
        success_avg = df['Success_Rate'].mean()
        if success_avg >= 98:
            insight = "System performing excellently with minimal errors"
            color = "success"
        elif success_avg >= 95:
            insight = "Good performance, monitor for improvements"
            color = "info"
        else:
            insight = "Success rate below target, investigate issues"
            color = "warning"
        
        st.info(f"💡 **Insight:** {insight}")
    
    with chart_tabs[3]:  # Resource Usage
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(df, x='Date', y='CPU_Usage', title='💻 CPU Usage Trends',
                         markers=True, line_shape="spline")
            fig.update_layout(height=350)
            fig.add_hline(y=80, line_dash="dash", line_color="red",
                         annotation_text="High Usage: 80%")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(df, x='Date', y='Memory_Usage', title='🧠 Memory Usage Trends',
                         markers=True, line_shape="spline")
            fig.update_layout(height=350)
            fig.add_hline(y=500, line_dash="dash", line_color="red",
                         annotation_text="Limit: 500MB")
            st.plotly_chart(fig, use_container_width=True)
        
        # Resource utilization cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cpu_avg = df['CPU_Usage'].mean()
            cpu_status = "Low" if cpu_avg < 50 else "Medium" if cpu_avg < 75 else "High"
            create_status_card(
                f"💻 CPU: {cpu_avg:.1f}%",
                f"Status: {cpu_status}<br>Peak: {df['CPU_Usage'].max():.1f}%",
                "success" if cpu_avg < 50 else "warning" if cpu_avg < 75 else "error",
                "⚡"
            )
        
        with col2:
            mem_avg = df['Memory_Usage'].mean()
            mem_status = "Low" if mem_avg < 300 else "Medium" if mem_avg < 400 else "High"
            create_status_card(
                f"🧠 Memory: {mem_avg:.0f}MB",
                f"Status: {mem_status}<br>Peak: {df['Memory_Usage'].max():.0f}MB",
                "success" if mem_avg < 300 else "warning" if mem_avg < 400 else "error",
                "📊"
            )
        
        with col3:
            # Calculate efficiency score
            efficiency = 100 - (cpu_avg * 0.6 + (mem_avg/500) * 100 * 0.4)
            create_status_card(
                f"⚡ Efficiency: {efficiency:.0f}%",
                f"Overall system efficiency<br>Based on resource usage",
                "success" if efficiency > 70 else "warning" if efficiency > 50 else "error",
                "🎯"
            )
    
    with chart_tabs[4]:  # Comparative Analysis
        st.markdown("**📊 Multi-Metric Comparison:**")
        
        # Normalize data for comparison
        df_normalized = df.copy()
        df_normalized['Requests_Norm'] = (df['Requests'] - df['Requests'].min()) / (df['Requests'].max() - df['Requests'].min()) * 100
        df_normalized['Response_Time_Norm'] = (1 - (df['Response_Time'] - df['Response_Time'].min()) / (df['Response_Time'].max() - df['Response_Time'].min())) * 100
        df_normalized['Success_Rate_Norm'] = df['Success_Rate']
        
        fig = px.line(df_normalized, x='Date', 
                     y=['Requests_Norm', 'Response_Time_Norm', 'Success_Rate_Norm'],
                     title='📈 Normalized Performance Comparison')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        correlation_data = df[['Requests', 'Response_Time', 'Success_Rate', 'CPU_Usage', 'Memory_Usage']].corr()
        
        fig_corr = px.imshow(correlation_data, 
                            title="🔗 Metric Correlation Matrix",
                            color_continuous_scale="RdBu",
                            aspect="auto")
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Agent-specific performance
    st.markdown("---")
    st.markdown("### 🤖 Agent-Specific Performance")
    
    # Mock agent performance data
    agent_performance = {
        "Agent": ["Research Agent", "Analysis Agent", "Writing Agent", "Coordinator", "Tool Agent"],
        "Requests": [245, 189, 167, 98, 134],
        "Avg Response (s)": [2.1, 1.8, 3.2, 0.8, 1.5],
        "Success Rate (%)": [97.2, 98.8, 94.1, 99.5, 96.7],
        "CPU Usage (%)": [22, 18, 35, 8, 15],
        "Errors": [7, 2, 10, 1, 4]
    }
    
    df_agents = pd.DataFrame(agent_performance)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Agent performance comparison
        fig = px.scatter(df_agents, x="Avg Response (s)", y="Success Rate (%)", 
                        size="Requests", hover_name="Agent", color="Agent",
                        title="🎯 Agent Performance Scatter Plot")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top performing agents
        df_agents['Performance Score'] = (df_agents['Success Rate (%)'] * 0.4 + 
                                        (100 - df_agents['Avg Response (s)'] * 20) * 0.3 +
                                        (100 - df_agents['CPU Usage (%)']) * 0.3)
        
        df_top = df_agents.nlargest(3, 'Performance Score')[['Agent', 'Performance Score']]
        
        st.markdown("**🏆 Top Performers:**")
        for i, (_, row) in enumerate(df_top.iterrows(), 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            st.markdown(f"{medal} **{row['Agent']}** - Score: {row['Performance Score']:.1f}")
    
    # Performance alerts and recommendations
    st.markdown("---")
    st.markdown("### 🚨 Alerts & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**⚠️ Current Alerts:**")
        
        alerts = []
        
        # Check for performance issues
        if df['Response_Time'].iloc[-1] > 2.0:
            alerts.append({"type": "warning", "message": "Response time above SLA threshold"})
        
        if df['Success_Rate'].iloc[-1] < 95:
            alerts.append({"type": "error", "message": "Success rate below target"})
        
        if df['CPU_Usage'].iloc[-1] > 75:
            alerts.append({"type": "warning", "message": "High CPU usage detected"})
        
        if not alerts:
            alerts.append({"type": "success", "message": "All systems operating normally"})
        
        for alert in alerts:
            if alert["type"] == "success":
                st.success(f"✅ {alert['message']}")
            elif alert["type"] == "warning":
                st.warning(f"⚠️ {alert['message']}")
            elif alert["type"] == "error":
                st.error(f"🚨 {alert['message']}")
    
    with col2:
        st.markdown("**💡 Optimization Recommendations:**")
        
        recommendations = [
            "Consider adding caching to reduce response times",
            "Implement load balancing for high-traffic periods", 
            "Optimize memory usage in analysis agents",
            "Add more monitoring for proactive issue detection",
            "Consider auto-scaling for peak demand periods"
        ]
        
        for rec in recommendations[:3]:  # Show top 3
            st.info(f"💡 {rec}")
    
    # Export and reporting
    st.markdown("---")
    st.markdown("### 📊 Export & Reporting")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Download CSV Report", use_container_width=True):
            csv_data = df.to_csv(index=False)
            st.download_button(
                "💾 Download Performance Data",
                csv_data,
                f"performance_report_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
    
    with col2:
        if st.button("📈 Generate Summary", use_container_width=True):
            create_notification("Performance summary generated successfully!", "success")
    
    with col3:
        if st.button("⚙️ Configure Alerts", use_container_width=True):
            create_notification("Alert configuration panel would open here", "info")


def show_evaluation_framework():
    """Show evaluation framework and testing strategies."""
    create_header("🎯 Evaluation Framework", "Comprehensive testing and validation tools for agent development")
    
    st.markdown("""
    Comprehensive evaluation is crucial for reliable agent development. This framework provides
    sophisticated tools for testing performance, reliability, accuracy, and production readiness.
    """)
    
    # Evaluation categories with enhanced tabs
    eval_tabs = create_tabs_with_icons([
        {"name": "Functionality Tests", "icon": "🔧"},
        {"name": "Performance Tests", "icon": "⚡"},
        {"name": "Integration Tests", "icon": "🔗"},
        {"name": "Security Tests", "icon": "🛡️"},
        {"name": "Load Testing", "icon": "📈"},
        {"name": "Reports", "icon": "📊"}
    ])
    
    with eval_tabs[0]:  # Functionality Tests
        st.markdown("### 🔧 Functionality Testing Suite")
        
        # Test configuration
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**📋 Test Configuration:**")
            
            selected_agent = st.selectbox("Agent to Test:", 
                                        ["Simple Agent", "Tool Agent", "Search Agent", "Stateful Agent"])
            
            test_categories = st.multiselect("Test Categories:", 
                                           ["Basic Response", "Tool Usage", "Error Handling", "Context Retention", "Edge Cases"],
                                           default=["Basic Response", "Tool Usage"])
            
            test_severity = st.selectbox("Test Severity:", ["Light", "Standard", "Comprehensive", "Stress"])
            
            if st.button("🚀 Run Functionality Tests", use_container_width=True, type="primary"):
                with st.spinner("🔄 Running functionality tests..."):
                    # Simulate test execution
                    st.session_state.func_test_results = {
                        "total_tests": 25,
                        "passed": 22,
                        "failed": 2,
                        "warnings": 1,
                        "duration": "2.3s"
                    }
                    create_notification("Functionality tests completed!", "success")
        
        with col2:
            st.markdown("**📊 Test Results:**")
            
            if 'func_test_results' in st.session_state:
                results = st.session_state.func_test_results
                
                # Test summary cards
                col_pass, col_fail, col_warn = st.columns(3)
                
                with col_pass:
                    create_metric_card(f"{results['passed']}", "Passed", f"{(results['passed']/results['total_tests']*100):.0f}%", "positive")
                
                with col_fail:
                    create_metric_card(f"{results['failed']}", "Failed", f"{(results['failed']/results['total_tests']*100):.0f}%", "negative")
                
                with col_warn:
                    create_metric_card(f"{results['warnings']}", "Warnings", "→", "neutral")
                
                # Detailed test results
                st.markdown("**📋 Detailed Results:**")
                
                test_details = [
                    {"Test": "Basic Response", "Status": "✅ Pass", "Duration": "0.5s", "Score": "100%"},
                    {"Test": "Tool Integration", "Status": "✅ Pass", "Duration": "1.2s", "Score": "95%"},
                    {"Test": "Error Handling", "Status": "⚠️ Warning", "Duration": "0.3s", "Score": "85%"},
                    {"Test": "Context Memory", "Status": "❌ Fail", "Duration": "0.8s", "Score": "60%"},
                    {"Test": "Response Quality", "Status": "✅ Pass", "Duration": "0.7s", "Score": "92%"}
                ]
                
                create_data_table(test_details, "Test Execution Results")
            else:
                st.info("🔄 Run tests to see detailed results here")
        
        # Test coverage analysis
        st.markdown("---")
        st.markdown("### 📊 Test Coverage Analysis")
        
        coverage_data = {
            "Component": ["Core Functions", "Tool Integration", "Error Handling", "Memory Management", "API Calls"],
            "Coverage (%)": [95, 88, 75, 82, 90],
            "Critical": [True, True, True, False, True]
        }
        
        import pandas as pd
        df_coverage = pd.DataFrame(coverage_data)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            import plotly.express as px
            fig = px.bar(df_coverage, x="Component", y="Coverage (%)", 
                        color="Critical", title="📈 Test Coverage by Component",
                        color_discrete_map={True: "#dc3545", False: "#28a745"})
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            avg_coverage = df_coverage["Coverage (%)"].mean()
            min_coverage = df_coverage["Coverage (%)"].min()
            
            create_status_card(
                f"📊 Overall Coverage: {avg_coverage:.0f}%",
                f"Minimum: {min_coverage}%<br>Critical components covered<br>Target: 90%+",
                "success" if avg_coverage >= 90 else "warning",
                "🎯"
            )
            
            # Coverage recommendations
            st.markdown("**💡 Recommendations:**")
            if min_coverage < 80:
                st.warning("⚠️ Improve error handling coverage")
            if avg_coverage < 90:
                st.info("💡 Add more edge case tests")
            else:
                st.success("✅ Excellent test coverage!")
    
    with eval_tabs[1]:  # Performance Tests
        st.markdown("### ⚡ Performance Testing Suite")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**⚙️ Performance Configuration:**")
            
            perf_test_type = st.selectbox("Test Type:", 
                                        ["Response Time", "Throughput", "Memory Usage", "CPU Efficiency", "Concurrent Users"])
            
            load_level = st.selectbox("Load Level:", ["Light", "Normal", "Heavy", "Extreme"])
            
            duration = st.slider("Test Duration (minutes):", 1, 30, 5)
            
            if st.button("📊 Run Performance Tests", use_container_width=True, type="primary"):
                with st.spinner("⚡ Running performance analysis..."):
                    # Simulate performance test
                    st.session_state.perf_results = {
                        "avg_response": 1.2,
                        "p95_response": 2.1,
                        "p99_response": 3.5,
                        "throughput": 45,
                        "cpu_usage": 22,
                        "memory_peak": 185
                    }
                    create_notification("Performance tests completed!", "success")
        
        with col2:
            st.markdown("**📈 Performance Metrics:**")
            
            if 'perf_results' in st.session_state:
                results = st.session_state.perf_results
                
                # Performance metrics
                metric_cols = st.columns(2)
                
                with metric_cols[0]:
                    create_metric_card(f"{results['avg_response']:.1f}s", "Avg Response", "-0.2s", "positive")
                    create_metric_card(f"{results['throughput']}", "Req/min", "+5", "positive")
                
                with metric_cols[1]:
                    create_metric_card(f"{results['cpu_usage']}%", "CPU Usage", "-3%", "positive")
                    create_metric_card(f"{results['memory_peak']}MB", "Peak Memory", "+10MB", "neutral")
                
                # Performance benchmarks
                st.markdown("**🎯 Benchmark Comparison:**")
                
                benchmark_data = [
                    {"Metric": "Response Time", "Current": f"{results['avg_response']:.1f}s", "Target": "< 2.0s", "Status": "✅ Pass"},
                    {"Metric": "Throughput", "Current": f"{results['throughput']}/min", "Target": "> 40/min", "Status": "✅ Pass"},
                    {"Metric": "P99 Latency", "Current": f"{results['p99_response']:.1f}s", "Target": "< 5.0s", "Status": "✅ Pass"},
                    {"Metric": "Memory Usage", "Current": f"{results['memory_peak']}MB", "Target": "< 500MB", "Status": "✅ Pass"}
                ]
                
                create_data_table(benchmark_data, "Performance Benchmarks")
            else:
                st.info("📊 Run performance tests to see metrics")
        
        # Performance trends
        st.markdown("---")
        st.markdown("### 📈 Performance Trends")
        
        # Mock trend data
        import random
        dates = pd.date_range(start='2025-06-05', end='2025-06-12', freq='D')
        trend_data = {
            'Date': dates,
            'Response_Time': [1.2 + random.uniform(-0.3, 0.3) for _ in range(len(dates))],
            'Throughput': [45 + random.randint(-5, 8) for _ in range(len(dates))],
            'Success_Rate': [98 + random.uniform(-2, 2) for _ in range(len(dates))]
        }
        
        df_trends = pd.DataFrame(trend_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(df_trends, x='Date', y='Response_Time', title='⚡ Response Time Trends',
                         markers=True, line_shape="spline")
            fig.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="SLA: 2.0s")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(df_trends, x='Date', y='Throughput', title='📊 Throughput Trends',
                         markers=True, line_shape="spline")
            fig.add_hline(y=40, line_dash="dash", line_color="orange", annotation_text="Target: 40/min")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with eval_tabs[2]:  # Integration Tests
        st.markdown("### 🔗 Integration Testing Suite")
        
        st.markdown("""
        Integration tests validate how different components work together in complex scenarios.
        """)
        
        # Integration test scenarios
        test_scenarios = [
            {
                "name": "Multi-Agent Coordination",
                "description": "Tests coordination between multiple agents",
                "components": ["Coordinator", "Research Agent", "Analysis Agent"],
                "status": "✅ Passing",
                "last_run": "2025-06-12 10:30",
                "duration": "45s"
            },
            {
                "name": "A2A Protocol Communication", 
                "description": "Tests cross-network agent communication",
                "components": ["A2A Agent 1", "A2A Agent 2", "Message Router"],
                "status": "⚠️ Warning",
                "last_run": "2025-06-12 09:15",
                "duration": "32s"
            },
            {
                "name": "Tool Chain Execution",
                "description": "Tests sequential tool usage and data flow",
                "components": ["Tool Agent", "Weather API", "Calculator", "File Manager"],
                "status": "✅ Passing",
                "last_run": "2025-06-12 11:45",
                "duration": "28s"
            },
            {
                "name": "Error Propagation",
                "description": "Tests how errors are handled across components",
                "components": ["All Agents", "Error Handler", "Logger"],
                "status": "❌ Failing",
                "last_run": "2025-06-12 08:20",
                "duration": "15s"
            }
        ]
        
        # Display integration test scenarios
        for scenario in test_scenarios:
            status_color = "success" if "✅" in scenario["status"] else "warning" if "⚠️" in scenario["status"] else "error"
            
            with st.expander(f"{scenario['status']} {scenario['name']}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Description:** {scenario['description']}")
                    st.markdown(f"**Components:** {', '.join(scenario['components'])}")
                    
                    if st.button(f"🔄 Run {scenario['name']}", key=f"run_{scenario['name']}"):
                        with st.spinner(f"Running {scenario['name']}..."):
                            create_notification(f"{scenario['name']} test completed!", "success")
                
                with col2:
                    create_status_card(
                        "Test Status",
                        f"Last run: {scenario['last_run']}<br>Duration: {scenario['duration']}<br>Components: {len(scenario['components'])}",
                        status_color,
                        "🔍"
                    )
        
        # Integration test matrix
        st.markdown("---")
        st.markdown("### 🧪 Integration Test Matrix")
          # Mock integration matrix
        matrix_data_dict = {
            "Component A": ["Simple Agent", "Tool Agent", "Multi-Agent", "A2A Agent"],
            "Component B": ["Tool Agent", "Search Agent", "A2A Agent", "Database"],
            "Test Status": ["✅ Pass", "⚠️ Warning", "✅ Pass", "❌ Fail"],
            "Compatibility": ["100%", "85%", "95%", "70%"],
            "Issues": ["None", "Timeout occasionally", "None", "Connection errors"]
        }
        
        # Convert to list of dictionaries format expected by create_data_table
        matrix_data = []
        num_rows = len(matrix_data_dict["Component A"])
        for i in range(num_rows):
            row = {}
            for key, values in matrix_data_dict.items():
                row[key] = values[i]
            matrix_data.append(row)
        
        create_data_table(matrix_data, "Component Integration Matrix")
    
    with eval_tabs[3]:  # Security Tests
        st.markdown("### 🛡️ Security Testing Suite")
        
        st.markdown("""
        Security testing ensures agents handle sensitive data properly and resist common attacks.
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**🔒 Security Test Categories:**")
            
            security_tests = [
                {"name": "Input Validation", "risk": "High", "status": "✅ Pass"},
                {"name": "Authentication", "risk": "Critical", "status": "✅ Pass"},
                {"name": "Data Encryption", "risk": "High", "status": "⚠️ Warning"},
                {"name": "API Security", "risk": "Medium", "status": "✅ Pass"},
                {"name": "Access Control", "risk": "High", "status": "❌ Fail"},
                {"name": "Session Management", "risk": "Medium", "status": "✅ Pass"}
            ]
            
            for test in security_tests:
                risk_color = "error" if test["risk"] == "Critical" else "warning" if test["risk"] == "High" else "info"
                status_icon = test["status"].split()[0]
                
                st.markdown(f"""
                <div style="display: flex; align-items: center; padding: 0.5rem; margin: 0.5rem 0; border-radius: 8px; border-left: 4px solid {'#dc3545' if 'Critical' in test['risk'] else '#ffc107' if 'High' in test['risk'] else '#17a2b8'};">
                    <div style="flex: 1;"><strong>{test['name']}</strong></div>
                    <div style="margin: 0 1rem;">{test['risk']} Risk</div>
                    <div>{test['status']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**🎯 Security Score:**")
            
            # Calculate security score
            passed_tests = sum(1 for test in security_tests if "✅" in test["status"])
            total_tests = len(security_tests)
            security_score = (passed_tests / total_tests) * 100
            
            create_metric_card(f"{security_score:.0f}%", "Security Score", "-5%", "negative" if security_score < 80 else "positive")
            
            # Security recommendations
            st.markdown("**🚨 Security Alerts:**")
            
            alerts = [
                {"type": "error", "message": "Access control failures detected"},
                {"type": "warning", "message": "Data encryption needs improvement"},
                {"type": "info", "message": "Regular security audits recommended"}
            ]
            
            for alert in alerts:
                if alert["type"] == "error":
                    st.error(f"🚨 {alert['message']}")
                elif alert["type"] == "warning":
                    st.warning(f"⚠️ {alert['message']}")
                else:
                    st.info(f"💡 {alert['message']}")
        
        # Security compliance
        st.markdown("---")
        st.markdown("### 📋 Security Compliance")
        
        compliance_data = [
            {"Standard": "OWASP Top 10", "Compliance": "85%", "Status": "⚠️ Partial", "Issues": "2 vulnerabilities"},
            {"Standard": "ISO 27001", "Compliance": "92%", "Status": "✅ Compliant", "Issues": "Minor gaps"},
            {"Standard": "GDPR", "Compliance": "78%", "Status": "❌ Non-compliant", "Issues": "Data handling"},
            {"Standard": "SOC 2", "Compliance": "88%", "Status": "⚠️ Partial", "Issues": "Logging incomplete"}
        ]
        
        create_data_table(compliance_data, "Security Compliance Status")
    
    with eval_tabs[4]:  # Load Testing
        st.markdown("### 📈 Load Testing Suite")
        
        st.markdown("""
        Load testing evaluates system behavior under various traffic conditions and identifies breaking points.
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**⚙️ Load Test Configuration:**")
            
            test_type = st.selectbox("Load Test Type:", 
                                   ["Stress Test", "Volume Test", "Spike Test", "Endurance Test"])
            
            concurrent_users = st.slider("Concurrent Users:", 1, 1000, 100)
            test_duration = st.slider("Duration (minutes):", 1, 60, 10)
            ramp_up_time = st.slider("Ramp-up Time (minutes):", 1, 30, 5)
            
            if st.button("🚀 Start Load Test", use_container_width=True, type="primary"):
                with st.spinner("📈 Running load test..."):
                    # Simulate load test
                    st.session_state.load_test_results = {
                        "max_users": concurrent_users,
                        "avg_response": 2.1,
                        "error_rate": 2.5,
                        "throughput": 180,
                        "breaking_point": 850
                    }
                    create_notification("Load test completed successfully!", "success")
        
        with col2:
            st.markdown("**📊 Load Test Results:**")
            
            if 'load_test_results' in st.session_state:
                results = st.session_state.load_test_results
                
                # Load test metrics
                col_metric1, col_metric2 = st.columns(2)
                
                with col_metric1:
                    create_metric_card(f"{results['max_users']}", "Max Users", f"+{results['max_users']//10}", "positive")
                    create_metric_card(f"{results['error_rate']:.1f}%", "Error Rate", "+0.5%", "negative")
                
                with col_metric2:
                    create_metric_card(f"{results['avg_response']:.1f}s", "Avg Response", "+0.3s", "negative")
                    create_metric_card(f"{results['throughput']}", "Throughput", "+20", "positive")
                
                # Performance under load
                st.markdown("**📈 Performance Under Load:**")
                
                # Simulate load curve data
                user_counts = list(range(0, results['max_users'] + 1, results['max_users']//10))
                response_times = [1.0 + (u/results['max_users']) * 2.5 for u in user_counts]
                
                fig = px.line(x=user_counts, y=response_times, 
                             title="📊 Response Time vs User Load")
                fig.add_hline(y=3.0, line_dash="dash", line_color="red", 
                             annotation_text="Breaking Point")
                fig.update_layout(height=300, 
                                xaxis_title="Concurrent Users", 
                                yaxis_title="Response Time (s)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("🔄 Run load test to see performance metrics")
        
        # Load test history
        st.markdown("---")
        st.markdown("### 📊 Load Test History")
        
        history_data = [
            {"Date": "2025-06-10", "Type": "Stress", "Max Users": 500, "Success Rate": "97%", "Breaking Point": "750 users"},
            {"Date": "2025-06-08", "Type": "Volume", "Max Users": 1000, "Success Rate": "94%", "Breaking Point": "1200 users"},
            {"Date": "2025-06-05", "Type": "Spike", "Max Users": 300, "Success Rate": "99%", "Breaking Point": "450 users"},
            {"Date": "2025-06-03", "Type": "Endurance", "Max Users": 200, "Success Rate": "98%", "Breaking Point": "N/A"}
        ]
        
        create_data_table(history_data, "Load Test History")
    
    with eval_tabs[5]:  # Reports
        st.markdown("### 📊 Evaluation Reports")
        
        st.markdown("""
        Comprehensive reports combining all testing results for stakeholder review and decision making.
        """)
        
        # Report generation
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**📋 Report Configuration:**")
            
            report_type = st.selectbox("Report Type:", 
                                     ["Executive Summary", "Technical Deep Dive", "Compliance Report", "Performance Analysis"])
            
            include_sections = st.multiselect("Include Sections:", 
                                            ["Functionality", "Performance", "Security", "Integration", "Load Testing"],
                                            default=["Functionality", "Performance", "Security"])
            
            report_format = st.selectbox("Output Format:", ["PDF", "HTML", "JSON", "CSV"])
            
            if st.button("📑 Generate Report", use_container_width=True, type="primary"):
                with st.spinner("📊 Generating comprehensive report..."):
                    create_notification("Report generated successfully!", "success")
                    st.session_state.report_generated = True
        
        with col2:
            st.markdown("**📈 Report Summary:**")
            
            if st.session_state.get('report_generated'):
                # Overall scores
                overall_scores = {
                    "Functionality": 88,
                    "Performance": 92,
                    "Security": 76,
                    "Integration": 85,
                    "Load Testing": 89
                }
                
                for category, score in overall_scores.items():
                    color = "success" if score >= 90 else "warning" if score >= 75 else "error"
                    create_status_card(
                        f"{category}: {score}%",
                        f"Score based on test results<br>Target: 85%+",
                        color,
                        "📊"
                    )
                
                # Download buttons
                st.markdown("**📥 Download Options:**")
                
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    if st.button("📄 Download PDF", use_container_width=True):
                        create_notification("PDF report download started", "info")
                
                with col_dl2:
                    if st.button("🔗 Share Report", use_container_width=True):
                        create_notification("Report sharing link generated", "info")
            else:
                st.info("📋 Generate a report to see summary and download options")
        
        # Historical reporting
        st.markdown("---")
        st.markdown("### 📈 Historical Analysis")
        
        # Mock historical data
        historical_dates = pd.date_range(start='2025-05-01', end='2025-06-12', freq='W')
        historical_data = {
            'Date': historical_dates,
            'Overall_Score': [75 + i*3 + random.randint(-5, 5) for i in range(len(historical_dates))],
            'Security_Score': [70 + i*2 + random.randint(-3, 7) for i in range(len(historical_dates))],
            'Performance_Score': [80 + i*2 + random.randint(-4, 6) for i in range(len(historical_dates))]
        }
        
        df_historical = pd.DataFrame(historical_data)
        
        fig = px.line(df_historical, x='Date', 
                     y=['Overall_Score', 'Security_Score', 'Performance_Score'],
                     title='📈 Evaluation Scores Over Time')
        fig.update_layout(height=400, yaxis_title="Score (%)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        recommendations = [
            {
                "priority": "High",
                "category": "Security",
                "issue": "Access control implementation needed",
                "impact": "Critical security vulnerability",
                "effort": "2-3 weeks"
            },
            {
                "priority": "Medium", 
                "category": "Performance",
                "issue": "Memory optimization opportunities",
                "impact": "10-15% performance improvement",
                "effort": "1 week"
            },
            {
                "priority": "Low",
                "category": "Integration",
                "issue": "Enhanced error messaging",
                "impact": "Better debugging experience",
                "effort": "3-5 days"
            }
        ]
        
        for rec in recommendations:
            priority_color = "error" if rec["priority"] == "High" else "warning" if rec["priority"] == "Medium" else "info"
            
            with st.expander(f"🔸 {rec['priority']} Priority: {rec['issue']}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Category:** {rec['category']}")
                    st.markdown(f"**Impact:** {rec['impact']}")
                    st.markdown(f"**Estimated Effort:** {rec['effort']}")
                
                with col2:
                    create_status_card(
                        f"{rec['priority']} Priority",
                        f"Category: {rec['category']}<br>Effort: {rec['effort']}",
                        priority_color,
                        "⚡"
                    )


def show_documentation():
    """Show comprehensive project documentation from markdown files."""
    st.markdown("## 📚 Project Documentation")
    st.markdown("Comprehensive guides, tutorials, and reference materials for mastering ADK & A2A.")
    
    # Documentation categories mapped to actual markdown files
    doc_files = {
        "Getting Started": "getting-started.md",
        "Agent Development": "agent-development.md", 
        "Multi-Agent Systems": "multi-agent-systems.md",
        "A2A Protocol": "a2a-protocol.md",
        "Best Practices": "best-practices.md",
        "Overview": "README.md"
    }
    
    doc_tabs = create_tabs_with_icons([
        {"name": "Getting Started", "icon": "🚀"},
        {"name": "Agent Development", "icon": "🤖"}, 
        {"name": "Multi-Agent Systems", "icon": "🔗"},
        {"name": "A2A Protocol", "icon": "🌐"},
        {"name": "Best Practices", "icon": "💡"},
        {"name": "Overview", "icon": "📖"}
    ])
    
    # Define the tab mapping
    tab_mapping = [
        "Getting Started",
        "Agent Development", 
        "Multi-Agent Systems",
        "A2A Protocol",
        "Best Practices",
        "Overview"
    ]
    
    # Render each tab with corresponding markdown content
    for i, tab_name in enumerate(tab_mapping):
        with doc_tabs[i]:
            if tab_name in doc_files:
                filename = doc_files[tab_name]
                
                # Load and render the markdown content
                markdown_content = load_markdown_file(filename)
                
                # Add a header based on the tab name
                if tab_name == "Getting Started":
                    st.markdown("### 🚀 Getting Started Guide")
                elif tab_name == "Agent Development":
                    st.markdown("### 🤖 Agent Development Guide")
                elif tab_name == "Multi-Agent Systems":
                    st.markdown("### 🔗 Multi-Agent Systems Guide")
                elif tab_name == "A2A Protocol":
                    st.markdown("### 🌐 A2A Protocol Guide")
                elif tab_name == "Best Practices":
                    st.markdown("### 💡 Best Practices Guide")
                elif tab_name == "Overview":
                    st.markdown("### 📖 Project Overview")
                
                # Render the markdown content
                render_markdown_content(markdown_content)
                
                # Add file information
                docs_path = Path(__file__).parent.parent / "docs" / filename
                if docs_path.exists():
                    file_size = docs_path.stat().st_size
                    modified_time = datetime.fromtimestamp(docs_path.stat().st_mtime)
                    
                    with st.expander("📄 Document Information"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("File Size", f"{file_size:,} bytes")
                        with col2:
                            st.metric("Last Modified", modified_time.strftime("%Y-%m-%d"))
                        with col3:
                            st.metric("File", filename)
            else:
                st.error(f"No documentation file mapped for {tab_name}")
    
    # Download documentation section
    st.markdown("---")
    st.markdown("### 📥 Download Documentation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Download All Docs", use_container_width=True):
            # Create a combined markdown file
            combined_content = "# ADK & A2A Documentation\n\n"
            
            for tab_name, filename in doc_files.items():
                content = load_markdown_file(filename)
                combined_content += f"\n\n---\n\n# {tab_name}\n\n{content}\n\n"
            
            st.download_button(
                "💾 Download Combined Documentation",
                combined_content,
                f"adk_a2a_docs_{datetime.now().strftime('%Y%m%d')}.md",
                "text/markdown",
                use_container_width=True
            )
    
    with col2:
        if st.button("📁 Individual Files", use_container_width=True):
            st.markdown("**📂 Available Documentation Files:**")
            for tab_name, filename in doc_files.items():
                docs_path = Path(__file__).parent.parent / "docs" / filename
                if docs_path.exists():
                    content = load_markdown_file(filename)
                    st.download_button(
                        f"📄 {tab_name}",
                        content,
                        filename,
                        "text/markdown",
                        key=f"download_{filename}"
                    )
    
    with col3:
        if st.button("🔗 Docs Folder", use_container_width=True):
            docs_path = Path(__file__).parent.parent / "docs"
            if docs_path.exists():
                st.info(f"📁 Documentation folder: `{docs_path.absolute()}`")
                
                # List all markdown files in docs folder
                md_files = list(docs_path.glob("*.md"))
                if md_files:
                    st.markdown("**📄 Available Files:**")
                    for md_file in md_files:
                        st.markdown(f"• `{md_file.name}`")
                else:
                    st.warning("No markdown files found in docs folder")
            else:
                st.error("Docs folder not found")
    
    # Documentation statistics
    st.markdown("---")
    st.markdown("### 📊 Documentation Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_files = len([f for f in doc_files.values() if (Path(__file__).parent.parent / "docs" / f).exists()])
        create_metric_card(f"{total_files}", "Total Docs", f"+{total_files}", "positive")
    
    with col2:
        # Calculate total word count
        total_words = 0
        for filename in doc_files.values():
            content = load_markdown_file(filename)
            total_words += len(content.split())
        create_metric_card(f"{total_words:,}", "Total Words", f"+{total_words//10}", "positive")
    
    with col3:
        # Calculate total lines
        total_lines = 0
        for filename in doc_files.values():
            content = load_markdown_file(filename)
            total_lines += len(content.split('\n'))
        create_metric_card(f"{total_lines:,}", "Total Lines", f"+{total_lines//10}", "positive")
    
    with col4:
        # Last update time
        newest_time = datetime.min
        for filename in doc_files.values():
            docs_path = Path(__file__).parent.parent / "docs" / filename
            if docs_path.exists():
                file_time = datetime.fromtimestamp(docs_path.stat().st_mtime)
                if file_time > newest_time:
                    newest_time = file_time
        
        if newest_time != datetime.min:
            days_ago = (datetime.now() - newest_time).days
            create_metric_card(f"{days_ago}", "Days Since Update", "→", "neutral")
        else:
            create_metric_card("N/A", "Days Since Update", "→", "neutral")


if __name__ == "__main__":
    main()