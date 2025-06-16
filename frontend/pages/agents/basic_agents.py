"""
Basic Agents page for creating and testing simple agents.
"""

import streamlit as st
import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from frontend.utils.ui_components import ui, charts

# Import agent-related modules with error handling
try:
    from agents.basic import create_agent, SimpleAgent, SearchAgent, ToolAgent, StatefulAgent
except ImportError as e:
    st.error(f"Error importing agent modules: {e}")
    SimpleAgent = None

def render():
    """Render the basic agents page."""
    ui.header(
        "Basic Agents",
        "Create and experiment with simple agents using Google ADK",
        "🤖"
    )
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Create Agent",
        "💬 Chat Interface", 
        "🧪 Agent Testing",
        "📊 Agent Analytics"
    ])
    
    with tab1:
        render_agent_creation()
    
    with tab2:
        render_chat_interface()
    
    with tab3:
        render_agent_testing()
    
    with tab4:
        render_agent_analytics()

def render_agent_creation():
    """Render agent creation interface."""
    st.subheader("🚀 Create New Agent")
    
    # Agent configuration form
    with st.form("agent_creation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            agent_name = st.text_input(
                "Agent Name",
                placeholder="My Awesome Agent",
                help="Give your agent a descriptive name"
            )
            
            agent_type = st.selectbox(
                "Agent Type",
                ["Simple Agent", "Search Agent", "Tool Agent", "Stateful Agent"],
                help="Choose the type of agent to create"
            )
            
            model_name = st.selectbox(
                "Model",
                ["gemini-pro", "gemini-pro-vision", "gemini-1.5-pro"],
                help="Select the AI model for your agent"
            )
        
        with col2:
            system_prompt = st.text_area(
                "System Prompt",
                placeholder="You are a helpful assistant...",
                height=150,
                help="Define your agent's personality and behavior"
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.7,
                step=0.1,
                help="Controls randomness in responses"
            )
            
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=100,
                max_value=4000,
                value=1000,
                help="Maximum response length"
            )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            enable_tools = st.checkbox("Enable Tools", value=False)
            enable_memory = st.checkbox("Enable Memory", value=False)
            enable_logging = st.checkbox("Enable Logging", value=True)
        
        submitted = st.form_submit_button("🚀 Create Agent", use_container_width=True)
        
        if submitted:
            if agent_name and system_prompt:
                create_new_agent(
                    agent_name, agent_type, model_name, system_prompt,
                    temperature, max_tokens, enable_tools, enable_memory, enable_logging
                )
            else:
                st.error("Please fill in all required fields")

def create_new_agent(name: str, agent_type: str, model: str, prompt: str,
                    temperature: float, max_tokens: int, tools: bool, 
                    memory: bool, logging: bool):
    """Create a new agent with the specified configuration."""
    try:
        # Initialize session state for agents if not exists
        if "created_agents" not in st.session_state:
            st.session_state.created_agents = []
        
        # Create agent configuration
        agent_config = {
            "id": f"agent_{len(st.session_state.created_agents) + 1}",
            "name": name,
            "type": agent_type,
            "model": model,
            "system_prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools_enabled": tools,
            "memory_enabled": memory,
            "logging_enabled": logging,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Add to session state
        st.session_state.created_agents.append(agent_config)
        
        # Success message
        st.success(f"✅ Agent '{name}' created successfully!")
        
        # Display agent info
        ui.info_box(
            "Agent Created",
            f"Name: {name}\nType: {agent_type}\nModel: {model}\nID: {agent_config['id']}",
            "success"
        )
        
    except Exception as e:
        st.error(f"Error creating agent: {e}")

def render_chat_interface():
    """Render chat interface for testing agents."""
    st.subheader("💬 Chat with Your Agents")
    
    # Check if any agents exist
    if "created_agents" not in st.session_state or not st.session_state.created_agents:
        ui.info_box(
            "No Agents Available",
            "Create an agent first to start chatting!",
            "info"
        )
        return
    
    # Agent selection
    agent_options = {agent["name"]: agent for agent in st.session_state.created_agents}
    selected_agent_name = st.selectbox(
        "Select Agent to Chat With",
        list(agent_options.keys())
    )
    
    if selected_agent_name:
        selected_agent = agent_options[selected_agent_name]
        
        # Display agent info
        col1, col2, col3 = st.columns(3)
        with col1:
            ui.status_indicator("online", f"Status: {selected_agent['status']}")
        with col2:
            st.write(f"**Model:** {selected_agent['model']}")
        with col3:
            st.write(f"**Type:** {selected_agent['type']}")
        
        st.markdown("---")
        
        # Chat interface
        render_chat_messages(selected_agent["id"])
        
        # Message input
        with st.form("chat_form"):
            message = st.text_area(
                "Your Message",
                placeholder="Type your message here...",
                height=100
            )
            
            col1, col2 = st.columns([3, 1])
            with col2:
                send_button = st.form_submit_button("Send 🚀", use_container_width=True)
            
            if send_button and message:
                handle_chat_message(selected_agent, message)

def render_chat_messages(agent_id: str):
    """Render chat messages for a specific agent."""
    chat_key = f"chat_history_{agent_id}"
    
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []
    
    # Display chat history
    for msg in st.session_state[chat_key]:
        if msg["role"] == "user":
            with st.container():
                st.markdown(f"""
                <div class="chat-message user">
                    <strong>You:</strong> {msg['content']}
                    <br><small>{msg['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            with st.container():
                st.markdown(f"""
                <div class="chat-message assistant">
                    <strong>{msg['agent_name']}:</strong> {msg['content']}
                    <br><small>{msg['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)

def handle_chat_message(agent_config: Dict[str, Any], message: str):
    """Handle a chat message and generate response."""
    chat_key = f"chat_history_{agent_config['id']}"
    
    # Add user message
    user_msg = {
        "role": "user",
        "content": message,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }
    st.session_state[chat_key].append(user_msg)
    
    # Generate mock response (replace with actual agent call)
    try:
        # This is a mock response - replace with actual agent implementation
        response = f"Hello! I'm {agent_config['name']}. I received your message: '{message}'. This is a simulated response using the {agent_config['model']} model."
        
        # Add agent response
        agent_msg = {
            "role": "assistant",
            "content": response,
            "agent_name": agent_config['name'],
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        st.session_state[chat_key].append(agent_msg)
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Error generating response: {e}")

def render_agent_testing():
    """Render agent testing interface."""
    st.subheader("🧪 Agent Testing")
    
    if "created_agents" not in st.session_state or not st.session_state.created_agents:
        ui.info_box(
            "No Agents Available",
            "Create an agent first to start testing!",
            "info"
        )
        return
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Basic Conversation",
            "description": "Test basic conversational abilities",
            "prompt": "Hello, can you introduce yourself?"
        },
        {
            "name": "Problem Solving",
            "description": "Test problem-solving capabilities",
            "prompt": "Solve this math problem: What is 15% of 240?"
        },
        {
            "name": "Creative Writing",
            "description": "Test creative abilities",
            "prompt": "Write a short poem about artificial intelligence"
        },
        {
            "name": "Technical Explanation",
            "description": "Test technical knowledge",
            "prompt": "Explain how machine learning works in simple terms"
        }
    ]
    
    # Agent selection for testing
    agent_options = {agent["name"]: agent for agent in st.session_state.created_agents}
    selected_agent = st.selectbox(
        "Select Agent to Test",
        list(agent_options.keys())
    )
    
    if selected_agent:
        st.markdown("### 📋 Test Scenarios")
        
        for i, scenario in enumerate(test_scenarios):
            with st.expander(f"🧪 {scenario['name']}"):
                st.write(scenario["description"])
                st.code(scenario["prompt"], language="text")
                
                if st.button(f"Run Test {i+1}", key=f"test_{i}"):
                    st.info("🔄 Running test scenario...")
                    # Mock test execution
                    st.success("✅ Test completed successfully!")
                    st.write("**Response:** This is a mock response for testing purposes.")

def render_agent_analytics():
    """Render agent analytics and metrics."""
    st.subheader("📊 Agent Analytics")
    
    if "created_agents" not in st.session_state or not st.session_state.created_agents:
        ui.info_box(
            "No Agents Available",
            "Create an agent first to view analytics!",
            "info"
        )
        return
    
    # Agent selection
    agent_options = {agent["name"]: agent for agent in st.session_state.created_agents}
    selected_agent = st.selectbox(
        "Select Agent for Analytics",
        list(agent_options.keys()),
        key="analytics_agent_select"
    )
    
    if selected_agent:
        agent_data = agent_options[selected_agent]
        
        # Metrics overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ui.metric_card("Total Messages", "23", "+5 today")
        with col2:
            ui.metric_card("Avg Response Time", "1.2s", "-0.3s from yesterday")
        with col3:
            ui.metric_card("Success Rate", "98.5%", "+1.2% this week")
        with col4:
            ui.metric_card("Uptime", "99.9%", "Excellent")
        
        # Charts
        st.markdown("### 📈 Performance Trends")
        
        # Mock data for charts
        import pandas as pd
        import random
        from datetime import timedelta
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='H')
        performance_data = pd.DataFrame({
            'Timestamp': dates,
            'Response Time (ms)': [random.randint(800, 2000) for _ in dates],
            'Messages': [random.randint(0, 5) for _ in dates]
        })
        
        charts.line_chart(
            performance_data,
            x='Timestamp',
            y='Response Time (ms)',
            title="Response Time Trend"
        )

if __name__ == "__main__":
    render()
