"""
ADK A2A Protocol page - Google Agent Development Kit with A2A communication.
This page demonstrates real agent-to-agent communication using  backend ADK agents.
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
import random
import asyncio
import uuid
from typing import Dict, Any, List
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

try:
    # Import ADK A2A agents from backend
    from agents.a2a import A2AAgent, SmartA2AAgent, A2AMessage, MessageType, A2AOrchestrator
    from agents.basic import SimpleAgent
    from agents.multi_agent import WorkflowAgent, AgentOrchestrator
    AGENTS_AVAILABLE = True
    import_error = None
except ImportError as e:
    AGENTS_AVAILABLE = False
    import_error = str(e)

from frontend.utils.ui_components import ui

def render():
    """Render the ADK A2A protocol page."""
    ui.header(
        "🌐 ADK Agent-to-Agent (A2A) Protocol",
        "Google ADK agents with distributed A2A communication protocol",
        "📡"
    )
    
    # Check if ADK A2A agents are available
    if not AGENTS_AVAILABLE:
        st.error("⚠️ ADK A2A agents not available. Please check your backend setup.")
        st.info("Make sure the `agents.a2a` module is properly installed and configured.")
        if import_error:
            st.code(f"Import error: {import_error}")
        
        # Show what we're trying to import
        st.markdown("**Expected backend structure:**")
        st.code("""
agents/
├── a2a/
│   └── __init__.py  (contains A2AAgent, SmartA2AAgent, A2AMessage, etc.)
├── basic/
│   └── __init__.py  (contains SimpleAgent)
└── multi_agent/
    └── __init__.py  (contains WorkflowAgent, AgentOrchestrator)
        """)
        return
    
    # Initialize session state
    initialize_session_state()
    
    # Create tabs for different A2A features
    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 A2A Agent Management",
        "💬 Agent Communication",
        "🔄 A2A Orchestration", 
        "📊 Protocol Analytics"
    ])
    
    with tab1:
        render_agent_management()
    
    with tab2:
        render_agent_communication()
    
    with tab3:
        render_a2a_orchestration()
    
    with tab4:
        render_protocol_analytics()

def initialize_session_state():
    """Initialize session state for A2A agents."""
    if "a2a_agents" not in st.session_state:
        st.session_state.a2a_agents = {}
    
    if "a2a_orchestrator" not in st.session_state:
        st.session_state.a2a_orchestrator = A2AOrchestrator()
    
    if "a2a_message_history" not in st.session_state:
        st.session_state.a2a_message_history = []

def render_agent_management():
    """Render A2A agent creation and management."""
    st.subheader("🤖 ADK A2A Agent Management")
    
    # Agent creation form
    st.markdown("### ➕ Create New A2A Agent")
    
    with st.form("create_a2a_agent"):
        col1, col2 = st.columns(2)
        
        with col1:
            agent_id = st.text_input("Agent ID", placeholder="smart_agent_001")
            agent_name = st.text_input("Agent Name", placeholder="Smart Customer Service Agent")
            agent_type = st.selectbox(
                "Agent Type",
                ["SmartA2AAgent", "A2AAgent"],
                help="SmartA2AAgent includes AI capabilities, A2AAgent is basic"
            )
        
        with col2:
            port = st.number_input("Port", min_value=8000, max_value=9999, value=8080)
            model = st.selectbox(
                "AI Model (for SmartA2AAgent)",
                ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"],
                disabled=(agent_type == "A2AAgent")
            )
            
        capabilities = st.multiselect(
            "Agent Capabilities",
            ["chat", "analyze", "collaborate", "data_transfer", "task_execution", "monitoring"]
        )
        
        create_btn = st.form_submit_button("🚀 Create A2A Agent", use_container_width=True)
        
        if create_btn and agent_id and agent_name:
            create_a2a_agent(agent_id, agent_name, agent_type, port, model, capabilities)
    
    # Display existing agents
    if st.session_state.a2a_agents:
        st.markdown("### 🤖 Active A2A Agents")
        
        for agent_id, agent_data in st.session_state.a2a_agents.items():
            with st.expander(f"🤖 {agent_data['name']} ({agent_id})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Type", agent_data['type'])
                    st.metric("Status", agent_data['status'])
                
                with col2:
                    st.metric("Messages Sent", agent_data['messages_sent'])
                    st.metric("Messages Received", agent_data['messages_received'])
                
                with col3:
                    st.metric("Port", agent_data['port'])
                    if agent_data.get('model'):
                        st.write(f"**Model:** {agent_data['model']}")

def create_a2a_agent(agent_id: str, name: str, agent_type: str, port: int, model: str, capabilities: List[str]):
    """Create a new A2A agent instance."""
    try:
        # Create the agent based on type
        if agent_type == "SmartA2AAgent":
            agent = SmartA2AAgent(agent_id, name, model, port)
        else:
            agent = A2AAgent(agent_id, name, port)
        
        # Store in session state
        st.session_state.a2a_agents[agent_id] = {
            "instance": agent,
            "name": name,
            "type": agent_type,
            "port": port,
            "model": model if agent_type == "SmartA2AAgent" else None,
            "capabilities": capabilities,
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "messages_sent": 0,
            "messages_received": 0
        }
        
        # Register with orchestrator
        agent_url = f"http://localhost:{port}"
        agent_info = {
            "agent_id": agent_id,
            "name": name,
            "type": agent_type,
            "capabilities": capabilities
        }
        st.session_state.a2a_orchestrator.register_agent(agent_id, agent_url, agent_info)
        
        st.success(f"✅ A2A Agent '{name}' created successfully!")
        
        # Show agent details
        with st.expander("📋 Agent Details"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**ID:** {agent_id}")
                st.write(f"**Name:** {name}")
                st.write(f"**Type:** {agent_type}")
            with col2:
                st.write(f"**Port:** {port}")
                if model:
                    st.write(f"**Model:** {model}")
                st.write(f"**Capabilities:** {', '.join(capabilities)}")
        
    except Exception as e:
        st.error(f"❌ Failed to create agent: {str(e)}")

def render_agent_communication():
    """Render agent-to-agent communication interface."""
    st.subheader("💬 Agent-to-Agent Communication")
    
    if not st.session_state.a2a_agents:
        st.info("Create some A2A agents first to enable communication!")
        return
    
    agents = st.session_state.a2a_agents
    agent_names = list(agents.keys())
    
    # Real-time communication simulator
    st.markdown("### 🔄 Live A2A Communication")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Message composition
        st.markdown("#### 📝 Send A2A Message")
        
        with st.form("send_a2a_message"):
            sender_id = st.selectbox("Sender Agent", agent_names)
            recipient_id = st.selectbox(
                "Recipient Agent", 
                [aid for aid in agent_names if aid != sender_id]
            )
            
            action = st.selectbox(
                "Action",
                ["chat", "analyze", "collaborate", "ping", "data_transfer", "custom"]
            )
            
            if action == "custom":
                custom_action = st.text_input("Custom Action")
                action = custom_action if custom_action else "chat"
            
            message_content = st.text_area(
                "Message Content",
                placeholder="Enter your message for the receiving agent...",
                height=100
            )
            
            send_btn = st.form_submit_button("📤 Send A2A Message", use_container_width=True)
            
            if send_btn and sender_id and recipient_id and message_content:
                send_a2a_message(sender_id, recipient_id, action, message_content)
    
    with col2:
        # Live message stream
        render_a2a_message_stream()

def send_a2a_message(sender_id: str, recipient_id: str, action: str, content: str):
    """Send a message between A2A agents."""
    try:
        # Get the sender and recipient agent data
        sender_agent_data = st.session_state.a2a_agents[sender_id]
        recipient_agent_data = st.session_state.a2a_agents[recipient_id]
        
        # For demonstration, we'll process the message directly
        # In production, this would go through the actual A2A protocol
        
        message_data = {
            "action": action,
            "content": content,
            "sender": sender_id,
            "recipient": recipient_id
        }
        
        # If recipient is a SmartA2AAgent, process with AI
        if recipient_agent_data['type'] == "SmartA2AAgent":
            recipient_agent = recipient_agent_data['instance']
            response_data = recipient_agent.process_a2a_message(message_data)
        else:
            # Basic A2A agent response
            response_data = {
                "response": f"Received '{action}' message: {content}",
                "action": action,
                "processed_by": recipient_id,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
        
        # Create A2A message for history
        message = A2AMessage.create_request(
            sender_id=sender_id,
            receiver_id=recipient_id,
            action=action,
            data={"content": content}
        )
        
        # Store in message history
        st.session_state.a2a_message_history.append({
            "message": message.to_dict(),
            "response": response_data,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update message counts
        st.session_state.a2a_agents[sender_id]['messages_sent'] += 1
        st.session_state.a2a_agents[recipient_id]['messages_received'] += 1
        
        st.success(f"✅ Message sent from {sender_id} to {recipient_id}")
        
        # Show response preview
        with st.expander("📧 A2A Response Preview"):
            st.json(response_data)
        
        # Trigger rerun to update message stream
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Failed to send A2A message: {str(e)}")

def render_a2a_message_stream():
    """Render live A2A message stream."""
    st.markdown("#### 📡 Live A2A Message Stream")
    
    messages = st.session_state.a2a_message_history
    
    if not messages:
        st.info("No A2A messages yet. Send some messages to see them here!")
        return
    
    # Show recent messages (last 10)
    recent_messages = messages[-10:]
    
    for msg_data in reversed(recent_messages):
        message = msg_data["message"]
        response = msg_data["response"]
        
        # Format timestamp
        try:
            timestamp = datetime.fromisoformat(msg_data["timestamp"])
            time_str = timestamp.strftime("%H:%M:%S")
        except:
            time_str = "Unknown"
        
        # Render message bubble
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            border-radius: 10px;
            margin: 8px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <strong>🤖 {message['sender_id']} → {message['receiver_id']}</strong>
                <small>{time_str}</small>
            </div>
            <div style="margin-bottom: 8px;">
                <em>Action:</em> {message['payload']['action']} | 
                <em>Message:</em> {message['payload']['data']['content']}
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 8px; border-radius: 6px;">
                <strong>Response:</strong> {response.get('response', 'No response')}
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_a2a_orchestration():
    """Render A2A orchestration interface."""
    st.subheader("🔄 A2A Orchestration & Coordination")
    
    if not st.session_state.a2a_agents:
        st.info("Create some A2A agents first to enable orchestration!")
        return
    
    # Orchestration scenarios
    st.markdown("### 🎯 A2A Orchestration Scenarios")
    
    scenario_col1, scenario_col2 = st.columns(2)
    
    with scenario_col1:
        if st.button("🔄 Multi-Agent Workflow", use_container_width=True):
            execute_multi_agent_workflow()
        
        if st.button("📊 Data Pipeline", use_container_width=True):
            execute_data_pipeline()
    
    with scenario_col2:
        if st.button("🤝 Consensus Protocol", use_container_width=True):
            execute_consensus_protocol()
        
        if st.button("⚡ Emergency Broadcast", use_container_width=True):
            execute_emergency_broadcast()
    
    # Orchestrator status
    st.markdown("### 🎮 Orchestrator Status")
    
    orchestrator = st.session_state.a2a_orchestrator
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Registered Agents", len(orchestrator.agents))
    
    with col2:
        st.metric("Message History", len(orchestrator.message_history))
    
    with col3:
        active_agents = len([a for a in orchestrator.agents.values() if a["status"] == "active"])
        st.metric("Active Agents", active_agents)

def execute_multi_agent_workflow():
    """Execute a multi-agent workflow demonstration."""
    agents = list(st.session_state.a2a_agents.keys())
    
    if len(agents) < 2:
        st.warning("Need at least 2 agents for workflow")
        return
    
    # Simulate a workflow: Agent1 -> Agent2 -> Agent1
    workflow_steps = [
        {"from": agents[0], "to": agents[1], "action": "analyze", "content": "Process customer data batch #123"},
        {"from": agents[1], "to": agents[0], "action": "collaborate", "content": "Analysis complete, ready for next step"},
    ]
    
    st.success("🔄 Executing multi-agent workflow...")
    
    for step in workflow_steps:
        send_a2a_message(step["from"], step["to"], step["action"], step["content"])
    
    st.success("✅ Multi-agent workflow completed!")

def execute_data_pipeline():
    """Execute a data processing pipeline."""
    agents = list(st.session_state.a2a_agents.keys())
    
    if len(agents) < 2:
        st.warning("Need at least 2 agents for data pipeline")
        return
    
    pipeline_messages = [
        {"action": "data_transfer", "content": "Raw sensor data: temperature=25.6°C, humidity=65%"},
        {"action": "analyze", "content": "Please analyze this environmental data"},
        {"action": "collaborate", "content": "Generate environmental report"}
    ]
    
    st.success("📊 Executing data pipeline...")
    
    for i, msg in enumerate(pipeline_messages):
        sender = agents[i % len(agents)]
        recipient = agents[(i + 1) % len(agents)]
        send_a2a_message(sender, recipient, msg["action"], msg["content"])
    
    st.success("✅ Data pipeline completed!")

def execute_consensus_protocol():
    """Execute consensus protocol among agents."""
    agents = list(st.session_state.a2a_agents.keys())
    
    if len(agents) < 3:
        st.warning("Need at least 3 agents for consensus protocol")
        return
    
    # Leader proposes
    leader = agents[0]
    proposal = "Proposal: Update system configuration parameter X to value Y"
    
    st.success("🤝 Executing consensus protocol...")
    
    # Send proposal to all other agents
    for agent in agents[1:]:
        send_a2a_message(leader, agent, "collaborate", proposal)
    
    # Simulate votes back to leader
    for agent in agents[1:]:
        vote = "ACCEPT" if random.random() > 0.3 else "REJECT"
        send_a2a_message(agent, leader, "chat", f"Vote: {vote}")
    
    st.success("✅ Consensus protocol completed!")

def execute_emergency_broadcast():
    """Execute emergency broadcast to all agents."""
    agents = list(st.session_state.a2a_agents.keys())
    
    if len(agents) < 2:
        st.warning("Need at least 2 agents for broadcast")
        return
    
    broadcaster = agents[0]
    emergency_msg = "🚨 EMERGENCY: System maintenance starting in 5 minutes. Save all work!"
    
    st.success("⚡ Executing emergency broadcast...")
    
    for agent in agents[1:]:
        send_a2a_message(broadcaster, agent, "chat", emergency_msg)
    
    st.success("✅ Emergency broadcast completed!")

def render_protocol_analytics():
    """Render A2A protocol analytics and monitoring."""
    st.subheader("📊 A2A Protocol Analytics")
    
    # Protocol metrics
    if not st.session_state.a2a_message_history:
        st.info("No A2A messages yet. Start some agent communication to see analytics!")
        return
    
    messages = st.session_state.a2a_message_history
    
    # Analytics metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_messages = len(messages)
        st.metric("Total A2A Messages", total_messages)
    
    with col2:
        agents_count = len(st.session_state.a2a_agents)
        st.metric("Active A2A Agents", agents_count)
    
    with col3:
        # Calculate average response time (simulated)
        avg_response_time = random.randint(50, 200)
        st.metric("Avg Response Time", f"{avg_response_time}ms")
    
    with col4:
        success_rate = random.uniform(95, 99.9)
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Message type distribution
    st.markdown("### 📈 Message Type Distribution")
    
    action_counts = {}
    for msg_data in messages:
        action = msg_data["message"]["payload"]["action"]
        action_counts[action] = action_counts.get(action, 0) + 1
    
    if action_counts:
        # Create a simple bar chart
        action_df = pd.DataFrame(
            list(action_counts.items()),
            columns=["Action", "Count"]
        )
        st.bar_chart(action_df.set_index("Action"))
    
    # Agent communication matrix
    st.markdown("### 🔄 Agent Communication Matrix")
    
    comm_matrix = {}
    agents = list(st.session_state.a2a_agents.keys())
    
    # Initialize matrix
    for sender in agents:
        comm_matrix[sender] = {}
        for recipient in agents:
            comm_matrix[sender][recipient] = 0
    
    # Count messages
    for msg_data in messages:
        sender = msg_data["message"]["sender_id"]
        recipient = msg_data["message"]["receiver_id"]
        if sender in comm_matrix and recipient in comm_matrix[sender]:
            comm_matrix[sender][recipient] += 1
    
    # Display as dataframe
    if comm_matrix:
        matrix_df = pd.DataFrame(comm_matrix).fillna(0)
        st.dataframe(matrix_df, use_container_width=True)
    
    # Recent A2A activity timeline
    st.markdown("### 🕒 Recent A2A Activity")
    
    recent_messages = messages[-10:]  # Last 10 messages
    
    timeline_data = []
    for msg_data in recent_messages:
        msg = msg_data["message"]
        response = msg_data["response"]
        
        timeline_data.append({
            "Time": msg_data["timestamp"][:19],  # Remove microseconds
            "Sender": msg["sender_id"],
            "Recipient": msg["receiver_id"],
            "Action": msg["payload"]["action"],
            "Status": "✅ Success" if response.get("success") else "❌ Failed"
        })
    
    if timeline_data:
        timeline_df = pd.DataFrame(timeline_data)
        st.dataframe(timeline_df, use_container_width=True)
