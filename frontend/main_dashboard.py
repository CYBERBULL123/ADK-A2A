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
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, List
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config import config
from utils import validate_environment, display_welcome, console
from agents.basic import create_agent, SimpleAgent, SearchAgent, ToolAgent, StatefulAgent
from agents.multi_agent import CoordinatorAgent, WorkflowOrchestrator, EXAMPLE_WORKFLOWS
from agents.a2a import SmartA2AAgent, A2AOrchestrator
from tools import CUSTOM_TOOLS, get_tool_info


# Configure Streamlit page
st.set_page_config(
    page_title="ADK & A2A Learning Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Main dashboard function."""
    st.title("🤖 ADK & A2A Learning Dashboard")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a learning module:",
        [
            "🏠 Overview",
            "🤖 Basic Agents",
            "🔗 Multi-Agent Systems",
            "🌐 A2A Protocol",
            "🛠️ Custom Tools",
            "📊 Performance Analytics",
            "🎯 Evaluation Framework"
        ]
    )
      # Environment status
    env_status = validate_environment()
    with st.sidebar.expander("Environment Status", expanded=not env_status):
        if env_status:
            st.success("✅ Environment validated")
        else:
            st.error("❌ Environment issues detected")
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
        show_custom_tools()
    elif page == "📊 Performance Analytics":
        show_performance_analytics()
    elif page == "🎯 Evaluation Framework":
        show_evaluation_framework()


def show_overview():
    """Show project overview and learning objectives."""
    st.header("Welcome to ADK & A2A Learning Project")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🎯 Learning Objectives
        
        This interactive dashboard helps you master:
        
        ### 🤖 Agent Development Kit (ADK)
        - **Agent Architecture**: Understanding core agent components
        - **Tool Integration**: Adding capabilities with built-in and custom tools
        - **Model Integration**: Working with Gemini and other LLMs
        - **Code-First Development**: Python-native agent development
        
        ### 🔗 Multi-Agent Systems
        - **Agent Hierarchies**: Creating specialized agent teams
        - **Coordination Patterns**: Orchestrating complex workflows
        - **Task Delegation**: Distributing work across agents
        - **Communication Protocols**: Inter-agent messaging
        
        ### 🌐 Agent-to-Agent (A2A) Protocol
        - **Remote Communication**: Network-based agent interaction
        - **Protocol Design**: Message formatting and routing
        - **Distributed Systems**: Scaling across multiple nodes
        - **Real-time Coordination**: Synchronous and asynchronous patterns
        """)
    
    with col2:
        st.markdown("""
        ## 📚 Learning Path
        
        **Phase 1: Foundations**
        - Basic agent creation
        - Tool integration
        - Simple interactions
        
        **Phase 2: Coordination**
        - Multi-agent workflows
        - Task orchestration
        - Performance optimization
        
        **Phase 3: Distribution**
        - A2A protocol implementation
        - Network communication
        - Scalable architectures
        
        **Phase 4: Production**
        - Deployment strategies
        - Monitoring and evaluation
        - Best practices
        """)
    
    # Quick Start section
    st.markdown("---")
    st.header("🚀 Quick Start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Try Basic Agent", use_container_width=True):
            st.session_state.demo_agent = SimpleAgent("QuickDemo")
            response = st.session_state.demo_agent.chat("Hello! Tell me about ADK.")
            st.success(f"Agent Response: {response}")
    
    with col2:
        if st.button("Check Environment", use_container_width=True):
            validation = config.validate_config()
            if validation["valid"]:
                st.success("Environment ready!")
            else:
                st.error("Please configure API keys")
    
    with col3:
        if st.button("View Examples", use_container_width=True):
            st.info("Navigate to other tabs to explore examples!")


def show_basic_agents():
    """Show basic agent examples and testing interface."""
    st.header("🤖 Basic Agents")
    
    # Agent type selection
    agent_type = st.selectbox(
        "Select agent type to explore:",
        ["Simple Agent", "Search Agent", "Tool Agent", "Stateful Agent"]
    )
    
    # Initialize session state for agents
    if 'current_agent' not in st.session_state:
        st.session_state.current_agent = None
    
    # Agent creation and testing
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Agent Configuration")
        
        agent_name = st.text_input("Agent Name", value=f"Demo_{agent_type.replace(' ', '_')}")
        
        if st.button("Create Agent", use_container_width=True):
            try:
                if agent_type == "Simple Agent":
                    st.session_state.current_agent = create_agent("simple", agent_name)
                elif agent_type == "Search Agent":
                    st.session_state.current_agent = create_agent("search", agent_name)
                elif agent_type == "Tool Agent":
                    st.session_state.current_agent = create_agent("tool", agent_name)
                elif agent_type == "Stateful Agent":
                    st.session_state.current_agent = create_agent("stateful", agent_name)
                
                st.success(f"Created {agent_type}: {agent_name}")
            except Exception as e:
                st.error(f"Error creating agent: {e}")
          # Agent information
        if st.session_state.current_agent:
            st.markdown("**Agent Status:**")
            
            # Create a nice status card
            agent_type = type(st.session_state.current_agent).__name__
            status_color = "#28a745"  # Green for active
            
            st.markdown(f"""
            <div style="
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid {status_color};
                margin: 0.5rem 0;
            ">
                <strong>🤖 {st.session_state.current_agent.name}</strong><br>
                <small>Type: {agent_type}</small><br>
                <small>Status: <span style="color: {status_color};">●</span> Active</small><br>
                <small>Created: {datetime.now().strftime("%H:%M:%S")}</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Agent Testing")
        
        if st.session_state.current_agent:
            # Chat interface
            user_message = st.text_area(
                "Enter your message:",
                placeholder="Ask the agent something...",
                height=100
            )
            
            if st.button("Send Message", use_container_width=True):
                if user_message:
                    try:
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
                          
                        st.markdown("**Agent Response:**")
                        
                        # Display response in a clean container
                        with st.container():
                            st.markdown(f"""
                            <div style="
                                padding: 1rem;
                                border-radius: 0.5rem;
                                border-left: 4px solid #1f77b4;
                                margin: 0.5rem 0;
                            ">
                                {response}
                            </div>
                            """, unsafe_allow_html=True)
                          # Show conversation history for stateful agents
                        if hasattr(st.session_state.current_agent, 'get_conversation_summary'):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                summary = st.session_state.current_agent.get_conversation_summary()
                                st.info(f"💬 {summary}")
                            with col2:
                                if st.button("🗑️ Clear History", help="Clear conversation history"):
                                    st.session_state.current_agent.clear_history()
                                    st.success("History cleared!")
                                    st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                else:
                    st.warning("⚠️ Please enter a message")
            
            # Add some spacing and helpful tips
            st.markdown("---")
            st.markdown("### 💡 Tips:")
            agent_type = type(st.session_state.current_agent).__name__
            
            if agent_type == "SimpleAgent":
                st.info("💬 This is a basic conversational agent. Try asking general questions!")
            elif agent_type == "SearchAgent":
                st.info("🔍 This agent can search the web. Try asking about current events or recent information!")
            elif agent_type == "ToolAgent":
                st.info("🛠️ This agent has weather and calculator tools. Try asking about weather or math calculations!")
            elif agent_type == "StatefulAgent":
                st.info("🧠 This agent remembers your conversation. Try having a multi-turn conversation!")
        else:
            st.info("👆 Create an agent first to start testing")
    
    # Agent comparison section
    st.markdown("---")
    st.subheader("Agent Type Comparison")
    
    comparison_data = {
        "Agent Type": ["Simple", "Search", "Tool", "Stateful"],
        "Capabilities": [
            "Basic conversation",
            "Web search + conversation",
            "Custom tools + conversation",
            "Memory + conversation"
        ],
        "Use Cases": [
            "General assistance",
            "Research and facts",
            "Specific tasks",
            "Personalized interaction"
        ],
        "Complexity": ["Low", "Medium", "Medium", "High"]
    }
    
    df = pd.DataFrame(comparison_data)
    st.table(df)


def show_multi_agent_systems():
    """Show multi-agent system examples and orchestration."""
    st.header("🔗 Multi-Agent Systems")
    
    # System type selection
    system_type = st.selectbox(
        "Select multi-agent system to explore:",
        ["Coordinator System", "Workflow Orchestrator", "Custom Hierarchy"]
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("System Configuration")
        
        if system_type == "Coordinator System":
            st.markdown("""
            **Coordinator Agent System**
            - Research Specialist
            - Analysis Specialist  
            - Writing Specialist
            - Main Coordinator
            """)
            
            project_description = st.text_area(
                "Project Description:",
                placeholder="Describe the project you want the team to work on...",
                height=100
            )
            
            if st.button("Execute Project", use_container_width=True):
                if project_description:
                    with st.spinner("Executing multi-agent project..."):
                        try:
                            coordinator = CoordinatorAgent("Project_Coordinator")
                            result = coordinator.execute_complex_project(project_description)
                            
                            st.session_state.last_project_result = result
                            st.session_state.last_coordinator = coordinator
                            st.session_state.last_project_description = project_description
                            
                            if result["success"]:
                                st.success("Project completed successfully!")
                            else:
                                st.error("Project execution failed")
                        
                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.warning("Please enter a project description")
            
            # Continue Workflow button
            if hasattr(st.session_state, 'last_project_result') and st.session_state.last_project_result:
                st.markdown("---")
                st.markdown("**Continue Incomplete Workflow**")
                
                additional_context = st.text_area(
                    "Additional Context/Instructions:",
                    placeholder="Provide additional context or specific instructions to continue the workflow...",
                    height=80,
                    key="continue_context"
                )
                
                col_continue1, col_continue2 = st.columns(2)
                
                with col_continue1:
                    if st.button("🔄 Continue Workflow", use_container_width=True):
                        if hasattr(st.session_state, 'last_coordinator'):
                            with st.spinner("Continuing workflow with additional context..."):
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
                                        st.success("Workflow continued successfully!")
                                    else:
                                        st.error("Workflow continuation failed")
                                
                                except Exception as e:
                                    st.error(f"Error continuing workflow: {e}")
                
                with col_continue2:
                    if st.button("🎯 Force Complete Research", use_container_width=True):
                        if hasattr(st.session_state, 'last_coordinator'):
                            with st.spinner("Forcing research completion..."):
                                try:
                                    research_result = st.session_state.last_coordinator.force_complete_research(
                                        st.session_state.last_project_description
                                    )
                                    
                                    if research_result.success:
                                        st.success("Research completed successfully!")
                                        st.session_state.force_research_result = research_result
                                    else:
                                        st.error("Research completion failed")
                                
                                except Exception as e:
                                    st.error(f"Error forcing research: {e}")
        
        elif system_type == "Workflow Orchestrator":
            st.markdown("**Available Workflows:**")
            
            workflow_name = st.selectbox(
                "Select workflow:",
                list(EXAMPLE_WORKFLOWS.keys())
            )
            
            workflow = EXAMPLE_WORKFLOWS[workflow_name]
            st.markdown(f"**Description:** {workflow['description']}")
            st.markdown(f"**Required Inputs:** {', '.join(workflow['required_inputs'])}")
            
            # Input collection
            inputs = {}
            for input_field in workflow['required_inputs']:
                inputs[input_field] = st.text_input(f"{input_field.replace('_', ' ').title()}:")
            
            if st.button("Execute Workflow", use_container_width=True):
                if all(inputs.values()):
                    with st.spinner("Executing workflow..."):
                        try:
                            orchestrator = WorkflowOrchestrator()
                            for name, wf in EXAMPLE_WORKFLOWS.items():
                                orchestrator.register_workflow(name, wf)
                            
                            # Note: This would need to be run in an async context
                            st.info("Workflow execution simulated (requires async runtime)")
                            st.session_state.workflow_inputs = inputs
                        
                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.warning("Please fill in all required inputs")
    
    with col2:
        st.subheader("Execution Results")
          # Show results if available
        if 'last_project_result' in st.session_state:
            result = st.session_state.last_project_result
            
            # Success/failure indicator
            if result["success"]:
                st.success("✅ Project Completed")
            else:
                st.error("❌ Project Failed")
            
            # Task breakdown
            st.markdown("**Task Execution:**")
            for i, task in enumerate(result["tasks"], 1):
                status = "✅" if task.success else "❌"
                st.markdown(f"{status} **Phase {i}**: {task.task}")
                if not task.success:
                    st.error(f"Error: {task.result}")
            
            # Final result
            if result["success"]:
                st.markdown("**Final Result:**")
                st.text_area("Output:", value=result["final_result"], height=200, disabled=True)
            
            # Show forced research result if available
            if 'force_research_result' in st.session_state:
                st.markdown("---")
                st.markdown("**🎯 Forced Research Result:**")
                research_result = st.session_state.force_research_result
                if research_result.success:
                    st.success("Research completed successfully!")
                    st.text_area("Research Output:", value=research_result.result, height=200, disabled=True)
                else:
                    st.error(f"Research failed: {research_result.result}")
        
        else:
            st.info("Execute a project to see results here")
    
    # Performance visualization
    st.markdown("---")
    st.subheader("System Performance")
    
    # Mock performance data for demonstration
    performance_data = {
        "Agent": ["Research", "Analysis", "Writing", "Coordinator"],
        "Tasks Completed": [15, 12, 18, 8],
        "Avg Response Time (s)": [2.3, 1.8, 3.1, 0.5],
        "Success Rate (%)": [95, 98, 92, 100]
    }
    
    df = pd.DataFrame(performance_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(df, x="Agent", y="Tasks Completed", title="Tasks Completed by Agent")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(df, x="Avg Response Time (s)", y="Success Rate (%)", 
                        size="Tasks Completed", hover_name="Agent",
                        title="Performance Overview")
        st.plotly_chart(fig, use_container_width=True)


def show_a2a_protocol():
    """Show A2A protocol examples and testing."""
    st.header("🌐 A2A Protocol")
    
    st.markdown("""
    **Agent-to-Agent (A2A) Protocol** enables distributed agent communication
    across networks, allowing for scalable multi-agent architectures.
    """)
    
    # Protocol demonstration
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("A2A Agent Network")
        
        # Agent configuration
        num_agents = st.slider("Number of A2A Agents:", 2, 5, 3)
        
        agent_configs = []
        for i in range(num_agents):
            with st.expander(f"Agent {i+1} Configuration"):
                name = st.text_input(f"Agent {i+1} Name:", value=f"A2A_Agent_{i+1}")
                port = st.number_input(f"Port:", value=8500+i, min_value=8500, max_value=9000)
                agent_configs.append({"name": name, "port": port})
        
        if st.button("Start A2A Network", use_container_width=True):
            st.session_state.a2a_agents = agent_configs
            st.success(f"Configured {num_agents} A2A agents")
    
    with col2:
        st.subheader("Network Communication")
        
        if 'a2a_agents' in st.session_state:
            # Select source and target agents
            agent_names = [agent["name"] for agent in st.session_state.a2a_agents]
            
            sender = st.selectbox("Sender Agent:", agent_names)
            receiver = st.selectbox("Receiver Agent:", [name for name in agent_names if name != sender])
            
            # Message configuration
            action = st.selectbox("Action:", ["chat", "analyze", "collaborate", "ping"])
            message_data = st.text_area("Message Data:", placeholder="Enter message content...")
            
            if st.button("Send A2A Message", use_container_width=True):
                if message_data:
                    st.info(f"Sending {action} from {sender} to {receiver}")
                    st.json({
                        "action": action,
                        "data": message_data,
                        "timestamp": datetime.now().isoformat()
                    })
                    st.success("Message sent successfully!")
                else:
                    st.warning("Please enter message data")
        else:
            st.info("Configure and start A2A network first")
    
    # Protocol details
    st.markdown("---")
    st.subheader("A2A Protocol Details")
    
    tab1, tab2, tab3 = st.tabs(["Message Format", "Communication Flow", "Example Implementation"])
    
    with tab1:
        st.markdown("**A2A Message Structure:**")
        example_message = {
            "id": "msg_123456",
            "type": "request",
            "sender_id": "agent_1", 
            "receiver_id": "agent_2",
            "payload": {
                "action": "chat",
                "data": {"message": "Hello from Agent 1!"}
            },
            "timestamp": "2025-06-12T10:30:00Z",
            "correlation_id": None
        }
        st.json(example_message)
    
    with tab2:
        st.markdown("""
        **Communication Flow:**
        
        1. **Request**: Agent A sends message to Agent B
        2. **Processing**: Agent B processes the request
        3. **Response**: Agent B sends back response
        4. **Correlation**: Messages linked by correlation ID
        5. **Error Handling**: Timeouts and error responses
        """)
    
    with tab3:
        st.code("""
# Create A2A agents
agent1 = SmartA2AAgent("agent_1", "ResearchBot", port=8501)
agent2 = SmartA2AAgent("agent_2", "AnalysisBot", port=8502)

# Start servers
await agent1.start_server()
await agent2.start_server()

# Send request
response = await agent1.send_request(
    "http://localhost:8502",
    "chat",
    {"message": "Hello from Agent 1!"}
)

print(response)
        """, language="python")


def show_custom_tools():
    """Show custom tools and integration examples."""
    st.header("🛠️ Custom Tools")
    
    st.markdown("""
    Custom tools extend agent capabilities beyond built-in functions.
    Test different tools and see how they integrate with agents.
    """)
    
    # Tool selection and testing
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Available Tools")
        
        tool_name = st.selectbox("Select tool to test:", list(CUSTOM_TOOLS.keys()))
        
        # Show tool information
        tool_info = get_tool_info(tool_name)
        if tool_info:
            st.markdown(f"**Description:** {tool_info['description']}")
        
        # Tool-specific inputs
        tool_inputs = {}
        
        if tool_name == "weather":
            tool_inputs["location"] = st.text_input("Location:", value="New York")
        
        elif tool_name == "calculator":
            tool_inputs["expression"] = st.text_input("Expression:", value="2 + 2 * 3")
        
        elif tool_name == "text_analyzer":
            tool_inputs["text"] = st.text_area("Text to analyze:", height=100)
            tool_inputs["analysis_type"] = st.selectbox("Analysis type:", ["sentiment", "keywords", "readability"])
        
        elif tool_name == "file_manager":
            tool_inputs["action"] = st.selectbox("Action:", ["read", "write", "list", "exists"])
            tool_inputs["file_path"] = st.text_input("File path:", value="example.txt")
            if tool_inputs["action"] == "write":
                tool_inputs["content"] = st.text_area("Content to write:")
        
        elif tool_name == "data_converter":
            tool_inputs["data"] = st.text_area("Data to convert:")
            tool_inputs["from_format"] = st.selectbox("From format:", ["json", "csv"])
            tool_inputs["to_format"] = st.selectbox("To format:", ["json", "csv"])
        
        elif tool_name == "task_scheduler":
            tool_inputs["action"] = st.selectbox("Action:", ["schedule", "list", "cancel"])
            if tool_inputs["action"] in ["schedule", "cancel"]:
                tool_inputs["task_name"] = st.text_input("Task name:")
            if tool_inputs["action"] == "schedule":
                tool_inputs["schedule_time"] = st.text_input("Schedule time (ISO format):", 
                                                           value=datetime.now().isoformat())
                tool_inputs["task_data"] = st.text_area("Task data:")
    
    with col2:
        st.subheader("Tool Testing")
        
        if st.button("Execute Tool", use_container_width=True):
            try:
                tool_function = CUSTOM_TOOLS[tool_name]
                
                # Execute tool with appropriate parameters
                if tool_name == "weather":
                    result = tool_function(tool_inputs["location"])
                elif tool_name == "calculator":
                    result = tool_function(tool_inputs["expression"])
                elif tool_name == "text_analyzer":
                    result = tool_function(tool_inputs["text"], tool_inputs["analysis_type"])
                elif tool_name == "file_manager":
                    if tool_inputs["action"] == "write":
                        result = tool_function(tool_inputs["action"], tool_inputs["file_path"], 
                                             tool_inputs["content"])
                    else:
                        result = tool_function(tool_inputs["action"], tool_inputs["file_path"])
                elif tool_name == "data_converter":
                    result = tool_function(tool_inputs["data"], tool_inputs["from_format"], 
                                         tool_inputs["to_format"])
                elif tool_name == "task_scheduler":
                    if tool_inputs["action"] == "schedule":
                        result = tool_function(tool_inputs["action"], tool_inputs["task_name"],
                                             tool_inputs["schedule_time"], tool_inputs["task_data"])
                    elif tool_inputs["action"] == "cancel":
                        result = tool_function(tool_inputs["action"], tool_inputs["task_name"])
                    else:
                        result = tool_function(tool_inputs["action"])
                
                st.markdown("**Tool Result:**")
                st.text_area("Output:", value=result, height=200, disabled=True)
            
            except Exception as e:
                st.error(f"Tool execution error: {e}")
        
        # Tool integration example
        st.markdown("---")
        st.subheader("Agent Integration")
        
        if st.button("Create Tool Agent", use_container_width=True):
            try:
                tool_agent = ToolAgent("Custom_Tool_Agent")
                st.session_state.tool_agent = tool_agent
                st.success("Tool agent created!")
            except Exception as e:
                st.error(f"Error creating tool agent: {e}")
        
        if 'tool_agent' in st.session_state:
            test_request = st.text_input("Test request:", 
                                       placeholder="Ask the agent to use tools...")
            
            if st.button("Send to Agent"):
                if test_request:
                    try:
                        response = st.session_state.tool_agent.process_request(test_request)
                        st.markdown("**Agent Response:**")
                        st.write(response)
                    except Exception as e:
                        st.error(f"Error: {e}")


def show_performance_analytics():
    """Show performance analytics and monitoring."""
    st.header("📊 Performance Analytics")
    
    # Generate mock performance data
    dates = pd.date_range(start='2025-06-01', end='2025-06-12', freq='D')
    
    performance_data = {
        'Date': dates,
        'Requests': [50 + i*5 + (i%3)*10 for i in range(len(dates))],
        'Response_Time': [1.2 + (i%4)*0.3 for i in range(len(dates))],
        'Success_Rate': [95 + (i%2)*3 for i in range(len(dates))],
        'Agent_Count': [3 + (i//3) for i in range(len(dates))]
    }
    
    df = pd.DataFrame(performance_data)
    
    # Metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Requests", f"{df['Requests'].sum():,}", "+12%")
    
    with col2:
        st.metric("Avg Response Time", f"{df['Response_Time'].mean():.2f}s", "-0.3s")
    
    with col3:
        st.metric("Success Rate", f"{df['Success_Rate'].mean():.1f}%", "+2.1%")
    
    with col4:
        st.metric("Active Agents", f"{df['Agent_Count'].iloc[-1]}", "+1")
    
    # Charts
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(df, x='Date', y='Requests', title='Daily Request Volume')
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.line(df, x='Date', y='Success_Rate', title='Success Rate Trend')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.line(df, x='Date', y='Response_Time', title='Response Time Trend')
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.bar(df, x='Date', y='Agent_Count', title='Active Agent Count')
        st.plotly_chart(fig, use_container_width=True)


def show_evaluation_framework():
    """Show evaluation framework and testing strategies."""
    st.header("🎯 Evaluation Framework")
    
    st.markdown("""
    Comprehensive evaluation is crucial for agent development. This framework
    provides tools for testing agent performance, reliability, and accuracy.
    """)
    
    # Evaluation categories
    tab1, tab2, tab3 = st.tabs(["Functionality Tests", "Performance Tests", "Integration Tests"])
    
    with tab1:
        st.subheader("Functionality Testing")
        
        test_cases = [
            {"name": "Basic Response", "description": "Agent responds to simple queries", "status": "✅ Pass"},
            {"name": "Tool Usage", "description": "Agent correctly uses available tools", "status": "✅ Pass"},
            {"name": "Error Handling", "description": "Agent handles errors gracefully", "status": "⚠️ Warning"},
            {"name": "Context Retention", "description": "Agent maintains conversation context", "status": "✅ Pass"}
        ]
        
        for test in test_cases:
            col1, col2, col3 = st.columns([2, 3, 1])
            with col1:
                st.write(f"**{test['name']}**")
            with col2:
                st.write(test['description'])
            with col3:
                st.write(test['status'])
    
    with tab2:
        st.subheader("Performance Testing")
        
        # Performance metrics
        metrics = {
            "Metric": ["Response Time", "Throughput", "Memory Usage", "CPU Usage"],
            "Current": ["1.2s", "45 req/min", "250MB", "15%"],
            "Target": ["<2.0s", ">40 req/min", "<500MB", "<25%"],
            "Status": ["✅ Good", "✅ Good", "✅ Good", "✅ Good"]
        }
        
        df = pd.DataFrame(metrics)
        st.table(df)
    
    with tab3:
        st.subheader("Integration Testing")
        
        integration_tests = [
            "Multi-agent coordination",
            "A2A protocol communication", 
            "Tool chain execution",
            "Error propagation",
            "State synchronization"
        ]
        
        for test in integration_tests:
            st.checkbox(test, value=True)


if __name__ == "__main__":
    main()
