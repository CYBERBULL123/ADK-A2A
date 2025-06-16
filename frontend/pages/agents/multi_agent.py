"""
Multi-Agent Systems page for creating and managing coordinated agents.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Dict, Any, List
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from frontend.utils.ui_components import ui, charts

def render():
    """Render the multi-agent systems page."""
    ui.header(
        "Multi-Agent Systems",
        "Build complex workflows with coordinated agents",
        "🔗"
    )
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏗️ Workflow Builder",
        "👥 Agent Teams",
        "📊 Coordination Analytics",
        "🎯 Use Cases"
    ])
    
    with tab1:
        render_workflow_builder()
    
    with tab2:
        render_agent_teams()
    
    with tab3:
        render_coordination_analytics()
    
    with tab4:
        render_use_cases()

def render_workflow_builder():
    """Render workflow builder interface."""
    st.subheader("🏗️ Multi-Agent Workflow Builder")
    
    # Workflow creation form
    with st.form("workflow_creation"):
        col1, col2 = st.columns(2)
        
        with col1:
            workflow_name = st.text_input(
                "Workflow Name",
                placeholder="Customer Support Pipeline",
                help="Give your workflow a descriptive name"
            )
            
            workflow_type = st.selectbox(
                "Workflow Type",
                ["Sequential", "Parallel", "Hierarchical", "Event-Driven"],
                help="Choose the coordination pattern"
            )
            
            priority = st.selectbox(
                "Priority Level",
                ["Low", "Medium", "High", "Critical"],
                index=1
            )
        
        with col2:
            description = st.text_area(
                "Description",
                placeholder="Describe what this workflow accomplishes...",
                height=100
            )
            
            max_agents = st.number_input(
                "Maximum Agents",
                min_value=2,
                max_value=50,
                value=5,
                help="Maximum number of agents in this workflow"
            )
            
            timeout = st.number_input(
                "Timeout (minutes)",
                min_value=1,
                max_value=1440,
                value=30
            )
        
        # Agent configuration
        st.markdown("### 🤖 Agent Configuration")
        
        num_agents = st.number_input(
            "Number of Agents",
            min_value=2,
            max_value=max_agents,
            value=3
        )
        
        # Create agent configuration for each agent
        agent_configs = []
        for i in range(num_agents):
            with st.expander(f"⚙️ Agent {i+1} Configuration"):
                col_a, col_b = st.columns(2)
                with col_a:
                    agent_name = st.text_input(f"Agent {i+1} Name", f"Agent_{i+1}", key=f"agent_name_{i}")
                    agent_role = st.selectbox(
                        f"Agent {i+1} Role",
                        ["Coordinator", "Worker", "Validator", "Notifier", "Custom"],
                        key=f"agent_role_{i}"
                    )
                with col_b:
                    agent_model = st.selectbox(
                        f"Agent {i+1} Model",
                        ["gemini-pro", "gemini-1.5-pro", "gpt-4", "claude-3"],
                        key=f"agent_model_{i}"
                    )
                    agent_tools = st.multiselect(
                        f"Agent {i+1} Tools",
                        ["Web Search", "Calculator", "Email", "Database", "File System"],
                        key=f"agent_tools_{i}"
                    )
                
                agent_configs.append({
                    "name": agent_name,
                    "role": agent_role,
                    "model": agent_model,
                    "tools": agent_tools
                })
        
        submitted = st.form_submit_button("🚀 Create Workflow", use_container_width=True)
        
        if submitted and workflow_name:
            create_workflow(workflow_name, workflow_type, description, agent_configs, priority, timeout)

def create_workflow(name: str, workflow_type: str, description: str, 
                   agents: List[Dict], priority: str, timeout: int):
    """Create a new multi-agent workflow."""
    
    # Initialize workflows in session state
    if "workflows" not in st.session_state:
        st.session_state.workflows = []
    
    workflow = {
        "id": f"workflow_{len(st.session_state.workflows) + 1}",
        "name": name,
        "type": workflow_type,
        "description": description,
        "agents": agents,
        "priority": priority,
        "timeout": timeout,
        "status": "Active",
        "created_at": datetime.now().isoformat(),
        "executions": 0,
        "success_rate": 100.0
    }
    
    st.session_state.workflows.append(workflow)
    
    st.success(f"✅ Workflow '{name}' created successfully!")
    
    # Display workflow summary
    with st.expander("📋 Workflow Summary"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Name:** {name}")
            st.write(f"**Type:** {workflow_type}")
            st.write(f"**Priority:** {priority}")
        with col2:
            st.write(f"**Agents:** {len(agents)}")
            st.write(f"**Timeout:** {timeout} minutes")
            st.write(f"**Status:** Active")

def render_agent_teams():
    """Render agent teams management."""
    st.subheader("👥 Agent Teams Management")
    
    # Check if workflows exist
    if "workflows" not in st.session_state or not st.session_state.workflows:
        ui.info_box(
            "No Workflows Available",
            "Create a workflow first to manage agent teams!",
            "info"
        )
        return
    
    # Workflow selection
    workflow_options = {wf["name"]: wf for wf in st.session_state.workflows}
    selected_workflow = st.selectbox(
        "Select Workflow",
        list(workflow_options.keys())
    )
    
    if selected_workflow:
        workflow = workflow_options[selected_workflow]
        
        # Display workflow info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ui.metric_card("Total Agents", len(workflow["agents"]))
        with col2:
            ui.metric_card("Workflow Type", workflow["type"])
        with col3:
            ui.metric_card("Priority", workflow["priority"])
        with col4:
            ui.metric_card("Status", workflow["status"])
        
        st.markdown("---")
        
        # Agent team visualization
        st.markdown("### 🔗 Agent Network")
        render_agent_network(workflow["agents"])
        
        # Agent details
        st.markdown("### 👥 Team Members")
        for i, agent in enumerate(workflow["agents"]):
            with st.expander(f"🤖 {agent['name']} - {agent['role']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Role:** {agent['role']}")
                    st.write(f"**Model:** {agent['model']}")
                with col2:
                    st.write(f"**Tools:** {', '.join(agent['tools']) if agent['tools'] else 'None'}")
                
                # Agent controls
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button(f"▶️ Start {agent['name']}", key=f"start_{i}"):
                        st.success(f"Started {agent['name']}")
                with col_b:
                    if st.button(f"⏸️ Pause {agent['name']}", key=f"pause_{i}"):
                        st.warning(f"Paused {agent['name']}")
                with col_c:
                    if st.button(f"🔄 Restart {agent['name']}", key=f"restart_{i}"):
                        st.info(f"Restarted {agent['name']}")

def render_agent_network(agents: List[Dict]):
    """Render agent network visualization."""
    # Create a simple network representation
    network_data = {
        "Agent": [agent["name"] for agent in agents],
        "Role": [agent["role"] for agent in agents],
        "Model": [agent["model"] for agent in agents],
        "Tools Count": [len(agent["tools"]) for agent in agents]
    }
    
    df = pd.DataFrame(network_data)
    
    # Create a scatter plot representing the network
    charts.scatter_plot(
        df,
        x="Tools Count",
        y="Role",
        title="Agent Network Overview",
        color="Model",
        size="Tools Count"
    )

def render_coordination_analytics():
    """Render coordination analytics."""
    st.subheader("📊 Coordination Analytics")
    
    # Generate mock analytics data
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ui.metric_card("Active Workflows", "8", "+2 today")
    with col2:
        ui.metric_card("Total Agents", "24", "+6 this week")
    with col3:
        ui.metric_card("Success Rate", "94.5%", "+2.1% this month")
    with col4:
        ui.metric_card("Avg Response Time", "2.3s", "-0.4s faster")
    
    # Charts
    st.markdown("### 📈 Performance Trends")
    
    # Generate sample data
    dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='H')
    performance_data = pd.DataFrame({
        'Timestamp': dates,
        'Workflow Executions': [random.randint(5, 25) for _ in dates],
        'Success Rate': [random.uniform(85, 100) for _ in dates],
        'Response Time': [random.uniform(1.0, 5.0) for _ in dates]
    })
    
    chart_tab1, chart_tab2 = st.tabs(["📊 Execution Volume", "⚡ Performance Metrics"])
    
    with chart_tab1:
        charts.line_chart(
            performance_data,
            x='Timestamp',
            y='Workflow Executions',
            title="Workflow Executions Over Time"
        )
    
    with chart_tab2:
        charts.line_chart(
            performance_data,
            x='Timestamp',
            y='Success Rate',
            title="Success Rate Trend"
        )

def render_use_cases():
    """Render multi-agent use cases."""
    st.subheader("🎯 Multi-Agent Use Cases")
    
    use_cases = [
        {
            "title": "Customer Support Pipeline",
            "description": "Automated customer inquiry handling with escalation",
            "agents": ["Classifier", "Responder", "Escalator", "Quality Checker"],
            "complexity": "Medium",
            "icon": "🎧"
        },
        {
            "title": "Content Creation Workflow",
            "description": "Research, write, edit, and publish content automatically",
            "agents": ["Researcher", "Writer", "Editor", "Publisher"],
            "complexity": "High",
            "icon": "✍️"
        },
        {
            "title": "Data Processing Pipeline",
            "description": "Extract, transform, validate, and load data",
            "agents": ["Extractor", "Transformer", "Validator", "Loader"],
            "complexity": "Medium",
            "icon": "📊"
        },
        {
            "title": "E-commerce Order Processing",
            "description": "Process orders from payment to fulfillment",
            "agents": ["Payment Processor", "Inventory Manager", "Shipping Coordinator", "Notifier"],
            "complexity": "High",
            "icon": "🛒"
        },
        {
            "title": "Code Review System",
            "description": "Automated code analysis and review process",
            "agents": ["Code Analyzer", "Security Checker", "Performance Reviewer", "Documentation Checker"],
            "complexity": "High",
            "icon": "💻"
        },
        {
            "title": "Social Media Management",
            "description": "Schedule, post, and monitor social media content",
            "agents": ["Content Scheduler", "Post Publisher", "Engagement Monitor", "Analytics Reporter"],
            "complexity": "Medium",
            "icon": "📱"
        }
    ]
    
    # Display use cases in a grid
    cols = st.columns(2)
    for i, use_case in enumerate(use_cases):
        col_idx = i % 2
        with cols[col_idx]:
            with st.container():
                st.markdown(f"### {use_case['icon']} {use_case['title']}")
                st.write(use_case['description'])
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**Complexity:** {use_case['complexity']}")
                with col_b:
                    st.write(f"**Agents:** {len(use_case['agents'])}")
                
                with st.expander("👥 Agent Roles"):
                    for agent in use_case['agents']:
                        st.write(f"• {agent}")
                
                if st.button(f"🚀 Create Template", key=f"template_{i}", use_container_width=True):
                    create_template_workflow(use_case)
                
                st.markdown("---")

def create_template_workflow(use_case: Dict):
    """Create a workflow from a template."""
    st.success(f"✅ Template '{use_case['title']}' created!")
    st.info("This would create a pre-configured workflow with the specified agents and settings.")

if __name__ == "__main__":
    render()
