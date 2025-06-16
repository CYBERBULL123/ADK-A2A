"""
Custom Tools page for creating and managing custom tools.
"""

import streamlit as st
import json
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from frontend.utils.ui_components import ui

def render():
    """Render the custom tools page."""
    ui.header(
        "Custom Tools",
        "Create and manage your own tools for agents",
        "🔧"
    )
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "🛠️ Tool Builder",
        "📚 My Tools",
        "🧪 Tool Tester",
        "📦 Tool Templates"
    ])
    
    with tab1:
        render_tool_builder()
    
    with tab2:
        render_my_tools()
    
    with tab3:
        render_tool_tester()
    
    with tab4:
        render_tool_templates()

def render_tool_builder():
    """Render custom tool builder interface."""
    st.subheader("🛠️ Custom Tool Builder")
    
    # Tool creation form
    with st.form("custom_tool_builder"):
        # Basic information
        st.markdown("### 📋 Basic Information")
        col1, col2 = st.columns(2)
        
        with col1:
            tool_name = st.text_input(
                "Tool Name",
                placeholder="my_awesome_tool",
                help="Use snake_case for function names"
            )
            
            tool_display_name = st.text_input(
                "Display Name",
                placeholder="My Awesome Tool"
            )
            
            tool_category = st.selectbox(
                "Category",
                ["Data Processing", "Web APIs", "File Operations", "AI/ML", "Communication", "Utility", "Custom"]
            )
        
        with col2:
            tool_description = st.text_area(
                "Description",
                placeholder="Describe what your tool does...",
                height=100
            )
            
            tool_version = st.text_input(
                "Version",
                value="1.0.0",
                help="Semantic versioning (e.g., 1.0.0)"
            )
            
            tool_author = st.text_input(
                "Author",
                placeholder="Your name"
            )
        
        # Parameters definition
        st.markdown("### ⚙️ Parameters")
        
        num_params = st.number_input(
            "Number of Parameters",
            min_value=0,
            max_value=20,
            value=1
        )
        
        parameters = []
        for i in range(num_params):
            with st.expander(f"📝 Parameter {i+1}"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    param_name = st.text_input(f"Parameter {i+1} Name", key=f"param_name_{i}")
                    param_type = st.selectbox(
                        f"Parameter {i+1} Type",
                        ["string", "integer", "float", "boolean", "list", "dict"],
                        key=f"param_type_{i}"
                    )
                    param_required = st.checkbox(f"Required", key=f"param_required_{i}")
                
                with col_b:
                    param_description = st.text_area(
                        f"Parameter {i+1} Description",
                        height=60,
                        key=f"param_desc_{i}"
                    )
                    param_default = st.text_input(
                        f"Default Value (optional)",
                        key=f"param_default_{i}"
                    )
                
                parameters.append({
                    "name": param_name,
                    "type": param_type,
                    "description": param_description,
                    "required": param_required,
                    "default": param_default if param_default else None
                })
        
        # Tool implementation
        st.markdown("### 💻 Implementation")
        
        implementation_type = st.selectbox(
            "Implementation Type",
            ["Python Function", "API Endpoint", "Shell Command", "Custom Script"]
        )
        
        if implementation_type == "Python Function":
            code_template = generate_python_template(tool_name, parameters)
            tool_code = st.text_area(
                "Python Code",
                value=code_template,
                height=300,
                help="Write your Python function implementation"
            )
        
        elif implementation_type == "API Endpoint":
            col_x, col_y = st.columns(2)
            with col_x:
                api_url = st.text_input("API URL", placeholder="https://api.example.com/endpoint")
                api_method = st.selectbox("HTTP Method", ["GET", "POST", "PUT", "DELETE"])
            with col_y:
                api_headers = st.text_area("Headers (JSON)", value='{"Content-Type": "application/json"}')
                api_auth = st.selectbox("Authentication", ["None", "API Key", "Bearer Token", "Basic Auth"])
        
        elif implementation_type == "Shell Command":
            shell_command = st.text_area(
                "Shell Command",
                placeholder="echo 'Hello from custom tool'",
                help="Use {param_name} for parameter substitution"
            )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            col_z, col_w = st.columns(2)
            with col_z:
                timeout = st.number_input("Timeout (seconds)", value=30)
                retry_attempts = st.number_input("Retry Attempts", value=0)
                async_execution = st.checkbox("Async Execution", value=False)
            with col_w:
                cache_results = st.checkbox("Cache Results", value=False)
                log_usage = st.checkbox("Log Usage", value=True)
                validate_inputs = st.checkbox("Validate Inputs", value=True)
        
        submitted = st.form_submit_button("🚀 Create Tool", use_container_width=True)
        
        if submitted and tool_name and tool_display_name:
            create_custom_tool(
                tool_name, tool_display_name, tool_description, tool_category,
                tool_version, tool_author, parameters, implementation_type,
                locals().get('tool_code', ''), timeout, async_execution
            )

def generate_python_template(tool_name: str, parameters: List[Dict]) -> str:
    """Generate Python function template."""
    
    param_list = []
    param_docs = []
    
    for param in parameters:
        param_signature = param['name']
        if not param['required'] and param['default']:
            param_signature += f"={repr(param['default'])}"
        elif not param['required']:
            param_signature += "=None"
        
        param_list.append(param_signature)
        param_docs.append(f"    {param['name']} ({param['type']}): {param['description']}")
    
    param_str = ", ".join(param_list)
    docs_str = "\n".join(param_docs)
    
    template = f'''def {tool_name}({param_str}):
    """
    {tool_name.replace('_', ' ').title()} tool implementation.
    
    Args:
{docs_str}
    
    Returns:
        dict: Result of the tool execution
    """
    try:
        # TODO: Implement your tool logic here
        result = {{
            "success": True,
            "data": "Tool executed successfully",
            "message": "Replace this with your actual implementation"
        }}
        
        return result
        
    except Exception as e:
        return {{
            "success": False,
            "error": str(e),
            "message": "Tool execution failed"
        }}'''
    
    return template

def create_custom_tool(name: str, display_name: str, description: str, category: str,
                      version: str, author: str, parameters: List[Dict], 
                      impl_type: str, code: str, timeout: int, async_exec: bool):
    """Create a new custom tool."""
    
    # Initialize custom tools in session state
    if "custom_tools" not in st.session_state:
        st.session_state.custom_tools = []
    
    tool = {
        "id": f"tool_{len(st.session_state.custom_tools) + 1}",
        "name": name,
        "display_name": display_name,
        "description": description,
        "category": category,
        "version": version,
        "author": author,
        "parameters": parameters,
        "implementation_type": impl_type,
        "code": code,
        "timeout": timeout,
        "async_execution": async_exec,
        "created_at": datetime.now().isoformat(),
        "usage_count": 0,
        "status": "Active"
    }
    
    st.session_state.custom_tools.append(tool)
    
    st.success(f"✅ Custom tool '{display_name}' created successfully!")
    
    # Display tool summary
    with st.expander("📋 Tool Summary"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Name:** {name}")
            st.write(f"**Category:** {category}")
            st.write(f"**Version:** {version}")
        with col2:
            st.write(f"**Parameters:** {len(parameters)}")
            st.write(f"**Implementation:** {impl_type}")
            st.write(f"**Author:** {author}")

def render_my_tools():
    """Render custom tools management."""
    st.subheader("📚 My Custom Tools")
    
    # Check if tools exist
    if "custom_tools" not in st.session_state or not st.session_state.custom_tools:
        ui.info_box(
            "No Custom Tools",
            "Create your first custom tool using the Tool Builder!",
            "info"
        )
        return
    
    # Tools overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ui.metric_card("Total Tools", len(st.session_state.custom_tools))
    with col2:
        active_tools = sum(1 for tool in st.session_state.custom_tools if tool["status"] == "Active")
        ui.metric_card("Active Tools", active_tools)
    with col3:
        total_usage = sum(tool["usage_count"] for tool in st.session_state.custom_tools)
        ui.metric_card("Total Usage", total_usage)
    with col4:
        categories = set(tool["category"] for tool in st.session_state.custom_tools)
        ui.metric_card("Categories", len(categories))
    
    st.markdown("---")
    
    # Tools list
    for tool in st.session_state.custom_tools:
        with st.expander(f"🔧 {tool['display_name']} - {tool['category']}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Description:** {tool['description']}")
                st.write(f"**Version:** {tool['version']} | **Author:** {tool['author']}")
                st.write(f"**Parameters:** {len(tool['parameters'])} | **Usage:** {tool['usage_count']} times")
                
                # Parameters list
                if tool['parameters']:
                    st.write("**Parameters:**")
                    for param in tool['parameters']:
                        required_text = "Required" if param['required'] else "Optional"
                        st.write(f"  • `{param['name']}` ({param['type']}) - {required_text}")
            
            with col2:
                st.write(f"**Status:** {tool['status']}")
                st.write(f"**Created:** {tool['created_at'][:10]}")
                
                # Actions
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("📝 Edit", key=f"edit_{tool['id']}"):
                        st.info("Edit functionality coming soon!")
                with col_b:
                    if st.button("🗑️ Delete", key=f"delete_{tool['id']}"):
                        delete_custom_tool(tool['id'])
                
                if st.button("📋 View Code", key=f"code_{tool['id']}", use_container_width=True):
                    show_tool_code(tool)

def delete_custom_tool(tool_id: str):
    """Delete a custom tool."""
    st.session_state.custom_tools = [
        tool for tool in st.session_state.custom_tools 
        if tool['id'] != tool_id
    ]
    st.success("🗑️ Tool deleted successfully!")
    st.rerun()

def show_tool_code(tool: Dict):
    """Show tool implementation code."""
    st.markdown(f"### 💻 {tool['display_name']} Implementation")
    
    if tool['implementation_type'] == "Python Function":
        ui.code_block(tool['code'], "python", "Python Implementation")
    else:
        st.write(f"**Implementation Type:** {tool['implementation_type']}")
        st.code(tool.get('code', 'No code available'), language="text")

def render_tool_tester():
    """Render tool testing interface."""
    st.subheader("🧪 Tool Tester")
    
    # Check if tools exist
    if "custom_tools" not in st.session_state or not st.session_state.custom_tools:
        ui.info_box(
            "No Tools to Test",
            "Create custom tools first to test them here!",
            "info"
        )
        return
    
    # Tool selection
    tool_options = {tool["display_name"]: tool for tool in st.session_state.custom_tools}
    selected_tool_name = st.selectbox(
        "Select Tool to Test",
        list(tool_options.keys())
    )
    
    if selected_tool_name:
        tool = tool_options[selected_tool_name]
        
        st.markdown(f"### 🔧 Testing: {tool['display_name']}")
        st.write(f"**Description:** {tool['description']}")
        
        # Parameter inputs
        if tool['parameters']:
            st.markdown("#### 📝 Parameters")
            
            test_params = {}
            for param in tool['parameters']:
                param_key = f"test_{tool['id']}_{param['name']}"
                
                if param['type'] == 'string':
                    value = st.text_input(
                        f"{param['name']} ({param['type']})",
                        value=param.get('default', ''),
                        help=param['description'],
                        key=param_key
                    )
                elif param['type'] == 'integer':
                    value = st.number_input(
                        f"{param['name']} ({param['type']})",
                        value=int(param.get('default', 0)) if param.get('default') else 0,
                        help=param['description'],
                        key=param_key
                    )
                elif param['type'] == 'float':
                    value = st.number_input(
                        f"{param['name']} ({param['type']})",
                        value=float(param.get('default', 0.0)) if param.get('default') else 0.0,
                        help=param['description'],
                        key=param_key
                    )
                elif param['type'] == 'boolean':
                    value = st.checkbox(
                        f"{param['name']} ({param['type']})",
                        value=bool(param.get('default', False)) if param.get('default') else False,
                        help=param['description'],
                        key=param_key
                    )
                else:
                    value = st.text_area(
                        f"{param['name']} ({param['type']})",
                        value=str(param.get('default', '')) if param.get('default') else '',
                        help=param['description'],
                        key=param_key
                    )
                
                test_params[param['name']] = value
            
            # Test execution
            if st.button("🧪 Run Test", use_container_width=True):
                execute_tool_test(tool, test_params)
        else:
            # No parameters, direct execution
            if st.button("🧪 Run Test", use_container_width=True):
                execute_tool_test(tool, {})

def execute_tool_test(tool: Dict, params: Dict):
    """Execute tool test with given parameters."""
    
    st.markdown("### 📊 Test Results")
    
    # Mock execution (replace with actual execution logic)
    try:
        # Simulate tool execution
        result = {
            "success": True,
            "execution_time": "0.245s",
            "output": {
                "message": f"Tool '{tool['display_name']}' executed successfully",
                "parameters_used": params,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Update usage count
        for i, t in enumerate(st.session_state.custom_tools):
            if t['id'] == tool['id']:
                st.session_state.custom_tools[i]['usage_count'] += 1
                break
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.success("✅ Test Passed")
            st.write(f"**Execution Time:** {result['execution_time']}")
        with col2:
            st.write(f"**Status:** {'Success' if result['success'] else 'Failed'}")
            st.write(f"**Timestamp:** {result['output']['timestamp'][:19]}")
        
        # Output details
        st.markdown("#### 📋 Output")
        st.json(result['output'])
        
    except Exception as e:
        st.error(f"❌ Test Failed: {str(e)}")

def render_tool_templates():
    """Render tool templates."""
    st.subheader("📦 Tool Templates")
    
    templates = [
        {
            "name": "Web API Client",
            "description": "Template for creating HTTP API client tools",
            "category": "Web APIs",
            "parameters": ["url", "method", "headers", "data"],
            "complexity": "Medium"
        },
        {
            "name": "File Processor",
            "description": "Template for file reading, writing, and processing",
            "category": "File Operations",
            "parameters": ["file_path", "operation", "content"],
            "complexity": "Easy"
        },
        {
            "name": "Data Transformer",
            "description": "Template for data transformation and manipulation",
            "category": "Data Processing",
            "parameters": ["input_data", "transformation_type", "options"],
            "complexity": "Medium"
        },
        {
            "name": "AI Model Wrapper",
            "description": "Template for wrapping AI/ML model endpoints",
            "category": "AI/ML",
            "parameters": ["model_name", "input_data", "parameters"],
            "complexity": "Hard"
        },
        {
            "name": "Database Query",
            "description": "Template for database query execution",
            "category": "Data Processing",
            "parameters": ["query", "database", "parameters"],
            "complexity": "Medium"
        },
        {
            "name": "Email Sender",
            "description": "Template for sending emails",
            "category": "Communication",
            "parameters": ["to", "subject", "body", "attachments"],
            "complexity": "Easy"
        }
    ]
    
    # Display templates in grid
    cols = st.columns(2)
    for i, template in enumerate(templates):
        col_idx = i % 2
        with cols[col_idx]:
            with st.container():
                st.markdown(f"### 📦 {template['name']}")
                st.write(template['description'])
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**Category:** {template['category']}")
                    st.write(f"**Complexity:** {template['complexity']}")
                with col_b:
                    st.write(f"**Parameters:** {len(template['parameters'])}")
                
                with st.expander("📋 Parameter List"):
                    for param in template['parameters']:
                        st.write(f"• {param}")
                
                if st.button(f"🚀 Use Template", key=f"template_{i}", use_container_width=True):
                    use_template(template)
                
                st.markdown("---")

def use_template(template: Dict):
    """Use a template to create a new tool."""
    st.success(f"✅ Template '{template['name']}' loaded!")
    st.info("Template will be loaded in the Tool Builder. This feature is coming soon!")

if __name__ == "__main__":
    render()
