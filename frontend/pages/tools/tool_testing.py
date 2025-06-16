"""
Tool Testing page.
"""

import streamlit as st
import json
import time
import pandas as pd
from datetime import datetime
from frontend.utils.ui_components import ui

def render():
    """Render the tool testing page."""
    ui.header(
        "Tool Testing",
        "Test and validate your tools",
        "🧪"
    )
    
    # Tool selection and testing interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔧 Tool Selection")
        
        # Available tools (mock data)
        available_tools = {
            "Web Scraper": {
                "description": "Extract content from web pages",
                "parameters": ["url", "selector", "timeout"],
                "type": "data_extraction"
            },
            "Data Analyzer": {
                "description": "Analyze datasets and generate insights",
                "parameters": ["data_file", "analysis_type", "output_format"],
                "type": "analytics"
            },
            "Email Sender": {
                "description": "Send emails with attachments",
                "parameters": ["to", "subject", "body", "attachments"],
                "type": "communication"
            },
            "File Converter": {
                "description": "Convert files between formats",
                "parameters": ["input_file", "output_format", "quality"],
                "type": "utility"
            },
            "API Client": {
                "description": "Make HTTP requests to APIs",
                "parameters": ["endpoint", "method", "headers", "payload"],
                "type": "integration"
            }
        }
        
        selected_tool = st.selectbox(
            "Select Tool to Test",
            options=list(available_tools.keys()),
            key="tool_selector"
        )
        
        if selected_tool:
            tool_info = available_tools[selected_tool]
            st.write(f"**Description:** {tool_info['description']}")
            st.write(f"**Type:** {tool_info['type']}")
            st.write(f"**Parameters:** {', '.join(tool_info['parameters'])}")
        
        st.markdown("---")
        
        # Test configuration
        st.subheader("⚙️ Test Configuration")
        
        test_mode = st.radio(
            "Test Mode",
            ["Single Test", "Batch Test", "Load Test"],
            key="test_mode"
        )
        
        if test_mode == "Load Test":
            concurrent_tests = st.slider("Concurrent Tests", 1, 100, 10)
            test_duration = st.slider("Duration (seconds)", 10, 300, 60)
        elif test_mode == "Batch Test":
            batch_size = st.slider("Batch Size", 1, 50, 5)
    
    with col2:
        st.subheader("📝 Test Parameters")
        
        if selected_tool:
            tool_info = available_tools[selected_tool]
            
            # Dynamic parameter input based on selected tool
            test_params = {}
            
            if selected_tool == "Web Scraper":
                test_params["url"] = st.text_input("URL", value="https://example.com")
                test_params["selector"] = st.text_input("CSS Selector", value="h1")
                test_params["timeout"] = st.number_input("Timeout (seconds)", value=30)
                
            elif selected_tool == "Data Analyzer":
                uploaded_file = st.file_uploader("Upload Data File", type=['csv', 'json', 'xlsx'])
                test_params["analysis_type"] = st.selectbox("Analysis Type", ["summary", "correlation", "trend"])
                test_params["output_format"] = st.selectbox("Output Format", ["json", "csv", "html"])
                
            elif selected_tool == "Email Sender":
                test_params["to"] = st.text_input("To Email", value="test@example.com")
                test_params["subject"] = st.text_input("Subject", value="Test Email")
                test_params["body"] = st.text_area("Email Body", value="This is a test email.")
                test_params["attachments"] = st.text_input("Attachments (comma-separated)")
                
            elif selected_tool == "File Converter":
                uploaded_file = st.file_uploader("Upload File to Convert")
                test_params["output_format"] = st.selectbox("Output Format", ["pdf", "docx", "txt", "html"])
                test_params["quality"] = st.selectbox("Quality", ["high", "medium", "low"])
                
            elif selected_tool == "API Client":
                test_params["endpoint"] = st.text_input("API Endpoint", value="https://api.example.com/data")
                test_params["method"] = st.selectbox("HTTP Method", ["GET", "POST", "PUT", "DELETE"])
                test_params["headers"] = st.text_area("Headers (JSON)", value='{"Content-Type": "application/json"}')
                test_params["payload"] = st.text_area("Payload (JSON)", value='{"key": "value"}')
            
            st.markdown("---")
            
            # Test execution
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🧪 Run Test", use_container_width=True, type="primary"):
                    run_tool_test(selected_tool, test_params, test_mode)
            
            with col2:
                if st.button("💾 Save Test", use_container_width=True):
                    save_test_configuration(selected_tool, test_params)
            
            with col3:
                if st.button("📋 Load Test", use_container_width=True):
                    load_test_configuration()
    
    st.markdown("---")
    
    # Test results section
    st.subheader("� Test Results")
    
    # Display test results in session state
    if "test_results" not in st.session_state:
        st.session_state.test_results = []
    if st.session_state.test_results:
        # Results summary
        col1, col2, col3, col4 = st.columns(4)
        
        latest_results = st.session_state.test_results[-10:]  # Last 10 tests
        success_rate = sum(1 for r in latest_results if r['status'] == 'Success') / len(latest_results) * 100
        avg_duration = sum(r['duration'] for r in latest_results) / len(latest_results)
        
        with col1:
            ui.metric_card(
                "Success Rate",
                f"{success_rate:.1f}%",
                f"Last {len(latest_results)} tests",
                icon="✅"
            )
        
        with col2:
            ui.metric_card(
                "Avg Duration",
                f"{avg_duration:.2f}s",
                "Response time",
                icon="⏱️"
            )
        
        with col3:
            ui.metric_card(
                "Total Tests",
                str(len(st.session_state.test_results)),
                "All time",
                icon="🧪"
            )
        
        with col4:
            failed_tests = sum(1 for r in st.session_state.test_results if r['status'] == 'Failed')
            ui.metric_card(
                "Failed Tests",
                str(failed_tests),
                "Needs attention",
                icon="❌"
            )
        
        # Detailed results table
        st.subheader("📋 Detailed Test Results")
        
        results_df = pd.DataFrame(st.session_state.test_results)
        if not results_df.empty:
            # Format the DataFrame for display
            display_df = results_df[['timestamp', 'tool', 'status', 'duration', 'message']].copy()
            display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df = display_df.sort_values('timestamp', ascending=False)
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Export functionality
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("📥 Export Results"):
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"tool_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    else:
        st.info("No test results yet. Run a test to see results here.")
    
    # Test history and analytics
    if st.session_state.test_results:
        st.markdown("---")
        st.subheader("📈 Test Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Success rate over time
            import plotly.express as px
            
            results_df = pd.DataFrame(st.session_state.test_results)
            results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
            results_df['success'] = results_df['status'] == 'Success'
            
            # Group by hour and calculate success rate
            hourly_success = results_df.set_index('timestamp').resample('H')['success'].mean().reset_index()
            
            fig = px.line(
                hourly_success,
                x='timestamp',
                y='success',
                title="Success Rate Over Time",
                labels={'success': 'Success Rate', 'timestamp': 'Time'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Duration distribution by tool
            fig2 = px.box(
                results_df,
                x='tool',
                y='duration',
                title="Duration Distribution by Tool"
            )
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)

def run_tool_test(tool_name: str, params: dict, test_mode: str):
    """Simulate running a tool test."""
    with st.spinner(f"Running {test_mode.lower()} for {tool_name}..."):
        # Simulate test execution
        time.sleep(2)  # Simulate processing time
        
        # Generate mock test result
        import random
        success = random.choice([True, True, True, False])  # 75% success rate
        duration = random.uniform(0.5, 3.0)
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'tool': tool_name,
            'status': 'Success' if success else 'Failed',
            'duration': duration,
            'message': 'Test completed successfully' if success else 'Test failed: Connection timeout',
            'parameters': params
        }
        
        # Store result in session state
        if "test_results" not in st.session_state:
            st.session_state.test_results = []
        
        st.session_state.test_results.append(result)
        
        # Show result
        if success:
            st.success(f"✅ Test completed successfully in {duration:.2f}s")
        else:
            st.error(f"❌ Test failed: {result['message']}")

def save_test_configuration(tool_name: str, params: dict):
    """Save test configuration for later use."""
    config = {
        'tool': tool_name,
        'parameters': params,
        'saved_at': datetime.now().isoformat()
    }
    
    # In a real app, this would save to a database or file
    st.success(f"✅ Test configuration saved for {tool_name}")

def load_test_configuration():
    """Load a previously saved test configuration."""
    # In a real app, this would load from a database or file
    st.info("📋 Test configuration loaded")

if __name__ == "__main__":
    render()
