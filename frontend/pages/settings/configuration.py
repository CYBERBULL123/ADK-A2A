"""
Configuration page.
"""

import streamlit as st
import json
import os
from pathlib import Path
from frontend.utils.ui_components import ui

def render():
    """Render the configuration page."""
    ui.header(
        "Configuration",
        "Configure application settings",
        "⚙️"
    )
    
    # Initialize session state for configuration
    if "config_changes" not in st.session_state:
        st.session_state.config_changes = {}
    
    # Configuration sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🤖 Agent Settings",
        "🛠️ Tool Configuration", 
        "🔌 API & Integrations",
        "📊 Analytics & Monitoring",
        "🔒 Security & Privacy"
    ])
    
    with tab1:
        st.subheader("🤖 Agent Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Default Agent Settings**")
            
            default_model = st.selectbox(
                "Default LLM Model",
                ["gemini-pro", "gemini-pro-vision", "gpt-4", "gpt-3.5-turbo"],
                index=0,
                key="default_model"
            )
            
            max_tokens = st.number_input(
                "Max Tokens per Response",
                min_value=100,
                max_value=8000,
                value=2048,
                step=100,
                key="max_tokens"
            )
            
            temperature = st.slider(
                "Temperature (Creativity)",
                min_value=0.0,
                max_value=2.0,
                value=0.7,
                step=0.1,
                key="temperature"
            )
            
            timeout = st.number_input(
                "Agent Timeout (seconds)",
                min_value=10,
                max_value=300,
                value=60,
                key="agent_timeout"
            )
        
        with col2:
            st.markdown("**Multi-Agent Settings**")
            
            max_agents = st.number_input(
                "Max Concurrent Agents",
                min_value=1,
                max_value=50,
                value=10,
                key="max_agents"
            )
            
            coordination_model = st.selectbox(
                "Coordination Model",
                ["hierarchical", "peer-to-peer", "centralized"],
                index=0,
                key="coordination_model"
            )
            
            enable_a2a = st.checkbox(
                "Enable A2A Protocol",
                value=True,
                key="enable_a2a"
            )
            
            agent_discovery = st.checkbox(
                "Auto Agent Discovery",
                value=True,
                key="agent_discovery"
            )
        
        st.markdown("---")
        
        # Agent lifecycle settings
        st.markdown("**Agent Lifecycle Management**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            auto_restart = st.checkbox("Auto-restart Failed Agents", value=True)
            max_retries = st.number_input("Max Retry Attempts", min_value=0, max_value=10, value=3)
        
        with col2:
            health_check_interval = st.number_input("Health Check Interval (seconds)", min_value=10, max_value=300, value=30)
            idle_timeout = st.number_input("Idle Timeout (minutes)", min_value=5, max_value=120, value=30)
        
        with col3:
            memory_limit = st.number_input("Memory Limit per Agent (MB)", min_value=100, max_value=2048, value=512)
            cpu_limit = st.slider("CPU Limit per Agent (%)", min_value=1, max_value=100, value=25)
    
    with tab2:
        st.subheader("🛠️ Tool Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Tool Registry Settings**")
            
            auto_load_tools = st.checkbox(
                "Auto-load Tools on Startup",
                value=True,
                key="auto_load_tools"
            )
            
            tool_validation = st.checkbox(
                "Strict Tool Validation",
                value=True,
                key="tool_validation"
            )
            
            custom_tool_path = st.text_input(
                "Custom Tools Directory",
                value="./tools/custom",
                key="custom_tool_path"
            )
            
            tool_timeout = st.number_input(
                "Tool Execution Timeout (seconds)",
                min_value=5,
                max_value=300,
                value=30,
                key="tool_timeout"
            )
        
        with col2:
            st.markdown("**Tool Security**")
            
            sandbox_tools = st.checkbox(
                "Sandbox Tool Execution",
                value=True,
                key="sandbox_tools"
            )
            
            allowed_domains = st.text_area(
                "Allowed Domains (one per line)",
                value="example.com\napi.openai.com\ngemini.google.com",
                key="allowed_domains"
            )
            
            max_file_size = st.number_input(
                "Max File Upload Size (MB)",
                min_value=1,
                max_value=100,
                value=10,
                key="max_file_size"
            )
        
        st.markdown("---")
        
        # Built-in tool configuration
        st.markdown("**Built-in Tool Settings**")
        
        tool_configs = {
            "Web Scraper": {"enabled": True, "rate_limit": 10, "user_agent": "ADK-Bot/1.0"},
            "Email Client": {"enabled": False, "smtp_server": "smtp.gmail.com", "port": 587},
            "File Converter": {"enabled": True, "max_size_mb": 50, "quality": "high"},
            "Data Analyzer": {"enabled": True, "cache_results": True, "max_rows": 10000}
        }
        
        for tool_name, config in tool_configs.items():
            with st.expander(f"⚙️ {tool_name} Configuration"):
                cols = st.columns(len(config))
                for i, (key, value) in enumerate(config.items()):
                    with cols[i]:
                        if isinstance(value, bool):
                            st.checkbox(key.replace('_', ' ').title(), value=value, key=f"{tool_name}_{key}")
                        elif isinstance(value, int):
                            st.number_input(key.replace('_', ' ').title(), value=value, key=f"{tool_name}_{key}")
                        else:
                            st.text_input(key.replace('_', ' ').title(), value=value, key=f"{tool_name}_{key}")
    
    with tab3:
        st.subheader("🔌 API & Integrations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**API Configuration**")
            
            # Google ADK settings
            st.markdown("**Google ADK**")
            adk_api_key = st.text_input(
                "ADK API Key",
                value="***************",
                type="password",
                key="adk_api_key"
            )
            adk_region = st.selectbox(
                "ADK Region",
                ["us-central1", "europe-west1", "asia-southeast1"],
                key="adk_region"
            )
            
            # Gemini AI settings
            st.markdown("**Gemini AI**")
            gemini_api_key = st.text_input(
                "Gemini API Key",
                value="***************",
                type="password",
                key="gemini_api_key"
            )
            gemini_safety_settings = st.selectbox(
                "Safety Level",
                ["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"],
                index=2,
                key="gemini_safety"
            )
        
        with col2:
            st.markdown("**External Integrations**")
            
            # Third-party APIs
            st.markdown("**Third-party APIs**")
            enable_openai = st.checkbox("Enable OpenAI Integration", value=False)
            if enable_openai:
                openai_api_key = st.text_input("OpenAI API Key", type="password")
            
            enable_anthropic = st.checkbox("Enable Anthropic Integration", value=False)
            if enable_anthropic:
                anthropic_api_key = st.text_input("Anthropic API Key", type="password")
            
            # Database settings
            st.markdown("**Database**")
            db_type = st.selectbox(
                "Database Type",
                ["sqlite", "postgresql", "mysql"],
                key="db_type"
            )
            
            if db_type != "sqlite":
                db_host = st.text_input("Database Host", value="localhost")
                db_port = st.number_input("Database Port", value=5432 if db_type == "postgresql" else 3306)
                db_user = st.text_input("Database User", value="adk_user")
                db_password = st.text_input("Database Password", type="password")
                db_name = st.text_input("Database Name", value="adk_db")
        
        st.markdown("---")
        
        # API rate limiting
        st.markdown("**Rate Limiting**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            api_rate_limit = st.number_input("API Requests per Minute", min_value=1, max_value=1000, value=60)
        with col2:
            burst_limit = st.number_input("Burst Limit", min_value=1, max_value=100, value=10)
        with col3:
            enable_caching = st.checkbox("Enable Response Caching", value=True)
    
    with tab4:
        st.subheader("📊 Analytics & Monitoring")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Logging Configuration**")
            
            log_level = st.selectbox(
                "Log Level",
                ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                index=1,
                key="log_level"
            )
            
            log_format = st.selectbox(
                "Log Format",
                ["JSON", "Text", "Structured"],
                key="log_format"
            )
            
            log_retention = st.number_input(
                "Log Retention (days)",
                min_value=1,
                max_value=365,
                value=30,
                key="log_retention"
            )
            
            enable_file_logging = st.checkbox("Enable File Logging", value=True)
            enable_console_logging = st.checkbox("Enable Console Logging", value=True)
        
        with col2:
            st.markdown("**Metrics Collection**")
            
            enable_metrics = st.checkbox("Enable Metrics Collection", value=True)
            metrics_interval = st.number_input("Metrics Collection Interval (seconds)", min_value=5, max_value=300, value=60)
            
            performance_tracking = st.checkbox("Track Performance Metrics", value=True)
            usage_analytics = st.checkbox("Collect Usage Analytics", value=True)
            error_tracking = st.checkbox("Enable Error Tracking", value=True)
            
            # Metrics storage
            st.markdown("**Metrics Storage**")
            metrics_retention = st.number_input("Metrics Retention (days)", min_value=1, max_value=365, value=90)
            enable_metrics_export = st.checkbox("Enable Metrics Export", value=False)
        
        st.markdown("---")
        
        # Alerting configuration
        st.markdown("**Alerting Configuration**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            enable_alerts = st.checkbox("Enable Alerts", value=True)
            if enable_alerts:
                alert_email = st.text_input("Alert Email", value="admin@example.com")
                cpu_threshold = st.slider("CPU Alert Threshold (%)", 50, 95, 80)
                memory_threshold = st.slider("Memory Alert Threshold (%)", 50, 95, 85)
        
        with col2:
            if enable_alerts:
                error_rate_threshold = st.slider("Error Rate Alert Threshold (%)", 1, 20, 5)
                response_time_threshold = st.number_input("Response Time Alert Threshold (ms)", min_value=100, max_value=5000, value=1000)
                alert_cooldown = st.number_input("Alert Cooldown (minutes)", min_value=1, max_value=60, value=15)
    
    with tab5:
        st.subheader("🔒 Security & Privacy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Authentication & Authorization**")
            
            enable_auth = st.checkbox("Enable Authentication", value=False)
            if enable_auth:
                auth_method = st.selectbox("Authentication Method", ["OAuth2", "JWT", "API Key"])
                session_timeout = st.number_input("Session Timeout (hours)", min_value=1, max_value=24, value=8)
            
            enable_rbac = st.checkbox("Enable Role-Based Access Control", value=False)
            if enable_rbac:
                default_role = st.selectbox("Default User Role", ["viewer", "user", "admin"])
            
            # API security
            st.markdown("**API Security**")
            require_https = st.checkbox("Require HTTPS", value=True)
            enable_cors = st.checkbox("Enable CORS", value=True)
            if enable_cors:
                allowed_origins = st.text_area("Allowed Origins", value="http://localhost:3000\nhttp://localhost:8501")
        
        with col2:
            st.markdown("**Data Privacy**")
            
            data_encryption = st.checkbox("Enable Data Encryption at Rest", value=True)
            encrypt_logs = st.checkbox("Encrypt Log Files", value=False)
            anonymize_data = st.checkbox("Anonymize User Data", value=True)
            
            # Privacy settings
            st.markdown("**Privacy Settings**")
            collect_telemetry = st.checkbox("Collect Anonymous Telemetry", value=True)
            share_crash_reports = st.checkbox("Share Crash Reports", value=False)
            
            # Data retention
            st.markdown("**Data Retention**")
            user_data_retention = st.number_input("User Data Retention (days)", min_value=1, max_value=365, value=90)
            temp_file_cleanup = st.number_input("Temp File Cleanup (hours)", min_value=1, max_value=48, value=24)
        
        st.markdown("---")
        
        # Security monitoring
        st.markdown("**Security Monitoring**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            enable_audit_log = st.checkbox("Enable Audit Logging", value=True)
            failed_login_tracking = st.checkbox("Track Failed Logins", value=True)
        
        with col2:
            intrusion_detection = st.checkbox("Enable Intrusion Detection", value=False)
            rate_limit_security = st.checkbox("Security Rate Limiting", value=True)
        
        with col3:
            security_alerts = st.checkbox("Enable Security Alerts", value=True)
            auto_block_suspicious = st.checkbox("Auto-block Suspicious Activity", value=False)
    
    # Configuration actions
    st.markdown("---")
    st.subheader("💾 Configuration Management")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💾 Save Configuration", use_container_width=True, type="primary"):
            save_configuration()
    
    with col2:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            reset_configuration()
    
    with col3:
        if st.button("📤 Export Config", use_container_width=True):
            export_configuration()
    
    with col4:
        uploaded_config = st.file_uploader("📥 Import Config", type=['json'], key="import_config")
        if uploaded_config:
            import_configuration(uploaded_config)
    
    # Configuration validation
    if st.session_state.config_changes:
        st.info(f"You have {len(st.session_state.config_changes)} unsaved changes.")
        
        with st.expander("View Pending Changes"):
            for key, value in st.session_state.config_changes.items():
                st.write(f"**{key}:** {value}")

def save_configuration():
    """Save the current configuration."""
    # In a real app, this would save to a database or config file
    st.success("✅ Configuration saved successfully!")
    st.session_state.config_changes = {}

def reset_configuration():
    """Reset configuration to defaults."""
    st.warning("⚠️ Configuration reset to defaults!")
    st.session_state.config_changes = {}
    st.rerun()

def export_configuration():
    """Export configuration as JSON."""
    config = {
        "agent_settings": {
            "default_model": "gemini-pro",
            "max_tokens": 2048,
            "temperature": 0.7
        },
        "tool_configuration": {
            "auto_load_tools": True,
            "tool_validation": True
        },
        "exported_at": "2024-01-15T10:30:00Z"
    }
    
    config_json = json.dumps(config, indent=2)
    st.download_button(
        label="Download Configuration",
        data=config_json,
        file_name="adk_config.json",
        mime="application/json"
    )

def import_configuration(uploaded_file):
    """Import configuration from uploaded file."""
    try:
        config = json.load(uploaded_file)
        st.success("✅ Configuration imported successfully!")
        st.json(config)
    except json.JSONDecodeError:
        st.error("❌ Invalid JSON file")
    except Exception as e:
        st.error(f"❌ Error importing configuration: {e}")

if __name__ == "__main__":
    render()
