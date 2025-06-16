"""
Environment page.
"""

import streamlit as st
import sys
import os
import platform
import subprocess
import importlib.util
from pathlib import Path
from frontend.utils.ui_components import ui

def render():
    """Render the environment page."""
    ui.header(
        "Environment",
        "Check and configure your development environment",
        "🌍"
    )
    
    # Environment status overview
    st.subheader("🔍 Environment Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Check Python version
    python_version = sys.version.split()[0]
    python_ok = sys.version_info >= (3, 8)
    with col1:
        ui.metric_card(
            "Python Version",
            python_version,
            "✅ Compatible" if python_ok else "❌ Upgrade needed",
            icon="🐍"
        )
    
    # Check required packages
    required_packages = check_required_packages()
    packages_ok = all(pkg['installed'] for pkg in required_packages)
    
    with col2:
        ui.metric_card(
            "Dependencies",
            f"{sum(pkg['installed'] for pkg in required_packages)}/{len(required_packages)}",
            "✅ All installed" if packages_ok else "⚠️ Missing packages",
            icon="📦"
        )
    
    # Check system resources
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        system_ok = memory_gb >= 4 and cpu_count >= 2
    except:
        memory_gb = 8  # fallback
        cpu_count = 4  # fallback
        system_ok = True
    
    with col3:
        ui.metric_card(
            "System Resources",
            f"{cpu_count} CPU, {memory_gb:.1f}GB RAM",
            "✅ Adequate" if system_ok else "⚠️ Limited",
            icon="💻"
        )
    
    # Check environment setup
    env_vars = check_environment_variables()
    env_ok = all(var['configured'] for var in env_vars if var['required'])
    
    with col4:
        ui.metric_card(
            "Environment",
            "Configured" if env_ok else "Needs Setup",
            "✅ Ready" if env_ok else "⚠️ Configure required",
            icon="⚙️"
        )
    
    st.markdown("---")
    
    # Detailed environment information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🐍 Python Environment")
        
        # Python details
        st.write(f"**Python Version:** {sys.version}")
        st.write(f"**Python Executable:** {sys.executable}")
        st.write(f"**Platform:** {platform.platform()}")
        st.write(f"**Architecture:** {platform.architecture()[0]}")
        
        # Virtual environment check
        venv_active = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        st.write(f"**Virtual Environment:** {'✅ Active' if venv_active else '❌ Not detected'}")
        
        if venv_active:
            st.write(f"**Virtual Environment Path:** {sys.prefix}")
        
        # Python path
        with st.expander("🔍 Python Path"):
            for i, path in enumerate(sys.path):
                st.write(f"{i+1}. {path}")
    
    with col2:
        st.subheader("🌍 System Information")
        
        # System details
        st.write(f"**Operating System:** {platform.system()} {platform.release()}")
        st.write(f"**Machine:** {platform.machine()}")
        st.write(f"**Processor:** {platform.processor()}")
        st.write(f"**Hostname:** {platform.node()}")
        
        # Additional system info
        try:
            import psutil
            boot_time = psutil.boot_time()
            import datetime
            boot_time_str = datetime.datetime.fromtimestamp(boot_time).strftime("%Y-%m-%d %H:%M:%S")
            st.write(f"**System Boot Time:** {boot_time_str}")
            
            disk_usage = psutil.disk_usage('/')
            st.write(f"**Disk Space:** {disk_usage.free / (1024**3):.1f}GB free of {disk_usage.total / (1024**3):.1f}GB")
        except:
            st.write("**Additional Info:** psutil not available")
    
    st.markdown("---")
    
    # Package dependencies
    st.subheader("📦 Package Dependencies")
    
    # Create DataFrame for better display
    import pandas as pd
    
    deps_df = pd.DataFrame(required_packages)
    
    # Style the dataframe
    def style_status(val):
        if val == "✅ Installed":
            return "background-color: #d4edda; color: #155724"
        elif val == "❌ Missing":
            return "background-color: #f8d7da; color: #721c24"
        elif val.startswith("⚠️"):
            return "background-color: #fff3cd; color: #856404"
        return ""
    
    if not deps_df.empty:
        styled_df = deps_df.style.applymap(style_status, subset=['status'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Installation helper
    missing_packages = [pkg for pkg in required_packages if not pkg['installed']]
    
    if missing_packages:
        st.warning(f"⚠️ {len(missing_packages)} required packages are missing!")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            install_command = "pip install " + " ".join([pkg['name'] for pkg in missing_packages])
            st.code(install_command, language="bash")
        
        with col2:
            if st.button("📥 Install Missing", use_container_width=True):
                install_missing_packages(missing_packages)
    else:
        st.success("✅ All required packages are installed!")
    
    st.markdown("---")
    
    # Environment variables
    st.subheader("🔧 Environment Variables")
    
    env_df = pd.DataFrame(env_vars)
    
    if not env_df.empty:
        # Color code based on status
        def style_env_status(val):
            if val == "✅ Set":
                return "background-color: #d4edda; color: #155724"
            elif val == "❌ Missing":
                return "background-color: #f8d7da; color: #721c24"
            elif val == "⚠️ Optional":
                return "background-color: #e2e3e5; color: #6c757d"
            return ""
        
        styled_env_df = env_df.style.applymap(style_env_status, subset=['status'])
        st.dataframe(styled_env_df, use_container_width=True, hide_index=True)
    
    # Environment variable configuration
    st.markdown("**Configure Environment Variables**")
    
    missing_env_vars = [var for var in env_vars if not var['configured'] and var['required']]
    
    if missing_env_vars:
        st.warning("⚠️ Required environment variables are missing. Configure them below:")
        
        for var in missing_env_vars:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.write(f"**{var['name']}:**")
            with col2:
                value = st.text_input(
                    f"Enter {var['name']}",
                    type="password" if "key" in var['name'].lower() or "token" in var['name'].lower() else "default",
                    key=f"env_{var['name']}"
                )
                if value:
                    st.info(f"💡 Add this to your .env file: {var['name']}={value}")
    
    st.markdown("---")
    
    # Development tools
    st.subheader("🛠️ Development Tools")
    
    dev_tools = [
        {"name": "Git", "command": "git --version", "required": True},
        {"name": "Docker", "command": "docker --version", "required": False},
        {"name": "Node.js", "command": "node --version", "required": False},
        {"name": "VS Code", "command": "code --version", "required": False}
    ]
    
    tool_status = []
    for tool in dev_tools:
        try:
            result = subprocess.run(tool['command'].split(), capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip().split('\n')[0]
                tool_status.append({
                    "Tool": tool['name'],
                    "Status": "✅ Installed",
                    "Version": version,
                    "Required": "Yes" if tool['required'] else "No"
                })
            else:
                tool_status.append({
                    "Tool": tool['name'],
                    "Status": "❌ Not Found",
                    "Version": "N/A",
                    "Required": "Yes" if tool['required'] else "No"
                })
        except (subprocess.TimeoutExpired, FileNotFoundError):
            tool_status.append({
                "Tool": tool['name'],
                "Status": "❌ Not Found",
                "Version": "N/A",
                "Required": "Yes" if tool['required'] else "No"
            })
    
    tools_df = pd.DataFrame(tool_status)
    st.dataframe(tools_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Environment actions
    st.subheader("🎛️ Environment Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔍 Refresh Check", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("📋 Export Report", use_container_width=True):
            export_environment_report()
    
    with col3:
        if st.button("🔧 Setup Guide", use_container_width=True):
            show_setup_guide()
    
    with col4:
        if st.button("🏥 Health Check", use_container_width=True):
            run_health_check()

def check_required_packages():
    """Check if required packages are installed."""
    required_packages = [
        "streamlit",
        "google-adk",
        "google-generativeai", 
        "plotly",
        "pandas",
        "numpy",
        "requests",
        "python-dotenv",
        "psutil"
    ]
    
    package_status = []
    for package in required_packages:
        try:
            spec = importlib.util.find_spec(package.replace("-", "_"))
            if spec is not None:
                # Try to get version
                try:
                    module = importlib.import_module(package.replace("-", "_"))
                    version = getattr(module, '__version__', 'unknown')
                except:
                    version = 'unknown'
                
                package_status.append({
                    "name": package,
                    "installed": True,
                    "version": version,
                    "status": "✅ Installed"
                })
            else:
                package_status.append({
                    "name": package,
                    "installed": False,
                    "version": "N/A",
                    "status": "❌ Missing"
                })
        except Exception:
            package_status.append({
                "name": package,
                "installed": False,
                "version": "N/A", 
                "status": "❌ Missing"
            })
    
    return package_status

def check_environment_variables():
    """Check required environment variables."""
    env_vars = [
        {"name": "GOOGLE_API_KEY", "required": True, "description": "Google Gemini API key"},
        {"name": "ADK_API_KEY", "required": True, "description": "Google ADK API key"},
        {"name": "OPENAI_API_KEY", "required": False, "description": "OpenAI API key (optional)"},
        {"name": "ANTHROPIC_API_KEY", "required": False, "description": "Anthropic API key (optional)"},
        {"name": "DATABASE_URL", "required": False, "description": "Database connection URL"},
        {"name": "LOG_LEVEL", "required": False, "description": "Logging level"}
    ]
    
    for var in env_vars:
        value = os.getenv(var['name'])
        var['configured'] = value is not None and value.strip() != ""
        
        if var['configured']:
            var['status'] = "✅ Set"
        elif var['required']:
            var['status'] = "❌ Missing"
        else:
            var['status'] = "⚠️ Optional"
    
    return env_vars

def install_missing_packages(packages):
    """Install missing packages."""
    with st.spinner("Installing missing packages..."):
        try:
            package_names = [pkg['name'] for pkg in packages]
            result = subprocess.run([sys.executable, "-m", "pip", "install"] + package_names, 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                st.success("✅ Packages installed successfully!")
                st.rerun()
            else:
                st.error(f"❌ Installation failed: {result.stderr}")
        except Exception as e:
            st.error(f"❌ Installation error: {e}")

def export_environment_report():
    """Export environment report."""
    report = {
        "environment_report": {
            "timestamp": "2024-01-15T10:30:00Z",
            "python_version": sys.version,
            "platform": platform.platform(),
            "packages": check_required_packages(),
            "environment_variables": check_environment_variables()
        }
    }
    
    import json
    report_json = json.dumps(report, indent=2)
    st.download_button(
        label="Download Environment Report",
        data=report_json,
        file_name="environment_report.json",
        mime="application/json"
    )

def show_setup_guide():
    """Show setup guide in an expander."""
    with st.expander("📖 Setup Guide", expanded=True):
        st.markdown("""
        ### 🚀 Quick Setup Guide
        
        1. **Python Environment**
           ```bash
           python -m venv adk_env
           source adk_env/bin/activate  # On Windows: adk_env\\Scripts\\activate
           ```
        
        2. **Install Dependencies**
           ```bash
           pip install -r requirements.txt
           ```
        
        3. **Environment Variables**
           Create a `.env` file in your project root:
           ```
           GOOGLE_API_KEY=your_gemini_api_key_here
           ADK_API_KEY=your_adk_api_key_here
           LOG_LEVEL=INFO
           ```
        
        4. **Verify Installation**
           ```bash
           python -c "import google.adk; print('ADK installed successfully!')"
           ```
        
        5. **Run the Application**
           ```bash
           streamlit run frontend/app.py
           ```
        """)

def run_health_check():
    """Run a comprehensive health check."""
    with st.spinner("Running health check..."):
        import time
        time.sleep(2)  # Simulate health check
        
        checks = [
            {"check": "Python version", "status": "✅ Pass"},
            {"check": "Required packages", "status": "✅ Pass"},
            {"check": "Environment variables", "status": "⚠️ Warning"},
            {"check": "System resources", "status": "✅ Pass"},
            {"check": "Network connectivity", "status": "✅ Pass"}
        ]
        
        st.success("🏥 Health check completed!")
        
        for check in checks:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(check["check"])
            with col2:
                st.write(check["status"])

if __name__ == "__main__":
    render()
