"""
Utility functions for the ADK & A2A Learning Project.

This module provides common utilities, logging setup, and helper functions
used throughout the project.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
import logging
from functools import wraps

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from config import config

# Initialize Rich console
console = Console()


def setup_logging():
    """Set up logging configuration with Rich handler."""
    
    # Create logs directory if it doesn't exist
    log_dir = Path(config.log_file).parent
    log_dir.mkdir(exist_ok=True)
    
    # Create a clean formatter for console output
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.log_level.upper()))
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler with clean formatting (only show warnings and errors)
    console_handler = RichHandler(
        console=console, 
        rich_tracebacks=True,
        show_time=False,
        show_path=False
    )
    console_handler.setLevel(logging.WARNING)  # Only show warnings and errors in console
    console_handler.setFormatter(console_formatter)
    
    # File handler with full formatting
    file_handler = logging.FileHandler(config.log_file)
    file_handler.setLevel(getattr(logging, config.log_level.upper()))
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Suppress some verbose loggers
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


logger = setup_logging()


def validate_environment() -> bool:
    """Validate that all required environment variables are set."""
    validation = config.validate_config()
    
    if not validation["valid"]:
        console.print(Panel(
            f"❌ Configuration Errors:\n" + "\n".join(validation["errors"]),
            title="Environment Validation Failed",
            style="red"
        ))
        return False
    
    if validation["warnings"]:
        console.print(Panel(
            f"⚠️ Warnings:\n" + "\n".join(validation["warnings"]),
            title="Environment Warnings",
            style="yellow"
        ))
    
    console.print(Panel(
        "✅ Environment validation passed!",
        title="Environment Status",
        style="green"
    ))
    
    return True


def display_welcome():
    """Display welcome message and project information."""
    console.print(Panel(
        """
🤖 ADK & A2A Learning Project

Welcome to the comprehensive learning environment for:
• Google Agent Development Kit (ADK)
• Agent-to-Agent (A2A) Protocols
• Multi-Agent System Development

Features:
• Interactive Streamlit Dashboard
• Progressive Learning Examples
• Multi-Agent Orchestration
• A2A Protocol Implementation
• Comprehensive Testing Framework
        """,
        title="Welcome",
        style="bold blue"
    ))


def create_agent_summary_table(agents: List[Dict[str, Any]]) -> Table:
    """Create a Rich table summarizing agent information."""
    table = Table(title="Agent Summary")
    
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Tools", style="yellow")
    table.add_column("Description")
    
    for agent in agents:
        table.add_row(
            agent.get("name", "Unknown"),
            agent.get("type", "Unknown"),
            agent.get("status", "Unknown"),
            str(len(agent.get("tools", []))),
            agent.get("description", "No description")[:50] + "..."
        )
    
    return table


def async_timer(func: Callable) -> Callable:
    """Decorator to time async function execution."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = await func(*args, **kwargs)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Function {func.__name__} took {duration:.2f} seconds")
        return result
    
    return wrapper


def timer(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Function {func.__name__} took {duration:.2f} seconds")
        return result
    
    return wrapper


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save data to JSON file with error handling."""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save data to {file_path}: {e}")
        raise


def load_json(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load data from JSON file with error handling."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Data loaded from {file_path}")
        return data
    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        return None


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_agent_response(response: str, agent_name: str = "Agent", include_metadata: bool = False) -> str:
    """Format agent response for display."""
    if include_metadata:
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"[{timestamp}] {agent_name}: {response}"
    else:
        # Return clean response for UI display
        return response.strip()


def format_agent_response_for_ui(response: str) -> str:
    """Format agent response specifically for UI display (clean format)."""
    return response.strip()


def format_agent_response_for_logs(response: str, agent_name: str = "Agent") -> str:
    """Format agent response specifically for logging (with metadata)."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    return f"[{timestamp}] {agent_name}: {response}"


class ProgressTracker:
    """Context manager for tracking progress with Rich."""
    
    def __init__(self, description: str):
        self.description = description
        self.progress = None
        self.task = None
    
    def __enter__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        )
        self.progress.start()
        self.task = self.progress.add_task(self.description)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.stop()
    
    def update(self, description: str):
        """Update progress description."""
        if self.progress and self.task:
            self.progress.update(self.task, description=description)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.absolute()


def get_examples_path() -> Path:
    """Get the examples directory path."""
    return get_project_root() / "examples"


def get_agents_path() -> Path:
    """Get the agents directory path."""
    return get_project_root() / "agents"
