"""
Configuration module for ADK & A2A Learning Project.

This module centralizes all configuration settings including API keys,
model configurations, and application settings.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the correct path
current_dir = Path(__file__).parent
env_path = current_dir / '.env'
load_dotenv(env_path)


@dataclass
class ModelConfig:
    """Configuration for AI models."""
    name: str
    provider: str
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30


@dataclass
class ADKConfig:
    """Configuration for ADK agents."""
    default_model: str = "gemini-2.0-flash"
    default_temperature: float = 0.7
    max_sub_agents: int = 10
    default_timeout: int = 60


@dataclass
class A2AConfig:
    """Configuration for A2A protocol."""
    server_host: str = "localhost"
    server_port: int = 8502
    protocol_version: str = "1.0"
    timeout: int = 30


@dataclass
class StreamlitConfig:
    """Configuration for Streamlit frontend."""
    host: str = "localhost"
    port: int = 8501
    debug: bool = True


class Config:
    """Main configuration class."""
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_cloud_project = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.openweather_api_key = os.getenv("OPENWEATHER_API_KEY")
        
        # Model configurations
        self.models = {
            "gemini-2.0-flash": ModelConfig(
                name="gemini-2.0-flash",
                provider="google",
                api_key=self.google_api_key,
                temperature=0.7,
                max_tokens=1000
            ),
            "gemini-1.5-pro": ModelConfig(
                name="gemini-1.5-pro",
                provider="google",
                api_key=self.google_api_key,
                temperature=0.7,
                max_tokens=2000
            ),
            "gpt-4": ModelConfig(
                name="gpt-4",
                provider="openai",
                api_key=self.openai_api_key,
                temperature=0.7,
                max_tokens=1000
            )
        }
        
        # Component configurations
        self.adk = ADKConfig()
        self.a2a = A2AConfig(
            server_host=os.getenv("A2A_SERVER_HOST", "localhost"),
            server_port=int(os.getenv("A2A_SERVER_PORT", "8502"))
        )
        self.streamlit = StreamlitConfig(
            host=os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost"),
            port=int(os.getenv("STREAMLIT_SERVER_PORT", "8501")),
            debug=os.getenv("DEBUG", "True").lower() == "true"
        )
        
        # Logging configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_file = os.getenv("LOG_FILE", "logs/app.log")
        
        # Environment
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "True").lower() == "true"
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        return self.models.get(model_name)
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return status."""
        status = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required API keys
        if not self.google_api_key:
            status["errors"].append("GOOGLE_API_KEY is required")
            status["valid"] = False
        
        if not self.google_cloud_project:
            status["warnings"].append("GOOGLE_CLOUD_PROJECT not set")
        
        return status


# Global configuration instance
config = Config()
