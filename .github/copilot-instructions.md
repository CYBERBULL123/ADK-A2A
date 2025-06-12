<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# ADK & A2A Learning Project - Copilot Instructions

This is a comprehensive learning project for Google's Agent Development Kit (ADK) and Agent-to-Agent (A2A) protocols.

## Project Context
- **Primary Framework**: Google ADK (Agent Development Kit)
- **Language**: Python 3.8+
- **Frontend**: Streamlit for interactive dashboards
- **Focus**: Multi-agent systems and A2A protocol implementation

## Coding Guidelines

### ADK-Specific Patterns
- Use `google.adk.agents.Agent` and `google.adk.agents.LlmAgent` for agent creation
- Follow ADK's code-first development approach
- Implement tools using `google.adk.tools` or custom function decorators
- Use proper agent hierarchies with `sub_agents` for multi-agent systems

### Code Style Preferences
- Use type hints throughout the codebase
- Follow PEP 8 with Black formatting
- Use async/await patterns for I/O operations
- Implement proper error handling and logging
- Add comprehensive docstrings for all classes and functions

### Project-Specific Patterns
- Organize agents by complexity: basic → multi-agent → a2a
- Create reusable tool modules in `tools/` directory
- Use Streamlit components for interactive demonstrations
- Implement evaluation frameworks for testing agents
- Follow progressive learning structure in examples

### Architecture Principles
- Modular design for easy learning and experimentation
- Clear separation between agent logic and UI components
- Comprehensive testing and evaluation strategies
- Production-ready deployment considerations

## Key Dependencies
- `google-adk`: Core agent development framework
- `streamlit`: Interactive frontend development
- `google-generativeai`: Gemini model integration
- `pytest`: Testing framework for agent evaluation

When suggesting code, prioritize educational value and demonstrate ADK best practices.
