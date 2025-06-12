# Getting Started with ADK & A2A Learning Project

Welcome to your comprehensive learning environment for Google's Agent Development Kit (ADK) and Agent-to-Agent (A2A) protocols!

## 🚀 Quick Start

### 1. Environment Setup

First, copy the environment template and configure your API keys:

```bash
copy .env.example .env
```

Edit `.env` and add your API keys:
```bash
# Essential for ADK functionality
GOOGLE_API_KEY=your_google_api_key_here

# Optional for extended functionality
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 2. Launch the Interactive Dashboard

**Option A: Using VS Code Task (Recommended)**
1. Press `Ctrl+Shift+P` in VS Code
2. Type "Tasks: Run Task"
3. Select "Launch Streamlit Dashboard"

**Option B: Command Line**
```bash
python -m streamlit run frontend/main_dashboard.py
```

The dashboard will open at http://localhost:8501

### 3. Run Example Scripts

**Basic Agents Example:**
```bash
python examples/example_01_basic_agents.py
```

**Multi-Agent Systems Example:**
```bash
python examples/example_02_multi_agent.py
```

## 📚 Learning Path

### Phase 1: Foundation (Start Here!)
- **Interactive Dashboard**: Explore different agent types
- **Basic Agents Example**: Run `example_01_basic_agents.py`
- **Understanding Tools**: Test custom tools in the dashboard

### Phase 2: Coordination
- **Multi-Agent Systems**: Run `example_02_multi_agent.py`
- **Workflow Orchestration**: Explore predefined workflows
- **Agent Communication**: Learn inter-agent messaging

### Phase 3: Advanced Concepts
- **A2A Protocol**: Network-based agent communication
- **Performance Evaluation**: Agent testing and benchmarking
- **Custom Development**: Create your own agents and tools

## 🎯 Key Learning Objectives

### Agent Development Kit (ADK) Mastery
- **Core Concepts**: Agent architecture, tools, and models
- **Code-First Development**: Python-native agent creation
- **Tool Integration**: Extending agent capabilities
- **Model Flexibility**: Working with different AI models

### Multi-Agent Systems
- **Specialization**: Creating role-specific agents
- **Coordination**: Orchestrating complex workflows  
- **Communication**: Inter-agent messaging patterns
- **Scalability**: Building systems that grow

### A2A Protocol Implementation
- **Network Communication**: Remote agent interaction
- **Message Formats**: Protocol design and standards
- **Distributed Systems**: Scaling across nodes
- **Real-time Coordination**: Synchronous and asynchronous patterns

## 🛠️ Project Structure Explained

```
├── agents/                    # Agent implementations
│   ├── basic/                # Single agents with different capabilities
│   ├── multi_agent/          # Coordinated agent systems
│   └── a2a/                  # Network-enabled agents
├── tools/                    # Custom tool definitions
├── evaluations/              # Testing and benchmarking
├── frontend/                 # Streamlit dashboard
├── examples/                 # Progressive learning scripts
└── docs/                     # Additional documentation
```

### Key Files to Explore:
- `config.py` - Centralized configuration
- `utils.py` - Common utilities and helpers
- `frontend/main_dashboard.py` - Interactive learning interface
- `agents/basic/__init__.py` - Basic agent examples
- `agents/multi_agent/__init__.py` - Multi-agent coordination
- `tools/__init__.py` - Custom tool implementations

## 🎮 Interactive Learning Features

### Streamlit Dashboard Modules:
1. **Overview** - Project introduction and quick tests
2. **Basic Agents** - Single agent creation and testing
3. **Multi-Agent Systems** - Coordination and orchestration
4. **A2A Protocol** - Network communication demos
5. **Custom Tools** - Tool development and integration
6. **Performance Analytics** - Monitoring and evaluation
7. **Evaluation Framework** - Testing strategies

### Hands-on Examples:
- **Interactive Playgrounds** - Test agents in real-time
- **Comparison Tools** - See how different agents handle tasks
- **Performance Monitoring** - Track agent efficiency
- **Workflow Builders** - Create custom agent workflows

## 🔧 Development Workflow

### Available VS Code Tasks:
- **Launch Streamlit Dashboard** - Start the interactive interface
- **Run Basic Agents Example** - Execute foundational examples
- **Run Multi-Agent Example** - Test coordination systems
- **Install Dependencies** - Set up the environment
- **Format Code with Black** - Maintain code quality

### Testing and Evaluation:
```bash
# Run comprehensive agent evaluation
python -c "import asyncio; from evaluations import evaluate_all_basic_agents; asyncio.run(evaluate_all_basic_agents())"

# Test specific agent types
python examples/example_01_basic_agents.py
python examples/example_02_multi_agent.py
```

## 💡 Best Practices

### Agent Development:
1. **Start Simple**: Begin with basic agents before moving to complex systems
2. **Use Type Hints**: Maintain code quality with proper typing
3. **Error Handling**: Implement robust error management
4. **Logging**: Use the built-in logging for debugging
5. **Testing**: Evaluate agents regularly with the evaluation framework

### Multi-Agent Design:
1. **Clear Roles**: Define specific responsibilities for each agent
2. **Communication Protocols**: Establish clear messaging patterns
3. **Coordination Logic**: Design efficient task delegation
4. **State Management**: Handle shared state carefully
5. **Performance Monitoring**: Track system efficiency

### A2A Implementation:
1. **Message Standards**: Use consistent message formats
2. **Network Resilience**: Handle connection failures gracefully
3. **Security**: Implement proper authentication and validation
4. **Scalability**: Design for growth and load distribution
5. **Documentation**: Maintain clear protocol documentation

## 🚀 Next Steps

1. **Environment Setup**: Configure your API keys in `.env`
2. **Launch Dashboard**: Start with the Streamlit interface
3. **Run Examples**: Execute the provided learning scripts
4. **Experiment**: Modify examples to understand concepts
5. **Build**: Create your own agents and workflows
6. **Deploy**: Scale your systems using the deployment guides

## 🆘 Troubleshooting

### Common Issues:

**"Module not found" errors:**
```bash
python -m pip install -r requirements.txt
```

**API key issues:**
- Ensure `.env` file is configured correctly
- Check that API keys are valid and have proper permissions

**Port conflicts:**
- Streamlit default port: 8501
- A2A default port: 8502
- Modify ports in `config.py` if needed

**Import errors:**
- Ensure you're running from the project root directory
- Check that all dependencies are installed

### Getting Help:
1. Check the example scripts for implementation patterns
2. Review the evaluation framework for testing strategies
3. Explore the interactive dashboard for hands-on learning
4. Examine the configuration files for customization options

## 🎯 Learning Outcomes

By completing this project, you will:

- **Master ADK Fundamentals**: Create, deploy, and manage AI agents
- **Understand Multi-Agent Systems**: Design coordinated agent workflows
- **Implement A2A Protocols**: Build network-enabled distributed systems
- **Apply Best Practices**: Develop production-ready agent applications
- **Evaluate Performance**: Test and optimize agent systems

Ready to start your ADK & A2A learning journey? Launch the Streamlit dashboard and begin exploring! 🚀
