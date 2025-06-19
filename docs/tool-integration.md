# Tool Integration with ADK Agents

This document explains how tools are integrated with different types of agents in the ADK & A2A learning project, with a focus on the Model Context Protocol (MCP) for dynamic tool discovery and sharing.

## Overview

The tool system in this project is designed to support three levels of agent architectures:

1. **Basic Agents**: Single agents with directly integrated tools
2. **Multi-Agent Systems**: Multiple agents sharing tools through MCP
3. **A2A Protocol**: Cross-network agent communication with tool sharing

## Why MCP Integration?

Model Context Protocol (MCP) provides several key benefits for agent-tool integration:

### 1. Dynamic Tool Discovery
- Agents can discover new tools at runtime
- No need to restart or reconfigure agents when new tools are added
- Tools can be added from external sources (MCP servers)

### 2. Cross-Agent Tool Sharing
- Multiple agents can share the same tool instances
- Prevents duplication and ensures consistency
- Enables collaborative workflows

### 3. A2A Protocol Support
- Tools can be exposed across different agent networks
- Enables cross-organizational agent collaboration
- Supports federated agent ecosystems

### 4. Standardized Interface
- Consistent tool interface across different agent types
- Simplified tool development and integration
- Better error handling and monitoring

## Tool Categories

### Built-in Tools (Direct Integration)
These tools are directly integrated with basic agents using ADK's native tool system:

```python
from google.adk.agents import Agent
from google.adk.tools import google_search
from tools import weather_api_tool

agent = Agent(
    name="MyAgent",
    model="gemini-1.5-pro",
    tools=[
        google_search,
        weather_api_tool
    ]
)
```

**Characteristics:**
- ✅ Direct integration with basic agents
- ✅ High performance (no protocol overhead)
- ✅ Type-safe integration
- ❌ Limited to single agent use
- ❌ No dynamic discovery

### MCP Tools (Protocol-Based Sharing)
These tools are registered through the MCP protocol and can be shared across multiple agents:

```python
from frontend.tools_page import mcp_manager

# Register tool for sharing
await mcp_manager.register_mcp_tool(
    "shared_weather_tool",
    {
        'type': 'api_service',
        'description': 'Weather API accessible to all agents',
        'parameters': ['location', 'forecast_days']
    },
    agent_types=['basic', 'multi_agent', 'a2a']
)

# Bind tool to specific agent
await mcp_manager.bind_tool_to_agent(
    "shared_weather_tool",
    "weather_agent_01",
    "multi_agent"
)
```

**Characteristics:**
- ✅ Cross-agent sharing
- ✅ Dynamic discovery and binding
- ✅ Permission-based access control
- ✅ Usage tracking and analytics
- ❌ Slight protocol overhead

### A2A Tools (Cross-Network Sharing)
These tools support the Agent-to-Agent protocol for cross-network communication:

```python
from examples.example_04_tool_integration import A2AToolProtocol

protocol = A2AToolProtocol()

# Expose tool for A2A sharing
protocol.expose_tool(
    "weather_service",
    weather_api_tool,
    permissions=['read', 'execute'],
    networks=['partner_network', 'research_network']
)

# Execute remote tool
result = await protocol.execute_remote_tool(
    'partner_network',
    'remote_data_processor',
    {'data': 'sample', 'operation': 'analyze'},
    'local_analysis_agent'
)
```

**Characteristics:**
- ✅ Cross-network agent communication
- ✅ Federated tool ecosystems
- ✅ Network-level security and permissions
- ✅ Scalable to multiple organizations
- ❌ Higher latency due to network communication

## Integration Patterns

### Pattern 1: Basic Agent with Tools

```python
class BasicAgentWithTools:
    def __init__(self):
        self.agent = Agent(
            name="ToolAgent",
            tools=[
                google_search,        # ADK built-in tool
                weather_api_tool,     # Custom tool
                database_query_tool   # Another custom tool
            ]
        )
    
    async def chat(self, message: str) -> str:
        # Agent automatically uses tools when needed
        return await self.agent.run(message)
```

**Use Cases:**
- Single-purpose agents
- Rapid prototyping
- Simple automations

### Pattern 2: Multi-Agent Tool Sharing

```python
class MultiAgentSystem:
    def __init__(self):
        self.mcp_manager = MCPToolManager()
        self.agents = {}
    
    async def register_shared_tool(self, tool_name: str, tool_func):
        await self.mcp_manager.register_mcp_tool(tool_name, tool_func)
        
        # Add to all existing agents
        for agent in self.agents.values():
            await self.mcp_manager.bind_tool_to_agent(
                tool_name, agent.id, "multi_agent"
            )
    
    async def create_agent(self, name: str, specialization: str):
        agent = Agent(name=name, specialization=specialization)
        
        # Bind all shared tools to new agent
        for tool_name in self.mcp_manager.mcp_tools:
            await self.mcp_manager.bind_tool_to_agent(
                tool_name, agent.id, "multi_agent"
            )
        
        self.agents[name] = agent
        return agent
```

**Use Cases:**
- Complex workflows requiring multiple specialized agents
- Collaborative problem-solving
- Resource optimization (shared tool instances)

### Pattern 3: A2A Protocol Integration

```python
class A2ANetworkNode:
    def __init__(self, network_id: str):
        self.network_id = network_id
        self.protocol = A2AToolProtocol()
        self.local_agents = {}
        
    async def expose_tools_to_network(self, tools: List[str]):
        for tool_name in tools:
            await self.protocol.expose_tool(
                tool_name,
                self.get_local_tool(tool_name),
                networks=['trusted_partners']
            )
    
    async def discover_network_tools(self, remote_network: str):
        return await self.protocol.discover_remote_tools(remote_network)
    
    async def execute_cross_network_task(self, task: Dict[str, Any]):
        # Coordinate task across multiple networks
        results = {}
        
        for network, subtask in task['network_tasks'].items():
            result = await self.protocol.execute_remote_tool(
                network,
                subtask['tool'],
                subtask['parameters'],
                self.network_id
            )
            results[network] = result
        
        return self.aggregate_results(results)
```

**Use Cases:**
- Cross-organizational collaboration
- Federated AI systems
- Large-scale distributed problem-solving

## Tool Development Guidelines

### Creating ADK-Compatible Tools

1. **Function Signature**: Use clear, typed parameters
```python
def my_tool(param1: str, param2: int = 10) -> str:
    \"\"\"Tool description for ADK.\"\"\"
    return f"Result: {param1} with {param2}"
```

2. **Error Handling**: Provide clear error messages
```python
def robust_tool(data: str) -> Dict[str, Any]:
    try:
        result = process_data(data)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

3. **Documentation**: Include comprehensive docstrings
```python
def weather_tool(location: str, units: str = "metric") -> str:
    \"\"\"
    Get weather information for a location.
    
    Args:
        location: City name or coordinates
        units: Temperature units (metric, imperial, kelvin)
        
    Returns:
        JSON string with weather data
        
    Raises:
        ValueError: If location is invalid
        ConnectionError: If API is unavailable
    \"\"\"
    # Implementation here
```

### MCP Tool Registration

```python
# Tool specification for MCP
tool_spec = {
    'name': 'my_custom_tool',
    'description': 'Processes data and returns analysis',
    'type': 'data_processor',
    'parameters': {
        'data': {'type': 'string', 'required': True},
        'analysis_type': {'type': 'string', 'default': 'basic'}
    },
    'returns': 'object',
    'agent_types': ['basic', 'multi_agent', 'a2a']
}

# Register with MCP manager
await mcp_manager.register_mcp_tool('my_custom_tool', tool_spec)
```

## Best Practices

### 1. Tool Design
- **Single Responsibility**: Each tool should have one clear purpose
- **Stateless**: Tools should not maintain state between calls
- **Idempotent**: Same inputs should always produce same outputs
- **Fast**: Keep tool execution time under 30 seconds

### 2. Error Handling
- **Graceful Degradation**: Handle failures without crashing
- **Clear Messages**: Provide actionable error messages
- **Logging**: Log tool usage and errors for debugging

### 3. Security
- **Input Validation**: Validate all parameters
- **Permission Checks**: Verify agent permissions for tool access
- **Rate Limiting**: Prevent abuse of expensive operations

### 4. Testing
- **Unit Tests**: Test each tool in isolation
- **Integration Tests**: Test tool-agent integration
- **Load Tests**: Verify performance under load

## Monitoring and Analytics

The MCP manager provides comprehensive monitoring:

```python
# Get tool usage statistics
stats = mcp_manager.get_tool_statistics()

# Monitor real-time usage
async for execution in mcp_manager.execution_history:
    if execution['success']:
        update_success_metrics(execution)
    else:
        alert_on_failure(execution)
```

## Future Enhancements

1. **Tool Marketplace**: Registry of community-contributed tools
2. **Automatic Discovery**: AI-powered tool recommendation
3. **Performance Optimization**: Caching and batching for better performance
4. **Security Enhancements**: Advanced permission models and audit trails

## Examples

See the following files for complete examples:
- `examples/example_04_tool_integration.py`: Comprehensive tool integration demos
- `frontend/tools_page.py`: Interactive tool management interface
- `tools/__init__.py`: Custom tool implementations

This tool integration system provides a solid foundation for building sophisticated agent systems that can evolve and scale as requirements grow.
