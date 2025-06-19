"""
Example: Tool Integration with ADK Agents

This example demonstrates how to integrate tools with different types of agents:
1. Basic agents with built-in tools
2. Multi-agent systems with shared tools
3. A2A protocol with cross-network tool sharing

This showcases the comprehensive tool ecosystem in the ADK & A2A learning project.
"""

import asyncio
from typing import Dict, Any, List
from datetime import datetime

# Import ADK components
from google.adk.agents import Agent, LlmAgent
from google.adk.tools import google_search
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# Import project modules
from config import config
from utils import logger, console
from tools import CUSTOM_TOOLS, weather_api_tool


class BasicAgentWithTools:
    """Example of basic agent with integrated tools."""
    
    def __init__(self, name: str = "ToolIntegratedAgent"):
        self.name = name
        
        # Create agent with tools
        self.agent = Agent(
            name=name,
            model=config.adk.default_model,
            instruction=(
                "You are a helpful assistant with access to various tools. "
                "Use the available tools to provide comprehensive answers."
            ),
            description="A basic agent with integrated custom tools",
            tools=[
                google_search,  # Built-in ADK tool
                weather_api_tool,  # Custom tool from our tools module
            ]
        )
        
        # Setup runner
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.agent,
            app_name="tool_integration_app",
            session_service=self.session_service
        )
        
        self.user_id = "tool_user"
        self.session_id = f"{name}_session"
    
    async def chat_with_tools(self, message: str) -> str:
        """Chat with the agent, demonstrating tool usage."""
        try:
            session = await self.session_service.create_session(
                app_name="tool_integration_app",
                user_id=self.user_id,
                session_id=self.session_id
            )
            
            console.print(f"[blue]User:[/blue] {message}")
            
            # Send message to agent
            from google.genai import types
            content = types.Content(role='user', parts=[types.Part(text=message)])
            
            response_text = "No response received"
            tool_calls = []
            
            async for event in self.runner.run_async(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=content
            ):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        response_text = event.content.parts[0].text
                
                # Track tool calls
                if hasattr(event, 'actions') and event.actions:
                    if hasattr(event.actions, 'function_calls'):
                        for call in event.actions.function_calls:
                            tool_calls.append({
                                'tool': call.name,
                                'args': call.args,
                                'timestamp': datetime.now().isoformat()
                            })
            
            console.print(f"[green]Agent:[/green] {response_text}")
            
            if tool_calls:
                console.print(f"[yellow]Tools used:[/yellow] {[call['tool'] for call in tool_calls]}")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error in tool-integrated chat: {e}")
            return f"Error: {str(e)}"


class MultiAgentToolSharing:
    """Example of multi-agent system with shared tools."""
    
    def __init__(self):
        self.agents = {}
        self.shared_tools = {}
        self.session_service = InMemorySessionService()
        self.tool_usage_log = []
        
    def register_shared_tool(self, tool_name: str, tool_func):
        """Register a tool that can be shared across agents."""
        self.shared_tools[tool_name] = tool_func
        console.print(f"[green]✓[/green] Registered shared tool: {tool_name}")
        
        # Add tool to all existing agents
        for agent_name, agent_data in self.agents.items():
            agent_data['agent'].tools.append(tool_func)
            console.print(f"[blue]→[/blue] Added {tool_name} to agent {agent_name}")
    
    def create_agent(self, name: str, specialization: str) -> Dict[str, Any]:
        """Create a new agent with access to shared tools."""
        agent = Agent(
            name=name,
            model=config.adk.default_model,
            instruction=(
                f"You are a {specialization} specialist. "
                f"Collaborate with other agents and use shared tools effectively."
            ),
            description=f"Specialized agent for {specialization}",
            tools=list(self.shared_tools.values())  # Start with all shared tools
        )
        
        runner = Runner(
            agent=agent,
            app_name="multi_agent_app",
            session_service=self.session_service
        )
        
        agent_data = {
            'agent': agent,
            'runner': runner,
            'specialization': specialization,
            'session_id': f"{name}_session",
            'tool_usage': []
        }
        
        self.agents[name] = agent_data
        console.print(f"[green]✓[/green] Created agent {name} ({specialization})")
        
        return agent_data
    
    async def agent_collaboration(self, task: str, participating_agents: List[str]) -> Dict[str, str]:
        """Demonstrate agents collaborating on a task using shared tools."""
        results = {}
        
        console.print(f"[bold blue]Task:[/bold blue] {task}")
        console.print(f"[blue]Participating agents:[/blue] {', '.join(participating_agents)}")
        
        for agent_name in participating_agents:
            if agent_name not in self.agents:
                console.print(f"[red]Error:[/red] Agent {agent_name} not found")
                continue
            
            agent_data = self.agents[agent_name]
            
            # Customize task for agent's specialization
            specialized_task = f"As a {agent_data['specialization']} specialist, help with: {task}"
            
            try:
                # Create session for this agent
                session = await self.session_service.create_session(
                    app_name="multi_agent_app",
                    user_id="collaboration_user",
                    session_id=agent_data['session_id']
                )
                
                from google.genai import types
                content = types.Content(role='user', parts=[types.Part(text=specialized_task)])
                
                response_text = "No response"
                
                async for event in agent_data['runner'].run_async(
                    user_id="collaboration_user",
                    session_id=agent_data['session_id'],
                    new_message=content
                ):
                    if event.is_final_response():
                        if event.content and event.content.parts:
                            response_text = event.content.parts[0].text
                
                results[agent_name] = response_text
                console.print(f"[green]{agent_name}:[/green] {response_text[:100]}...")
                
            except Exception as e:
                logger.error(f"Error with agent {agent_name}: {e}")
                results[agent_name] = f"Error: {str(e)}"
        
        return results


class A2AToolProtocol:
    """Example of A2A protocol with cross-network tool sharing."""
    
    def __init__(self):
        self.local_agents = {}
        self.remote_networks = {}
        self.exposed_tools = {}
        self.tool_permissions = {}
        
    def expose_tool(self, tool_name: str, tool_func, 
                   permissions: List[str] = None, 
                   networks: List[str] = None):
        """Expose a tool for A2A sharing."""
        self.exposed_tools[tool_name] = {
            'function': tool_func,
            'permissions': permissions or ['read', 'execute'],
            'allowed_networks': networks or ['*'],  # All networks
            'usage_count': 0,
            'exposed_at': datetime.now().isoformat()
        }
        
        console.print(f"[green]✓[/green] Exposed tool {tool_name} for A2A sharing")
        
    def register_remote_network(self, network_id: str, network_info: Dict[str, Any]):
        """Register a remote network for A2A communication."""
        self.remote_networks[network_id] = {
            'info': network_info,
            'available_tools': [],
            'trust_level': network_info.get('trust_level', 'low'),
            'registered_at': datetime.now().isoformat()
        }
        
        console.print(f"[blue]→[/blue] Registered remote network: {network_id}")
        
    async def discover_remote_tools(self, network_id: str) -> List[Dict[str, Any]]:
        """Discover tools available from a remote network."""
        if network_id not in self.remote_networks:
            raise ValueError(f"Network {network_id} not registered")
        
        # Simulate tool discovery
        discovered_tools = [
            {
                'name': f'{network_id}_weather_service',
                'description': 'Weather data from remote network',
                'type': 'api_service',
                'permissions_required': ['read']
            },
            {
                'name': f'{network_id}_data_processor',
                'description': 'Data processing service',
                'type': 'computation',
                'permissions_required': ['read', 'execute']
            }
        ]
        
        self.remote_networks[network_id]['available_tools'] = discovered_tools
        
        console.print(f"[yellow]🔍[/yellow] Discovered {len(discovered_tools)} tools from {network_id}")
        
        return discovered_tools
        
    async def execute_remote_tool(self, network_id: str, tool_name: str, 
                                parameters: Dict[str, Any], 
                                requesting_agent: str) -> Dict[str, Any]:
        """Execute a tool on a remote network via A2A protocol."""
        try:
            # Check permissions
            if network_id not in self.remote_networks:
                raise ValueError(f"Network {network_id} not accessible")
            
            # Simulate A2A protocol communication
            console.print(f"[blue]🌐[/blue] A2A Request: {requesting_agent} → {network_id}:{tool_name}")
            
            # Simulate execution
            await asyncio.sleep(0.5)  # Network delay
            
            result = {
                'success': True,
                'result': f"Remote execution of {tool_name} completed",
                'network': network_id,
                'tool': tool_name,
                'parameters': parameters,
                'execution_time': '0.5s',
                'timestamp': datetime.now().isoformat()
            }
            
            console.print(f"[green]✓[/green] A2A tool execution successful")
            
            return result
            
        except Exception as e:
            logger.error(f"A2A tool execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


async def demo_basic_agent_tools():
    """Demonstrate basic agent with tools."""
    console.print("\n[bold cyan]🤖 Demo: Basic Agent with Tools[/bold cyan]")
    
    agent = BasicAgentWithTools("WeatherAgent")
    
    # Test tool integration
    questions = [
        "What's the weather like in London?",
        "Search for information about ADK agents",
        "Can you help me with both weather and search?"
    ]
    
    for question in questions:
        response = await agent.chat_with_tools(question)
        console.print("=" * 50)


async def demo_multi_agent_tools():
    """Demonstrate multi-agent system with shared tools."""
    console.print("\n[bold cyan]🔄 Demo: Multi-Agent Tool Sharing[/bold cyan]")
    
    system = MultiAgentToolSharing()
    
    # Register shared tools
    system.register_shared_tool("weather_api", weather_api_tool)
    system.register_shared_tool("google_search", google_search)
    
    # Create specialized agents
    system.create_agent("WeatherAnalyst", "weather analysis")
    system.create_agent("ResearchAgent", "information research")
    system.create_agent("TaskCoordinator", "task coordination")
    
    # Demonstrate collaboration
    task = "Plan a trip to Tokyo considering weather and travel information"
    results = await system.agent_collaboration(
        task, 
        ["WeatherAnalyst", "ResearchAgent", "TaskCoordinator"]
    )
    
    console.print(f"\n[green]Collaboration Results:[/green]")
    for agent, result in results.items():
        console.print(f"[blue]{agent}:[/blue] {result[:100]}...")


async def demo_a2a_tool_protocol():
    """Demonstrate A2A protocol tool sharing."""
    console.print("\n[bold cyan]🌐 Demo: A2A Tool Protocol[/bold cyan]")
    
    protocol = A2AToolProtocol()
    
    # Expose local tools
    protocol.expose_tool("weather_api", weather_api_tool, 
                        permissions=['read', 'execute'],
                        networks=['network_b', 'network_c'])
    
    # Register remote networks
    protocol.register_remote_network("network_b", {
        'name': 'European Weather Network',
        'trust_level': 'high',
        'capabilities': ['weather', 'climate_data']
    })
    
    protocol.register_remote_network("network_c", {
        'name': 'Data Processing Cluster',
        'trust_level': 'medium',
        'capabilities': ['computation', 'analytics']
    })
    
    # Discover remote tools
    for network_id in ['network_b', 'network_c']:
        tools = await protocol.discover_remote_tools(network_id)
        console.print(f"[yellow]Tools from {network_id}:[/yellow]")
        for tool in tools:
            console.print(f"  • {tool['name']}: {tool['description']}")
    
    # Execute remote tools
    result = await protocol.execute_remote_tool(
        'network_b', 
        'network_b_weather_service',
        {'location': 'Berlin', 'forecast_days': 3},
        'local_weather_agent'
    )
    
    console.print(f"\n[green]A2A Execution Result:[/green]")
    console.print(f"Success: {result['success']}")
    console.print(f"Result: {result.get('result', 'N/A')}")


async def main():
    """Run all tool integration demonstrations."""
    console.print("[bold green]🚀 ADK Tool Integration Demonstrations[/bold green]")
    
    try:
        # Demo 1: Basic agent with tools
        await demo_basic_agent_tools()
        
        # Demo 2: Multi-agent tool sharing
        await demo_multi_agent_tools()
        
        # Demo 3: A2A tool protocol
        await demo_a2a_tool_protocol()
        
        console.print("\n[bold green]✅ All demonstrations completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Demo failed: {e}[/bold red]")
        logger.error(f"Tool integration demo error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
