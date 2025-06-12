"""
Basic single agent examples using Google ADK.

This module demonstrates fundamental agent creation patterns:
1. Simple conversational agent
2. Agent with custom tools
3. Agent with external API integration
4. Agent with memory and state management
"""

from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

from google.adk.agents import Agent, LlmAgent
from google.adk.tools import google_search
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from config import config
from utils import logger, timer, format_agent_response_for_ui, format_agent_response_for_logs


class SimpleAgent:
    """A basic conversational agent using ADK."""
    
    def __init__(self, name: str = "SimpleAgent"):
        self.name = name
        self.agent = Agent(
            name=name,
            model=config.adk.default_model,
            instruction=(
                "You are a helpful AI assistant. Provide clear, "
                "concise, and informative responses to user questions."
            ),
            description="A basic conversational agent for general assistance."
        )
        
        # Set up session service and runner
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.agent,
            app_name="basic_agent_app",
            session_service=self.session_service
        )
          # Session details
        self.user_id = "default_user"
        self.session_id = f"{name}_session"
        self.session = None  # Will be initialized when needed
    
    async def _initialize_session(self):
        """Initialize the session."""
        if self.session is None:
            try:
                self.session = await self.session_service.create_session(
                    app_name="basic_agent_app",
                    user_id=self.user_id,
                    session_id=self.session_id
                )
            except Exception as e:
                logger.error(f"Failed to initialize session: {e}")
                raise
    
    @timer
    def chat(self, message: str) -> str:
        """Send a message to the agent and get a response."""
        try:
            # Run the async chat method
            return asyncio.run(self._chat_async(message))
        except Exception as e:
            logger.error(f"Error in agent chat: {e}")
            return f"Error: {str(e)}"
        
    async def _chat_async(self, message: str) -> str:
        """Async implementation of chat."""
        try:
            # Ensure session is initialized
            await self._initialize_session()
            
            # Create message content
            content = types.Content(role='user', parts=[types.Part(text=message)])
            
            # Get response from agent
            final_response_text = "Agent did not produce a final response."
            
            async for event in self.runner.run_async(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=content
            ):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        final_response_text = event.content.parts[0].text
                    elif event.actions and event.actions.escalate:
                        final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                    break
            logger.info(f"Agent {self.name} responding to user message")
            
            # Log the full response for debugging (but not shown to user)
            logger.debug(format_agent_response_for_logs(final_response_text, self.name))
            
            # Return clean response for UI
            return format_agent_response_for_ui(final_response_text)
            
        except Exception as e:
            logger.error(f"Error in async chat: {e}")
            return f"Error: {str(e)}"
    
    def _extract_response(self, events) -> str:
        """Extract the final response text from ADK events."""
        for event in reversed(events):
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts') and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            return part.text
        return "No response generated"


class SearchAgent:
    """An agent with Google Search capabilities."""
    
    def __init__(self, name: str = "SearchAgent"):
        self.name = name
        self.agent = Agent(
            name=name,
            model=config.adk.default_model,
            instruction=(
                "You are a research assistant with access to Google Search. "
                "Use search when users ask for current information, facts, "
                "or recent events. Provide well-researched, accurate responses."
            ),
            description="An agent capable of searching the web for information.",
            tools=[google_search]
        )
        
        # Set up session service and runner
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.agent,
            app_name="search_agent_app",
            session_service=self.session_service
        )
          # Session details
        self.user_id = "default_user"
        self.session_id = f"{name}_session"
        self.session = None  # Will be initialized when needed
    
    async def _initialize_session(self):
        """Initialize the session."""
        if self.session is None:
            try:
                self.session = await self.session_service.create_session(
                    app_name="search_agent_app",
                    user_id=self.user_id,
                    session_id=self.session_id
                )
            except Exception as e:
                logger.error(f"Failed to initialize session: {e}")
                raise
    
    @timer
    def search_and_respond(self, query: str) -> str:
        """Search for information and provide a comprehensive response."""
        try:
            return asyncio.run(self._search_async(query))
        except Exception as e:
            logger.error(f"Error in search agent: {e}")
            return f"Error: {str(e)}"
        
    async def _search_async(self, query: str) -> str:
        """Async implementation of search."""
        try:
            # Ensure session is initialized
            await self._initialize_session()
            
            # Create message content
            content = types.Content(
                role='user', 
                parts=[types.Part(text=f"Search and provide information about: {query}")]
            )
            
            # Get response from agent
            final_response_text = "Agent did not produce a final response."
            
            async for event in self.runner.run_async(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=content
            ):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        final_response_text = event.content.parts[0].text
                    elif event.actions and event.actions.escalate:
                        final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                    break
                logger.info(f"Search agent processing query")
            
            # Log the full response for debugging
            logger.debug(format_agent_response_for_logs(final_response_text, self.name))
            
            # Return clean response for UI
            return format_agent_response_for_ui(final_response_text)
            
        except Exception as e:
            logger.error(f"Error in async search: {e}")
            return f"Error: {str(e)}"
    
    def _extract_response(self, events) -> str:
        """Extract the final response text from ADK events."""
        for event in reversed(events):
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts') and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            return part.text
        return "No response generated"


def weather_tool(location: str) -> str:
    """
    Custom tool to get weather information.
    
    Args:
        location: The location to get weather for
        
    Returns:
        Weather information for the location
    """
    # This is a mock implementation
    # In practice, you would integrate with a real weather API
    return f"The weather in {location} is sunny with a temperature of 22°C."


def calculator_tool(expression: str) -> str:
    """
    Custom tool to perform mathematical calculations.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        Result of the calculation
    """
    try:
        # Simple calculator - in production, use a proper math parser
        result = eval(expression)  # Note: eval is unsafe, use for demo only
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


class ToolAgent:
    """An agent with custom tools for specific tasks."""
    
    def __init__(self, name: str = "ToolAgent"):
        self.name = name
        self.agent = Agent(
            name=name,
            model=config.adk.default_model,
            instruction=(
                "You are a multi-purpose assistant with access to weather "
                "and calculation tools. Use the appropriate tool based on "
                "the user's request. Always explain your actions clearly."
            ),
            description="An agent with weather and calculator tools.",
            tools=[weather_tool, calculator_tool]
        )
        
        # Set up session service and runner
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.agent,
            app_name="tool_agent_app",
            session_service=self.session_service
        )
          # Session details
        self.user_id = "default_user"
        self.session_id = f"{name}_session"
        self.session = None  # Will be initialized when needed
    
    async def _initialize_session(self):
        """Initialize the session."""
        if self.session is None:
            try:
                self.session = await self.session_service.create_session(
                    app_name="tool_agent_app",
                    user_id=self.user_id,
                    session_id=self.session_id
                )
            except Exception as e:
                logger.error(f"Failed to initialize session: {e}")
                raise
    
    @timer
    def process_request(self, request: str) -> str:
        """Process a user request using available tools."""
        try:
            return asyncio.run(self._process_async(request))
        except Exception as e:
            logger.error(f"Error in tool agent: {e}")
            return f"Error: {str(e)}"
        
    async def _process_async(self, request: str) -> str:
        """Async implementation of process request."""
        try:
            # Ensure session is initialized
            await self._initialize_session()
            
            # Create message content
            content = types.Content(role='user', parts=[types.Part(text=request)])
            
            # Get response from agent
            final_response_text = "Agent did not produce a final response."
            
            async for event in self.runner.run_async(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=content
            ):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        final_response_text = event.content.parts[0].text
                    elif event.actions and event.actions.escalate:
                        final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                    break
                logger.info(f"Tool agent processing request")
            
            # Log the full response for debugging
            logger.debug(format_agent_response_for_logs(final_response_text, self.name))
            
            # Return clean response for UI
            return format_agent_response_for_ui(final_response_text)
            
        except Exception as e:
            logger.error(f"Error in async process: {e}")
            return f"Error: {str(e)}"
    
    def _extract_response(self, events) -> str:
        """Extract the final response text from ADK events."""
        for event in reversed(events):
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts') and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            return part.text
        return "No response generated"


class StatefulAgent:
    """An agent that maintains conversation history and context."""
    
    def __init__(self, name: str = "StatefulAgent"):
        self.name = name
        self.conversation_history: List[Dict[str, str]] = []
        self.agent = Agent(  # Changed from LlmAgent to Agent
            name=name,
            model=config.adk.default_model,
            instruction=(
                "You are a helpful assistant that remembers previous "
                "conversations. Use the conversation history to provide "
                "contextual and personalized responses."
            ),
            description="An agent that maintains conversation state."
        )
        
        # Set up session service and runner
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.agent,
            app_name="stateful_agent_app",
            session_service=self.session_service
        )
          # Session details
        self.user_id = "default_user"
        self.session_id = f"{name}_session"
        self.session = None  # Will be initialized when needed
    
    async def _initialize_session(self):
        """Initialize the session."""
        if self.session is None:
            try:
                self.session = await self.session_service.create_session(
                    app_name="stateful_agent_app",
                    user_id=self.user_id,
                    session_id=self.session_id
                )
            except Exception as e:
                logger.error(f"Failed to initialize session: {e}")
                raise
    
    def add_to_history(self, role: str, content: str):
        """Add a message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 20 messages to prevent context overflow
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    @timer
    def chat_with_memory(self, message: str) -> str:
        """Chat with the agent while maintaining conversation history."""
        try:
            return asyncio.run(self._chat_with_memory_async(message))
        except Exception as e:
            logger.error(f"Error in stateful agent: {e}")
            return f"Error: {str(e)}"
        
    async def _chat_with_memory_async(self, message: str) -> str:
        """Async implementation of chat with memory."""
        try:
            # Ensure session is initialized
            await self._initialize_session()
            
            # Add user message to history
            self.add_to_history("user", message)
            
            # Create context with history
            context = "Previous conversation:\n"
            for msg in self.conversation_history[-5:]:  # Last 5 messages
                context += f"{msg['role']}: {msg['content']}\n"
            
            full_prompt = f"{context}\nCurrent message: {message}"
            
            # Create message content
            content = types.Content(role='user', parts=[types.Part(text=full_prompt)])
            
            # Get response from agent
            final_response_text = "Agent did not produce a final response."
            
            async for event in self.runner.run_async(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=content
            ):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        final_response_text = event.content.parts[0].text
                    elif event.actions and event.actions.escalate:
                        final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                    break
            
            # Add agent response to history
            self.add_to_history("assistant", final_response_text)
            logger.info(f"Stateful agent responding with memory context")
            
            # Log the full response for debugging
            logger.debug(format_agent_response_for_logs(final_response_text, self.name))
            
            # Return clean response for UI
            return format_agent_response_for_ui(final_response_text)
            
        except Exception as e:
            logger.error(f"Error in async chat with memory: {e}")
            return f"Error: {str(e)}"
    
    def _extract_response(self, events) -> str:
        """Extract the final response text from ADK events."""
        for event in reversed(events):
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts') and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            return part.text
        return "No response generated"
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation history."""
        if not self.conversation_history:
            return "No conversation history available."
        
        total_messages = len(self.conversation_history)
        user_messages = len([msg for msg in self.conversation_history if msg["role"] == "user"])
        
        return f"Conversation includes {total_messages} total messages ({user_messages} from user)"
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")


# Factory function to create different types of agents
def create_agent(agent_type: str, name: Optional[str] = None) -> Any:
    """
    Factory function to create different types of basic agents.
    
    Args:
        agent_type: Type of agent to create ('simple', 'search', 'tool', 'stateful')
        name: Optional name for the agent
        
    Returns:
        An instance of the requested agent type
    """
    agents = {
        "simple": SimpleAgent,
        "search": SearchAgent,
        "tool": ToolAgent,
        "stateful": StatefulAgent
    }
    
    if agent_type not in agents:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    agent_class = agents[agent_type]
    
    if name:
        return agent_class(name)
    else:
        return agent_class()


# Example usage function
async def demonstrate_basic_agents():
    """Demonstrate all basic agent types."""
    logger.info("Starting basic agent demonstrations")
    
    # Simple agent demo
    simple = create_agent("simple", "Demo_Simple")
    simple_response = simple.chat("Hello! What can you help me with?")
    print(f"Simple Agent: {simple_response}")
    
    # Search agent demo (if API key available)
    if config.google_api_key:
        search = create_agent("search", "Demo_Search")
        search_response = search.search_and_respond("Latest AI developments")
        print(f"Search Agent: {search_response}")
    
    # Tool agent demo
    tool = create_agent("tool", "Demo_Tool")
    tool_response = tool.process_request("What's the weather in New York?")
    print(f"Tool Agent: {tool_response}")
    
    # Stateful agent demo
    stateful = create_agent("stateful", "Demo_Stateful")
    response1 = stateful.chat_with_memory("My name is Alice")
    response2 = stateful.chat_with_memory("What's my name?")
    print(f"Stateful Agent 1: {response1}")
    print(f"Stateful Agent 2: {response2}")
    
    logger.info("Basic agent demonstrations completed")


if __name__ == "__main__":
    asyncio.run(demonstrate_basic_agents())
