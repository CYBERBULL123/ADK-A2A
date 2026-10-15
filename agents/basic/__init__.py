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
    Get real weather information using OpenWeatherMap API.
    
    Args:
        location: The location to get weather for (city name, city,country, or coordinates)
        
    Returns:
        Comprehensive weather information for the location
    """
    import requests
    import json
    from datetime import datetime
    
    try:
        # Get API key from config
        api_key = config.openweather_api_key
        if not api_key:
            return "❌ Weather service unavailable: API key not configured. Please add OPENWEATHER_API_KEY to your .env file."
        
        # Clean and validate location input
        location = location.strip()
        if not location:
            return "❌ Please provide a valid location name."
        
        # OpenWeatherMap API endpoint
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        
        # API parameters
        params = {
            'q': location,
            'appid': api_key,
            'units': 'metric',  # Celsius, use 'imperial' for Fahrenheit
            'lang': 'en'
        }
        
        # Make API request with timeout
        response = requests.get(base_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract weather information
            city = data['name']
            country = data['sys']['country']
            temp = data['main']['temp']
            feels_like = data['main']['feels_like']
            humidity = data['main']['humidity']
            pressure = data['main']['pressure']
            wind_speed = data['wind'].get('speed', 0)
            wind_deg = data['wind'].get('deg', 0)
            visibility = data.get('visibility', 0) / 1000  # Convert to km
            
            weather_main = data['weather'][0]['main']
            weather_desc = data['weather'][0]['description'].title()
            weather_icon = data['weather'][0]['icon']
            
            # Sunrise and sunset
            sunrise = datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M')
            sunset = datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M')
            
            # Weather emoji mapping
            weather_emojis = {
                'Clear': '☀️',
                'Clouds': '☁️',
                'Rain': '🌧️',
                'Drizzle': '🌦️',
                'Thunderstorm': '⛈️',
                'Snow': '❄️',
                'Mist': '🌫️',
                'Smoke': '💨',
                'Haze': '🌫️',
                'Dust': '💨',
                'Fog': '🌫️',
                'Sand': '💨',
                'Ash': '💨',
                'Squall': '💨',
                'Tornado': '🌪️'
            }
            
            weather_emoji = weather_emojis.get(weather_main, '🌤️')
            
            # Wind direction
            wind_directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                             'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
            wind_dir = wind_directions[round(wind_deg / 22.5) % 16] if wind_speed > 0 else 'N/A'
            
            # Temperature conversion helper
            temp_f = (temp * 9/5) + 32
            feels_like_f = (feels_like * 9/5) + 32
            
            # Format comprehensive weather report
            weather_report = f"""
🌍 **Weather Report for {city}, {country}**

{weather_emoji} **Current Conditions:** {weather_desc}

🌡️ **Temperature:**
   • Current: {temp:.1f}°C ({temp_f:.1f}°F)
   • Feels like: {feels_like:.1f}°C ({feels_like_f:.1f}°F)

💧 **Humidity:** {humidity}%
🔽 **Pressure:** {pressure} hPa
💨 **Wind:** {wind_speed:.1f} m/s ({wind_dir})
👁️ **Visibility:** {visibility:.1f} km

🌅 **Sunrise:** {sunrise}
🌇 **Sunset:** {sunset}

📊 **Weather Summary:** The weather in {city} is currently {weather_desc.lower()} with a temperature of {temp:.1f}°C. {_get_weather_advice(weather_main, temp, wind_speed, humidity)}
            """.strip()
            
            logger.info(f"Successfully fetched weather data for {city}, {country}")
            return weather_report
            
        elif response.status_code == 404:
            return f"❌ Location '{location}' not found. Please check the spelling and try again. You can use formats like 'London', 'Paris,FR', or 'New York,US'."
        
        elif response.status_code == 401:
            return "❌ Weather service authentication failed. Please check your API key configuration."
        
        else:
            return f"❌ Weather service temporarily unavailable (Error {response.status_code}). Please try again later."
    
    except requests.exceptions.Timeout:
        return f"⏱️ Weather service request timed out for '{location}'. Please try again."
    
    except requests.exceptions.ConnectionError:
        return f"🌐 Unable to connect to weather service. Please check your internet connection."
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Weather API request error: {e}")
        return f"❌ Weather service error: {str(e)}"
    
    except KeyError as e:
        logger.error(f"Weather data parsing error: {e}")
        return f"❌ Unexpected weather data format. Please try again or contact support."
    
    except Exception as e:
        logger.error(f"Unexpected weather tool error: {e}")
        return f"❌ An unexpected error occurred while fetching weather data: {str(e)}"


def _get_weather_advice(weather_main: str, temp: float, wind_speed: float, humidity: float) -> str:
    """
    Get weather-based advice for the current conditions.
    
    Args:
        weather_main: Main weather condition
        temp: Temperature in Celsius
        wind_speed: Wind speed in m/s
        humidity: Humidity percentage
        
    Returns:
        Helpful advice based on current weather conditions
    """
    advice = []
    
    # Temperature-based advice
    if temp < 0:
        advice.append("It's freezing! Dress warmly and watch for icy conditions.")
    elif temp < 10:
        advice.append("It's quite cold. Consider wearing layers and a warm jacket.")
    elif temp > 30:
        advice.append("It's very warm! Stay hydrated and seek shade when possible.")
    elif temp > 25:
        advice.append("Pleasant warm weather. Perfect for outdoor activities!")
    
    # Weather condition advice
    if weather_main in ['Rain', 'Drizzle']:
        advice.append("Don't forget your umbrella or raincoat!")
    elif weather_main == 'Thunderstorm':
        advice.append("Thunderstorms expected. Stay indoors if possible and avoid outdoor activities.")
    elif weather_main == 'Snow':
        advice.append("Snowy conditions. Drive carefully and allow extra travel time.")
    elif weather_main in ['Mist', 'Fog']:
        advice.append("Reduced visibility due to fog. Drive carefully and use headlights.")
    
    # Wind advice
    if wind_speed > 10:
        advice.append("Very windy conditions. Secure loose objects and be cautious outdoors.")
    elif wind_speed > 5:
        advice.append("Breezy conditions. Light jacket recommended.")
    
    # Humidity advice
    if humidity > 80:
        advice.append("High humidity may make it feel warmer than it is.")
    elif humidity < 30:
        advice.append("Low humidity. Consider using moisturizer and stay hydrated.")
    
    return " ".join(advice) if advice else "Enjoy the weather!"


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
        "stateful": StatefulAgent,
        "code_generator": CodeGenerationAgent
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

class CodeGenerationAgent:
    """Specialized agent for generating Python tool code."""
    
    def __init__(self, name: str = "CodeGenerationAgent"):
        self.name = name
        self.agent = Agent(
            name=name,
            model=config.adk.default_model,
            instruction=(
                "You are an expert Python developer specializing in creating MCP (Model Context Protocol) tools. "
                "Your role is to generate clean, functional, and well-documented Python functions based on user descriptions. "
                
                "IMPORTANT GUIDELINES:\n"
                "1. Always create complete, executable Python functions\n"
                "2. Include proper type hints and docstrings\n"
                "3. Handle errors gracefully with try-catch blocks\n"
                "4. Return structured data (Dict[str, Any]) with 'success' and 'result' fields\n"
                "5. Import required modules within the function\n"
                "6. Create realistic implementations, not just placeholders\n"
                "7. Follow PEP 8 coding standards\n"
                "8. Make the function name descriptive and snake_case\n"
                
                "RESPONSE FORMAT:\n"
                "Provide ONLY the Python function code, no explanations or markdown formatting.\n"
                "The function should be ready to execute immediately."
            ),
            description="A specialized agent for generating Python tool functions."
        )
        
        # Set up session service and runner
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.agent,
            app_name="code_gen_agent_app",
            session_service=self.session_service
        )
        
        # Session details
        self.user_id = "default_user"
        self.session_id = f"{name}_session"
        self.session = None
    
    async def _initialize_session(self):
        """Initialize the session."""
        if self.session is None:
            try:
                self.session = await self.session_service.create_session(
                    app_name="code_gen_agent_app",
                    user_id=self.user_id,
                    session_id=self.session_id
                )
            except Exception as e:
                logger.error(f"Failed to initialize session: {e}")
                raise
    
    @timer
    def generate_tool_code(self, description: str, category: str = "", complexity: str = "") -> str:
        """Generate Python tool code based on description."""
        try:
            # Check if we're already in an event loop (like Streamlit)
            try:
                loop = asyncio.get_running_loop()
                # We're in a running event loop, so we need to use a different approach
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._run_code_generation, description, category, complexity)
                    return future.result(timeout=30)  # 30 second timeout
            except RuntimeError:
                # No running event loop, safe to use asyncio.run()
                return asyncio.run(self._generate_code_async(description, category, complexity))
        except Exception as e:
            logger.error(f"Error in code generation: {e}")
            return self._fallback_code_generation(description, category)
    
    def _run_code_generation(self, description: str, category: str, complexity: str) -> str:
        """Helper method to run code generation in a separate thread."""
        return asyncio.run(self._generate_code_async(description, category, complexity))
    
    async def _generate_code_async(self, description: str, category: str, complexity: str) -> str:
        """Async implementation of code generation."""
        try:
            # Ensure session is initialized
            await self._initialize_session()
            
            # Create detailed prompt for code generation
            prompt = self._create_code_generation_prompt(description, category, complexity)
            
            # Create message content
            content = types.Content(role='user', parts=[types.Part(text=prompt)])
            
            # Get response from agent
            final_response_text = ""
            
            async for event in self.runner.run_async(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=content
            ):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        final_response_text = event.content.parts[0].text
            
            logger.info(f"Code generation agent generated code for: {description[:50]}...")
            
            # Clean up the response to ensure it's valid Python code
            return self._clean_generated_code(final_response_text)
            
        except Exception as e:
            logger.error(f"Error in async code generation: {e}")
            return self._fallback_code_generation(description, category)
    
    def _create_code_generation_prompt(self, description: str, category: str, complexity: str) -> str:
        """Create a detailed prompt for code generation."""
        return f"""
Generate a Python function for this tool:

DESCRIPTION: {description}
CATEGORY: {category}
COMPLEXITY: {complexity}

REQUIREMENTS:
- Function name should be descriptive and snake_case
- Include proper type hints: -> Dict[str, Any]
- Add comprehensive docstring
- Handle all errors with try-except
- Return dict with 'success' boolean and 'result' or 'error' fields
- Import modules inside the function
- Create a working implementation, not placeholder code

EXAMPLES OF GOOD IMPLEMENTATIONS:

For password generation:
```python
def generate_secure_password(length: int = 12, include_symbols: bool = True) -> Dict[str, Any]:
    \"\"\"Generate a secure password with customizable options.\"\"\"
    import random
    import string
    try:
        chars = string.ascii_letters + string.digits
        if include_symbols:
            chars += "!@#$%^&*"
        password = ''.join(random.choice(chars) for _ in range(length))
        return {{'success': True, 'result': password, 'length': len(password)}}
    except Exception as e:
        return {{'success': False, 'error': str(e)}}
```

For file processing:
```python
def process_text_file(file_path: str, operation: str = "count_lines") -> Dict[str, Any]:
    \"\"\"Process a text file with various operations.\"\"\"
    import os
    try:
        if not os.path.exists(file_path):
            return {{'success': False, 'error': 'File not found'}}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if operation == "count_lines":
            result = len(content.splitlines())
        elif operation == "word_count":
            result = len(content.split())
        else:
            result = content
            
        return {{'success': True, 'result': result, 'operation': operation}}
    except Exception as e:
        return {{'success': False, 'error': str(e)}}
```

Generate a similar function for the given description. Return ONLY the Python code, no markdown or explanations.
"""
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean and validate generated code."""
        # Remove markdown code blocks if present
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        # Clean up extra whitespace
        code = code.strip()
        
        # Ensure proper indentation
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip():  # Skip empty lines
                cleaned_lines.append(line)
            else:
                cleaned_lines.append("")
        
        return '\n'.join(cleaned_lines)
    
    def _fallback_code_generation(self, description: str, category: str) -> str:
        """Fallback code generation when AI is not available."""
        # Extract function name from description
        words = description.lower().split()
        func_name = '_'.join([w for w in words if w.isalnum()][:3])
        if not func_name:
            func_name = "generated_tool"
        
        return f'''def {func_name}(input_data: str) -> Dict[str, Any]:
    """
    {description}
    
    Auto-generated fallback implementation.
    """
    from datetime import datetime
    from typing import Dict, Any
    
    try:
        # Process the input data
        result = f"Processed: {{input_data}}"
        
        return {{
            'success': True,
            'result': result,
            'description': '{description}',
            'category': '{category}',
            'timestamp': datetime.now().isoformat()
        }}
    except Exception as e:
        return {{
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }}'''
