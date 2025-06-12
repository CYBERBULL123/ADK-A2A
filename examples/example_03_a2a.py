"""
Example 03: Agent-to-Agent (A2A) Protocol Implementation

This example demonstrates:
1. HTTP-based agent communication
2. Message routing and protocol handling
3. Distributed agent networks
4. Real-time agent interaction
"""

import asyncio
import json
import logging
from typing import Dict, Any, List
import aiohttp
from aiohttp import web
import threading
import time

from agents.a2a import (
    A2AMessage, A2AAgent, AgentServer, 
    MessageRouter, HttpAgentClient
)
from utils import setup_logging, format_output

# Setup logging
logger = setup_logging(__name__)


class WeatherAgent(A2AAgent):
    """Agent that provides weather information"""
    
    def __init__(self, agent_id: str, port: int):
        super().__init__(agent_id, port)
        self.weather_data = {
            "New York": {"temp": 72, "condition": "Sunny", "humidity": 45},
            "London": {"temp": 65, "condition": "Cloudy", "humidity": 70},
            "Tokyo": {"temp": 78, "condition": "Rainy", "humidity": 85},
            "Sydney": {"temp": 68, "condition": "Partly Cloudy", "humidity": 55}
        }
    
    async def handle_message(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle incoming A2A messages"""
        try:
            if message.message_type == "weather_request":
                city = message.payload.get("city", "")
                weather = self.weather_data.get(city)
                
                if weather:
                    response_payload = {
                        "city": city,
                        "weather": weather,
                        "status": "success"
                    }
                else:
                    response_payload = {
                        "error": f"Weather data not available for {city}",
                        "status": "error"
                    }
                
                return {
                    "message_id": f"resp_{message.message_id}",
                    "sender_id": self.agent_id,
                    "receiver_id": message.sender_id,
                    "message_type": "weather_response",
                    "payload": response_payload
                }
            
            return {"error": "Unknown message type"}
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return {"error": str(e)}


class NewsAgent(A2AAgent):
    """Agent that provides news information"""
    
    def __init__(self, agent_id: str, port: int):
        super().__init__(agent_id, port)
        self.news_data = [
            {"title": "Tech Innovation Summit 2025", "category": "technology", "importance": "high"},
            {"title": "Global Climate Agreement Reached", "category": "environment", "importance": "high"},
            {"title": "AI Breakthrough in Medicine", "category": "health", "importance": "medium"},
            {"title": "Space Mission Launch Success", "category": "science", "importance": "medium"},
            {"title": "Economic Markets Show Growth", "category": "finance", "importance": "low"}
        ]
    
    async def handle_message(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle incoming A2A messages"""
        try:
            if message.message_type == "news_request":
                category = message.payload.get("category", "all")
                limit = message.payload.get("limit", 3)
                
                if category == "all":
                    filtered_news = self.news_data[:limit]
                else:
                    filtered_news = [
                        news for news in self.news_data 
                        if news["category"] == category
                    ][:limit]
                
                response_payload = {
                    "news": filtered_news,
                    "category": category,
                    "count": len(filtered_news),
                    "status": "success"
                }
                
                return {
                    "message_id": f"resp_{message.message_id}",
                    "sender_id": self.agent_id,
                    "receiver_id": message.sender_id,
                    "message_type": "news_response",
                    "payload": response_payload
                }
            
            return {"error": "Unknown message type"}
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return {"error": str(e)}


class CoordinatorAgent(A2AAgent):
    """Central coordinator that orchestrates multiple agents"""
    
    def __init__(self, agent_id: str, port: int):
        super().__init__(agent_id, port)
        self.known_agents = {}
        self.client = HttpAgentClient()
    
    def register_agent(self, agent_id: str, address: str, capabilities: List[str]):
        """Register an agent with its capabilities"""
        self.known_agents[agent_id] = {
            "address": address,
            "capabilities": capabilities,
            "status": "active"
        }
        logger.info(f"Registered agent {agent_id} with capabilities: {capabilities}")
    
    async def handle_message(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle incoming A2A messages"""
        try:
            if message.message_type == "info_request":
                query_type = message.payload.get("type", "")
                
                if query_type == "weather":
                    # Find weather agent and forward request
                    weather_agent = None
                    for agent_id, info in self.known_agents.items():
                        if "weather" in info["capabilities"]:
                            weather_agent = agent_id
                            break
                    
                    if weather_agent:
                        weather_message = A2AMessage(
                            message_id=f"fwd_{message.message_id}",
                            sender_id=self.agent_id,
                            receiver_id=weather_agent,
                            message_type="weather_request",
                            payload={"city": message.payload.get("city", "New York")}
                        )
                        
                        # Forward request to weather agent
                        agent_info = self.known_agents[weather_agent]
                        response = await self.client.send_message(
                            agent_info["address"], weather_message
                        )
                        
                        return {
                            "message_id": f"coord_resp_{message.message_id}",
                            "sender_id": self.agent_id,
                            "receiver_id": message.sender_id,
                            "message_type": "info_response",
                            "payload": response
                        }
                
                elif query_type == "news":
                    # Find news agent and forward request
                    news_agent = None
                    for agent_id, info in self.known_agents.items():
                        if "news" in info["capabilities"]:
                            news_agent = agent_id
                            break
                    
                    if news_agent:
                        news_message = A2AMessage(
                            message_id=f"fwd_{message.message_id}",
                            sender_id=self.agent_id,
                            receiver_id=news_agent,
                            message_type="news_request",
                            payload={
                                "category": message.payload.get("category", "all"),
                                "limit": message.payload.get("limit", 3)
                            }
                        )
                        
                        # Forward request to news agent
                        agent_info = self.known_agents[news_agent]
                        response = await self.client.send_message(
                            agent_info["address"], news_message
                        )
                        
                        return {
                            "message_id": f"coord_resp_{message.message_id}",
                            "sender_id": self.agent_id,
                            "receiver_id": message.sender_id,
                            "message_type": "info_response",
                            "payload": response
                        }
                
                return {"error": f"No agent available for {query_type}"}
            
            return {"error": "Unknown message type"}
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return {"error": str(e)}


async def run_agent_server(agent: A2AAgent):
    """Run an agent server in the background"""
    try:
        await agent.start_server()
    except Exception as e:
        logger.error(f"Error running agent server {agent.agent_id}: {e}")


async def test_a2a_communication():
    """Test A2A communication between multiple agents"""
    print(format_output("🤖 Starting A2A Communication Test", "info"))
    
    # Create agents
    weather_agent = WeatherAgent("weather_agent", 8001)
    news_agent = NewsAgent("news_agent", 8002)
    coordinator = CoordinatorAgent("coordinator", 8000)
    
    # Start servers in background tasks
    tasks = []
    
    # Start agent servers
    tasks.append(asyncio.create_task(run_agent_server(weather_agent)))
    tasks.append(asyncio.create_task(run_agent_server(news_agent)))
    tasks.append(asyncio.create_task(run_agent_server(coordinator)))
    
    # Wait a moment for servers to start
    await asyncio.sleep(2)
    
    try:
        # Register agents with coordinator
        coordinator.register_agent(
            "weather_agent", 
            "http://localhost:8001", 
            ["weather", "temperature"]
        )
        coordinator.register_agent(
            "news_agent", 
            "http://localhost:8002", 
            ["news", "headlines"]
        )
        
        # Create a client for testing
        client = HttpAgentClient()
        
        print(format_output("📡 Testing Weather Request", "info"))
        
        # Test weather request through coordinator
        weather_request = A2AMessage(
            message_id="test_weather_001",
            sender_id="test_client",
            receiver_id="coordinator",
            message_type="info_request",
            payload={"type": "weather", "city": "New York"}
        )
        
        weather_response = await client.send_message(
            "http://localhost:8000", weather_request
        )
        print(f"Weather Response: {json.dumps(weather_response, indent=2)}")
        
        print(format_output("📰 Testing News Request", "info"))
        
        # Test news request through coordinator
        news_request = A2AMessage(
            message_id="test_news_001",
            sender_id="test_client",
            receiver_id="coordinator",
            message_type="info_request",
            payload={"type": "news", "category": "technology", "limit": 2}
        )
        
        news_response = await client.send_message(
            "http://localhost:8000", news_request
        )
        print(f"News Response: {json.dumps(news_response, indent=2)}")
        
        print(format_output("🔄 Testing Direct Agent Communication", "info"))
        
        # Test direct communication with weather agent
        direct_weather_request = A2AMessage(
            message_id="direct_weather_001",
            sender_id="test_client",
            receiver_id="weather_agent",
            message_type="weather_request",
            payload={"city": "London"}
        )
        
        direct_response = await client.send_message(
            "http://localhost:8001", direct_weather_request
        )
        print(f"Direct Weather Response: {json.dumps(direct_response, indent=2)}")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        print(format_output(f"❌ Test failed: {e}", "error"))
    
    finally:
        # Cancel all background tasks
        for task in tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)


async def demonstrate_agent_discovery():
    """Demonstrate agent discovery and capability matching"""
    print(format_output("🔍 Agent Discovery Demonstration", "info"))
    
    coordinator = CoordinatorAgent("discovery_coordinator", 8003)
    
    # Start coordinator server
    server_task = asyncio.create_task(run_agent_server(coordinator))
    await asyncio.sleep(1)
    
    try:
        # Simulate agent registration
        agents_to_register = [
            ("weather_service", "http://localhost:8101", ["weather", "climate", "forecast"]),
            ("news_service", "http://localhost:8102", ["news", "headlines", "articles"]),
            ("translation_service", "http://localhost:8103", ["translate", "language", "text"]),
            ("calculator_service", "http://localhost:8104", ["math", "calculate", "compute"])
        ]
        
        for agent_id, address, capabilities in agents_to_register:
            coordinator.register_agent(agent_id, address, capabilities)
        
        # Display registered agents
        print("\n📋 Registered Agents:")
        for agent_id, info in coordinator.known_agents.items():
            print(f"  • {agent_id}: {info['capabilities']} @ {info['address']}")
        
        # Demonstrate capability search
        print(f"\n🔎 Finding agents with 'weather' capability:")
        weather_agents = [
            agent_id for agent_id, info in coordinator.known_agents.items()
            if "weather" in info["capabilities"]
        ]
        print(f"  Found: {weather_agents}")
        
        print(f"\n🔎 Finding agents with 'news' capability:")
        news_agents = [
            agent_id for agent_id, info in coordinator.known_agents.items()
            if "news" in info["capabilities"]
        ]
        print(f"  Found: {news_agents}")
        
    except Exception as e:
        logger.error(f"Error during discovery demo: {e}")
    
    finally:
        server_task.cancel()
        await asyncio.gather(server_task, return_exceptions=True)


def run_a2a_examples():
    """Run all A2A protocol examples"""
    print(format_output("🚀 Starting A2A Protocol Examples", "title"))
    
    # Run async examples
    async def main():
        await test_a2a_communication()
        print("\n" + "="*60 + "\n")
        await demonstrate_agent_discovery()
    
    try:
        asyncio.run(main())
        print(format_output("✅ A2A Examples completed successfully!", "success"))
    except KeyboardInterrupt:
        print(format_output("⏹️ A2A Examples interrupted by user", "warning"))
    except Exception as e:
        logger.error(f"Error running A2A examples: {e}")
        print(format_output(f"❌ A2A Examples failed: {e}", "error"))


if __name__ == "__main__":
    run_a2a_examples()
