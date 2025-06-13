"""
Agent-to-Agent (A2A) Protocol Implementation.

This module demonstrates A2A communication patterns:
1. Basic A2A protocol setup
2. Remote agent communication
3. Distributed agent coordination
4. Message passing and serialization
5. Network-based multi-agent systems
"""

import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
from aiohttp import web
import websockets

from config import config
from utils import logger, timer, format_agent_response_for_ui, format_agent_response_for_logs


class MessageType(Enum):
    """Types of A2A messages."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


@dataclass
class A2AMessage:
    """Standard A2A message format."""
    id: str
    type: MessageType
    sender_id: str
    receiver_id: str
    payload: Dict[str, Any]
    timestamp: str
    correlation_id: Optional[str] = None
    
    @classmethod
    def create_request(cls, sender_id: str, receiver_id: str, 
                      action: str, data: Dict[str, Any]) -> 'A2AMessage':
        """Create a request message."""
        return cls(
            id=str(uuid.uuid4()),
            type=MessageType.REQUEST,
            sender_id=sender_id,
            receiver_id=receiver_id,
            payload={"action": action, "data": data},
            timestamp=datetime.now().isoformat()
        )
    
    @classmethod
    def create_response(cls, sender_id: str, receiver_id: str,
                       success: bool, data: Dict[str, Any],
                       correlation_id: str) -> 'A2AMessage':
        """Create a response message."""
        return cls(
            id=str(uuid.uuid4()),
            type=MessageType.RESPONSE,
            sender_id=sender_id,
            receiver_id=receiver_id,
            payload={"success": success, "data": data},
            timestamp=datetime.now().isoformat(),
            correlation_id=correlation_id
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        data = asdict(self)
        data['type'] = self.type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'A2AMessage':
        """Create message from dictionary."""
        data['type'] = MessageType(data['type'])
        return cls(**data)


class A2AAgent:
    """Base class for A2A-enabled agents."""
    
    def __init__(self, agent_id: str, name: str, port: int = None):
        self.agent_id = agent_id
        self.name = name
        self.port = port or config.a2a.server_port
        self.message_handlers: Dict[str, Callable] = {}
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.is_running = False
        self.server = None
        
        # Register default handlers
        self.register_handler("ping", self._handle_ping)
        self.register_handler("get_info", self._handle_get_info)
    
    def register_handler(self, action: str, handler: Callable):
        """Register a message handler for an action."""
        self.message_handlers[action] = handler
        logger.info(f"Registered handler for action: {action}")
    
    async def _handle_ping(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ping requests."""
        return {"pong": True, "timestamp": datetime.now().isoformat()}
    
    async def _handle_get_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle info requests."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "port": self.port,
            "handlers": list(self.message_handlers.keys()),
            "status": "running" if self.is_running else "stopped"
        }
    
    async def handle_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Process incoming A2A message."""
        try:
            if message.type == MessageType.REQUEST:
                action = message.payload.get("action")
                data = message.payload.get("data", {})
                
                if action in self.message_handlers:
                    handler = self.message_handlers[action]
                    result = await handler(data)
                    
                    return A2AMessage.create_response(
                        sender_id=self.agent_id,
                        receiver_id=message.sender_id,
                        success=True,
                        data=result,
                        correlation_id=message.id
                    )
                else:
                    return A2AMessage.create_response(
                        sender_id=self.agent_id,
                        receiver_id=message.sender_id,
                        success=False,
                        data={"error": f"Unknown action: {action}"},
                        correlation_id=message.id
                    )
            
            elif message.type == MessageType.RESPONSE:
                # Handle response to our request
                correlation_id = message.correlation_id
                if correlation_id in self.pending_requests:
                    future = self.pending_requests.pop(correlation_id)
                    future.set_result(message)
            
            return None
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return A2AMessage.create_response(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                success=False,
                data={"error": str(e)},
                correlation_id=message.id
            )
    
    async def send_request(self, target_agent_url: str, action: str, 
                          data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request to another agent."""
        message = A2AMessage.create_request(
            sender_id=self.agent_id,
            receiver_id="unknown",  # Will be filled by target
            action=action,
            data=data
        )
        
        # Store pending request
        future = asyncio.Future()
        self.pending_requests[message.id] = future
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{target_agent_url}/a2a/message",
                    json=message.to_dict(),
                    timeout=aiohttp.ClientTimeout(total=config.a2a.timeout)
                ) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        response_message = A2AMessage.from_dict(response_data)
                        return response_message.payload
                    else:
                        raise Exception(f"Request failed with status {response.status}")
        
        except Exception as e:
            if message.id in self.pending_requests:
                self.pending_requests.pop(message.id)
            logger.error(f"Failed to send request: {e}")
            raise
    
    async def start_server(self):
        """Start the A2A server."""
        app = web.Application()
        app.router.add_post('/a2a/message', self._handle_http_message)
        app.router.add_get('/a2a/info', self._handle_http_info)
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, config.a2a.server_host, self.port)
        await site.start()
        
        self.is_running = True
        logger.info(f"A2A agent {self.name} started on port {self.port}")
    
    async def stop_server(self):
        """Stop the A2A server."""
        self.is_running = False
        logger.info(f"A2A agent {self.name} stopped")
    
    async def _handle_http_message(self, request):
        """Handle HTTP message requests."""
        try:
            data = await request.json()
            message = A2AMessage.from_dict(data)
            
            response_message = await self.handle_message(message)
            
            if response_message:
                return web.json_response(response_message.to_dict())
            else:
                return web.Response(status=204)  # No content
        
        except Exception as e:
            logger.error(f"Error handling HTTP message: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_http_info(self, request):
        """Handle HTTP info requests."""
        info = await self._handle_get_info({})
        return web.json_response(info)


class SmartA2AAgent(A2AAgent):
    """A2A agent with AI capabilities using ADK."""
    def __init__(self, agent_id: str, name: str, model: str = None, port: int = None):
        super().__init__(agent_id, name, port)
        
        from google.adk.agents import LlmAgent
        
        self.llm_agent = LlmAgent(
            name=name,
            model=model or config.adk.default_model,
            instruction=(
                f"You are {name}, an AI agent capable of communicating with "
                "other agents via A2A protocol. Process requests logically "
                "and provide helpful responses."
            ),
            description=f"Smart A2A agent: {name}"
        )
          # Register AI-powered handlers
        self.register_handler("chat", self._handle_chat)
        self.register_handler("analyze", self._handle_analyze)
        self.register_handler("collaborate", self._handle_collaborate)

    def process_a2a_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process A2A message and return response for UI demonstration."""
        try:
            action = message_data.get("action", "chat")
            content = message_data.get("content", "")
            sender = message_data.get("sender", "unknown")
            
            # Process based on action type
            if action == "chat":
                prompt = f"Agent {sender} says: {content}. Please respond naturally."
            elif action == "analyze":
                prompt = f"Please analyze this request from {sender}: {content}"
            elif action == "collaborate":
                prompt = f"Agent {sender} wants to collaborate on: {content}. How can you help?"
            elif action == "ping":
                return {
                    "response": f"Pong from {self.name}! I'm online and ready.",
                    "status": "active",
                    "agent": self.name,
                    "timestamp": datetime.now().isoformat()
                }
            elif action == "data_transfer":
                prompt = f"Agent {sender} sent data: {content}. Please acknowledge and summarize."
            elif action == "task_request":
                prompt = f"Agent {sender} requests task: {content}. Please respond with your capability to handle this."
            else:
                prompt = f"Agent {sender} sent: {content}. Action type: {action}. Please respond appropriately."            # Run the LLM to generate response
            try:
                # Use the LLM agent chat method directly instead of runner
                response = self.llm_agent.chat(prompt)
            except Exception as llm_error:
                # If LLM agent fails, provide fallback response
                response = f"I'm {self.name}, an AI agent. I encountered an issue processing your request: {llm_error}"
            
            return {
                "response": response,
                "action": action,
                "processed_by": self.name,
                "original_sender": sender,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
        except Exception as e:
            return {
                "response": f"Error processing message: {str(e)}",
                "action": message_data.get("action", "unknown"),
                "processed_by": self.name,
                "error": True,
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
    
    async def _handle_chat(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle chat requests using AI."""
        message = data.get("message", "")
        try:
            # Use LLM agent chat method directly
            response = self.llm_agent.chat(message)
            
            return {
                "response": response,
                "agent": self.name,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": f"Chat processing failed: {str(e)}"}
        
    async def _handle_analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle analysis requests using AI."""
        content = data.get("content", "")
        analysis_type = data.get("type", "general")       
        try:
            prompt = f"Analyze the following content using {analysis_type} analysis:\n{content}"
            # Use LLM agent chat method directly
            analysis = self.llm_agent.chat(prompt)
            
            return {
                "analysis": analysis,
                "type": analysis_type,
                "agent": self.name,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
        
    async def _handle_collaborate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle collaboration requests with other agents."""
        task = data.get("task", "")
        context = data.get("context", {})       
        try:
            prompt = (
                f"Collaborate on this task: {task}\n"
                f"Context: {json.dumps(context, indent=2)}\n"
                "Provide your contribution to this collaborative effort."
            )
            
            # Use LLM agent chat method directly
            contribution = self.llm_agent.chat(prompt)
            
            return {
                "contribution": contribution,
                "task": task,
                "agent": self.name,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": f"Collaboration failed: {str(e)}"}


class A2AOrchestrator:
    """Orchestrates communication between multiple A2A agents."""
    
    def __init__(self):
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.message_history: List[A2AMessage] = []
    
    def register_agent(self, agent_id: str, agent_url: str, agent_info: Dict[str, Any]):
        """Register an agent with the orchestrator."""
        self.agents[agent_id] = {
            "url": agent_url,
            "info": agent_info,
            "last_seen": datetime.now(),
            "status": "active"
        }
        logger.info(f"Registered agent: {agent_id}")
    
    async def discover_agents(self, agent_urls: List[str]) -> List[Dict[str, Any]]:
        """Discover agents at given URLs."""
        discovered = []
        
        for url in agent_urls:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{url}/a2a/info") as response:
                        if response.status == 200:
                            info = await response.json()
                            discovered.append({
                                "url": url,
                                "info": info
                            })
                            self.register_agent(info["agent_id"], url, info)
            except Exception as e:
                logger.warning(f"Failed to discover agent at {url}: {e}")
        
        return discovered
    
    async def broadcast_message(self, sender_id: str, action: str, 
                              data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Broadcast a message to all registered agents."""
        results = []
        
        for agent_id, agent_info in self.agents.items():
            if agent_id != sender_id:  # Don't send to self
                try:
                    # Create a temporary agent to send the message
                    temp_agent = A2AAgent(sender_id, "orchestrator")
                    result = await temp_agent.send_request(
                        agent_info["url"], action, data
                    )
                    results.append({
                        "agent_id": agent_id,
                        "success": True,
                        "result": result
                    })
                except Exception as e:
                    results.append({
                        "agent_id": agent_id,
                        "success": False,
                        "error": str(e)
                    })
        
        return results
    
    async def coordinate_collaboration(self, task: str, participant_ids: List[str]) -> Dict[str, Any]:
        """Coordinate a collaborative task between multiple agents."""
        collaboration_results = {
            "task": task,
            "participants": participant_ids,
            "contributions": [],
            "start_time": datetime.now().isoformat()
        }
        
        # Send collaboration request to each participant
        for agent_id in participant_ids:
            if agent_id in self.agents:
                try:
                    temp_agent = A2AAgent("orchestrator", "orchestrator")
                    result = await temp_agent.send_request(
                        self.agents[agent_id]["url"],
                        "collaborate",
                        {
                            "task": task,
                            "context": {
                                "participants": participant_ids,
                                "orchestrator": "A2A_Orchestrator"
                            }
                        }
                    )
                    
                    collaboration_results["contributions"].append({
                        "agent_id": agent_id,
                        "contribution": result,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    collaboration_results["contributions"].append({
                        "agent_id": agent_id,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
        
        collaboration_results["end_time"] = datetime.now().isoformat()
        return collaboration_results


# Example usage and demonstration
async def demonstrate_a2a_protocol():
    """Demonstrate A2A protocol capabilities."""
    logger.info("Starting A2A protocol demonstration")
    
    # Create multiple smart agents
    agent1 = SmartA2AAgent("agent_1", "ResearchBot", port=8501)
    agent2 = SmartA2AAgent("agent_2", "AnalysisBot", port=8502)
    agent3 = SmartA2AAgent("agent_3", "WritingBot", port=8503)
    
    # Start agents
    await agent1.start_server()
    await agent2.start_server()
    await agent3.start_server()
    
    # Wait a moment for servers to start
    await asyncio.sleep(1)
    
    try:
        # Test direct communication
        response = await agent1.send_request(
            "http://localhost:8502",
            "chat",
            {"message": "Hello from Agent 1!"}
        )
        print(f"Direct communication result: {response}")
        
        # Test orchestrated collaboration
        orchestrator = A2AOrchestrator()
        
        # Discover agents
        agent_urls = [
            "http://localhost:8501",
            "http://localhost:8502",
            "http://localhost:8503"
        ]
        
        discovered = await orchestrator.discover_agents(agent_urls)
        print(f"Discovered {len(discovered)} agents")
        
        # Coordinate collaboration
        collaboration = await orchestrator.coordinate_collaboration(
            "Create a comprehensive market analysis report",
            ["agent_1", "agent_2", "agent_3"]
        )
        
        print("Collaboration Results:")
        for contribution in collaboration["contributions"]:
            if "error" not in contribution:
                print(f"- {contribution['agent_id']}: {contribution['contribution']['contribution'][:100]}...")
        
    finally:
        # Clean up
        await agent1.stop_server()
        await agent2.stop_server()
        await agent3.stop_server()
    
    logger.info("A2A protocol demonstration completed")


if __name__ == "__main__":
    asyncio.run(demonstrate_a2a_protocol())
