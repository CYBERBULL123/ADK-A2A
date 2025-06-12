# A2A Protocol Guide

## 🌐 Distributed Agent Communication

The Agent-to-Agent (A2A) protocol enables seamless communication between agents across different processes, servers, and networks. This guide covers protocol design, implementation, and best practices.

## Table of Contents
- [Protocol Overview](#protocol-overview)
- [Message Format](#message-format)
- [Communication Patterns](#communication-patterns)
- [Implementation Guide](#implementation-guide)
- [Security Considerations](#security-considerations)
- [Advanced Features](#advanced-features)

## Protocol Overview

### What is A2A Protocol?
A2A (Agent-to-Agent) is a communication protocol that enables:
- **Remote Communication**: Agents can communicate across networks
- **Standardized Messaging**: Common message format for interoperability
- **Async Operations**: Non-blocking communication patterns
- **Scalable Architecture**: Support for large-scale distributed systems

### Key Benefits
- **Distribution**: Agents can run on different machines
- **Scalability**: Add agents across multiple servers
- **Reliability**: Built-in error handling and retries
- **Flexibility**: Support for various communication patterns

### Protocol Stack
```
Application Layer    │ Agent Logic & Business Rules
Message Layer        │ A2A Message Format & Routing
Transport Layer      │ HTTP/WebSocket/TCP
Network Layer        │ IP/Internet
```

## Message Format

### Standard Message Structure
```json
{
  "id": "unique_message_id",
  "type": "request|response|notification|error",
  "sender_id": "agent_identifier",
  "receiver_id": "target_agent_identifier",
  "payload": {
    "action": "method_to_call",
    "data": {...},
    "metadata": {...}
  },
  "timestamp": "2025-06-12T10:30:00Z",
  "correlation_id": "for_request_response_matching",
  "priority": 1,
  "ttl": 30000,
  "route": ["agent1", "agent2", "agent3"]
}
```

### Message Types

#### 1. Request Message
```json
{
  "id": "req_123456",
  "type": "request",
  "sender_id": "agent_research",
  "receiver_id": "agent_analysis",
  "payload": {
    "action": "analyze_data",
    "data": {
      "dataset": "research_results.json",
      "analysis_type": "sentiment"
    },
    "metadata": {
      "timeout": 30,
      "priority": "high"
    }
  },
  "timestamp": "2025-06-12T10:30:00Z",
  "correlation_id": null,
  "priority": 1,
  "ttl": 30000
}
```

#### 2. Response Message
```json
{
  "id": "resp_123457",
  "type": "response",
  "sender_id": "agent_analysis",
  "receiver_id": "agent_research",
  "payload": {
    "action": "analyze_data",
    "data": {
      "sentiment_score": 0.75,
      "confidence": 0.89,
      "analysis_details": {...}
    },
    "metadata": {
      "processing_time": 2.5,
      "model_version": "v2.1"
    }
  },
  "timestamp": "2025-06-12T10:30:02Z",
  "correlation_id": "req_123456",
  "status": "success"
}
```

#### 3. Notification Message
```json
{
  "id": "notif_123458",
  "type": "notification",
  "sender_id": "agent_monitor",
  "receiver_id": "broadcast",
  "payload": {
    "action": "system_alert",
    "data": {
      "alert_type": "high_load",
      "affected_agents": ["agent_1", "agent_2"],
      "severity": "warning"
    }
  },
  "timestamp": "2025-06-12T10:31:00Z",
  "priority": 2
}
```

#### 4. Error Message
```json
{
  "id": "err_123459",
  "type": "error",
  "sender_id": "agent_analysis",
  "receiver_id": "agent_research",
  "payload": {
    "error_code": "INVALID_DATA",
    "error_message": "Dataset format not supported",
    "data": {
      "supported_formats": ["json", "csv", "xml"],
      "received_format": "binary"
    }
  },
  "timestamp": "2025-06-12T10:30:01Z",
  "correlation_id": "req_123456"
}
```

## Communication Patterns

### 1. Request-Response Pattern
```python
class A2AAgent(LlmAgent):
    """Agent with A2A communication capabilities."""
    
    def __init__(self, agent_id: str, name: str, port: int = 8500):
        super().__init__(name)
        self.agent_id = agent_id
        self.port = port
        self.server = None
        self.pending_requests = {}
        self.message_handlers = {}
    
    async def send_request(
        self, 
        target_url: str, 
        action: str, 
        data: dict,
        timeout: int = 30
    ) -> dict:
        """Send a request to another agent."""
        message = {
            "id": self.generate_message_id(),
            "type": "request",
            "sender_id": self.agent_id,
            "receiver_id": self.extract_agent_id(target_url),
            "payload": {
                "action": action,
                "data": data
            },
            "timestamp": datetime.now().isoformat(),
            "ttl": timeout * 1000
        }
        
        # Store pending request
        self.pending_requests[message["id"]] = {
            "message": message,
            "future": asyncio.Future(),
            "timestamp": datetime.now()
        }
        
        try:
            # Send HTTP request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{target_url}/message",
                    json=message,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        # Wait for response
                        result = await asyncio.wait_for(
                            self.pending_requests[message["id"]]["future"],
                            timeout=timeout
                        )
                        return result
                    else:
                        raise A2AError(f"Request failed: {response.status}")
        
        except asyncio.TimeoutError:
            raise A2AError("Request timeout")
        finally:
            # Clean up pending request
            self.pending_requests.pop(message["id"], None)
    
    async def handle_request(self, message: dict) -> dict:
        """Handle incoming request message."""
        action = message["payload"]["action"]
        data = message["payload"]["data"]
        
        try:
            # Execute requested action
            if action in self.message_handlers:
                result = await self.message_handlers[action](data)
            else:
                result = await self.default_handler(action, data)
            
            # Create response message
            response = {
                "id": self.generate_message_id(),
                "type": "response",
                "sender_id": self.agent_id,
                "receiver_id": message["sender_id"],
                "payload": {
                    "action": action,
                    "data": result
                },
                "timestamp": datetime.now().isoformat(),
                "correlation_id": message["id"],
                "status": "success"
            }
            
            return response
        
        except Exception as e:
            # Create error response
            error_response = {
                "id": self.generate_message_id(),
                "type": "error",
                "sender_id": self.agent_id,
                "receiver_id": message["sender_id"],
                "payload": {
                    "error_code": "EXECUTION_ERROR",
                    "error_message": str(e)
                },
                "timestamp": datetime.now().isoformat(),
                "correlation_id": message["id"]
            }
            
            return error_response
```

### 2. Pub-Sub Pattern
```python
class A2APubSubAgent(A2AAgent):
    """Agent with publish-subscribe capabilities."""
    
    def __init__(self, agent_id: str, name: str, port: int = 8500):
        super().__init__(agent_id, name, port)
        self.subscriptions = set()
        self.subscribers = {}
    
    async def publish(self, topic: str, data: dict):
        """Publish a message to a topic."""
        message = {
            "id": self.generate_message_id(),
            "type": "notification",
            "sender_id": self.agent_id,
            "receiver_id": "broadcast",
            "payload": {
                "action": "publish",
                "data": {
                    "topic": topic,
                    "content": data
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to all subscribers
        if topic in self.subscribers:
            tasks = []
            for subscriber_url in self.subscribers[topic]:
                tasks.append(self.send_notification(subscriber_url, message))
            
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def subscribe(self, topic: str, publisher_url: str):
        """Subscribe to a topic from a publisher."""
        subscription_message = {
            "id": self.generate_message_id(),
            "type": "request",
            "sender_id": self.agent_id,
            "receiver_id": self.extract_agent_id(publisher_url),
            "payload": {
                "action": "subscribe",
                "data": {
                    "topic": topic,
                    "subscriber_url": f"http://localhost:{self.port}"
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
        response = await self.send_request(
            publisher_url, "subscribe", 
            subscription_message["payload"]["data"]
        )
        
        if response.get("status") == "success":
            self.subscriptions.add(topic)
    
    async def handle_subscribe_request(self, data: dict) -> dict:
        """Handle subscription request."""
        topic = data["topic"]
        subscriber_url = data["subscriber_url"]
        
        if topic not in self.subscribers:
            self.subscribers[topic] = set()
        
        self.subscribers[topic].add(subscriber_url)
        
        return {"status": "success", "message": f"Subscribed to {topic}"}
```

### 3. Pipeline Pattern
```python
class A2APipelineAgent(A2AAgent):
    """Agent that participates in processing pipelines."""
    
    def __init__(self, agent_id: str, name: str, port: int = 8500):
        super().__init__(agent_id, name, port)
        self.next_agent = None
        self.previous_agent = None
    
    def set_next_agent(self, agent_url: str):
        """Set the next agent in the pipeline."""
        self.next_agent = agent_url
    
    def set_previous_agent(self, agent_url: str):
        """Set the previous agent in the pipeline."""
        self.previous_agent = agent_url
    
    async def process_and_forward(self, data: dict) -> dict:
        """Process data and forward to next agent."""
        # Process data locally
        processed_data = await self.process_data(data)
        
        # Forward to next agent if exists
        if self.next_agent:
            response = await self.send_request(
                self.next_agent,
                "process_pipeline_data",
                processed_data
            )
            return response
        else:
            # End of pipeline
            return {"status": "completed", "final_result": processed_data}
    
    async def process_data(self, data: dict) -> dict:
        """Process data - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement process_data")
```

## Implementation Guide

### Basic A2A Agent Implementation
```python
import aiohttp
import asyncio
from aiohttp import web
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Callable

class SmartA2AAgent(LlmAgent):
    """Advanced A2A agent with full protocol support."""
    
    def __init__(self, agent_id: str, name: str, port: int = 8500):
        super().__init__(name)
        self.agent_id = agent_id
        self.port = port
        self.server = None
        self.app = web.Application()
        self.setup_routes()
        
        # Message handling
        self.pending_requests = {}
        self.message_handlers = {
            "ping": self.handle_ping,
            "chat": self.handle_chat,
            "analyze": self.handle_analyze
        }
        
        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0,
            "uptime_start": datetime.now()
        }
    
    def setup_routes(self):
        """Setup HTTP routes for A2A communication."""
        self.app.router.add_post('/message', self.receive_message)
        self.app.router.add_get('/status', self.get_status)
        self.app.router.add_get('/health', self.health_check)
    
    async def start_server(self):
        """Start the A2A communication server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, 'localhost', self.port)
        await site.start()
        
        self.server = runner
        print(f"A2A Agent {self.agent_id} started on port {self.port}")
    
    async def stop_server(self):
        """Stop the A2A communication server."""
        if self.server:
            await self.server.cleanup()
            self.server = None
    
    async def receive_message(self, request):
        """HTTP endpoint for receiving A2A messages."""
        try:
            message = await request.json()
            self.stats["messages_received"] += 1
            
            # Validate message format
            if not self.validate_message(message):
                return web.json_response(
                    {"error": "Invalid message format"}, 
                    status=400
                )
            
            # Handle different message types
            if message["type"] == "request":
                response = await self.handle_request(message)
                return web.json_response(response)
            
            elif message["type"] == "response":
                await self.handle_response(message)
                return web.json_response({"status": "received"})
            
            elif message["type"] == "notification":
                await self.handle_notification(message)
                return web.json_response({"status": "received"})
            
            elif message["type"] == "error":
                await self.handle_error(message)
                return web.json_response({"status": "received"})
            
            else:
                return web.json_response(
                    {"error": "Unknown message type"}, 
                    status=400
                )
        
        except Exception as e:
            self.stats["errors"] += 1
            return web.json_response(
                {"error": str(e)}, 
                status=500
            )
    
    def validate_message(self, message: dict) -> bool:
        """Validate A2A message format."""
        required_fields = ["id", "type", "sender_id", "payload", "timestamp"]
        return all(field in message for field in required_fields)
    
    async def send_request(
        self, 
        target_url: str, 
        action: str, 
        data: dict,
        timeout: int = 30
    ) -> dict:
        """Send a request to another A2A agent."""
        message_id = str(uuid.uuid4())
        
        message = {
            "id": message_id,
            "type": "request",
            "sender_id": self.agent_id,
            "receiver_id": self.extract_agent_id(target_url),
            "payload": {
                "action": action,
                "data": data,
                "metadata": {
                    "timeout": timeout,
                    "sender_name": self.name
                }
            },
            "timestamp": datetime.now().isoformat(),
            "ttl": timeout * 1000
        }
        
        # Create future for response
        response_future = asyncio.Future()
        self.pending_requests[message_id] = {
            "future": response_future,
            "timestamp": datetime.now()
        }
        
        try:
            # Send HTTP request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{target_url}/message",
                    json=message,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        self.stats["messages_sent"] += 1
                        
                        # Wait for actual response
                        result = await asyncio.wait_for(
                            response_future, timeout=timeout
                        )
                        return result
                    else:
                        raise A2AError(f"HTTP {response.status}: {await response.text()}")
        
        except asyncio.TimeoutError:
            raise A2AError("Request timeout")
        except Exception as e:
            self.stats["errors"] += 1
            raise A2AError(f"Request failed: {str(e)}")
        finally:
            # Clean up
            self.pending_requests.pop(message_id, None)
    
    async def handle_request(self, message: dict) -> dict:
        """Handle incoming request and generate response."""
        action = message["payload"]["action"]
        data = message["payload"]["data"]
        
        try:
            # Execute the requested action
            if action in self.message_handlers:
                result = await self.message_handlers[action](data)
            else:
                result = await self.default_action_handler(action, data)
            
            # Create success response
            response = {
                "id": str(uuid.uuid4()),
                "type": "response",
                "sender_id": self.agent_id,
                "receiver_id": message["sender_id"],
                "payload": {
                    "action": action,
                    "data": result,
                    "metadata": {
                        "processing_time": 0.1,  # Calculate actual time
                        "agent_name": self.name
                    }
                },
                "timestamp": datetime.now().isoformat(),
                "correlation_id": message["id"],
                "status": "success"
            }
            
            return response
        
        except Exception as e:
            # Create error response
            error_response = {
                "id": str(uuid.uuid4()),
                "type": "error",
                "sender_id": self.agent_id,
                "receiver_id": message["sender_id"],
                "payload": {
                    "error_code": "EXECUTION_ERROR",
                    "error_message": str(e),
                    "action": action
                },
                "timestamp": datetime.now().isoformat(),
                "correlation_id": message["id"]
            }
            
            return error_response
    
    async def handle_response(self, message: dict):
        """Handle response to a previous request."""
        correlation_id = message.get("correlation_id")
        
        if correlation_id and correlation_id in self.pending_requests:
            pending = self.pending_requests[correlation_id]
            
            if not pending["future"].done():
                if message.get("status") == "success":
                    pending["future"].set_result(message["payload"]["data"])
                else:
                    error = A2AError(f"Remote error: {message['payload']}")
                    pending["future"].set_exception(error)
    
    # Message handlers
    async def handle_ping(self, data: dict) -> dict:
        """Handle ping request."""
        return {
            "message": "pong",
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "timestamp": datetime.now().isoformat()
        }
    
    async def handle_chat(self, data: dict) -> dict:
        """Handle chat request."""
        message = data.get("message", "")
        response = await self.chat(message)
        
        return {
            "response": response,
            "agent_id": self.agent_id
        }
    
    async def handle_analyze(self, data: dict) -> dict:
        """Handle analysis request."""
        # Implement analysis logic
        analysis_result = {
            "analyzed_data": data,
            "analysis_type": "basic",
            "confidence": 0.85,
            "timestamp": datetime.now().isoformat()
        }
        
        return analysis_result
    
    async def default_action_handler(self, action: str, data: dict) -> dict:
        """Default handler for unknown actions."""
        return {
            "error": f"Unknown action: {action}",
            "available_actions": list(self.message_handlers.keys())
        }
    
    def extract_agent_id(self, url: str) -> str:
        """Extract agent ID from URL."""
        # Simple implementation - in real world, this might query the agent
        return url.split(":")[-1]  # Use port as agent ID
    
    async def get_status(self, request):
        """HTTP endpoint for agent status."""
        uptime = datetime.now() - self.stats["uptime_start"]
        
        status = {
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "status": "active",
            "uptime_seconds": uptime.total_seconds(),
            "messages_sent": self.stats["messages_sent"],
            "messages_received": self.stats["messages_received"],
            "errors": self.stats["errors"],
            "available_actions": list(self.message_handlers.keys())
        }
        
        return web.json_response(status)
    
    async def health_check(self, request):
        """HTTP endpoint for health check."""
        return web.json_response({"status": "healthy"})

class A2AError(Exception):
    """A2A protocol specific error."""
    pass
```

### A2A Orchestrator
```python
class A2AOrchestrator:
    """Orchestrator for managing multiple A2A agents."""
    
    def __init__(self):
        self.agents = {}
        self.agent_urls = {}
        self.communication_graph = {}
    
    def register_agent(self, agent_id: str, agent_url: str, capabilities: list):
        """Register an A2A agent."""
        self.agent_urls[agent_id] = agent_url
        self.agents[agent_id] = {
            "url": agent_url,
            "capabilities": capabilities,
            "status": "unknown",
            "last_ping": None
        }
    
    async def discover_agents(self) -> dict:
        """Discover all registered agents and their status."""
        discovery_results = {}
        
        tasks = []
        for agent_id, agent_info in self.agents.items():
            task = self.ping_agent(agent_id, agent_info["url"])
            tasks.append((agent_id, task))
        
        results = await asyncio.gather(
            *[task for _, task in tasks], 
            return_exceptions=True
        )
        
        for (agent_id, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                discovery_results[agent_id] = {
                    "status": "error",
                    "error": str(result)
                }
            else:
                discovery_results[agent_id] = {
                    "status": "active",
                    "info": result
                }
                self.agents[agent_id]["status"] = "active"
                self.agents[agent_id]["last_ping"] = datetime.now()
        
        return discovery_results
    
    async def ping_agent(self, agent_id: str, agent_url: str) -> dict:
        """Ping an agent to check if it's alive."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{agent_url}/message",
                    json={
                        "id": str(uuid.uuid4()),
                        "type": "request",
                        "sender_id": "orchestrator",
                        "receiver_id": agent_id,
                        "payload": {
                            "action": "ping",
                            "data": {}
                        },
                        "timestamp": datetime.now().isoformat()
                    },
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        raise Exception(f"HTTP {response.status}")
        except Exception as e:
            raise Exception(f"Ping failed: {str(e)}")
    
    async def execute_distributed_workflow(
        self, 
        workflow_steps: list,
        initial_data: dict
    ) -> dict:
        """Execute a workflow across multiple A2A agents."""
        current_data = initial_data
        execution_log = []
        
        for step in workflow_steps:
            agent_id = step["agent_id"]
            action = step["action"]
            
            if agent_id not in self.agent_urls:
                raise ValueError(f"Unknown agent: {agent_id}")
            
            agent_url = self.agent_urls[agent_id]
            
            try:
                # Send request to agent
                async with aiohttp.ClientSession() as session:
                    message = {
                        "id": str(uuid.uuid4()),
                        "type": "request",
                        "sender_id": "orchestrator",
                        "receiver_id": agent_id,
                        "payload": {
                            "action": action,
                            "data": current_data
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    async with session.post(
                        f"{agent_url}/message",
                        json=message,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            current_data = result["payload"]["data"]
                            
                            execution_log.append({
                                "step": step["name"],
                                "agent_id": agent_id,
                                "action": action,
                                "status": "success",
                                "timestamp": datetime.now().isoformat()
                            })
                        else:
                            raise Exception(f"HTTP {response.status}")
            
            except Exception as e:
                execution_log.append({
                    "step": step["name"],
                    "agent_id": agent_id,
                    "action": action,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                break
        
        return {
            "status": "completed" if all(log["status"] == "success" for log in execution_log) else "failed",
            "final_data": current_data,
            "execution_log": execution_log
        }
```

## Security Considerations

### Authentication and Authorization
```python
import jwt
import hashlib
from datetime import datetime, timedelta

class SecureA2AAgent(SmartA2AAgent):
    """A2A agent with security features."""
    
    def __init__(self, agent_id: str, name: str, port: int = 8500, secret_key: str = None):
        super().__init__(agent_id, name, port)
        self.secret_key = secret_key or "default_secret"
        self.authorized_agents = set()
        self.rate_limits = {}
    
    def generate_token(self, target_agent_id: str) -> str:
        """Generate JWT token for authentication."""
        payload = {
            "sender_id": self.agent_id,
            "target_id": target_agent_id,
            "issued_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=1)).isoformat()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def verify_token(self, token: str, sender_id: str) -> bool:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            
            # Check expiration
            expires_at = datetime.fromisoformat(payload["expires_at"])
            if datetime.now() > expires_at:
                return False
            
            # Check sender
            if payload["sender_id"] != sender_id:
                return False
            
            return True
        
        except jwt.InvalidTokenError:
            return False
    
    async def send_secure_request(
        self, 
        target_url: str, 
        action: str, 
        data: dict,
        timeout: int = 30
    ) -> dict:
        """Send authenticated request."""
        target_agent_id = self.extract_agent_id(target_url)
        token = self.generate_token(target_agent_id)
        
        # Add authentication to message
        message = {
            "id": str(uuid.uuid4()),
            "type": "request",
            "sender_id": self.agent_id,
            "receiver_id": target_agent_id,
            "payload": {
                "action": action,
                "data": data,
                "auth_token": token
            },
            "timestamp": datetime.now().isoformat(),
            "ttl": timeout * 1000
        }
        
        # Send request (implementation similar to parent class)
        return await super().send_request(target_url, action, data, timeout)
    
    def check_rate_limit(self, sender_id: str) -> bool:
        """Check if sender is within rate limits."""
        now = datetime.now()
        window_start = now - timedelta(minutes=1)
        
        if sender_id not in self.rate_limits:
            self.rate_limits[sender_id] = []
        
        # Remove old requests
        self.rate_limits[sender_id] = [
            timestamp for timestamp in self.rate_limits[sender_id]
            if timestamp > window_start
        ]
        
        # Check limit (e.g., 60 requests per minute)
        if len(self.rate_limits[sender_id]) >= 60:
            return False
        
        # Add current request
        self.rate_limits[sender_id].append(now)
        return True
    
    async def receive_message(self, request):
        """Enhanced message reception with security checks."""
        try:
            message = await request.json()
            sender_id = message.get("sender_id")
            
            # Rate limiting
            if not self.check_rate_limit(sender_id):
                return web.json_response(
                    {"error": "Rate limit exceeded"}, 
                    status=429
                )
            
            # Authentication for requests
            if message["type"] == "request":
                auth_token = message["payload"].get("auth_token")
                if not auth_token or not self.verify_token(auth_token, sender_id):
                    return web.json_response(
                        {"error": "Authentication failed"}, 
                        status=401
                    )
            
            # Continue with normal message processing
            return await super().receive_message(request)
        
        except Exception as e:
            return web.json_response(
                {"error": str(e)}, 
                status=500
            )
```

### Message Encryption
```python
from cryptography.fernet import Fernet

class EncryptedA2AAgent(SecureA2AAgent):
    """A2A agent with message encryption."""
    
    def __init__(self, agent_id: str, name: str, port: int = 8500, encryption_key: bytes = None):
        super().__init__(agent_id, name, port)
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
    
    def encrypt_payload(self, payload: dict) -> str:
        """Encrypt message payload."""
        payload_json = json.dumps(payload)
        encrypted_payload = self.cipher.encrypt(payload_json.encode())
        return encrypted_payload.decode()
    
    def decrypt_payload(self, encrypted_payload: str) -> dict:
        """Decrypt message payload."""
        encrypted_bytes = encrypted_payload.encode()
        decrypted_bytes = self.cipher.decrypt(encrypted_bytes)
        payload_json = decrypted_bytes.decode()
        return json.loads(payload_json)
    
    async def send_encrypted_request(
        self, 
        target_url: str, 
        action: str, 
        data: dict,
        timeout: int = 30
    ) -> dict:
        """Send encrypted request."""
        # Encrypt the payload
        payload = {
            "action": action,
            "data": data,
            "auth_token": self.generate_token(self.extract_agent_id(target_url))
        }
        
        encrypted_payload = self.encrypt_payload(payload)
        
        message = {
            "id": str(uuid.uuid4()),
            "type": "request",
            "sender_id": self.agent_id,
            "receiver_id": self.extract_agent_id(target_url),
            "payload": encrypted_payload,
            "encrypted": True,
            "timestamp": datetime.now().isoformat()
        }
        
        # Send the encrypted message
        # Implementation continues...
```

## Advanced Features

### Load Balancing and Discovery
```python
class A2ALoadBalancer:
    """Load balancer for A2A agents."""
    
    def __init__(self):
        self.agent_pools = {}
        self.health_status = {}
        self.request_counts = {}
    
    def register_agent_pool(self, service_name: str, agent_urls: list):
        """Register a pool of agents for a service."""
        self.agent_pools[service_name] = agent_urls
        for url in agent_urls:
            self.health_status[url] = "unknown"
            self.request_counts[url] = 0
    
    async def get_healthy_agent(self, service_name: str) -> str:
        """Get a healthy agent from the pool."""
        if service_name not in self.agent_pools:
            raise ValueError(f"Unknown service: {service_name}")
        
        healthy_agents = [
            url for url in self.agent_pools[service_name]
            if self.health_status.get(url) == "healthy"
        ]
        
        if not healthy_agents:
            # Try to find any available agent
            await self.check_agent_health(service_name)
            healthy_agents = [
                url for url in self.agent_pools[service_name]
                if self.health_status.get(url) == "healthy"
            ]
        
        if not healthy_agents:
            raise Exception(f"No healthy agents available for {service_name}")
        
        # Simple round-robin selection
        selected_agent = min(healthy_agents, key=lambda x: self.request_counts[x])
        self.request_counts[selected_agent] += 1
        
        return selected_agent
    
    async def check_agent_health(self, service_name: str):
        """Check health of all agents in a service pool."""
        if service_name not in self.agent_pools:
            return
        
        tasks = []
        for agent_url in self.agent_pools[service_name]:
            task = self.ping_agent_health(agent_url)
            tasks.append((agent_url, task))
        
        results = await asyncio.gather(
            *[task for _, task in tasks], 
            return_exceptions=True
        )
        
        for (agent_url, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                self.health_status[agent_url] = "unhealthy"
            else:
                self.health_status[agent_url] = "healthy"
    
    async def ping_agent_health(self, agent_url: str) -> bool:
        """Ping agent health endpoint."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{agent_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except:
            return False
```

### Monitoring and Metrics
```python
class A2AMonitor:
    """Monitor A2A network performance and health."""
    
    def __init__(self):
        self.metrics = {
            "total_messages": 0,
            "successful_messages": 0,
            "failed_messages": 0,
            "average_response_time": 0,
            "agent_health": {},
            "network_topology": {}
        }
        self.response_times = []
    
    def record_message(self, message_type: str, response_time: float = None, success: bool = True):
        """Record message metrics."""
        self.metrics["total_messages"] += 1
        
        if success:
            self.metrics["successful_messages"] += 1
        else:
            self.metrics["failed_messages"] += 1
        
        if response_time:
            self.response_times.append(response_time)
            self.metrics["average_response_time"] = sum(self.response_times) / len(self.response_times)
    
    async def generate_network_report(self) -> dict:
        """Generate comprehensive network report."""
        return {
            "summary": {
                "total_agents": len(self.metrics["agent_health"]),
                "healthy_agents": sum(
                    1 for status in self.metrics["agent_health"].values() 
                    if status == "healthy"
                ),
                "total_messages": self.metrics["total_messages"],
                "success_rate": self.metrics["successful_messages"] / max(1, self.metrics["total_messages"]),
                "average_response_time": self.metrics["average_response_time"]
            },
            "agent_details": self.metrics["agent_health"],
            "network_topology": self.metrics["network_topology"]
        }
```

## Best Practices

### 1. Message Design
- **Idempotency**: Design messages to be safely retried
- **Versioning**: Include version information in messages
- **Timeouts**: Always set appropriate timeouts
- **Validation**: Validate all incoming messages

### 2. Error Handling
- **Graceful Degradation**: Handle network failures gracefully
- **Retry Logic**: Implement exponential backoff for retries
- **Circuit Breakers**: Prevent cascade failures
- **Dead Letter Queues**: Handle undeliverable messages

### 3. Performance
- **Connection Pooling**: Reuse HTTP connections
- **Batching**: Group multiple operations when possible
- **Compression**: Compress large payloads
- **Caching**: Cache frequently accessed data

### 4. Security
- **Authentication**: Verify agent identity
- **Authorization**: Control access to actions
- **Encryption**: Encrypt sensitive data
- **Rate Limiting**: Prevent abuse

## Example Use Cases

### Distributed Data Processing
```python
# Research Agent -> Analysis Agent -> Report Agent
workflow = [
    {"name": "research", "agent_id": "research_agent", "action": "gather_data"},
    {"name": "analysis", "agent_id": "analysis_agent", "action": "analyze_data"},
    {"name": "reporting", "agent_id": "report_agent", "action": "generate_report"}
]

orchestrator = A2AOrchestrator()
result = await orchestrator.execute_distributed_workflow(workflow, initial_data)
```

### Real-time Collaboration
```python
# Agents collaborating on a shared task
collaboration_network = A2APubSubAgent("collab_coordinator", "Coordinator")

# Agent publishes progress updates
await collaboration_network.publish("task_progress", {
    "task_id": "task_123",
    "progress": 0.5,
    "agent_id": "worker_1"
})

# Other agents subscribe to updates
await other_agent.subscribe("task_progress", "http://localhost:8500")
```

## Troubleshooting

### Common Issues
1. **Connection Refused**: Check if target agent is running
2. **Timeout Errors**: Increase timeout or check network latency
3. **Authentication Failures**: Verify tokens and keys
4. **Message Format Errors**: Validate message structure

### Debugging Tools
- **Message Logging**: Log all A2A communications
- **Health Checks**: Regular agent health monitoring
- **Network Tracing**: Track message paths
- **Performance Metrics**: Monitor response times and throughput

## Next Steps

1. **Production Deployment** - Scale A2A networks in production
2. **Advanced Patterns** - Explore complex communication patterns
3. **Integration** - Connect with existing systems
4. **Optimization** - Fine-tune performance and reliability

## Resources

- [Production Deployment Guide](deployment.md)
- [Performance Optimization](performance.md)
- [Security Best Practices](security.md)
- [API Reference](api-reference.md)
