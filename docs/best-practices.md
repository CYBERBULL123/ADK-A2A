# Best Practices Guide

## 💡 Production-Ready Agent Development

This guide outlines best practices for developing, deploying, and maintaining agent systems that are robust, scalable, and production-ready.

## Table of Contents
- [Development Principles](#development-principles)
- [Code Quality](#code-quality)
- [Architecture Patterns](#architecture-patterns)
- [Performance Optimization](#performance-optimization)
- [Security Guidelines](#security-guidelines)
- [Testing Strategies](#testing-strategies)
- [Deployment Practices](#deployment-practices)
- [Monitoring and Observability](#monitoring-and-observability)

## Development Principles

### 1. Single Responsibility Principle
Each agent should have one clear, well-defined purpose.

**✅ Good Example:**
```python
class ResearchAgent(LlmAgent):
    """Agent specialized in information gathering and research."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.add_tool(self.web_search)
        self.add_tool(self.academic_search)
        self.add_tool(self.summarize_sources)
    
    def research_topic(self, topic: str) -> dict:
        """Focus on research tasks only."""
        pass
```

**❌ Bad Example:**
```python
class SuperAgent(LlmAgent):
    """Agent that does everything."""
    
    def do_everything(self, request: str):
        # Handles research, analysis, writing, calculations, etc.
        # Too many responsibilities!
        pass
```

### 2. Separation of Concerns
Separate different types of logic into distinct components.

```python
class WellStructuredAgent(LlmAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.data_processor = DataProcessor()
        self.validator = InputValidator()
        self.cache = CacheManager()
        self.logger = LoggingService()
    
    async def process_request(self, request: dict) -> dict:
        # Validate input
        self.validator.validate(request)
        
        # Check cache
        cache_key = self.cache.generate_key(request)
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Process request
        result = await self.data_processor.process(request)
        
        # Cache result
        await self.cache.set(cache_key, result)
        
        # Log operation
        self.logger.log_operation(request, result)
        
        return result
```

### 3. Dependency Injection
Make dependencies explicit and configurable.

```python
from abc import ABC, abstractmethod

class SearchService(ABC):
    @abstractmethod
    async def search(self, query: str) -> list:
        pass

class GoogleSearchService(SearchService):
    async def search(self, query: str) -> list:
        # Google search implementation
        pass

class BingSearchService(SearchService):
    async def search(self, query: str) -> list:
        # Bing search implementation
        pass

class ConfigurableSearchAgent(LlmAgent):
    def __init__(self, name: str, search_service: SearchService):
        super().__init__(name)
        self.search_service = search_service
    
    async def search_and_analyze(self, query: str) -> dict:
        results = await self.search_service.search(query)
        return self.analyze_results(results)

# Usage
google_agent = ConfigurableSearchAgent("GoogleAgent", GoogleSearchService())
bing_agent = ConfigurableSearchAgent("BingAgent", BingSearchService())
```

## Code Quality

### 1. Type Hints
Use comprehensive type hints for better code clarity and IDE support.

```python
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass

@dataclass
class AgentMessage:
    sender: str
    receiver: str
    content: str
    timestamp: datetime
    priority: int = 1

class TypedAgent(LlmAgent):
    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        super().__init__(name)
        self.config: Dict[str, Any] = config
        self.message_handlers: Dict[str, Callable] = {}
    
    async def send_message(
        self, 
        recipient: str, 
        content: str, 
        priority: int = 1
    ) -> Optional[AgentMessage]:
        """Send a message to another agent."""
        message = AgentMessage(
            sender=self.name,
            receiver=recipient,
            content=content,
            timestamp=datetime.now(),
            priority=priority
        )
        
        return await self.deliver_message(message)
    
    async def process_batch(
        self, 
        items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process a batch of items."""
        results: List[Dict[str, Any]] = []
        
        for item in items:
            result = await self.process_item(item)
            results.append(result)
        
        return results
```

### 2. Error Handling
Implement comprehensive error handling with specific exception types.

```python
class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass

class ValidationError(AgentError):
    """Raised when input validation fails."""
    pass

class ProcessingError(AgentError):
    """Raised when processing fails."""
    pass

class ExternalServiceError(AgentError):
    """Raised when external service calls fail."""
    pass

class RobustAgent(LlmAgent):
    async def process_request(self, request: dict) -> dict:
        try:
            # Validate input
            self.validate_request(request)
            
            # Process request
            result = await self.execute_processing(request)
            
            return {"status": "success", "data": result}
        
        except ValidationError as e:
            self.logger.warning(f"Validation error: {e}")
            return {"status": "error", "error_type": "validation", "message": str(e)}
        
        except ExternalServiceError as e:
            self.logger.error(f"External service error: {e}")
            # Try fallback
            fallback_result = await self.try_fallback(request)
            if fallback_result:
                return {"status": "success", "data": fallback_result, "note": "fallback_used"}
            return {"status": "error", "error_type": "service_unavailable", "message": str(e)}
        
        except ProcessingError as e:
            self.logger.error(f"Processing error: {e}")
            return {"status": "error", "error_type": "processing", "message": str(e)}
        
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}", exc_info=True)
            return {"status": "error", "error_type": "unexpected", "message": "Internal error occurred"}
    
    def validate_request(self, request: dict) -> None:
        """Validate request format and content."""
        if not isinstance(request, dict):
            raise ValidationError("Request must be a dictionary")
        
        required_fields = ["action", "data"]
        for field in required_fields:
            if field not in request:
                raise ValidationError(f"Missing required field: {field}")
    
    async def try_fallback(self, request: dict) -> Optional[dict]:
        """Try fallback processing when primary method fails."""
        try:
            # Implement fallback logic
            return await self.fallback_processor.process(request)
        except Exception:
            return None
```

### 3. Logging
Implement structured logging for better observability.

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = logging.getLogger(agent_name)
        
        # Configure structured logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_event(self, event_type: str, data: dict, level: str = "info"):
        """Log structured event data."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.agent_name,
            "event_type": event_type,
            "data": data
        }
        
        message = json.dumps(log_entry)
        
        if level == "debug":
            self.logger.debug(message)
        elif level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)

class LoggingAgent(LlmAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.logger = StructuredLogger(name)
    
    async def process_request(self, request: dict) -> dict:
        request_id = str(uuid.uuid4())
        
        # Log request start
        self.logger.log_event("request_start", {
            "request_id": request_id,
            "action": request.get("action"),
            "data_size": len(str(request))
        })
        
        start_time = time.time()
        
        try:
            result = await self.execute_request(request)
            
            # Log successful completion
            self.logger.log_event("request_complete", {
                "request_id": request_id,
                "duration": time.time() - start_time,
                "result_size": len(str(result))
            })
            
            return result
        
        except Exception as e:
            # Log error
            self.logger.log_event("request_error", {
                "request_id": request_id,
                "duration": time.time() - start_time,
                "error": str(e),
                "error_type": type(e).__name__
            }, level="error")
            
            raise
```

## Architecture Patterns

### 1. Factory Pattern for Agent Creation
```python
from enum import Enum
from typing import Type

class AgentType(Enum):
    RESEARCH = "research"
    ANALYSIS = "analysis"
    WRITING = "writing"
    COORDINATION = "coordination"

class AgentFactory:
    """Factory for creating different types of agents."""
    
    _agent_classes: Dict[AgentType, Type[LlmAgent]] = {
        AgentType.RESEARCH: ResearchAgent,
        AgentType.ANALYSIS: AnalysisAgent,
        AgentType.WRITING: WritingAgent,
        AgentType.COORDINATION: CoordinatorAgent
    }
    
    @classmethod
    def create_agent(
        cls, 
        agent_type: AgentType, 
        name: str, 
        config: Dict[str, Any] = None
    ) -> LlmAgent:
        """Create an agent of the specified type."""
        if agent_type not in cls._agent_classes:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_class = cls._agent_classes[agent_type]
        agent = agent_class(name)
        
        if config:
            agent.configure(config)
        
        return agent
    
    @classmethod
    def register_agent_type(cls, agent_type: AgentType, agent_class: Type[LlmAgent]):
        """Register a new agent type."""
        cls._agent_classes[agent_type] = agent_class

# Usage
factory = AgentFactory()
research_agent = factory.create_agent(AgentType.RESEARCH, "ResearchBot")
analysis_agent = factory.create_agent(AgentType.ANALYSIS, "AnalysisBot")
```

### 2. Strategy Pattern for Tool Selection
```python
class ToolSelectionStrategy(ABC):
    @abstractmethod
    def select_tool(self, tools: List[Tool], context: dict) -> Tool:
        pass

class PerformanceBasedStrategy(ToolSelectionStrategy):
    def __init__(self):
        self.tool_performance = {}
    
    def select_tool(self, tools: List[Tool], context: dict) -> Tool:
        # Select tool based on historical performance
        best_tool = max(tools, key=lambda t: self.tool_performance.get(t.name, 0))
        return best_tool

class ContextBasedStrategy(ToolSelectionStrategy):
    def select_tool(self, tools: List[Tool], context: dict) -> Tool:
        # Select tool based on context
        task_type = context.get("task_type")
        for tool in tools:
            if task_type in tool.capabilities:
                return tool
        return tools[0]  # Default

class StrategicAgent(LlmAgent):
    def __init__(self, name: str, strategy: ToolSelectionStrategy):
        super().__init__(name)
        self.strategy = strategy
        self.available_tools = []
    
    async def execute_task(self, task: dict) -> dict:
        # Use strategy to select appropriate tool
        selected_tool = self.strategy.select_tool(self.available_tools, task)
        return await selected_tool.execute(task)
```

### 3. Observer Pattern for Event Handling
```python
class EventObserver(ABC):
    @abstractmethod
    async def handle_event(self, event_type: str, data: dict):
        pass

class LoggingObserver(EventObserver):
    async def handle_event(self, event_type: str, data: dict):
        print(f"LOG: {event_type} - {data}")

class MetricsObserver(EventObserver):
    def __init__(self):
        self.metrics = {}
    
    async def handle_event(self, event_type: str, data: dict):
        if event_type not in self.metrics:
            self.metrics[event_type] = 0
        self.metrics[event_type] += 1

class ObservableAgent(LlmAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.observers: List[EventObserver] = []
    
    def add_observer(self, observer: EventObserver):
        self.observers.append(observer)
    
    async def notify_observers(self, event_type: str, data: dict):
        tasks = [observer.handle_event(event_type, data) for observer in self.observers]
        await asyncio.gather(*tasks)
    
    async def process_request(self, request: dict) -> dict:
        await self.notify_observers("request_start", {"request": request})
        
        try:
            result = await self.execute_request(request)
            await self.notify_observers("request_success", {"result": result})
            return result
        except Exception as e:
            await self.notify_observers("request_error", {"error": str(e)})
            raise
```

## Performance Optimization

### 1. Caching Strategies
```python
import asyncio
from functools import lru_cache
import hashlib
import json

class CacheManager:
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def generate_key(self, data: dict) -> str:
        """Generate cache key from data."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["value"]
            else:
                del self.cache[key]
        return None
    
    async def set(self, key: str, value: Any):
        """Set value in cache."""
        # Implement LRU eviction if needed
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            "value": value,
            "timestamp": time.time()
        }

class CachedAgent(LlmAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.cache = CacheManager()
    
    @lru_cache(maxsize=100)
    def expensive_computation(self, input_data: str) -> str:
        """Expensive computation with caching."""
        # Simulate expensive operation
        time.sleep(1)
        return f"Result for {input_data}"
    
    async def cached_request(self, request: dict) -> dict:
        """Process request with caching."""
        cache_key = self.cache.generate_key(request)
        
        # Check cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Compute result
        result = await self.compute_result(request)
        
        # Cache result
        await self.cache.set(cache_key, result)
        
        return result
```

### 2. Connection Pooling
```python
import aiohttp
from aiohttp import ClientSession, TCPConnector

class ConnectionPoolManager:
    def __init__(self, max_connections: int = 100):
        self.connector = TCPConnector(
            limit=max_connections,
            limit_per_host=20,
            keepalive_timeout=60
        )
        self.session = None
    
    async def __aenter__(self):
        self.session = ClientSession(connector=self.connector)
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

class OptimizedHTTPAgent(LlmAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.connection_pool = ConnectionPoolManager()
    
    async def make_multiple_requests(self, urls: List[str]) -> List[dict]:
        """Make multiple HTTP requests efficiently."""
        async with self.connection_pool as session:
            tasks = [self.fetch_url(session, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
    
    async def fetch_url(self, session: ClientSession, url: str) -> dict:
        """Fetch data from URL using shared session."""
        try:
            async with session.get(url) as response:
                data = await response.json()
                return {"url": url, "data": data, "status": "success"}
        except Exception as e:
            return {"url": url, "error": str(e), "status": "error"}
```

### 3. Batch Processing
```python
class BatchProcessor:
    def __init__(self, batch_size: int = 10, max_wait_time: float = 1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_items = []
        self.batch_task = None
    
    async def add_item(self, item: Any) -> Any:
        """Add item to batch and get result."""
        future = asyncio.Future()
        self.pending_items.append((item, future))
        
        # Start batch processing if needed
        if not self.batch_task or self.batch_task.done():
            self.batch_task = asyncio.create_task(self.process_batch())
        
        return await future
    
    async def process_batch(self):
        """Process items in batches."""
        while self.pending_items:
            # Wait for batch to fill or timeout
            start_time = time.time()
            while (len(self.pending_items) < self.batch_size and 
                   time.time() - start_time < self.max_wait_time):
                await asyncio.sleep(0.1)
            
            # Process current batch
            if self.pending_items:
                batch = self.pending_items[:self.batch_size]
                self.pending_items = self.pending_items[self.batch_size:]
                
                # Extract items and futures
                items = [item for item, _ in batch]
                futures = [future for _, future in batch]
                
                try:
                    # Process batch
                    results = await self.process_items_batch(items)
                    
                    # Set results
                    for future, result in zip(futures, results):
                        future.set_result(result)
                        
                except Exception as e:
                    # Set error for all futures
                    for future in futures:
                        future.set_exception(e)
    
    async def process_items_batch(self, items: List[Any]) -> List[Any]:
        """Process a batch of items - to be implemented by subclasses."""
        raise NotImplementedError

class BatchProcessingAgent(LlmAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.batch_processor = BatchProcessor()
    
    async def process_item(self, item: dict) -> dict:
        """Process single item using batch processor."""
        return await self.batch_processor.add_item(item)
```

## Security Guidelines

### 1. Input Validation and Sanitization
```python
import re
from typing import Union

class InputValidator:
    @staticmethod
    def validate_string(value: str, max_length: int = 1000, pattern: str = None) -> str:
        """Validate and sanitize string input."""
        if not isinstance(value, str):
            raise ValidationError("Value must be a string")
        
        if len(value) > max_length:
            raise ValidationError(f"String too long (max {max_length} characters)")
        
        if pattern and not re.match(pattern, value):
            raise ValidationError(f"String doesn't match required pattern")
        
        # Basic sanitization
        sanitized = value.strip()
        
        # Remove potential script tags
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        return sanitized
    
    @staticmethod
    def validate_number(value: Union[int, float], min_val: float = None, max_val: float = None) -> Union[int, float]:
        """Validate numeric input."""
        if not isinstance(value, (int, float)):
            raise ValidationError("Value must be a number")
        
        if min_val is not None and value < min_val:
            raise ValidationError(f"Value must be >= {min_val}")
        
        if max_val is not None and value > max_val:
            raise ValidationError(f"Value must be <= {max_val}")
        
        return value
    
    @staticmethod
    def validate_email(email: str) -> str:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            raise ValidationError("Invalid email format")
        return email.lower()

class SecureAgent(LlmAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.validator = InputValidator()
    
    async def process_user_input(self, user_data: dict) -> dict:
        """Process user input with validation."""
        try:
            # Validate each field
            validated_data = {}
            
            if "name" in user_data:
                validated_data["name"] = self.validator.validate_string(
                    user_data["name"], max_length=100, pattern=r'^[a-zA-Z\s]+$'
                )
            
            if "email" in user_data:
                validated_data["email"] = self.validator.validate_email(user_data["email"])
            
            if "age" in user_data:
                validated_data["age"] = self.validator.validate_number(
                    user_data["age"], min_val=0, max_val=150
                )
            
            return await self.process_validated_data(validated_data)
        
        except ValidationError as e:
            return {"error": "Validation failed", "details": str(e)}
```

### 2. Rate Limiting
```python
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_times = defaultdict(list)
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for identifier."""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        self.request_times[identifier] = [
            req_time for req_time in self.request_times[identifier]
            if req_time > minute_ago
        ]
        
        # Check limit
        if len(self.request_times[identifier]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.request_times[identifier].append(now)
        return True

class RateLimitedAgent(LlmAgent):
    def __init__(self, name: str, rate_limit: int = 60):
        super().__init__(name)
        self.rate_limiter = RateLimiter(rate_limit)
    
    async def handle_request(self, request: dict, client_id: str) -> dict:
        """Handle request with rate limiting."""
        if not self.rate_limiter.is_allowed(client_id):
            return {
                "error": "Rate limit exceeded",
                "retry_after": 60
            }
        
        return await self.process_request(request)
```

### 3. Secrets Management
```python
import os
from typing import Optional

class SecretsManager:
    def __init__(self):
        self.secrets_cache = {}
    
    def get_secret(self, key: str) -> Optional[str]:
        """Get secret from environment or cache."""
        if key in self.secrets_cache:
            return self.secrets_cache[key]
        
        # Try environment variable
        value = os.getenv(key)
        if value:
            self.secrets_cache[key] = value
            return value
        
        # Try external secrets service (AWS Secrets Manager, etc.)
        value = self.fetch_from_external_service(key)
        if value:
            self.secrets_cache[key] = value
            return value
        
        return None
    
    def fetch_from_external_service(self, key: str) -> Optional[str]:
        """Fetch secret from external service."""
        # Implementation for external secrets service
        pass

class SecureConfiguredAgent(LlmAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.secrets = SecretsManager()
        self.api_key = self.secrets.get_secret("AGENT_API_KEY")
        
        if not self.api_key:
            raise ValueError("Required API key not found")
    
    async def make_authenticated_request(self, url: str, data: dict) -> dict:
        """Make request with API key authentication."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                return await response.json()
```

## Testing Strategies

### 1. Unit Testing
```python
import pytest
from unittest.mock import Mock, AsyncMock, patch

class TestMyAgent:
    @pytest.fixture
    def agent(self):
        return MyAgent("TestAgent")
    
    @pytest.mark.asyncio
    async def test_basic_chat(self, agent):
        """Test basic chat functionality."""
        response = await agent.chat("Hello")
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_tool_execution(self, agent):
        """Test tool execution."""
        with patch.object(agent, 'external_api_call') as mock_api:
            mock_api.return_value = {"result": "success"}
            
            result = await agent.execute_tool("test_tool", {"input": "test"})
            
            assert result["result"] == "success"
            mock_api.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test error handling."""
        with patch.object(agent, 'external_api_call') as mock_api:
            mock_api.side_effect = Exception("API Error")
            
            with pytest.raises(ProcessingError):
                await agent.execute_tool("test_tool", {"input": "test"})
    
    def test_input_validation(self, agent):
        """Test input validation."""
        with pytest.raises(ValidationError):
            agent.validate_input({"invalid": "data"})
        
        # Valid input should not raise
        agent.validate_input({"valid": "data", "action": "test"})
```

### 2. Integration Testing
```python
@pytest.mark.integration
class TestAgentIntegration:
    @pytest.fixture
    async def agent_system(self):
        """Setup agent system for integration tests."""
        coordinator = CoordinatorAgent("Coordinator")
        research_agent = ResearchAgent("Researcher")
        analysis_agent = AnalysisAgent("Analyzer")
        
        coordinator.add_agent(research_agent)
        coordinator.add_agent(analysis_agent)
        
        return {
            "coordinator": coordinator,
            "research": research_agent,
            "analysis": analysis_agent
        }
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, agent_system):
        """Test complete workflow execution."""
        coordinator = agent_system["coordinator"]
        
        result = await coordinator.execute_project("Test project")
        
        assert result["status"] == "success"
        assert "research_data" in result
        assert "analysis_results" in result
        assert "final_report" in result
    
    @pytest.mark.asyncio
    async def test_agent_communication(self, agent_system):
        """Test inter-agent communication."""
        research_agent = agent_system["research"]
        analysis_agent = agent_system["analysis"]
        
        # Send message from research to analysis
        message = await research_agent.send_message(
            analysis_agent.name,
            {"type": "data", "content": "test data"}
        )
        
        assert message is not None
        assert message.receiver == analysis_agent.name
```

### 3. Load Testing
```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

class LoadTester:
    def __init__(self, agent: LlmAgent):
        self.agent = agent
        self.results = []
    
    async def run_load_test(
        self, 
        concurrent_requests: int = 10, 
        total_requests: int = 100
    ) -> dict:
        """Run load test on agent."""
        start_time = time.time()
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def make_request(request_id: int):
            async with semaphore:
                request_start = time.time()
                try:
                    result = await self.agent.chat(f"Test request {request_id}")
                    duration = time.time() - request_start
                    return {
                        "request_id": request_id,
                        "duration": duration,
                        "success": True,
                        "result_length": len(str(result))
                    }
                except Exception as e:
                    duration = time.time() - request_start
                    return {
                        "request_id": request_id,
                        "duration": duration,
                        "success": False,
                        "error": str(e)
                    }
        
        # Execute all requests
        tasks = [make_request(i) for i in range(total_requests)]
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        avg_response_time = sum(r["duration"] for r in successful_requests) / len(successful_requests) if successful_requests else 0
        
        return {
            "total_requests": total_requests,
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": len(successful_requests) / total_requests,
            "average_response_time": avg_response_time,
            "total_time": total_time,
            "requests_per_second": total_requests / total_time
        }

# Usage
@pytest.mark.load
async def test_agent_load():
    agent = MyAgent("LoadTestAgent")
    tester = LoadTester(agent)
    
    results = await tester.run_load_test(concurrent_requests=20, total_requests=200)
    
    assert results["success_rate"] > 0.95  # 95% success rate
    assert results["average_response_time"] < 2.0  # Under 2 seconds
```

## Deployment Practices

### 1. Environment Configuration
```python
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class AgentConfig:
    name: str
    model: str
    api_key: str
    max_tokens: int = 1000
    temperature: float = 0.7
    rate_limit: int = 60
    cache_ttl: int = 3600
    log_level: str = "INFO"
    
    @classmethod
    def from_environment(cls, name: str) -> 'AgentConfig':
        """Create config from environment variables."""
        return cls(
            name=name,
            model=os.getenv("AGENT_MODEL", "gemini-pro"),
            api_key=os.getenv("AGENT_API_KEY"),
            max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "1000")),
            temperature=float(os.getenv("AGENT_TEMPERATURE", "0.7")),
            rate_limit=int(os.getenv("AGENT_RATE_LIMIT", "60")),
            cache_ttl=int(os.getenv("AGENT_CACHE_TTL", "3600")),
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )

class ConfigurableAgent(LlmAgent):
    def __init__(self, config: AgentConfig):
        super().__init__(config.name)
        self.config = config
        self.configure_from_config()
    
    def configure_from_config(self):
        """Configure agent based on config."""
        self.set_model(self.config.model)
        self.set_api_key(self.config.api_key)
        self.set_max_tokens(self.config.max_tokens)
        self.set_temperature(self.config.temperature)
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
```

### 2. Health Checks
```python
class HealthChecker:
    def __init__(self, agent: LlmAgent):
        self.agent = agent
        self.last_check = None
        self.health_status = "unknown"
    
    async def check_health(self) -> dict:
        """Perform comprehensive health check."""
        checks = {
            "agent_responsive": await self.check_agent_responsive(),
            "external_services": await self.check_external_services(),
            "memory_usage": self.check_memory_usage(),
            "disk_space": self.check_disk_space()
        }
        
        all_healthy = all(checks.values())
        self.health_status = "healthy" if all_healthy else "unhealthy"
        self.last_check = datetime.now()
        
        return {
            "status": self.health_status,
            "timestamp": self.last_check.isoformat(),
            "checks": checks
        }
    
    async def check_agent_responsive(self) -> bool:
        """Check if agent responds to basic requests."""
        try:
            response = await asyncio.wait_for(
                self.agent.chat("Health check"), timeout=5.0
            )
            return isinstance(response, str) and len(response) > 0
        except:
            return False
    
    async def check_external_services(self) -> bool:
        """Check external service connectivity."""
        # Implementation depends on services used
        return True
    
    def check_memory_usage(self) -> bool:
        """Check memory usage."""
        import psutil
        memory_percent = psutil.virtual_memory().percent
        return memory_percent < 90  # Less than 90% memory usage
    
    def check_disk_space(self) -> bool:
        """Check disk space."""
        import psutil
        disk_usage = psutil.disk_usage('/').percent
        return disk_usage < 90  # Less than 90% disk usage

class HealthyAgent(LlmAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.health_checker = HealthChecker(self)
    
    async def get_health_status(self) -> dict:
        """Get current health status."""
        return await self.health_checker.check_health()
```

### 3. Graceful Shutdown
```python
import signal
import asyncio

class GracefulAgent(LlmAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.shutdown_event = asyncio.Event()
        self.active_requests = set()
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, self.handle_shutdown_signal)
    
    def handle_shutdown_signal(self, signum, frame):
        """Handle shutdown signal."""
        print(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.graceful_shutdown())
    
    async def graceful_shutdown(self):
        """Perform graceful shutdown."""
        print("Starting graceful shutdown...")
        
        # Stop accepting new requests
        self.shutdown_event.set()
        
        # Wait for active requests to complete
        while self.active_requests:
            print(f"Waiting for {len(self.active_requests)} active requests...")
            await asyncio.sleep(1)
        
        # Cleanup resources
        await self.cleanup_resources()
        
        print("Graceful shutdown complete")
    
    async def process_request(self, request: dict) -> dict:
        """Process request with shutdown awareness."""
        if self.shutdown_event.is_set():
            return {"error": "Service shutting down"}
        
        request_id = str(uuid.uuid4())
        self.active_requests.add(request_id)
        
        try:
            result = await self.execute_request(request)
            return result
        finally:
            self.active_requests.discard(request_id)
    
    async def cleanup_resources(self):
        """Cleanup resources during shutdown."""
        # Close database connections, file handles, etc.
        pass
```

## Monitoring and Observability

### 1. Metrics Collection
```python
from dataclasses import dataclass, field
from typing import Dict, List
import time

@dataclass
class AgentMetrics:
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_response_time: float = 0.0
    response_times: List[float] = field(default_factory=list)
    error_types: Dict[str, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.success_count / self.request_count
    
    @property
    def average_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    @property
    def error_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count

class MetricsCollector:
    def __init__(self):
        self.metrics = AgentMetrics()
        self.start_time = time.time()
    
    def record_request_start(self) -> str:
        """Record start of request and return request ID."""
        request_id = str(uuid.uuid4())
        self.metrics.request_count += 1
        return request_id
    
    def record_request_success(self, request_id: str, response_time: float):
        """Record successful request completion."""
        self.metrics.success_count += 1
        self.metrics.total_response_time += response_time
        self.metrics.response_times.append(response_time)
    
    def record_request_error(self, request_id: str, error_type: str, response_time: float):
        """Record request error."""
        self.metrics.error_count += 1
        self.metrics.total_response_time += response_time
        self.metrics.response_times.append(response_time)
        
        if error_type not in self.metrics.error_types:
            self.metrics.error_types[error_type] = 0
        self.metrics.error_types[error_type] += 1
    
    def get_metrics_summary(self) -> dict:
        """Get metrics summary."""
        uptime = time.time() - self.start_time
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.metrics.request_count,
            "successful_requests": self.metrics.success_count,
            "failed_requests": self.metrics.error_count,
            "success_rate": self.metrics.success_rate,
            "error_rate": self.metrics.error_rate,
            "average_response_time": self.metrics.average_response_time,
            "requests_per_second": self.metrics.request_count / uptime if uptime > 0 else 0,
            "error_breakdown": self.metrics.error_types
        }

class MonitoredAgent(LlmAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.metrics_collector = MetricsCollector()
    
    async def process_request(self, request: dict) -> dict:
        """Process request with metrics collection."""
        request_id = self.metrics_collector.record_request_start()
        start_time = time.time()
        
        try:
            result = await self.execute_request(request)
            response_time = time.time() - start_time
            self.metrics_collector.record_request_success(request_id, response_time)
            return result
        
        except Exception as e:
            response_time = time.time() - start_time
            error_type = type(e).__name__
            self.metrics_collector.record_request_error(request_id, error_type, response_time)
            raise
    
    def get_metrics(self) -> dict:
        """Get current metrics."""
        return self.metrics_collector.get_metrics_summary()
```

### 2. Distributed Tracing
```python
import uuid
from contextvars import ContextVar
from typing import Optional

# Context variable for trace ID
trace_id_var: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)

class TracingAgent(LlmAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.trace_spans = {}
    
    async def process_request(self, request: dict, trace_id: str = None) -> dict:
        """Process request with distributed tracing."""
        # Set or generate trace ID
        if not trace_id:
            trace_id = str(uuid.uuid4())
        
        trace_id_var.set(trace_id)
        
        # Create span
        span_id = self.start_span("process_request", trace_id)
        
        try:
            # Add request info to span
            self.add_span_info(span_id, {
                "agent_name": self.name,
                "request_type": request.get("action"),
                "request_size": len(str(request))
            })
            
            result = await self.execute_request(request)
            
            # Add result info to span
            self.add_span_info(span_id, {
                "result_size": len(str(result)),
                "success": True
            })
            
            return result
        
        except Exception as e:
            # Add error info to span
            self.add_span_info(span_id, {
                "error": str(e),
                "error_type": type(e).__name__,
                "success": False
            })
            raise
        
        finally:
            self.end_span(span_id)
    
    def start_span(self, operation: str, trace_id: str) -> str:
        """Start a new span."""
        span_id = str(uuid.uuid4())
        self.trace_spans[span_id] = {
            "trace_id": trace_id,
            "span_id": span_id,
            "operation": operation,
            "start_time": time.time(),
            "info": {}
        }
        return span_id
    
    def add_span_info(self, span_id: str, info: dict):
        """Add information to span."""
        if span_id in self.trace_spans:
            self.trace_spans[span_id]["info"].update(info)
    
    def end_span(self, span_id: str):
        """End span and send to tracing system."""
        if span_id in self.trace_spans:
            span = self.trace_spans[span_id]
            span["end_time"] = time.time()
            span["duration"] = span["end_time"] - span["start_time"]
            
            # Send to tracing system (e.g., Jaeger, Zipkin)
            self.send_span_to_tracing_system(span)
            
            del self.trace_spans[span_id]
    
    def send_span_to_tracing_system(self, span: dict):
        """Send span to external tracing system."""
        # Implementation depends on tracing system
        print(f"TRACE: {span}")
```

## Checklist for Production Readiness

### Code Quality ✅
- [ ] Type hints throughout codebase
- [ ] Comprehensive error handling
- [ ] Input validation and sanitization
- [ ] Structured logging implemented
- [ ] Code documentation complete

### Testing ✅
- [ ] Unit tests with >90% coverage
- [ ] Integration tests for critical paths
- [ ] Load testing completed
- [ ] Error scenarios tested
- [ ] Mock external dependencies

### Security ✅
- [ ] Input validation implemented
- [ ] Rate limiting configured
- [ ] Secrets management in place
- [ ] Authentication/authorization implemented
- [ ] Security scan completed

### Performance ✅
- [ ] Caching strategy implemented
- [ ] Connection pooling configured
- [ ] Batch processing where applicable
- [ ] Performance benchmarks established
- [ ] Resource usage optimized

### Monitoring ✅
- [ ] Metrics collection implemented
- [ ] Health checks configured
- [ ] Logging strategy in place
- [ ] Alerting rules defined
- [ ] Dashboard created

### Deployment ✅
- [ ] Environment configuration externalized
- [ ] Graceful shutdown implemented
- [ ] Container images built
- [ ] Infrastructure as code
- [ ] Rollback strategy defined

This comprehensive best practices guide ensures your agent systems are production-ready, maintainable, and scalable.
