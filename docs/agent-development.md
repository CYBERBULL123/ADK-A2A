# Agent Development Guide

## 🤖 Building Intelligent Agents with ADK

This guide covers everything you need to know about building sophisticated agents using Google's Agent Development Kit (ADK).

## Table of Contents
- [Core Concepts](#core-concepts)
- [Agent Architecture](#agent-architecture)
- [Agent Types](#agent-types)
- [Tool Integration](#tool-integration)
- [State Management](#state-management)
- [Advanced Patterns](#advanced-patterns)

## Core Concepts

### What is an Agent?
An agent is an autonomous software entity that:
- **Perceives** its environment through inputs
- **Reasons** about information using AI models
- **Acts** through tools and responses
- **Learns** from interactions and feedback

### ADK Philosophy
ADK embraces a **code-first** approach:
- Agents are Python classes, not configuration files
- Tools are functions with decorators
- Behavior is programmed, not prompted
- Integration is native, not bolted-on

## Agent Architecture

### Base Agent Structure
```python
from google.adk.agents import LlmAgent
from google.adk.tools import Tool

class MyAgent(LlmAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.setup_tools()
        self.configure_model()
    
    def setup_tools(self):
        """Register tools for this agent."""
        self.add_tool(self.custom_tool)
    
    def configure_model(self):
        """Configure the LLM model."""
        self.set_model("gemini-pro")
    
    @Tool
    def custom_tool(self, input_data: str) -> str:
        """A custom tool implementation."""
        return f"Processed: {input_data}"
```

### Key Components

#### 1. Model Integration
```python
# Configure different models
agent.set_model("gemini-pro")           # Google Gemini
agent.set_model("gpt-4")                # OpenAI GPT-4
agent.set_model("claude-3-opus")        # Anthropic Claude

# Model-specific settings
agent.configure_model({
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 0.9
})
```

#### 2. Tool Management
```python
from google.adk.tools import Tool

class WeatherAgent(LlmAgent):
    @Tool
    def get_weather(self, location: str) -> dict:
        """Get current weather for a location."""
        # Implementation here
        return {"location": location, "temp": "22°C"}
    
    @Tool  
    def forecast(self, location: str, days: int = 5) -> list:
        """Get weather forecast."""
        # Implementation here
        return [{"day": i, "temp": f"{20+i}°C"} for i in range(days)]
```

#### 3. State Management
```python
class StatefulAgent(LlmAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.conversation_history = []
        self.user_preferences = {}
    
    def chat_with_memory(self, message: str) -> str:
        """Chat with conversation memory."""
        # Add to history
        self.conversation_history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now()
        })
        
        # Generate response with context
        response = self.generate_response(message, self.conversation_history)
        
        # Store response
        self.conversation_history.append({
            "role": "assistant", 
            "content": response,
            "timestamp": datetime.now()
        })
        
        return response
```

## Agent Types

### 1. Simple Agent
**Purpose**: Basic conversational AI
**Use Cases**: Customer support, Q&A, content generation

```python
class SimpleAgent(LlmAgent):
    """Basic conversational agent."""
    
    def chat(self, message: str) -> str:
        """Simple chat interface."""
        return self.generate_response(message)
```

**Best Practices**:
- Keep prompts clear and focused
- Handle edge cases gracefully
- Provide consistent response format

### 2. Tool Agent
**Purpose**: Task automation with external capabilities
**Use Cases**: Data processing, API integration, calculations

```python
class ToolAgent(LlmAgent):
    """Agent with multiple tools."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.add_tool(self.calculator)
        self.add_tool(self.web_search)
        self.add_tool(self.file_operations)
    
    @Tool
    def calculator(self, expression: str) -> float:
        """Safe calculator tool."""
        try:
            # Use safe evaluation
            return eval(expression, {"__builtins__": {}})
        except:
            return "Invalid expression"
    
    @Tool
    def web_search(self, query: str) -> list:
        """Web search tool."""
        # Implementation
        pass
```

**Best Practices**:
- Validate all tool inputs
- Handle tool failures gracefully
- Provide clear tool descriptions
- Use type hints for parameters

### 3. Search Agent
**Purpose**: Information retrieval and research
**Use Cases**: Research, fact-checking, data gathering

```python
class SearchAgent(LlmAgent):
    """Agent specialized in information retrieval."""
    
    @Tool
    def web_search(self, query: str) -> dict:
        """Search the web for information."""
        # Implementation with search API
        pass
    
    @Tool
    def summarize_results(self, results: list) -> str:
        """Summarize search results."""
        # Implementation
        pass
    
    def search_and_respond(self, question: str) -> str:
        """Search for information and provide response."""
        # 1. Search for relevant information
        results = self.web_search(question)
        
        # 2. Summarize findings
        summary = self.summarize_results(results)
        
        # 3. Generate informed response
        return self.generate_response(question, context=summary)
```

### 4. Stateful Agent
**Purpose**: Personalized, context-aware interactions
**Use Cases**: Personal assistants, learning systems, long-term relationships

```python
class StatefulAgent(LlmAgent):
    """Agent with persistent memory and state."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.memory = ConversationMemory()
        self.user_profile = UserProfile()
    
    def chat_with_memory(self, message: str) -> str:
        """Chat with full context awareness."""
        # Retrieve relevant memories
        relevant_context = self.memory.retrieve_relevant(message)
        
        # Update user profile
        self.user_profile.update_from_message(message)
        
        # Generate contextual response
        response = self.generate_response(
            message, 
            context=relevant_context,
            user_profile=self.user_profile
        )
        
        # Store interaction
        self.memory.store_interaction(message, response)
        
        return response
```

## Tool Integration

### Creating Custom Tools

#### 1. Simple Function Tool
```python
@Tool
def current_time() -> str:
    """Get the current time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

#### 2. Parameterized Tool
```python
@Tool
def convert_temperature(temp: float, from_unit: str, to_unit: str) -> float:
    """Convert temperature between units."""
    if from_unit == "C" and to_unit == "F":
        return (temp * 9/5) + 32
    elif from_unit == "F" and to_unit == "C":
        return (temp - 32) * 5/9
    else:
        raise ValueError("Unsupported conversion")
```

#### 3. Async Tool
```python
@Tool
async def fetch_data(url: str) -> dict:
    """Fetch data from an API endpoint."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

#### 4. Complex Tool with State
```python
class DatabaseTool:
    def __init__(self, connection_string: str):
        self.db = Database(connection_string)
    
    @Tool
    def query_database(self, sql: str) -> list:
        """Execute SQL query and return results."""
        return self.db.execute(sql)
    
    @Tool
    def insert_record(self, table: str, data: dict) -> bool:
        """Insert a record into the database."""
        return self.db.insert(table, data)
```

### Tool Best Practices

#### Error Handling
```python
@Tool
def safe_division(a: float, b: float) -> float:
    """Safely divide two numbers."""
    try:
        if b == 0:
            raise ValueError("Division by zero")
        return a / b
    except Exception as e:
        return f"Error: {str(e)}"
```

#### Input Validation
```python
@Tool
def process_email(email: str) -> dict:
    """Process and validate email address."""
    import re
    
    # Validate email format
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return {"valid": False, "error": "Invalid email format"}
    
    # Process email
    domain = email.split('@')[1]
    return {
        "valid": True,
        "email": email,
        "domain": domain,
        "local_part": email.split('@')[0]
    }
```

#### Documentation
```python
@Tool
def analyze_text(text: str, analysis_type: str = "sentiment") -> dict:
    """
    Analyze text using various NLP techniques.
    
    Args:
        text: The text to analyze
        analysis_type: Type of analysis ("sentiment", "entities", "keywords")
    
    Returns:
        Dictionary containing analysis results
    
    Example:
        result = analyze_text("I love this product!", "sentiment")
        # Returns: {"sentiment": "positive", "confidence": 0.95}
    """
    # Implementation here
    pass
```

## State Management

### Memory Systems

#### 1. Simple Memory
```python
class SimpleMemory:
    def __init__(self):
        self.messages = []
    
    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })
    
    def get_recent(self, count: int = 10) -> list:
        return self.messages[-count:]
```

#### 2. Semantic Memory
```python
class SemanticMemory:
    def __init__(self):
        self.embeddings = EmbeddingStore()
        self.memories = []
    
    def store_memory(self, content: str, metadata: dict = None):
        embedding = self.embeddings.embed(content)
        memory = {
            "content": content,
            "embedding": embedding,
            "metadata": metadata or {},
            "timestamp": datetime.now()
        }
        self.memories.append(memory)
    
    def retrieve_relevant(self, query: str, limit: int = 5) -> list:
        query_embedding = self.embeddings.embed(query)
        similarities = []
        
        for memory in self.memories:
            similarity = cosine_similarity(query_embedding, memory["embedding"])
            similarities.append((similarity, memory))
        
        # Return top similar memories
        similarities.sort(reverse=True)
        return [memory for _, memory in similarities[:limit]]
```

### User Profiling
```python
class UserProfile:
    def __init__(self):
        self.preferences = {}
        self.interaction_count = 0
        self.topics_of_interest = set()
        self.communication_style = "neutral"
    
    def update_from_message(self, message: str):
        self.interaction_count += 1
        
        # Extract topics (simplified)
        topics = self.extract_topics(message)
        self.topics_of_interest.update(topics)
        
        # Analyze communication style
        style = self.analyze_style(message)
        self.update_communication_style(style)
    
    def get_personalization_context(self) -> dict:
        return {
            "interaction_count": self.interaction_count,
            "preferred_topics": list(self.topics_of_interest),
            "communication_style": self.communication_style,
            "preferences": self.preferences
        }
```

## Advanced Patterns

### 1. Agent Composition
```python
class CompositeAgent(LlmAgent):
    """Agent that combines multiple specialized agents."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.research_agent = ResearchAgent("Researcher")
        self.analysis_agent = AnalysisAgent("Analyzer")
        self.writer_agent = WriterAgent("Writer")
    
    def comprehensive_response(self, query: str) -> str:
        # Research phase
        research = self.research_agent.research(query)
        
        # Analysis phase
        analysis = self.analysis_agent.analyze(research)
        
        # Writing phase
        response = self.writer_agent.write_response(query, analysis)
        
        return response
```

### 2. Adaptive Behavior
```python
class AdaptiveAgent(LlmAgent):
    """Agent that adapts its behavior based on context."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.performance_metrics = {}
        self.strategy_weights = {"detailed": 0.5, "concise": 0.5}
    
    def adaptive_response(self, message: str, context: dict) -> str:
        # Choose strategy based on context and performance
        strategy = self.select_strategy(context)
        
        # Generate response using selected strategy
        response = self.generate_with_strategy(message, strategy)
        
        # Learn from interaction
        self.update_strategy_weights(strategy, context, response)
        
        return response
```

### 3. Error Recovery
```python
class ResilientAgent(LlmAgent):
    """Agent with robust error handling and recovery."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.fallback_responses = [
            "I apologize, but I'm experiencing technical difficulties.",
            "Let me try a different approach to help you.",
            "I'm having trouble with that request. Could you rephrase it?"
        ]
    
    def resilient_chat(self, message: str) -> str:
        try:
            return self.generate_response(message)
        except ModelError as e:
            self.log_error(e)
            return self.fallback_response("model_error")
        except ToolError as e:
            self.log_error(e)
            return self.fallback_response("tool_error")
        except Exception as e:
            self.log_error(e)
            return self.fallback_response("general_error")
    
    def fallback_response(self, error_type: str) -> str:
        """Provide contextual fallback responses."""
        responses = {
            "model_error": "I'm having trouble processing your request right now.",
            "tool_error": "Some of my tools are temporarily unavailable.",
            "general_error": "I encountered an unexpected issue."
        }
        return responses.get(error_type, "I apologize for the inconvenience.")
```

## Testing and Validation

### Unit Testing Agents
```python
import pytest
from unittest.mock import Mock, patch

class TestMyAgent:
    def setup_method(self):
        self.agent = MyAgent("TestAgent")
    
    def test_basic_chat(self):
        response = self.agent.chat("Hello")
        assert isinstance(response, str)
        assert len(response) > 0
    
    @patch('external_api.call')
    def test_tool_with_mock(self, mock_api):
        mock_api.return_value = {"result": "success"}
        result = self.agent.external_tool("test_input")
        assert result["result"] == "success"
        mock_api.assert_called_once_with("test_input")
    
    def test_error_handling(self):
        with pytest.raises(ValueError):
            self.agent.validate_input("invalid_input")
```

### Integration Testing
```python
class TestAgentIntegration:
    def test_full_workflow(self):
        agent = WorkflowAgent("IntegrationTest")
        
        # Test complete workflow
        result = agent.execute_workflow({
            "input": "test_data",
            "steps": ["research", "analyze", "summarize"]
        })
        
        assert result["status"] == "completed"
        assert "summary" in result
```

## Performance Optimization

### 1. Caching
```python
from functools import lru_cache

class OptimizedAgent(LlmAgent):
    @lru_cache(maxsize=100)
    def cached_tool(self, input_data: str) -> str:
        """Tool with caching for expensive operations."""
        # Expensive computation here
        return self.expensive_operation(input_data)
```

### 2. Async Operations
```python
class AsyncAgent(LlmAgent):
    async def parallel_tools(self, inputs: list) -> list:
        """Execute multiple tools in parallel."""
        tasks = [self.async_tool(input_data) for input_data in inputs]
        results = await asyncio.gather(*tasks)
        return results
```

### 3. Batch Processing
```python
class BatchAgent(LlmAgent):
    def batch_process(self, items: list, batch_size: int = 10) -> list:
        """Process items in batches for efficiency."""
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = self.process_batch(batch)
            results.extend(batch_results)
        return results
```

## Next Steps

Once you've mastered basic agent development:

1. **Multi-Agent Systems** - Learn to coordinate multiple agents
2. **A2A Protocol** - Implement distributed agent communication
3. **Advanced Tools** - Build sophisticated tool ecosystems
4. **Production Deployment** - Scale your agents for real-world use

## Resources

- [Multi-Agent Systems Guide](multi-agent-systems.md)
- [A2A Protocol Documentation](a2a-protocol.md)
- [Tool Development Guide](tools-integration.md)
- [API Reference](api-reference.md)
