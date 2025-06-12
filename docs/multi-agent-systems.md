# Multi-Agent Systems Guide

## 🔗 Building Coordinated Agent Teams

Multi-agent systems enable complex problem-solving by coordinating multiple specialized agents. This guide covers design patterns, implementation strategies, and best practices.

## Table of Contents
- [Core Concepts](#core-concepts)
- [Coordination Patterns](#coordination-patterns)
- [Implementation Strategies](#implementation-strategies)
- [Communication Protocols](#communication-protocols)
- [Workflow Management](#workflow-management)
- [Advanced Topics](#advanced-topics)

## Core Concepts

### What are Multi-Agent Systems?
Multi-agent systems (MAS) consist of multiple autonomous agents that:
- **Collaborate** to achieve common goals
- **Communicate** through defined protocols
- **Coordinate** their actions and decisions
- **Specialize** in different domains or tasks

### Benefits of Multi-Agent Approaches
- **Modularity**: Each agent has specific responsibilities
- **Scalability**: Add agents as needs grow
- **Resilience**: System continues if one agent fails
- **Specialization**: Agents can be domain experts
- **Parallel Processing**: Multiple agents work simultaneously

## Coordination Patterns

### 1. Hierarchical Coordination
**Structure**: One coordinator manages multiple sub-agents
**Best For**: Clear command structures, complex workflows

```python
class HierarchicalCoordinator(LlmAgent):
    """Coordinator that manages specialized sub-agents."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.research_agent = ResearchAgent("Researcher")
        self.analysis_agent = AnalysisAgent("Analyzer")
        self.writing_agent = WritingAgent("Writer")
        self.sub_agents = [
            self.research_agent,
            self.analysis_agent, 
            self.writing_agent
        ]
    
    async def execute_project(self, description: str) -> dict:
        """Execute a complex project using sub-agents."""
        try:
            # Phase 1: Research
            research_result = await self.delegate_task(
                self.research_agent,
                "research",
                {"topic": description}
            )
            
            # Phase 2: Analysis
            analysis_result = await self.delegate_task(
                self.analysis_agent,
                "analyze",
                {"data": research_result}
            )
            
            # Phase 3: Writing
            final_result = await self.delegate_task(
                self.writing_agent,
                "write",
                {"analysis": analysis_result, "topic": description}
            )
            
            return {
                "status": "success",
                "result": final_result,
                "phases": ["research", "analysis", "writing"]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "completed_phases": self.get_completed_phases()
            }
    
    async def delegate_task(self, agent: LlmAgent, task: str, data: dict):
        """Delegate a task to a specific agent."""
        self.log_delegation(agent.name, task, data)
        result = await agent.execute_task(task, data)
        self.log_completion(agent.name, task, result)
        return result
```

### 2. Pipeline Coordination
**Structure**: Agents process data sequentially
**Best For**: Data transformation, assembly lines

```python
class PipelineCoordinator(LlmAgent):
    """Coordinate agents in a processing pipeline."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.pipeline_stages = [
            DataIngestionAgent("Ingestion"),
            DataCleaningAgent("Cleaning"),
            DataAnalysisAgent("Analysis"),
            ReportGenerationAgent("Reporting")
        ]
    
    async def process_pipeline(self, initial_data: dict) -> dict:
        """Process data through the agent pipeline."""
        current_data = initial_data
        results = {"stages": []}
        
        for stage_agent in self.pipeline_stages:
            try:
                stage_result = await stage_agent.process(current_data)
                results["stages"].append({
                    "agent": stage_agent.name,
                    "status": "success",
                    "output_size": len(str(stage_result))
                })
                current_data = stage_result
                
            except Exception as e:
                results["stages"].append({
                    "agent": stage_agent.name,
                    "status": "error",
                    "error": str(e)
                })
                break
        
        results["final_output"] = current_data
        return results
```

### 3. Market-Based Coordination
**Structure**: Agents bid for tasks and negotiate
**Best For**: Resource allocation, dynamic task distribution

```python
class MarketCoordinator(LlmAgent):
    """Coordinate agents using market-based mechanisms."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.available_agents = []
        self.task_queue = []
        self.agent_capabilities = {}
    
    def register_agent(self, agent: LlmAgent, capabilities: list, cost_per_task: float):
        """Register an agent with its capabilities and cost."""
        self.available_agents.append(agent)
        self.agent_capabilities[agent.name] = {
            "capabilities": capabilities,
            "cost": cost_per_task,
            "workload": 0,
            "performance_history": []
        }
    
    async def allocate_task(self, task: dict) -> dict:
        """Allocate task to the best available agent."""
        required_capability = task.get("required_capability")
        
        # Find capable agents
        capable_agents = [
            agent for agent in self.available_agents
            if required_capability in self.agent_capabilities[agent.name]["capabilities"]
        ]
        
        if not capable_agents:
            return {"status": "error", "message": "No capable agents available"}
        
        # Select agent based on cost and performance
        selected_agent = self.select_optimal_agent(capable_agents, task)
        
        # Execute task
        result = await selected_agent.execute_task(task["type"], task["data"])
        
        # Update performance metrics
        self.update_agent_performance(selected_agent.name, result)
        
        return {
            "status": "success",
            "agent": selected_agent.name,
            "result": result
        }
    
    def select_optimal_agent(self, agents: list, task: dict) -> LlmAgent:
        """Select the optimal agent based on multiple criteria."""
        scores = []
        
        for agent in agents:
            metrics = self.agent_capabilities[agent.name]
            
            # Calculate score based on cost, workload, and performance
            cost_score = 1.0 / (metrics["cost"] + 1)
            workload_score = 1.0 / (metrics["workload"] + 1)
            performance_score = self.calculate_performance_score(agent.name)
            
            total_score = (cost_score + workload_score + performance_score) / 3
            scores.append((total_score, agent))
        
        # Return agent with highest score
        scores.sort(reverse=True)
        return scores[0][1]
```

### 4. Peer-to-Peer Coordination
**Structure**: Agents communicate directly with each other
**Best For**: Distributed systems, emergent behavior

```python
class P2PAgent(LlmAgent):
    """Agent that can communicate directly with peers."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.peers = []
        self.message_queue = asyncio.Queue()
        self.running = False
    
    def add_peer(self, peer: 'P2PAgent'):
        """Add another agent as a peer."""
        if peer not in self.peers:
            self.peers.append(peer)
            peer.peers.append(self)
    
    async def send_message(self, recipient: 'P2PAgent', message: dict):
        """Send a message to a peer agent."""
        message_envelope = {
            "sender": self.name,
            "recipient": recipient.name,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        await recipient.receive_message(message_envelope)
    
    async def receive_message(self, message_envelope: dict):
        """Receive a message from a peer."""
        await self.message_queue.put(message_envelope)
    
    async def broadcast_message(self, message: dict):
        """Broadcast a message to all peers."""
        tasks = [
            self.send_message(peer, message)
            for peer in self.peers
        ]
        await asyncio.gather(*tasks)
    
    async def process_messages(self):
        """Process incoming messages continuously."""
        while self.running:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(), timeout=1.0
                )
                await self.handle_message(message)
            except asyncio.TimeoutError:
                continue
    
    async def handle_message(self, message_envelope: dict):
        """Handle a received message."""
        message = message_envelope["message"]
        sender = message_envelope["sender"]
        
        if message["type"] == "request_help":
            response = await self.provide_help(message["data"])
            await self.send_message(
                self.get_peer_by_name(sender),
                {"type": "help_response", "data": response}
            )
        elif message["type"] == "share_knowledge":
            self.learn_from_peer(message["data"])
```

## Implementation Strategies

### Agent Specialization

#### Research Agent
```python
class ResearchAgent(LlmAgent):
    """Specialized agent for information gathering."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.add_tool(self.web_search)
        self.add_tool(self.academic_search)
        self.add_tool(self.summarize_sources)
    
    @Tool
    async def web_search(self, query: str, num_results: int = 10) -> list:
        """Search the web for information."""
        # Implementation with search API
        pass
    
    @Tool
    async def academic_search(self, query: str) -> list:
        """Search academic databases."""
        # Implementation with academic APIs
        pass
    
    @Tool
    def summarize_sources(self, sources: list) -> str:
        """Summarize multiple information sources."""
        # Implementation
        pass
    
    async def research_topic(self, topic: str) -> dict:
        """Comprehensive research on a topic."""
        web_results = await self.web_search(topic)
        academic_results = await self.academic_search(topic)
        
        all_sources = web_results + academic_results
        summary = self.summarize_sources(all_sources)
        
        return {
            "topic": topic,
            "sources": all_sources,
            "summary": summary,
            "confidence": self.assess_confidence(all_sources)
        }
```

#### Analysis Agent
```python
class AnalysisAgent(LlmAgent):
    """Specialized agent for data analysis."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.add_tool(self.statistical_analysis)
        self.add_tool(self.pattern_detection)
        self.add_tool(self.trend_analysis)
    
    @Tool
    def statistical_analysis(self, data: dict) -> dict:
        """Perform statistical analysis on data."""
        # Implementation
        pass
    
    @Tool
    def pattern_detection(self, data: dict) -> list:
        """Detect patterns in data."""
        # Implementation
        pass
    
    async def analyze_research(self, research_data: dict) -> dict:
        """Analyze research findings."""
        statistics = self.statistical_analysis(research_data)
        patterns = self.pattern_detection(research_data)
        trends = self.trend_analysis(research_data)
        
        return {
            "statistics": statistics,
            "patterns": patterns,
            "trends": trends,
            "insights": self.generate_insights(statistics, patterns, trends)
        }
```

#### Writing Agent
```python
class WritingAgent(LlmAgent):
    """Specialized agent for content creation."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.add_tool(self.structure_content)
        self.add_tool(self.improve_readability)
        self.add_tool(self.fact_check)
    
    @Tool
    def structure_content(self, content: str, format_type: str) -> str:
        """Structure content according to format."""
        # Implementation
        pass
    
    @Tool
    def improve_readability(self, content: str) -> str:
        """Improve content readability."""
        # Implementation
        pass
    
    async def write_report(self, analysis_data: dict, topic: str) -> str:
        """Write a comprehensive report."""
        # Generate initial content
        initial_content = self.generate_content(analysis_data, topic)
        
        # Structure the content
        structured_content = self.structure_content(initial_content, "report")
        
        # Improve readability
        final_content = self.improve_readability(structured_content)
        
        # Fact-check
        verified_content = await self.fact_check(final_content)
        
        return verified_content
```

## Communication Protocols

### Message Standards
```python
class AgentMessage:
    """Standard message format for inter-agent communication."""
    
    def __init__(
        self,
        sender: str,
        receiver: str, 
        message_type: str,
        content: dict,
        priority: int = 1
    ):
        self.id = self.generate_id()
        self.sender = sender
        self.receiver = receiver
        self.message_type = message_type
        self.content = content
        self.priority = priority
        self.timestamp = datetime.now()
        self.status = "pending"
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.message_type,
            "content": self.content,
            "priority": self.priority,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AgentMessage':
        message = cls(
            data["sender"],
            data["receiver"],
            data["type"],
            data["content"],
            data.get("priority", 1)
        )
        message.id = data["id"]
        message.timestamp = datetime.fromisoformat(data["timestamp"])
        message.status = data["status"]
        return message
```

### Communication Manager
```python
class CommunicationManager:
    """Manages communication between agents."""
    
    def __init__(self):
        self.agents = {}
        self.message_queue = asyncio.PriorityQueue()
        self.message_history = []
        self.running = False
    
    def register_agent(self, agent: LlmAgent):
        """Register an agent for communication."""
        self.agents[agent.name] = agent
        agent.communication_manager = self
    
    async def send_message(self, message: AgentMessage):
        """Send a message between agents."""
        # Validate sender and receiver exist
        if message.sender not in self.agents:
            raise ValueError(f"Unknown sender: {message.sender}")
        if message.receiver not in self.agents:
            raise ValueError(f"Unknown receiver: {message.receiver}")
        
        # Add to queue with priority
        await self.message_queue.put((-message.priority, message))
        
        # Log message
        self.message_history.append(message)
    
    async def process_messages(self):
        """Process messages in the queue."""
        while self.running:
            try:
                _, message = await asyncio.wait_for(
                    self.message_queue.get(), timeout=1.0
                )
                await self.deliver_message(message)
            except asyncio.TimeoutError:
                continue
    
    async def deliver_message(self, message: AgentMessage):
        """Deliver a message to the target agent."""
        receiver = self.agents[message.receiver]
        
        try:
            await receiver.handle_message(message)
            message.status = "delivered"
        except Exception as e:
            message.status = "failed"
            self.log_error(f"Message delivery failed: {e}")
```

## Workflow Management

### Workflow Definition
```python
class WorkflowStep:
    """Represents a single step in a workflow."""
    
    def __init__(
        self,
        name: str,
        agent_type: str,
        action: str,
        inputs: dict,
        outputs: list,
        dependencies: list = None
    ):
        self.name = name
        self.agent_type = agent_type
        self.action = action
        self.inputs = inputs
        self.outputs = outputs
        self.dependencies = dependencies or []
        self.status = "pending"
        self.result = None

class Workflow:
    """Represents a complete workflow."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.steps = []
        self.execution_order = []
    
    def add_step(self, step: WorkflowStep):
        """Add a step to the workflow."""
        self.steps.append(step)
        self._recalculate_execution_order()
    
    def _recalculate_execution_order(self):
        """Calculate the order of step execution based on dependencies."""
        # Topological sort implementation
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(step):
            if step.name in temp_visited:
                raise ValueError("Circular dependency detected")
            if step.name in visited:
                return
            
            temp_visited.add(step.name)
            
            for dep_name in step.dependencies:
                dep_step = next(s for s in self.steps if s.name == dep_name)
                visit(dep_step)
            
            temp_visited.remove(step.name)
            visited.add(step.name)
            order.append(step)
        
        for step in self.steps:
            if step.name not in visited:
                visit(step)
        
        self.execution_order = order
```

### Workflow Orchestrator
```python
class WorkflowOrchestrator(LlmAgent):
    """Orchestrates complex multi-agent workflows."""
    
    def __init__(self, name: str = "WorkflowOrchestrator"):
        super().__init__(name)
        self.registered_workflows = {}
        self.agent_pool = {}
        self.execution_history = []
    
    def register_workflow(self, workflow: Workflow):
        """Register a workflow for execution."""
        self.registered_workflows[workflow.name] = workflow
    
    def register_agent(self, agent: LlmAgent, agent_type: str):
        """Register an agent for workflow execution."""
        if agent_type not in self.agent_pool:
            self.agent_pool[agent_type] = []
        self.agent_pool[agent_type].append(agent)
    
    async def execute_workflow(self, workflow_name: str, initial_data: dict) -> dict:
        """Execute a registered workflow."""
        if workflow_name not in self.registered_workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        
        workflow = self.registered_workflows[workflow_name]
        execution_context = {"data": initial_data, "results": {}}
        
        try:
            for step in workflow.execution_order:
                await self.execute_step(step, execution_context)
            
            return {
                "status": "success",
                "workflow": workflow_name,
                "results": execution_context["results"]
            }
        
        except Exception as e:
            return {
                "status": "error",
                "workflow": workflow_name,
                "error": str(e),
                "completed_steps": [
                    s.name for s in workflow.steps if s.status == "completed"
                ]
            }
    
    async def execute_step(self, step: WorkflowStep, context: dict):
        """Execute a single workflow step."""
        # Check dependencies
        for dep_name in step.dependencies:
            if dep_name not in context["results"]:
                raise ValueError(f"Dependency {dep_name} not satisfied")
        
        # Get agent for step
        agent = self.get_agent_for_step(step)
        
        # Prepare inputs
        step_inputs = self.prepare_step_inputs(step, context)
        
        # Execute step
        step.status = "executing"
        try:
            result = await agent.execute_task(step.action, step_inputs)
            step.result = result
            step.status = "completed"
            
            # Store results for dependent steps
            for output_name in step.outputs:
                context["results"][output_name] = result
                
        except Exception as e:
            step.status = "failed"
            step.result = str(e)
            raise
```

## Advanced Topics

### Dynamic Agent Creation
```python
class AgentFactory:
    """Factory for creating agents dynamically."""
    
    @staticmethod
    def create_agent(agent_type: str, config: dict) -> LlmAgent:
        """Create an agent based on type and configuration."""
        if agent_type == "research":
            agent = ResearchAgent(config["name"])
            agent.configure_search_apis(config.get("apis", []))
        elif agent_type == "analysis":
            agent = AnalysisAgent(config["name"])
            agent.configure_analysis_tools(config.get("tools", []))
        elif agent_type == "writing":
            agent = WritingAgent(config["name"])
            agent.configure_writing_style(config.get("style", "professional"))
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        return agent

class DynamicCoordinator(LlmAgent):
    """Coordinator that creates agents on demand."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.active_agents = {}
        self.agent_factory = AgentFactory()
    
    async def handle_request(self, request: dict) -> dict:
        """Handle request by creating appropriate agents."""
        required_capabilities = self.analyze_requirements(request)
        
        # Create agents for required capabilities
        agents = []
        for capability in required_capabilities:
            agent_config = {
                "name": f"{capability}_{len(self.active_agents)}",
                "type": capability
            }
            agent = self.agent_factory.create_agent(capability, agent_config)
            agents.append(agent)
            self.active_agents[agent.name] = agent
        
        # Execute request using created agents
        result = await self.coordinate_agents(agents, request)
        
        # Clean up agents if needed
        await self.cleanup_agents(agents)
        
        return result
```

### Load Balancing
```python
class LoadBalancer:
    """Load balancer for distributing tasks among agents."""
    
    def __init__(self):
        self.agent_loads = {}
        self.agent_capabilities = {}
        self.performance_history = {}
    
    def register_agent(self, agent: LlmAgent, capabilities: list):
        """Register an agent with its capabilities."""
        self.agent_loads[agent.name] = 0
        self.agent_capabilities[agent.name] = capabilities
        self.performance_history[agent.name] = []
    
    def select_agent(self, task_type: str) -> LlmAgent:
        """Select the best agent for a task type."""
        # Filter capable agents
        capable_agents = [
            name for name, caps in self.agent_capabilities.items()
            if task_type in caps
        ]
        
        if not capable_agents:
            raise ValueError(f"No agents capable of {task_type}")
        
        # Select agent with lowest load
        selected_agent = min(capable_agents, key=lambda x: self.agent_loads[x])
        
        return selected_agent
    
    def update_load(self, agent_name: str, delta: int):
        """Update agent load."""
        self.agent_loads[agent_name] += delta
```

### Monitoring and Metrics
```python
class MultiAgentMonitor:
    """Monitor multi-agent system performance."""
    
    def __init__(self):
        self.metrics = {
            "message_count": 0,
            "task_completion_time": [],
            "agent_utilization": {},
            "error_count": 0
        }
    
    def log_message(self, message: AgentMessage):
        """Log inter-agent message."""
        self.metrics["message_count"] += 1
    
    def log_task_completion(self, agent_name: str, duration: float):
        """Log task completion time."""
        self.metrics["task_completion_time"].append(duration)
        
        if agent_name not in self.metrics["agent_utilization"]:
            self.metrics["agent_utilization"][agent_name] = []
        self.metrics["agent_utilization"][agent_name].append(duration)
    
    def log_error(self, agent_name: str, error: str):
        """Log agent error."""
        self.metrics["error_count"] += 1
    
    def get_performance_report(self) -> dict:
        """Generate performance report."""
        return {
            "total_messages": self.metrics["message_count"],
            "avg_completion_time": statistics.mean(self.metrics["task_completion_time"]) if self.metrics["task_completion_time"] else 0,
            "agent_utilization": {
                name: statistics.mean(times)
                for name, times in self.metrics["agent_utilization"].items()
            },
            "error_rate": self.metrics["error_count"] / max(1, self.metrics["message_count"])
        }
```

## Best Practices

### 1. Design Principles
- **Single Responsibility**: Each agent should have one clear purpose
- **Loose Coupling**: Agents should be independent and replaceable
- **Clear Interfaces**: Well-defined communication protocols
- **Fault Tolerance**: System should handle agent failures gracefully

### 2. Communication Guidelines
- **Async Operations**: Use async/await for non-blocking communication
- **Message Validation**: Validate all inter-agent messages
- **Timeout Handling**: Set timeouts for agent responses
- **Priority Systems**: Important messages should be prioritized

### 3. Performance Optimization
- **Load Balancing**: Distribute work evenly among agents
- **Caching**: Cache frequently used results
- **Batch Processing**: Group similar tasks together
- **Resource Management**: Monitor and limit resource usage

### 4. Testing Strategies
- **Unit Testing**: Test each agent individually
- **Integration Testing**: Test agent interactions
- **Load Testing**: Test system under high load
- **Failure Testing**: Test system resilience

## Example Workflows

### Research and Analysis Workflow
```python
research_workflow = Workflow(
    "Research and Analysis",
    "Comprehensive research and analysis workflow"
)

research_workflow.add_step(WorkflowStep(
    name="initial_research",
    agent_type="research",
    action="research_topic",
    inputs={"topic": "workflow_input"},
    outputs=["research_data"]
))

research_workflow.add_step(WorkflowStep(
    name="deep_analysis",
    agent_type="analysis",
    action="analyze_research",
    inputs={"research_data": "initial_research"},
    outputs=["analysis_results"],
    dependencies=["initial_research"]
))

research_workflow.add_step(WorkflowStep(
    name="report_generation",
    agent_type="writing",
    action="write_report",
    inputs={"analysis": "deep_analysis", "topic": "workflow_input"},
    outputs=["final_report"],
    dependencies=["deep_analysis"]
))
```

## Next Steps

1. **A2A Protocol** - Learn distributed agent communication
2. **Advanced Coordination** - Explore complex coordination patterns
3. **Performance Optimization** - Scale your multi-agent systems
4. **Production Deployment** - Deploy real-world applications

## Resources

- [A2A Protocol Guide](a2a-protocol.md)
- [Performance Optimization](performance.md)
- [Production Deployment](deployment.md)
- [API Reference](api-reference.md)
