"""
Multi-agent system examples using Google ADK.

This module demonstrates advanced multi-agent patterns:
1. Hierarchical agent coordination
2. Specialized agent roles and workflows
3. Agent communication and task delegation
4. Complex orchestration scenarios
"""

from typing import List, Dict, Any, Optional, Tuple
import asyncio
from datetime import datetime
from dataclasses import dataclass

from google.adk.agents import Agent, LlmAgent, BaseAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from config import config
from utils import logger, timer, format_agent_response_for_ui, format_agent_response_for_logs


@dataclass
class TaskResult:
    """Represents the result of a task executed by an agent."""
    agent_name: str
    task: str
    result: str
    success: bool
    execution_time: float
    timestamp: datetime


class ResearchAgent:
    """Specialized agent for research tasks."""
    
    def __init__(self, name: str = "ResearchAgent"):
        self.name = name
        self.agent = Agent(
            name=name,
            model=config.adk.default_model,
            instruction=(
                "You are a research specialist. You provide factual information and analysis "
                "based on publicly available data and market trends. For investment-related topics, "
                "provide educational information about market performance, company fundamentals, "
                "and sector analysis. Always include appropriate disclaimers about consulting "
                "financial advisors, but DO provide the requested research and analysis. "
                "Your role is to inform, not to advise - present data and let users make decisions."            ),
            description="Specialized agent for research and information gathering."
        )
        
        # Set up session service and runner
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.agent,
            app_name="research_agent_app",
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
                    app_name="research_agent_app",
                    user_id=self.user_id,
                    session_id=self.session_id
                )
            except Exception as e:
                logger.error(f"Failed to initialize session: {e}")
                raise
    
    @timer
    def research_topic(self, topic: str) -> TaskResult:
        """Conduct research on a specific topic."""
        try:
            return asyncio.run(self._research_topic_async(topic))
        except Exception as e:
            logger.error(f"Error in research agent: {e}")
            return TaskResult(
                agent_name=self.name,
                task=f"Research: {topic}",
                result=f"Error: {str(e)}",
                success=False,
                execution_time=0.0,
                timestamp=datetime.now()
            )
        
    async def _research_topic_async(self, topic: str) -> TaskResult:
        """Async implementation of research topic."""
        start_time = datetime.now()
        
        try:
            # Ensure session is initialized
            await self._initialize_session()
            
            prompt = (
                f"Research the topic: {topic}\n"
                "Provide a comprehensive report including:\n"
                "1. Key findings with specific data and examples\n"
                "2. Current market developments and trends\n"
                "3. Important considerations and factors\n"
                "4. Relevant performance metrics where applicable\n"
                "Note: Provide educational information and data analysis. "
                "Include appropriate disclaimers but complete the full research analysis."
            )
            
            # Create message content
            content = types.Content(role='user', parts=[types.Part(text=prompt)])
            
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
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Research agent completed topic research")
            logger.debug(format_agent_response_for_logs(final_response_text, self.name))
            
            return TaskResult(
                agent_name=self.name,
                task=f"Research: {topic}",
                result=format_agent_response_for_ui(final_response_text),
                success=True,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Research agent error: {e}")
            
            return TaskResult(
                agent_name=self.name,
                task=f"Research: {topic}",
                result=f"Error: {str(e)}",
                success=False,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
    
    def _extract_response(self, events) -> str:
        """Extract the final response text from ADK events."""
        for event in reversed(events):
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts') and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            return part.text
        return "No response generated"


class AnalysisAgent:
    """Specialized agent for data analysis and insights."""
    
    def __init__(self, name: str = "AnalysisAgent"):
        self.name = name
        self.agent = Agent(
            name=name,
            model=config.adk.default_model,
            instruction=(
                "You are an analysis specialist. Analyze information provided to you, "
                "identify patterns, draw insights, and provide data-driven analysis. "
                "For financial data, focus on objective metrics, trends, and patterns. "
                "Present findings clearly and include appropriate disclaimers when needed. "
                "Your role is analytical - provide insights based on the data given to you."
            ),
            description="Specialized agent for analysis and insights."
        )
        
        # Set up session service and runner
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.agent,
            app_name="analysis_agent_app",
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
                    app_name="analysis_agent_app",
                    user_id=self.user_id,
                    session_id=self.session_id
                )
            except Exception as e:                
                logger.error(f"Failed to initialize session: {e}")
                raise
    
    @timer
    def analyze_data(self, data: str, analysis_type: str = "general") -> TaskResult:
        """Analyze provided data and generate insights."""
        try:
            return asyncio.run(self._analyze_data_async(data, analysis_type))
        except Exception as e:
            logger.error(f"Error in analysis agent: {e}")
            return TaskResult(
                agent_name=self.name,
                task=f"Analysis: {analysis_type}",
                result=f"Error: {str(e)}",
                success=False,
                execution_time=0.0,
                timestamp=datetime.now()
            )
    
    async def _analyze_data_async(self, data: str, analysis_type: str = "general") -> TaskResult:
        """Async implementation of analyze data."""
        start_time = datetime.now()
        
        try:
            # Ensure session is initialized
            await self._initialize_session()
            
            prompt = (
                f"Analyze the following data using {analysis_type} analysis:\n\n"
                f"{data}\n\n"
                "Provide:\n"
                "1. Key insights\n"
                "2. Patterns identified\n"
                "3. Recommendations\n"
                "4. Next steps"
            )
            
            # Create message content
            content = types.Content(role='user', parts=[types.Part(text=prompt)])
            
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
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Analysis agent completed data analysis")
            logger.debug(format_agent_response_for_logs(final_response_text, self.name))
            
            return TaskResult(
                agent_name=self.name,
                task=f"Analysis: {analysis_type}",
                result=format_agent_response_for_ui(final_response_text),
                success=True,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Analysis agent error: {e}")
            
            return TaskResult(
                agent_name=self.name,
                task=f"Analysis: {analysis_type}",
                result=f"Error: {str(e)}",
                success=False,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
    
    def _extract_response(self, events) -> str:
        """Extract the final response text from ADK events."""
        for event in reversed(events):
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts') and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            return part.text
        return "No response generated"


class WritingAgent:
    """Specialized agent for content creation and writing."""
    
    def __init__(self, name: str = "WritingAgent"):
        self.name = name
        self.agent = Agent(
            name=name,
            model=config.adk.default_model,
            instruction=(
                "You are a writing specialist. Create high-quality content "
                "including reports, summaries, articles, and documentation. "
                "Focus on clarity, structure, and engaging presentation. "
                "When creating financial reports, present information objectively "
                "with appropriate disclaimers. Always complete the full requested content."
            ),
            description="Specialized agent for content creation and writing."
        )
        
        # Set up session service and runner
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.agent,
            app_name="writing_agent_app",
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
                    app_name="writing_agent_app",
                    user_id=self.user_id,
                    session_id=self.session_id
                )
            except Exception as e:
                logger.error(f"Failed to initialize session: {e}")
                raise
    
    @timer
    def create_content(self, content_type: str, inputs: Dict[str, Any]) -> TaskResult:
        """Create content based on provided inputs."""
        try:
            return asyncio.run(self._create_content_async(content_type, inputs))
        except Exception as e:
            logger.error(f"Error in writing agent: {e}")
            return TaskResult(
                agent_name=self.name,
                task=f"Create {content_type}",
                result=f"Error: {str(e)}",
                success=False,
                execution_time=0.0,
                timestamp=datetime.now()
            )
    
    async def _create_content_async(self, content_type: str, inputs: Dict[str, Any]) -> TaskResult:
        """Async implementation of create content."""
        start_time = datetime.now()
        
        try:
            # Ensure session is initialized
            await self._initialize_session()
            
            prompt = f"Create a {content_type} using the following inputs:\n\n"            
            for key, value in inputs.items():
                prompt += f"{key}:\n{value}\n\n"
            
            prompt += (
                f"Requirements:\n"
                f"1. Professional and engaging tone\n"
                f"2. Clear structure and organization\n"
                f"3. Appropriate length for {content_type}\n"                f"4. Include relevant examples if applicable"
            )
            
            # Create message content
            content = types.Content(role='user', parts=[types.Part(text=prompt)])
            
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
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Writing agent completed content creation")
            logger.debug(format_agent_response_for_logs(final_response_text, self.name))
            
            return TaskResult(
                agent_name=self.name,
                task=f"Create {content_type}",
                result=format_agent_response_for_ui(final_response_text),
                success=True,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Writing agent error: {e}")
            
            return TaskResult(
                agent_name=self.name,
                task=f"Create {content_type}",
                result=f"Error: {str(e)}",
                success=False,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
    
    def _extract_response(self, events) -> str:
        """Extract the final response text from ADK events."""
        for event in reversed(events):
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts') and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            return part.text
        return "No response generated"


class CoordinatorAgent:
    """Coordinator agent that manages and orchestrates multiple specialized agents."""
    
    def __init__(self, name: str = "CoordinatorAgent"):
        self.name = name
        self.research_agent = ResearchAgent("Research_Specialist")
        self.analysis_agent = AnalysisAgent("Analysis_Specialist")
        self.writing_agent = WritingAgent("Writing_Specialist")
        
        # Create the main coordinator using ADK's multi-agent pattern
        self.agent = Agent(
            name=name,
            model=config.adk.default_model,
            instruction=(
                "You are a project coordinator managing a team of specialists. "
                "Break down complex tasks into subtasks and delegate to the "
                "appropriate specialists: research, analysis, and writing. "
                "Coordinate their work and synthesize final results."
            ),
            description="Coordinator agent managing specialized sub-agents.",
            sub_agents=[
                self.research_agent.agent,
                self.analysis_agent.agent,
                self.writing_agent.agent
            ]
        )
        
        self.task_history: List[TaskResult] = []
        # Set up session service and runner
    @timer
    def execute_complex_project(self, project_description: str) -> Dict[str, Any]:
        """Execute a complex project using multiple specialized agents."""
        logger.info(f"Starting complex project: {project_description}")
        
        project_results = {
            "project": project_description,
            "start_time": datetime.now(),
            "tasks": [],
            "final_result": "",
            "success": False
        }
        
        try:
            # Step 1: Research phase with enhanced prompt
            logger.info("Phase 1: Research")
            enhanced_research_prompt = (
                f"{project_description}\n\n"
                "IMPORTANT: Provide complete factual information and data. "
                "Do not ask for permission to proceed. For financial/investment topics, "
                "provide educational information including specific examples, metrics, "
                "and data analysis. Include appropriate disclaimers but complete the full analysis."
            )
            
            research_result = self.research_agent.research_topic(enhanced_research_prompt)
            project_results["tasks"].append(research_result)
            self.task_history.append(research_result)
            
            if not research_result.success:
                project_results["final_result"] = "Project failed during research phase"
                return project_results
            
            # Step 2: Analysis phase with research data
            logger.info("Phase 2: Analysis")
            analysis_prompt = (
                f"RESEARCH DATA PROVIDED:\n{research_result.result}\n\n"
                f"Based on the above research data, perform {project_description} analysis. "
                "Analyze the provided information and identify patterns, insights, and recommendations."
            )
            
            analysis_result = self.analysis_agent.analyze_data(
                analysis_prompt, 
                "strategic"
            )
            project_results["tasks"].append(analysis_result)
            self.task_history.append(analysis_result)
            
            if not analysis_result.success:
                project_results["final_result"] = "Project failed during analysis phase"
                return project_results
            
            # Step 3: Writing phase with all data
            logger.info("Phase 3: Content Creation")
            writing_inputs = {
                "Research Findings": research_result.result,
                "Analysis Results": analysis_result.result,
                "Project Goal": project_description,
                "Instructions": "Create a comprehensive, complete report. Do not ask for permission or additional information."
            }
            
            writing_result = self.writing_agent.create_content(
                "comprehensive report", 
                writing_inputs
            )
            project_results["tasks"].append(writing_result)
            self.task_history.append(writing_result)
            
            if not writing_result.success:
                project_results["final_result"] = "Project failed during writing phase"
                return project_results
            
            # Final coordination
            project_results["final_result"] = writing_result.result
            project_results["success"] = True
            project_results["end_time"] = datetime.now()
            
            total_time = (project_results["end_time"] - project_results["start_time"]).total_seconds()
            logger.info(f"Complex project completed in {total_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Complex project error: {e}")
            project_results["final_result"] = f"Project failed with error: {str(e)}"
        
        return project_results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of all agents."""
        if not self.task_history:
            return {"message": "No tasks executed yet"}
        
        total_tasks = len(self.task_history)
        successful_tasks = len([task for task in self.task_history if task.success])
        avg_execution_time = sum(task.execution_time for task in self.task_history) / total_tasks
        
        agent_performance = {}
        for task in self.task_history:
            if task.agent_name not in agent_performance:
                agent_performance[task.agent_name] = {
                    "total_tasks": 0,
                    "successful_tasks": 0,
                    "avg_time": 0,
                    "total_time": 0
                }
            
            agent_perf = agent_performance[task.agent_name]
            agent_perf["total_tasks"] += 1
            agent_perf["total_time"] += task.execution_time
            
            if task.success:
                agent_perf["successful_tasks"] += 1
        
        # Calculate averages
        for agent_name, perf in agent_performance.items():
            perf["avg_time"] = perf["total_time"] / perf["total_tasks"]
            perf["success_rate"] = perf["successful_tasks"] / perf["total_tasks"]
        
        return {
            "overall": {
                "total_tasks": total_tasks,
                "successful_tasks": successful_tasks,
                "success_rate": successful_tasks / total_tasks,
                "avg_execution_time": avg_execution_time
            },
            "by_agent": agent_performance
        }
    
    def continue_workflow(self, project_description: str, context: str = "") -> Dict[str, Any]:
        """Continue an incomplete workflow with additional context."""
        logger.info(f"Continuing workflow for: {project_description}")
        
        enhanced_description = f"{project_description}\n\nAdditional Context: {context}\n\nPlease provide a complete analysis without asking for permission to proceed."
        
        return self.execute_complex_project(enhanced_description)
    
    def force_complete_research(self, topic: str) -> TaskResult:
        """Force completion of research task with explicit instructions."""
        enhanced_prompt = (
            f"Research the topic: {topic}\n"
            "IMPORTANT: You must provide complete factual information. Do not ask for permission to proceed.\n"
            "For financial topics, provide educational information about:\n"
            "- Market performance data\n"
            "- Company fundamentals\n"
            "- Sector analysis\n"
            "- Historical trends\n"
            "Include disclaimers but complete the full analysis as requested."
        )
        
        return self.research_agent.research_topic(enhanced_prompt)
    
    def retry_failed_phase(self, phase_name: str, original_input: str, additional_context: str = "") -> TaskResult:
        """Retry a specific failed phase with enhanced instructions."""
        logger.info(f"Retrying failed phase: {phase_name}")
        
        enhanced_input = (
            f"{original_input}\n\n"
            f"RETRY INSTRUCTIONS: {additional_context}\n"
            "IMPORTANT: Provide complete information. Do not ask for permission or additional data."
        )
        
        if phase_name.lower() == "research":
            return self.research_agent.research_topic(enhanced_input)
        elif phase_name.lower() == "analysis":
            return self.analysis_agent.analyze_data(enhanced_input, "comprehensive")
        elif phase_name.lower() == "writing":
            return self.writing_agent.create_content("complete report", {"Content": enhanced_input})
        else:
            raise ValueError(f"Unknown phase: {phase_name}")

# Advanced multi-agent workflow example
class WorkflowOrchestrator:
    """Orchestrates complex workflows with multiple agents."""
    
    def __init__(self):
        self.coordinator = CoordinatorAgent("Main_Coordinator")
        self.workflows = {}
    
    def register_workflow(self, name: str, workflow_definition: Dict[str, Any]):
        """Register a new workflow definition."""
        self.workflows[name] = workflow_definition
        logger.info(f"Registered workflow: {name}")
    
    async def execute_workflow(self, workflow_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a registered workflow with given inputs."""
        if workflow_name not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        
        workflow = self.workflows[workflow_name]
        logger.info(f"Executing workflow: {workflow_name}")
        
        # For this example, we'll use the coordinator to execute the workflow
        project_description = f"{workflow['description']} with inputs: {inputs}"
        
        result = self.coordinator.execute_complex_project(project_description)
        
        return {
            "workflow_name": workflow_name,
            "inputs": inputs,
            "result": result,
            "performance": self.coordinator.get_performance_summary()
        }


# Example workflow definitions
EXAMPLE_WORKFLOWS = {
    "market_analysis": {
        "description": "Comprehensive market analysis for a product or service",
        "required_inputs": ["product", "target_market", "timeframe"],
        "expected_outputs": ["market_size", "competition", "opportunities"]
    },
    "content_strategy": {
        "description": "Content strategy development for marketing campaigns",
        "required_inputs": ["brand", "audience", "goals"],
        "expected_outputs": ["content_plan", "channels", "metrics"]
    },
    "risk_assessment": {
        "description": "Risk assessment for business decisions",
        "required_inputs": ["decision", "context", "constraints"],
        "expected_outputs": ["risk_factors", "mitigation", "recommendations"]
    }
}


async def demonstrate_multi_agent_system():
    """Demonstrate the multi-agent system capabilities."""
    logger.info("Starting multi-agent system demonstration")
    
    # Create orchestrator and register workflows
    orchestrator = WorkflowOrchestrator()
    
    for name, workflow in EXAMPLE_WORKFLOWS.items():
        orchestrator.register_workflow(name, workflow)
    
    # Execute a sample workflow
    sample_inputs = {
        "product": "AI-powered productivity app",
        "target_market": "remote workers",
        "timeframe": "next 12 months"
    }
    
    result = await orchestrator.execute_workflow("market_analysis", sample_inputs)
    
    print("Multi-Agent System Results:")
    print(f"Success: {result['result']['success']}")
    print(f"Tasks Executed: {len(result['result']['tasks'])}")
    print(f"Performance Summary: {result['performance']}")
    
    logger.info("Multi-agent system demonstration completed")


if __name__ == "__main__":
    asyncio.run(demonstrate_multi_agent_system())
