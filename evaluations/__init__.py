"""
Evaluation Framework for ADK & A2A Agents.

This module provides comprehensive testing and evaluation tools for:
- Agent functionality and correctness
- Performance benchmarking
- Multi-agent coordination assessment
- A2A protocol validation
- Tool integration testing
"""

import asyncio
import time
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import statistics

from config import config
from utils import logger, timer, save_json, load_json


@dataclass
class TestResult:
    """Represents the result of a single test."""
    test_name: str
    passed: bool
    score: float
    execution_time: float
    details: Dict[str, Any]
    timestamp: datetime
    error_message: Optional[str] = None


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report."""
    agent_name: str
    test_suite: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    average_score: float
    total_execution_time: float
    test_results: List[TestResult]
    timestamp: datetime
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        return (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0


class AgentEvaluator:
    """Base class for agent evaluation."""
    
    def __init__(self, name: str):
        self.name = name
        self.test_results: List[TestResult] = []
    
    async def run_test(self, test_name: str, test_func: Callable, 
                      *args, **kwargs) -> TestResult:
        """Run a single test and record results."""
        start_time = time.time()
        
        try:
            result = await test_func(*args, **kwargs) if asyncio.iscoroutinefunction(test_func) else test_func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            # Determine pass/fail and score based on result
            if isinstance(result, dict):
                passed = result.get("passed", True)
                score = result.get("score", 1.0 if passed else 0.0)
                details = result.get("details", {})
            elif isinstance(result, bool):
                passed = result
                score = 1.0 if passed else 0.0
                details = {}
            else:
                passed = True
                score = 1.0
                details = {"result": str(result)}
            
            test_result = TestResult(
                test_name=test_name,
                passed=passed,
                score=score,
                execution_time=execution_time,
                details=details,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            test_result = TestResult(
                test_name=test_name,
                passed=False,
                score=0.0,
                execution_time=execution_time,
                details={},
                timestamp=datetime.now(),
                error_message=str(e)
            )
            
            logger.error(f"Test {test_name} failed with error: {e}")
        
        self.test_results.append(test_result)
        return test_result
    
    def generate_report(self, test_suite_name: str, agent_name: str) -> EvaluationReport:
        """Generate evaluation report from test results."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.passed)
        failed_tests = total_tests - passed_tests
        
        average_score = statistics.mean([result.score for result in self.test_results]) if self.test_results else 0.0
        total_execution_time = sum(result.execution_time for result in self.test_results)
        
        return EvaluationReport(
            agent_name=agent_name,
            test_suite=test_suite_name,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            average_score=average_score,
            total_execution_time=total_execution_time,
            test_results=self.test_results.copy(),
            timestamp=datetime.now()
        )


class BasicAgentEvaluator(AgentEvaluator):
    """Evaluator for basic agent functionality."""
    
    def __init__(self):
        super().__init__("BasicAgentEvaluator")
    
    def test_response_generation(self, agent, prompt: str) -> Dict[str, Any]:
        """Test if agent generates a response to a prompt."""
        try:
            if hasattr(agent, 'chat'):
                response = agent.chat(prompt)
            elif hasattr(agent, 'run'):
                response = agent.run(prompt)
            else:
                return {"passed": False, "details": {"error": "No chat or run method found"}}
            
            # Basic validation
            is_valid = isinstance(response, str) and len(response.strip()) > 0
            
            return {
                "passed": is_valid,
                "score": 1.0 if is_valid else 0.0,
                "details": {
                    "prompt": prompt,
                    "response_length": len(response) if isinstance(response, str) else 0,
                    "response_type": type(response).__name__
                }
            }
        
        except Exception as e:
            return {"passed": False, "score": 0.0, "details": {"error": str(e)}}
    
    def test_tool_usage(self, agent, tool_prompt: str) -> Dict[str, Any]:
        """Test if agent can use tools when appropriate."""
        try:
            if hasattr(agent, 'process_request'):
                response = agent.process_request(tool_prompt)
            elif hasattr(agent, 'chat'):
                response = agent.chat(tool_prompt)
            else:
                return {"passed": False, "details": {"error": "No suitable method found"}}
            
            # Check if response indicates tool usage
            tool_indicators = ["result:", "calculation:", "weather:", "analysis:"]
            used_tool = any(indicator in response.lower() for indicator in tool_indicators)
            
            return {
                "passed": used_tool,
                "score": 1.0 if used_tool else 0.5,  # Partial credit if response exists
                "details": {
                    "prompt": tool_prompt,
                    "tool_usage_detected": used_tool,
                    "response": response[:200] + "..." if len(response) > 200 else response
                }
            }
        
        except Exception as e:
            return {"passed": False, "score": 0.0, "details": {"error": str(e)}}
    
    def test_error_handling(self, agent, invalid_input: str) -> Dict[str, Any]:
        """Test how agent handles invalid or problematic input."""
        try:
            if hasattr(agent, 'chat'):
                response = agent.chat(invalid_input)
            else:
                return {"passed": False, "details": {"error": "No chat method found"}}
            
            # Check if agent handled error gracefully (no crash, some response)
            handled_gracefully = isinstance(response, str) and len(response) > 0
            
            # Check for error indicators in response
            error_handled = any(word in response.lower() for word in 
                              ["error", "sorry", "cannot", "unable", "unclear"])
            
            score = 1.0 if handled_gracefully else 0.0
            if handled_gracefully and error_handled:
                score = 1.0
            elif handled_gracefully:
                score = 0.7
            
            return {
                "passed": handled_gracefully,
                "score": score,
                "details": {
                    "input": invalid_input,
                    "graceful_handling": handled_gracefully,
                    "error_acknowledgment": error_handled
                }
            }
        
        except Exception as e:
            # If agent crashed, it failed the test
            return {"passed": False, "score": 0.0, "details": {"error": str(e)}}
    
    async def evaluate_agent(self, agent, agent_name: str) -> EvaluationReport:
        """Run comprehensive evaluation of a basic agent."""
        logger.info(f"Starting evaluation of {agent_name}")
        
        # Test cases
        test_cases = [
            ("Response Generation - Simple", self.test_response_generation, 
             agent, "Hello, how are you?"),
            ("Response Generation - Complex", self.test_response_generation, 
             agent, "Explain the concept of artificial intelligence in simple terms."),
            ("Tool Usage - Weather", self.test_tool_usage, 
             agent, "What's the weather like in Paris?"),
            ("Tool Usage - Calculation", self.test_tool_usage, 
             agent, "Calculate 15 * 23 + 7"),
            ("Error Handling - Empty Input", self.test_error_handling, 
             agent, ""),
            ("Error Handling - Nonsense", self.test_error_handling, 
             agent, "xjkldsjflkdsjf random nonsense input"),
        ]
        
        # Run tests
        for test_name, test_func, *args in test_cases:
            await self.run_test(test_name, test_func, *args)
        
        report = self.generate_report("Basic Agent Functionality", agent_name)
        logger.info(f"Evaluation completed for {agent_name}: {report.success_rate:.1f}% success rate")
        
        return report


class PerformanceEvaluator(AgentEvaluator):
    """Evaluator for agent performance metrics."""
    
    def __init__(self):
        super().__init__("PerformanceEvaluator")
    
    def test_response_time(self, agent, prompt: str, max_time: float = 5.0) -> Dict[str, Any]:
        """Test agent response time."""
        start_time = time.time()
        
        try:
            if hasattr(agent, 'chat'):
                response = agent.chat(prompt)
            else:
                return {"passed": False, "details": {"error": "No chat method found"}}
            
            response_time = time.time() - start_time
            passed = response_time <= max_time
            score = max(0.0, (max_time - response_time) / max_time) if response_time <= max_time else 0.0
            
            return {
                "passed": passed,
                "score": score,
                "details": {
                    "response_time": response_time,
                    "max_allowed_time": max_time,
                    "prompt_length": len(prompt)
                }
            }
        
        except Exception as e:
            return {"passed": False, "score": 0.0, "details": {"error": str(e)}}
    
    def test_throughput(self, agent, prompts: List[str]) -> Dict[str, Any]:
        """Test agent throughput with multiple requests."""
        start_time = time.time()
        successful_responses = 0
        
        try:
            for prompt in prompts:
                if hasattr(agent, 'chat'):
                    response = agent.chat(prompt)
                    if isinstance(response, str) and len(response) > 0:
                        successful_responses += 1
            
            total_time = time.time() - start_time
            throughput = successful_responses / total_time  # responses per second
            
            # Score based on throughput (higher is better)
            # Assume 1 response per second is baseline
            score = min(1.0, throughput / 1.0)
            
            return {
                "passed": successful_responses == len(prompts),
                "score": score,
                "details": {
                    "total_requests": len(prompts),
                    "successful_responses": successful_responses,
                    "total_time": total_time,
                    "throughput_rps": throughput
                }
            }
        
        except Exception as e:
            return {"passed": False, "score": 0.0, "details": {"error": str(e)}}
    
    async def evaluate_performance(self, agent, agent_name: str) -> EvaluationReport:
        """Run performance evaluation."""
        logger.info(f"Starting performance evaluation of {agent_name}")
        
        # Performance test cases
        test_cases = [
            ("Response Time - Short Prompt", self.test_response_time, 
             agent, "Hello", 2.0),
            ("Response Time - Medium Prompt", self.test_response_time, 
             agent, "Explain the benefits of using AI agents in business applications.", 5.0),
            ("Response Time - Long Prompt", self.test_response_time, 
             agent, "Write a detailed analysis of the current state of artificial intelligence technology, including its applications, limitations, and future prospects.", 10.0),
            ("Throughput Test", self.test_throughput, 
             agent, ["Hello", "How are you?", "What's 2+2?", "Tell me a joke", "Goodbye"]),
        ]
        
        # Run tests
        for test_name, test_func, *args in test_cases:
            await self.run_test(test_name, test_func, *args)
        
        report = self.generate_report("Performance Evaluation", agent_name)
        logger.info(f"Performance evaluation completed for {agent_name}")
        
        return report


class MultiAgentEvaluator(AgentEvaluator):
    """Evaluator for multi-agent system coordination."""
    
    def __init__(self):
        super().__init__("MultiAgentEvaluator")
    
    def test_task_delegation(self, coordinator, task: str) -> Dict[str, Any]:
        """Test if coordinator properly delegates tasks."""
        try:
            if hasattr(coordinator, 'execute_complex_project'):
                result = coordinator.execute_complex_project(task)
                
                success = result.get("success", False)
                tasks_executed = len(result.get("tasks", []))
                
                # Score based on successful task completion
                score = 1.0 if success else (tasks_executed / 3.0)  # Assume 3 phases
                
                return {
                    "passed": success,
                    "score": score,
                    "details": {
                        "task": task,
                        "success": success,
                        "tasks_executed": tasks_executed,
                        "execution_phases": [t.task for t in result.get("tasks", [])]
                    }
                }
            else:
                return {"passed": False, "details": {"error": "No execute_complex_project method"}}
        
        except Exception as e:
            return {"passed": False, "score": 0.0, "details": {"error": str(e)}}
    
    def test_coordination_quality(self, coordinator, complex_task: str) -> Dict[str, Any]:
        """Test quality of multi-agent coordination."""
        try:
            if hasattr(coordinator, 'execute_complex_project'):
                result = coordinator.execute_complex_project(complex_task)
                
                # Analyze coordination quality
                tasks = result.get("tasks", [])
                all_successful = all(task.success for task in tasks)
                coordination_score = sum(1 for task in tasks if task.success) / len(tasks) if tasks else 0
                
                # Check if final result integrates all phases
                final_result = result.get("final_result", "")
                integration_quality = len(final_result) > 100  # Basic check for substantial output
                
                score = (coordination_score + (1.0 if integration_quality else 0.0)) / 2.0
                
                return {
                    "passed": all_successful and integration_quality,
                    "score": score,
                    "details": {
                        "task": complex_task,
                        "coordination_score": coordination_score,
                        "integration_quality": integration_quality,
                        "result_length": len(final_result)
                    }
                }
            else:
                return {"passed": False, "details": {"error": "No execute_complex_project method"}}
        
        except Exception as e:
            return {"passed": False, "score": 0.0, "details": {"error": str(e)}}
    
    async def evaluate_multi_agent_system(self, coordinator, system_name: str) -> EvaluationReport:
        """Run multi-agent system evaluation."""
        logger.info(f"Starting multi-agent evaluation of {system_name}")
        
        # Multi-agent test cases
        test_cases = [
            ("Task Delegation - Simple", self.test_task_delegation, 
             coordinator, "Research renewable energy technologies"),
            ("Task Delegation - Complex", self.test_task_delegation, 
             coordinator, "Create a comprehensive business plan for a tech startup"),
            ("Coordination Quality", self.test_coordination_quality, 
             coordinator, "Analyze market trends and write a strategic report with recommendations"),
        ]
        
        # Run tests
        for test_name, test_func, *args in test_cases:
            await self.run_test(test_name, test_func, *args)
        
        report = self.generate_report("Multi-Agent System Evaluation", system_name)
        logger.info(f"Multi-agent evaluation completed for {system_name}")
        
        return report


class EvaluationSuite:
    """Comprehensive evaluation suite for all agent types."""
    
    def __init__(self, output_dir: str = "evaluations/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.basic_evaluator = BasicAgentEvaluator()
        self.performance_evaluator = PerformanceEvaluator()
        self.multi_agent_evaluator = MultiAgentEvaluator()
    
    async def evaluate_basic_agent(self, agent, agent_name: str) -> EvaluationReport:
        """Evaluate a basic agent."""
        return await self.basic_evaluator.evaluate_agent(agent, agent_name)
    
    async def evaluate_performance(self, agent, agent_name: str) -> EvaluationReport:
        """Evaluate agent performance."""
        return await self.performance_evaluator.evaluate_performance(agent, agent_name)
    
    async def evaluate_multi_agent_system(self, coordinator, system_name: str) -> EvaluationReport:
        """Evaluate a multi-agent system."""
        return await self.multi_agent_evaluator.evaluate_multi_agent_system(coordinator, system_name)
    
    async def run_comprehensive_evaluation(self, agent, agent_name: str, 
                                         include_performance: bool = True) -> Dict[str, EvaluationReport]:
        """Run comprehensive evaluation including multiple test suites."""
        logger.info(f"Starting comprehensive evaluation of {agent_name}")
        
        reports = {}
        
        # Basic functionality evaluation
        reports["functionality"] = await self.evaluate_basic_agent(agent, agent_name)
        
        # Performance evaluation
        if include_performance:
            reports["performance"] = await self.evaluate_performance(agent, agent_name)
        
        # Save reports
        self.save_reports(reports, agent_name)
        
        logger.info(f"Comprehensive evaluation completed for {agent_name}")
        return reports
    
    def save_reports(self, reports: Dict[str, EvaluationReport], agent_name: str):
        """Save evaluation reports to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for report_type, report in reports.items():
            filename = f"{agent_name}_{report_type}_{timestamp}.json"
            filepath = self.output_dir / filename
            
            # Convert report to dictionary for JSON serialization
            report_dict = asdict(report)
            report_dict["timestamp"] = report.timestamp.isoformat()
            
            # Convert test results
            report_dict["test_results"] = [
                {**asdict(result), "timestamp": result.timestamp.isoformat()}
                for result in report.test_results
            ]
            
            save_json(report_dict, filepath)
            logger.info(f"Saved {report_type} report to {filepath}")
    
    def generate_summary_report(self, reports: Dict[str, EvaluationReport]) -> Dict[str, Any]:
        """Generate summary report from multiple evaluation reports."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_test_suites": len(reports),
            "overall_metrics": {},
            "test_suite_summaries": {}
        }
        
        total_tests = 0
        total_passed = 0
        total_score = 0.0
        
        for suite_name, report in reports.items():
            total_tests += report.total_tests
            total_passed += report.passed_tests
            total_score += report.average_score
            
            summary["test_suite_summaries"][suite_name] = {
                "success_rate": report.success_rate,
                "average_score": report.average_score,
                "execution_time": report.total_execution_time,
                "tests": report.total_tests
            }
        
        summary["overall_metrics"] = {
            "overall_success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
            "overall_average_score": total_score / len(reports) if reports else 0,
            "total_tests": total_tests,
            "total_passed": total_passed
        }
        
        return summary


# Example evaluation scripts
async def evaluate_all_basic_agents():
    """Evaluate all basic agent types."""
    from agents.basic import create_agent
    
    evaluation_suite = EvaluationSuite()
    
    agent_types = ["simple", "search", "tool", "stateful"]
    all_reports = {}
    
    for agent_type in agent_types:
        try:
            agent = create_agent(agent_type, f"Test_{agent_type.title()}")
            reports = await evaluation_suite.run_comprehensive_evaluation(agent, f"{agent_type}_agent")
            all_reports[agent_type] = reports
            
        except Exception as e:
            logger.error(f"Failed to evaluate {agent_type} agent: {e}")
    
    # Generate overall summary
    summary = evaluation_suite.generate_summary_report({
        f"{agent_type}_{report_type}": report 
        for agent_type, reports in all_reports.items()
        for report_type, report in reports.items()
    })
    
    return all_reports, summary


if __name__ == "__main__":
    asyncio.run(evaluate_all_basic_agents())
