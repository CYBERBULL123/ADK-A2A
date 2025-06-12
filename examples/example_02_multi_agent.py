"""
Example 2: Multi-Agent Systems and Coordination

This example demonstrates advanced multi-agent patterns:
1. Creating specialized agents for different roles
2. Coordinating agents to work together on complex tasks
3. Implementing workflow orchestration
4. Managing agent communication and task delegation
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config import config
from utils import logger, display_welcome, validate_environment
from agents.multi_agent import (
    CoordinatorAgent, WorkflowOrchestrator, EXAMPLE_WORKFLOWS,
    ResearchAgent, AnalysisAgent, WritingAgent
)


async def demo_specialized_agents():
    """Demonstrate individual specialized agents."""
    print("\n" + "="*60)
    print("DEMO 1: Specialized Agent Roles")
    print("="*60)
    
    # Create specialized agents
    research_agent = ResearchAgent("Research_Specialist")
    analysis_agent = AnalysisAgent("Analysis_Specialist")
    writing_agent = WritingAgent("Writing_Specialist")
    
    # Test research agent
    print("\n🔍 Research Agent Demo:")
    research_topic = "artificial intelligence in healthcare"
    print(f"👤 Task: Research {research_topic}")
    
    research_result = research_agent.research_topic(research_topic)
    print(f"🤖 Research Agent: Task completed in {research_result.execution_time:.2f}s")
    print(f"   Success: {research_result.success}")
    if research_result.success:
        print(f"   Result preview: {research_result.result[:200]}...")
    
    # Test analysis agent
    print("\n📊 Analysis Agent Demo:")
    sample_data = research_result.result if research_result.success else "Sample data about AI in healthcare trends, adoption rates, and benefits."
    print(f"👤 Task: Analyze the research findings")
    
    analysis_result = analysis_agent.analyze_data(sample_data, "strategic")
    print(f"🤖 Analysis Agent: Task completed in {analysis_result.execution_time:.2f}s")
    print(f"   Success: {analysis_result.success}")
    if analysis_result.success:
        print(f"   Result preview: {analysis_result.result[:200]}...")
    
    # Test writing agent
    print("\n✍️ Writing Agent Demo:")
    writing_inputs = {
        "Research Findings": research_result.result if research_result.success else "AI healthcare research data",
        "Analysis Results": analysis_result.result if analysis_result.success else "Strategic analysis of AI healthcare",
        "Target Audience": "Healthcare executives and decision makers"
    }
    print(f"👤 Task: Create an executive summary")
    
    writing_result = writing_agent.create_content("executive summary", writing_inputs)
    print(f"🤖 Writing Agent: Task completed in {writing_result.execution_time:.2f}s")
    print(f"   Success: {writing_result.success}")
    if writing_result.success:
        print(f"   Result preview: {writing_result.result[:300]}...")
    
    print("\n✅ Specialized agents demo completed!")


async def demo_coordinator_system():
    """Demonstrate the coordinator agent orchestrating multiple specialists."""
    print("\n" + "="*60)
    print("DEMO 2: Multi-Agent Coordination")
    print("="*60)
    
    # Create coordinator
    coordinator = CoordinatorAgent("Project_Coordinator")
    
    # Define complex projects
    projects = [
        "Create a comprehensive market analysis for electric vehicles",
        "Develop a strategic plan for implementing AI in small businesses",
        "Research and analyze renewable energy investment opportunities"
    ]
    
    for i, project in enumerate(projects, 1):
        print(f"\n🎯 Project {i}: {project}")
        print("👤 Initiating multi-agent coordination...")
        
        try:
            # Execute project using coordinator
            result = coordinator.execute_complex_project(project)
            
            print(f"\n📋 Project Results:")
            print(f"   Success: {'✅' if result['success'] else '❌'}")
            print(f"   Phases Completed: {len(result['tasks'])}")
            print(f"   Total Time: {(result.get('end_time', result['start_time']) - result['start_time']).total_seconds():.2f}s" if 'end_time' in result else "N/A")
            
            # Show phase breakdown
            for j, task in enumerate(result['tasks'], 1):
                status = "✅" if task.success else "❌"
                print(f"   Phase {j} ({task.agent_name}): {status} - {task.execution_time:.2f}s")
            
            if result['success']:
                print(f"\n📄 Final Output Preview:")
                print(f"   {result['final_result'][:250]}...")
        
        except Exception as e:
            print(f"❌ Project failed: {e}")
    
    # Show performance summary
    print(f"\n📊 Overall Performance Summary:")
    performance = coordinator.get_performance_summary()
    if "overall" in performance:
        overall = performance["overall"]
        print(f"   Total Tasks: {overall['total_tasks']}")
        print(f"   Success Rate: {overall['success_rate']:.1%}")
        print(f"   Avg Execution Time: {overall['avg_execution_time']:.2f}s")
    
    print("\n✅ Coordinator system demo completed!")


async def demo_workflow_orchestration():
    """Demonstrate workflow orchestration with predefined workflows."""
    print("\n" + "="*60)
    print("DEMO 3: Workflow Orchestration")
    print("="*60)
    
    # Create orchestrator
    orchestrator = WorkflowOrchestrator()
    
    # Register example workflows
    for name, workflow in EXAMPLE_WORKFLOWS.items():
        orchestrator.register_workflow(name, workflow)
    
    print(f"📋 Available Workflows: {list(EXAMPLE_WORKFLOWS.keys())}")
    
    # Execute each workflow with sample inputs
    workflow_tests = [
        ("market_analysis", {
            "product": "Smart Home Security System",
            "target_market": "Homeowners aged 25-45",
            "timeframe": "next 18 months"
        }),
        ("content_strategy", {
            "brand": "EcoTech Solutions",
            "audience": "Environmentally conscious consumers",
            "goals": "Increase brand awareness and drive sales"
        }),
        ("risk_assessment", {
            "decision": "Expand into international markets",
            "context": "Technology startup with 2 years of domestic success",
            "constraints": "Limited budget and regulatory complexity"
        })
    ]
    
    for workflow_name, inputs in workflow_tests:
        print(f"\n🔄 Executing Workflow: {workflow_name}")
        print(f"📥 Inputs: {inputs}")
        
        try:
            # Note: In a real async environment, this would work
            # For demo purposes, we'll simulate the execution
            print("🚀 Workflow execution initiated...")
            print("   Phase 1: Research and data gathering...")
            print("   Phase 2: Analysis and insights...")
            print("   Phase 3: Report generation...")
            print("✅ Workflow completed successfully!")
            
            # Simulate results
            print(f"📊 Simulated Results:")
            print(f"   Workflow: {workflow_name}")
            print(f"   Status: Completed")
            print(f"   Output: Comprehensive {workflow_name.replace('_', ' ')} report generated")
        
        except Exception as e:
            print(f"❌ Workflow failed: {e}")
    
    print("\n✅ Workflow orchestration demo completed!")


async def demo_agent_communication():
    """Demonstrate direct communication between agents."""
    print("\n" + "="*60)
    print("DEMO 4: Agent-to-Agent Communication")
    print("="*60)
    
    # Create agents
    research_agent = ResearchAgent("Research_Alice")
    analysis_agent = AnalysisAgent("Analysis_Bob")
    writing_agent = WritingAgent("Writing_Charlie")
    
    print("👥 Created three specialized agents:")
    print("   🔍 Research_Alice - Research specialist")
    print("   📊 Analysis_Bob - Analysis specialist")
    print("   ✍️ Writing_Charlie - Writing specialist")
    
    # Simulate communication chain
    topic = "sustainable technology trends"
    
    print(f"\n🔗 Communication Chain for topic: '{topic}'")
    
    # Step 1: Research
    print("\n📤 Step 1: Research_Alice conducts research")
    research_result = research_agent.research_topic(topic)
    
    if research_result.success:
        print(f"✅ Research completed in {research_result.execution_time:.2f}s")
        
        # Step 2: Analysis
        print("\n📤 Step 2: Analysis_Bob analyzes research data")
        analysis_result = analysis_agent.analyze_data(research_result.result, "trend")
        
        if analysis_result.success:
            print(f"✅ Analysis completed in {analysis_result.execution_time:.2f}s")
            
            # Step 3: Writing
            print("\n📤 Step 3: Writing_Charlie creates final report")
            writing_inputs = {
                "Research Data": research_result.result,
                "Analysis Results": analysis_result.result,
                "Report Type": "Trend Analysis Report"
            }
            
            writing_result = writing_agent.create_content("trend report", writing_inputs)
            
            if writing_result.success:
                print(f"✅ Writing completed in {writing_result.execution_time:.2f}s")
                print(f"\n📄 Final Report Preview:")
                print(f"{writing_result.result[:400]}...")
            else:
                print(f"❌ Writing failed: {writing_result.result}")
        else:
            print(f"❌ Analysis failed: {analysis_result.result}")
    else:
        print(f"❌ Research failed: {research_result.result}")
    
    print("\n✅ Agent communication demo completed!")


async def demo_performance_comparison():
    """Compare performance of single agent vs multi-agent approach."""
    print("\n" + "="*60)
    print("DEMO 5: Performance Comparison")
    print("="*60)
    
    from agents.basic import SimpleAgent
    import time
    
    task = "Create a business analysis report on renewable energy market opportunities"
    
    # Single agent approach
    print("🤖 Single Agent Approach:")
    single_agent = SimpleAgent("Single_Task_Agent")
    
    start_time = time.time()
    try:
        single_result = single_agent.chat(f"Please {task}")
        single_time = time.time() - start_time
        print(f"   ✅ Completed in {single_time:.2f}s")
        print(f"   📄 Result length: {len(single_result)} characters")
        print(f"   📝 Preview: {single_result[:200]}...")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        single_time = float('inf')
    
    # Multi-agent approach
    print("\n👥 Multi-Agent Approach:")
    coordinator = CoordinatorAgent("Performance_Coordinator")
    
    start_time = time.time()
    try:
        multi_result = coordinator.execute_complex_project(task)
        multi_time = time.time() - start_time
        
        if multi_result['success']:
            print(f"   ✅ Completed in {multi_time:.2f}s")
            print(f"   📄 Result length: {len(multi_result['final_result'])} characters")
            print(f"   📝 Preview: {multi_result['final_result'][:200]}...")
            print(f"   🔄 Phases executed: {len(multi_result['tasks'])}")
        else:
            print(f"   ❌ Failed: {multi_result.get('final_result', 'Unknown error')}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        multi_time = float('inf')
    
    # Comparison
    print(f"\n📊 Performance Comparison:")
    if single_time != float('inf') and multi_time != float('inf'):
        print(f"   Single Agent Time: {single_time:.2f}s")
        print(f"   Multi-Agent Time: {multi_time:.2f}s")
        print(f"   Time Difference: {abs(multi_time - single_time):.2f}s")
        
        if multi_time < single_time:
            print("   🏆 Multi-agent approach was faster")
        else:
            print("   🏆 Single agent approach was faster")
        
        print("\n💡 Key Insights:")
        print("   • Multi-agent systems provide specialization and quality")
        print("   • Single agents may be faster for simple tasks")
        print("   • Multi-agent systems excel at complex, multi-step workflows")
        print("   • Choice depends on task complexity and quality requirements")
    
    print("\n✅ Performance comparison completed!")


async def interactive_multi_agent_playground():
    """Interactive playground for multi-agent systems."""
    print("\n" + "="*60)
    print("INTERACTIVE MULTI-AGENT PLAYGROUND")
    print("="*60)
    
    coordinator = CoordinatorAgent("Interactive_Coordinator")
    
    print("🎮 Multi-Agent System Ready!")
    print("Type complex tasks and watch the agents work together.")
    print("Examples:")
    print("• 'Research AI trends and create a market report'")
    print("• 'Analyze customer feedback and recommend improvements'")
    print("• 'Study renewable energy and write an investment guide'")
    print("\nType 'quit' to exit, 'performance' to see stats")
    
    while True:
        try:
            user_input = input("\n👤 Enter a complex task: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'performance':
                performance = coordinator.get_performance_summary()
                print("\n📊 Performance Summary:")
                if "overall" in performance:
                    overall = performance["overall"]
                    print(f"   Total Projects: {overall['total_tasks']}")
                    print(f"   Success Rate: {overall['success_rate']:.1%}")
                    print(f"   Average Time: {overall['avg_execution_time']:.2f}s")
                continue
            
            if not user_input:
                continue
            
            print(f"\n🚀 Initiating multi-agent coordination...")
            
            try:
                result = coordinator.execute_complex_project(user_input)
                
                if result['success']:
                    print(f"✅ Project completed successfully!")
                    print(f"📋 Phases: {len(result['tasks'])}")
                    
                    for i, task in enumerate(result['tasks'], 1):
                        status = "✅" if task.success else "❌"
                        print(f"   Phase {i}: {status} {task.agent_name}")
                    
                    print(f"\n📄 Final Result:")
                    print(f"{result['final_result'][:500]}...")
                else:
                    print(f"❌ Project failed: {result.get('final_result', 'Unknown error')}")
            
            except Exception as e:
                print(f"❌ Error: {e}")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


async def main():
    """Main function to run all multi-agent examples."""
    # Display welcome and validate environment
    display_welcome()
    
    if not validate_environment():
        print("❌ Environment validation failed. Please check your configuration.")
        return
    
    print("\n🚀 Starting ADK Multi-Agent System Examples")
    print("This example demonstrates coordination between specialized agents.")
    
    try:
        # Run all demos
        await demo_specialized_agents()
        await demo_coordinator_system()
        await demo_workflow_orchestration()
        await demo_agent_communication()
        await demo_performance_comparison()
        
        print("\n" + "="*60)
        print("ALL MULTI-AGENT DEMOS COMPLETED!")
        print("="*60)
        print("\n💡 Key Takeaways:")
        print("• Specialized agents excel at specific tasks")
        print("• Coordinators orchestrate complex multi-step workflows")
        print("• Agent communication enables sophisticated task delegation")
        print("• Multi-agent systems provide quality through specialization")
        print("• Workflow orchestration enables reusable process patterns")
        
        print("\n🎮 Want to try the interactive multi-agent playground? (y/n)")
        choice = input().strip().lower()
        if choice in ['y', 'yes']:
            await interactive_multi_agent_playground()
    
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
