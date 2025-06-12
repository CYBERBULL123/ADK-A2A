"""
Example 1: Basic Agent Creation and Testing

This example demonstrates the fundamentals of creating and using ADK agents.
You'll learn how to:
1. Create a simple conversational agent
2. Add tools to extend agent capabilities
3. Test agent responses and behavior
4. Handle different types of user input
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config import config
from utils import logger, display_welcome, validate_environment
from agents.basic import SimpleAgent, SearchAgent, ToolAgent, StatefulAgent
from tools import weather_api_tool, calculator_tool


async def demo_simple_agent():
    """Demonstrate basic agent creation and interaction."""
    print("\n" + "="*60)
    print("DEMO 1: Simple Conversational Agent")
    print("="*60)
    
    # Create a simple agent
    agent = SimpleAgent("MyFirstAgent")
    
    # Test basic conversation
    test_prompts = [
        "Hello! What can you help me with?",
        "Tell me about artificial intelligence",
        "What are the benefits of using AI agents?",
        "Can you help me solve problems?"
    ]
    
    for prompt in test_prompts:
        print(f"\n👤 User: {prompt}")
        response = agent.chat(prompt)
        print(f"🤖 Agent: {response}")
    
    print(f"\n✅ Simple agent demo completed!")


async def demo_tool_agent():
    """Demonstrate agent with custom tools."""
    print("\n" + "="*60)
    print("DEMO 2: Agent with Custom Tools")
    print("="*60)
    
    # Create a tool-enabled agent
    agent = ToolAgent("ToolDemoAgent")
    
    # Test tool usage
    tool_prompts = [
        "What's the weather like in Tokyo?",
        "Calculate 15 * 23 + 47",
        "What's 25% of 400?",
        "Check the weather in London and tell me if I need an umbrella"
    ]
    
    for prompt in tool_prompts:
        print(f"\n👤 User: {prompt}")
        response = agent.process_request(prompt)
        print(f"🤖 Agent: {response}")
    
    print(f"\n✅ Tool agent demo completed!")


async def demo_stateful_agent():
    """Demonstrate agent with memory and context."""
    print("\n" + "="*60)
    print("DEMO 3: Stateful Agent with Memory")
    print("="*60)
    
    # Create a stateful agent
    agent = StatefulAgent("MemoryDemoAgent")
    
    # Test conversation with context
    conversation_flow = [
        "Hi, my name is Alice and I'm a software developer",
        "What did I tell you my name was?",
        "What's my profession?",
        "Can you recommend some programming languages for AI development?",
        "Based on what you know about me, what projects might I be interested in?"
    ]
    
    for message in conversation_flow:
        print(f"\n👤 Alice: {message}")
        response = agent.chat_with_memory(message)
        print(f"🤖 Agent: {response}")
    
    # Show conversation summary
    summary = agent.get_conversation_summary()
    print(f"\n📊 Conversation Summary: {summary}")
    
    print(f"\n✅ Stateful agent demo completed!")


async def demo_search_agent():
    """Demonstrate agent with search capabilities."""
    print("\n" + "="*60)
    print("DEMO 4: Search-Enabled Agent")
    print("="*60)
    
    if not config.google_api_key:
        print("⚠️  Google API key not configured. Skipping search demo.")
        return
    
    # Create a search agent
    agent = SearchAgent("SearchDemoAgent")
    
    # Test search functionality
    search_prompts = [
        "What are the latest developments in AI?",
        "Current weather trends globally",
        "Recent advances in renewable energy",
        "What's happening in the tech industry this week?"
    ]
    
    for prompt in search_prompts:
        print(f"\n👤 User: {prompt}")
        try:
            response = agent.search_and_respond(prompt)
            print(f"🤖 Agent: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n✅ Search agent demo completed!")


async def compare_agent_responses():
    """Compare how different agent types handle the same prompt."""
    print("\n" + "="*60)
    print("DEMO 5: Agent Response Comparison")
    print("="*60)
    
    # Create different agent types
    simple_agent = SimpleAgent("Simple")
    tool_agent = ToolAgent("Tool")
    stateful_agent = StatefulAgent("Stateful")
    
    # Test prompt that could benefit from tools
    test_prompt = "I need to plan a trip to Paris. What's the weather like there and how much would a hotel cost for 3 nights?"
    
    print(f"👤 User: {test_prompt}\n")
    
    # Simple agent response
    print("🤖 Simple Agent:")
    try:
        response = simple_agent.chat(test_prompt)
        print(f"   {response}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # Tool agent response
    print("🛠️  Tool Agent:")
    try:
        response = tool_agent.process_request(test_prompt)
        print(f"   {response}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # Stateful agent response
    print("🧠 Stateful Agent:")
    try:
        response = stateful_agent.chat_with_memory(test_prompt)
        print(f"   {response}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    print("💡 Notice how different agent types provide different capabilities!")
    print("✅ Agent comparison completed!")


async def interactive_agent_playground():
    """Interactive playground for testing agents."""
    print("\n" + "="*60)
    print("INTERACTIVE AGENT PLAYGROUND")
    print("="*60)
    print("Choose an agent type to interact with:")
    print("1. Simple Agent")
    print("2. Tool Agent") 
    print("3. Stateful Agent")
    print("4. Search Agent (if API key available)")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "5":
                print("👋 Goodbye!")
                break
            
            # Create selected agent
            agent = None
            agent_name = ""
            
            if choice == "1":
                agent = SimpleAgent("Interactive_Simple")
                agent_name = "Simple Agent"
            elif choice == "2":
                agent = ToolAgent("Interactive_Tool")
                agent_name = "Tool Agent"
            elif choice == "3":
                agent = StatefulAgent("Interactive_Stateful")
                agent_name = "Stateful Agent"
            elif choice == "4":
                if config.google_api_key:
                    agent = SearchAgent("Interactive_Search")
                    agent_name = "Search Agent"
                else:
                    print("❌ Google API key required for search agent")
                    continue
            else:
                print("❌ Invalid choice. Please try again.")
                continue
            
            print(f"\n🤖 {agent_name} is ready! Type 'quit' to return to menu.")
            
            # Interactive chat loop
            while True:
                user_input = input(f"\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'back']:
                    break
                
                if not user_input:
                    continue
                
                try:
                    # Route to appropriate method based on agent type
                    if hasattr(agent, 'chat_with_memory'):
                        response = agent.chat_with_memory(user_input)
                    elif hasattr(agent, 'process_request'):
                        response = agent.process_request(user_input)
                    elif hasattr(agent, 'search_and_respond'):
                        response = agent.search_and_respond(user_input)
                    elif hasattr(agent, 'chat'):
                        response = agent.chat(user_input)
                    else:
                        response = "Error: Agent method not found"
                    
                    print(f"🤖 {agent_name}: {response}")
                
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


async def main():
    """Main function to run all basic agent examples."""
    # Display welcome and validate environment
    display_welcome()
    
    if not validate_environment():
        print("❌ Environment validation failed. Please check your configuration.")
        return
    
    print("\n🚀 Starting ADK Basic Agent Examples")
    print("This example will demonstrate different types of agents and their capabilities.")
    
    try:
        # Run all demos
        await demo_simple_agent()
        await demo_tool_agent()
        await demo_stateful_agent()
        await demo_search_agent()
        await compare_agent_responses()
        
        print("\n" + "="*60)
        print("ALL DEMOS COMPLETED!")
        print("="*60)
        print("\n💡 Key Takeaways:")
        print("• Simple agents provide basic conversational capabilities")
        print("• Tool agents can perform specific tasks using custom functions")
        print("• Stateful agents remember conversation context")
        print("• Search agents can access real-time information")
        print("• Different agent types have different strengths and use cases")
        
        print("\n🎮 Want to try the interactive playground? (y/n)")
        choice = input().strip().lower()
        if choice in ['y', 'yes']:
            await interactive_agent_playground()
    
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
