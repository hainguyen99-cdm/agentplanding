"""Example usage of AI Agent"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from agent import AIAgent

# Load environment variables
load_dotenv()


def example_basic_chat():
    """Example 1: Basic chat with knowledge extraction"""
    print("=" * 60)
    print("Example 1: Basic Chat with Knowledge Extraction")
    print("=" * 60)
    
    agent = AIAgent("config.yaml")
    
    # Chat with knowledge extraction
    message = "Python là một ngôn ngữ lập trình mạnh mẽ được tạo bởi Guido van Rossum. Nó được sử dụng rộng rãi trong AI, data science, và web development."
    
    result = agent.process_message(message)
    
    print(f"\nUser: {message}")
    print(f"\nAgent: {result['response']}")
    
    if result["knowledge_extraction"]:
        ke = result["knowledge_extraction"]
        print(f"\n✅ Extracted: {len(ke['extracted'])} entries")
        print(f"✅ Stored: {len(ke['stored'])} entries")
        print(f"⚠️ Rejected: {len(ke['rejected'])} entries")
    
    print(f"\n📚 Knowledge base stats: {agent.get_knowledge_stats()}")


def example_rag_retrieval():
    """Example 2: RAG retrieval"""
    print("\n" + "=" * 60)
    print("Example 2: RAG Retrieval")
    print("=" * 60)
    
    agent = AIAgent("config.yaml")
    
    # Add some knowledge
    print("\nAdding knowledge entries...")
    agent.add_knowledge("Việt Nam là một quốc gia ở Đông Nam Á")
    agent.add_knowledge("Thủ đô của Việt Nam là Hà Nội")
    agent.add_knowledge("Tiếng Việt là ngôn ngữ chính thức của Việt Nam")
    
    # Query
    query = "Việt Nam ở đâu?"
    result = agent.process_message(query)
    
    print(f"\nQuery: {query}")
    print(f"\nAgent: {result['response']}")
    
    if result["rag_context"]:
        print(f"\n📚 Retrieved {result['rag_context']['retrieved_count']} entries:")
        for entry in result["rag_context"]["entries"]:
            print(f"  - {entry['content']}")


def example_config_update():
    """Example 3: Update agent configuration"""
    print("\n" + "=" * 60)
    print("Example 3: Update Agent Configuration")
    print("=" * 60)
    
    agent = AIAgent("config.yaml")
    
    print(f"\nOriginal config:")
    print(f"  Name: {agent.config.agent.name}")
    print(f"  Personality: {agent.config.agent.personality}")
    
    # Update configuration
    agent.update_config(
        name="Sakura",
        personality="professional",
        speaking_style="formal"
    )
    
    print(f"\nUpdated config:")
    print(f"  Name: {agent.config.agent.name}")
    print(f"  Personality: {agent.config.agent.personality}")
    print(f"  Speaking style: {agent.config.agent.speaking_style}")
    
    # Test with new config
    result = agent.process_message("Xin chào!")
    print(f"\nAgent response: {result['response']}")


def example_duplicate_detection():
    """Example 4: Duplicate detection"""
    print("\n" + "=" * 60)
    print("Example 4: Duplicate Detection")
    print("=" * 60)
    
    agent = AIAgent("config.yaml")
    
    # Add original knowledge
    print("\nAdding original knowledge...")
    success1, id1 = agent.add_knowledge("Machine Learning là một nhánh của AI")
    print(f"  Result: {success1}, ID: {id1}")
    
    # Try to add similar knowledge (should be rejected)
    print("\nAdding similar knowledge (should be rejected)...")
    success2, id2 = agent.add_knowledge("Machine Learning là một phần của trí tuệ nhân tạo")
    print(f"  Result: {success2}, Message: {id2}")
    
    # Add different knowledge (should succeed)
    print("\nAdding different knowledge...")
    success3, id3 = agent.add_knowledge("Deep Learning sử dụng neural networks")
    print(f"  Result: {success3}, ID: {id3}")


def example_tool_usage():
    """Example 5: Tool usage"""
    print("\n" + "=" * 60)
    print("Example 5: Tool Usage")
    print("=" * 60)
    
    agent = AIAgent("config.yaml")
    
    from tools import ToolManager
    tool_manager = ToolManager(agent.rag_pipeline)
    
    # Use calculator tool
    print("\nUsing calculator tool...")
    result = tool_manager.execute_tool("calculator", expression="2 + 2 * 3")
    print(f"  Expression: 2 + 2 * 3")
    print(f"  Result: {result['result']}")
    
    # Use knowledge base tool
    print("\nUsing knowledge base tool...")
    agent.add_knowledge("Python được phát hành lần đầu năm 1991")
    result = tool_manager.execute_tool("knowledge_base", query="Python")
    print(f"  Query: Python")
    print(f"  Results: {len(result['results'])} entries found")


def example_multi_language():
    """Example 6: Multi-language support"""
    print("\n" + "=" * 60)
    print("Example 6: Multi-language Support")
    print("=" * 60)
    
    # Vietnamese
    print("\nVietnamese Agent:")
    agent_vi = AIAgent("config.yaml")
    result = agent_vi.process_message("Xin chào!")
    print(f"  Response: {result['response'][:100]}...")
    
    # Update to English
    print("\nEnglish Agent:")
    agent_vi.update_config(language="en")
    result = agent_vi.process_message("Hello!")
    print(f"  Response: {result['response'][:100]}...")


if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set")
        print("Please set your OpenAI API key in .env file")
        sys.exit(1)
    
    # Run examples
    try:
        example_basic_chat()
        example_rag_retrieval()
        example_config_update()
        example_duplicate_detection()
        example_tool_usage()
        example_multi_language()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

