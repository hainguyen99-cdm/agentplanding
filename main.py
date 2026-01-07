"""Main entry point for AI Agent"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from agent import AIAgent
from ui import launch_ui


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Agent with Long-term Memory")
    parser.add_argument(
        "--mode",
        choices=["cli", "ui"],
        default="ui",
        help="Run mode: cli for command line, ui for Gradio interface"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    if args.mode == "ui":
        print("🚀 Launching Gradio UI...")
        launch_ui(args.config)
    else:
        print("💬 Starting CLI mode...")
        cli_mode(args.config)


def cli_mode(config_path: str):
    """CLI mode for testing"""
    agent = AIAgent(config_path)
    
    print(f"\n🤖 Agent: {agent.config.agent.name}")
    print(f"Language: {agent.config.agent.language}")
    print(f"Personality: {agent.config.agent.personality}")
    print("\nType 'quit' to exit, 'stats' for knowledge stats, 'clear' to clear knowledge\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            
            if user_input.lower() == "stats":
                stats = agent.get_knowledge_stats()
                print(f"\n📊 Knowledge Stats:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                print()
                continue
            
            if user_input.lower() == "clear":
                agent.clear_knowledge()
                print("✅ Knowledge cleared\n")
                continue
            
            # Process message
            result = agent.process_message(user_input)
            
            if result["error"]:
                print(f"❌ Error: {result['error']}\n")
            else:
                print(f"\n{agent.config.agent.name}: {result['response']}\n")
                
                # Show knowledge extraction info
                if result["knowledge_extraction"]:
                    ke = result["knowledge_extraction"]
                    if ke["stored"]:
                        print(f"✅ Stored {len(ke['stored'])} knowledge entries")
                    if ke["rejected"]:
                        print(f"⚠️ Rejected {len(ke['rejected'])} entries")
                
                # Show RAG context
                if result["rag_context"]:
                    ctx = result["rag_context"]
                    if ctx["retrieved_count"] > 0:
                        print(f"\n📚 Retrieved {ctx['retrieved_count']} knowledge entries:")
                        for entry in ctx["entries"]:
                            print(f"  - (relevance: {entry['similarity']:.2f}) {entry['content'][:80]}...")
                print()
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    main()

