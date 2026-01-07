# Quick Reference Guide

## Installation & Setup (5 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key
export OPENAI_API_KEY="sk-..."

# 3. Run UI
python main.py --mode ui
```

## Basic Usage

### Chat with Agent
```python
from src.agent import AIAgent

agent = AIAgent("config.yaml")
result = agent.process_message("Hello!")
print(result["response"])
```

### Add Knowledge
```python
success, entry_id = agent.add_knowledge("Python is a programming language")
print(f"Stored: {success}, ID: {entry_id}")
```

### Get Statistics
```python
stats = agent.get_knowledge_stats()
print(f"Total entries: {stats['total_entries']}")
```

### Update Configuration
```python
agent.update_config(
    name="MyAgent",
    personality="professional",
    language="en"
)
```

## Configuration Quick Reference

### Agent Personality
```yaml
agent:
  name: "Luna"                    # Agent name
  age: 25                        # Age
  gender: "female"               # male/female/other
  language: "vi"                 # vi/en/ja/zh
  personality: "friendly"        # friendly/professional/casual/humorous
  speaking_style: "natural"      # natural/formal/casual/poetic
```

### Knowledge Settings
```yaml
knowledge:
  max_entries: 10000            # Max knowledge entries
  similarity_threshold: 0.85    # Duplicate detection threshold
  min_confidence: 0.6           # Min confidence to store
  retention_days: 365           # How long to keep
```

### RAG Settings
```yaml
rag:
  top_k: 5                      # Retrieve top 5 entries
  chunk_size: 512               # Text chunk size
  chunk_overlap: 50             # Overlap between chunks
```

## RAG Pipeline Stages

```
Extract     → Judge      → Embed      → Store      → Retrieve
Extract     Judge if     Convert to   Add to       Search
knowledge   worth        vectors      database     similar
from text   storing                                entries
```

## File Structure

```
ai-agent/
├── config.yaml              # Configuration
├── requirements.txt         # Dependencies
├── main.py                 # Entry point
├── example_usage.py        # Examples
├── test_agent.py          # Tests
├── README.md              # Full documentation
├── ARCHITECTURE.md        # System design
├── IMPLEMENTATION_GUIDE.md # How-to guide
├── QUICK_REFERENCE.md     # This file
└── src/
    ├── agent.py           # Main agent
    ├── vector_db.py       # FAISS database
    ├── rag_pipeline.py    # RAG workflow
    ├── knowledge_extractor.py  # Knowledge extraction
    ├── embeddings.py      # Text embeddings
    ├── tools.py           # Tool system
    ├── ui.py              # Gradio UI
    └── config.py          # Configuration
```

## Common Commands

```bash
# Run Gradio UI
python main.py --mode ui

# Run CLI mode
python main.py --mode cli

# Run examples
python example_usage.py

# Run tests
python test_agent.py

# View logs
tail -f logs/agent.log

# Clear knowledge
# (In Python: agent.clear_knowledge())
```

## API Quick Reference

### Agent Methods
```python
# Process message with full pipeline
result = agent.process_message(text, extract_knowledge=True)

# Add knowledge manually
success, id = agent.add_knowledge(content, source="manual", confidence=0.95)

# Get statistics
stats = agent.get_knowledge_stats()

# Update configuration
agent.update_config(name="NewName", personality="professional")

# Get current configuration
config = agent.get_config()

# Clear knowledge base
agent.clear_knowledge()

# Clear conversation history
agent.clear_conversation()
```

### RAG Pipeline Methods
```python
# Process new information (extract → judge → embed → store)
result = pipeline.process_new_information(text, source="chat")

# Retrieve context for query
context = pipeline.retrieve_context(query, top_k=5)

# Build prompt with context
prompt = pipeline.build_rag_prompt(query, context)

# Get knowledge statistics
stats = pipeline.get_knowledge_stats()

# Clear all knowledge
pipeline.clear_knowledge()
```

### Vector DB Methods
```python
# Add entry
success, id = vector_db.add_entry(content, source="chat", confidence=0.9)

# Find similar entries
results = vector_db.find_similar(text, top_k=5)

# Check for duplicates
is_dup, dup_id = vector_db.is_duplicate(text)

# Get entry by ID
entry = vector_db.get_entry(entry_id)

# Delete entry
success = vector_db.delete_entry(entry_id)

# Get statistics
stats = vector_db.get_stats()
```

### Tool Manager Methods
```python
# Register tool
tool_manager.register_tool("my_tool", MyTool())

# Execute tool
result = tool_manager.execute_tool("calculator", expression="2+2")

# List available tools
tools = tool_manager.list_tools()

# Get tool descriptions
descriptions = tool_manager.get_tools_description()
```

## Response Structure

### process_message() Response
```python
{
    "user_message": "...",
    "knowledge_extraction": {
        "input": "...",
        "source": "user_chat",
        "extracted": ["...", "..."],
        "stored": [{"entry_id": "...", "content": "...", "confidence": 0.9}],
        "rejected": [{"content": "...", "reason": "...", "confidence": 0.5}],
        "errors": []
    },
    "rag_context": {
        "retrieved_count": 3,
        "entries": [
            {"content": "...", "similarity": 0.95},
            {"content": "...", "similarity": 0.87},
            {"content": "...", "similarity": 0.82}
        ],
        "context_text": "..."
    },
    "response": "Agent's response text",
    "response_knowledge": {...},
    "error": None
}
```

## Configuration Examples

### Friendly Vietnamese Agent
```yaml
agent:
  name: "Luna"
  age: 25
  gender: "female"
  language: "vi"
  personality: "friendly"
  speaking_style: "natural"
```

### Professional English Agent
```yaml
agent:
  name: "Alex"
  age: 30
  gender: "male"
  language: "en"
  personality: "professional"
  speaking_style: "formal"
```

### Casual Japanese Agent
```yaml
agent:
  name: "Sakura"
  age: 22
  gender: "female"
  language: "ja"
  personality: "casual"
  speaking_style: "casual"
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OPENAI_API_KEY not set | `export OPENAI_API_KEY="sk-..."` |
| FAISS index error | `rm -rf data/vector_db/` |
| Slow responses | Check network, reduce `top_k` |
| High memory usage | `agent.clear_knowledge()` |
| Port already in use | `python main.py --mode ui --port 7861` |

## Performance Tips

1. **Cache embeddings** - Avoid re-embedding same text
2. **Batch operations** - Process multiple items together
3. **Reduce top_k** - Retrieve fewer entries per query
4. **Increase similarity_threshold** - Stricter duplicate detection
5. **Periodic cleanup** - Clear old knowledge entries

## Security Checklist

- [ ] Use environment variables for API keys
- [ ] Don't commit .env file
- [ ] Validate user inputs
- [ ] Enable HTTPS in production
- [ ] Implement rate limiting
- [ ] Regular backups
- [ ] Monitor API usage
- [ ] Encrypt sensitive data

## Useful Links

- [OpenAI API](https://platform.openai.com/docs)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [Gradio Docs](https://gradio.app)
- [Project README](README.md)
- [Architecture Guide](ARCHITECTURE.md)
- [Implementation Guide](IMPLEMENTATION_GUIDE.md)

## Example Workflows

### Workflow 1: Simple Chat
```python
agent = AIAgent("config.yaml")
result = agent.process_message("Hello!")
print(result["response"])
```

### Workflow 2: Knowledge Building
```python
# Add knowledge
agent.add_knowledge("Fact 1")
agent.add_knowledge("Fact 2")

# Query with context
result = agent.process_message("Related question?")
# Agent automatically uses stored facts
```

### Workflow 3: Configuration Change
```python
# Change personality
agent.update_config(personality="professional")

# Chat with new personality
result = agent.process_message("Hello!")
```

### Workflow 4: Tool Usage
```python
# Use calculator
result = tool_manager.execute_tool("calculator", expression="2+2")

# Use knowledge base
result = tool_manager.execute_tool("knowledge_base", query="Python")
```

## Key Concepts

**RAG**: Retrieval-Augmented Generation
- Retrieve relevant knowledge
- Augment prompt with knowledge
- Generate response using context

**Vector DB**: Stores text as vectors
- Fast similarity search
- Duplicate detection
- Long-term memory

**Embedding**: Convert text to numbers
- 1536 dimensions (OpenAI)
- Captures semantic meaning
- Used for similarity search

**Knowledge Entry**: Stored fact
- Content (text)
- Embedding (vector)
- Metadata (source, confidence, etc.)

**Duplicate Detection**: Prevent storing similar facts
- Compare similarity scores
- Threshold-based filtering
- Maintains knowledge quality

---

**Last Updated**: 2024
**Version**: 1.0

