# 🤖 AI Agent with Long-term Memory

An intelligent AI agent with OpenAI integration, long-term memory management using Vector Database (FAISS), and RAG (Retrieval-Augmented Generation) pipeline.

## ✨ Features

### Core Features
- ✅ **OpenAI Integration**: Uses GPT-4 for intelligent responses
- ✅ **Long-term Memory**: FAISS-based vector database for knowledge storage
- ✅ **RAG Pipeline**: Extract → Judge → Embed → Store → Retrieve
- ✅ **Automatic Knowledge Extraction**: Automatically extracts knowledge from conversations
- ✅ **Duplicate Detection**: Prevents storing duplicate or similar knowledge
- ✅ **Noise Filtering**: Filters out low-confidence or irrelevant information
- ✅ **Configurable Personality**: Customize agent name, age, gender, language, personality, and speaking style
- ✅ **Tool Integration**: Extensible tool system (calculator, knowledge base, etc.)
- ✅ **Gradio UI**: User-friendly web interface

### Architecture

```
User Input
    ↓
Knowledge Extraction (LLM)
    ↓
Judgment (Should store?)
    ↓
Embedding (OpenAI)
    ↓
Duplicate Detection (FAISS)
    ↓
Storage (FAISS Vector DB)
    ↓
Retrieval (RAG for queries)
    ↓
Response Generation (GPT-4)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repo-url>
cd ai-agent

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 2. Configuration

Edit `config.yaml` to customize your agent:

```yaml
agent:
  name: "Luna"           # Agent name
  age: 25               # Agent age
  gender: "female"      # Agent gender
  language: "vi"        # Language (vi, en, ja, zh)
  personality: "friendly"  # Personality type
  speaking_style: "natural"  # Speaking style

openai:
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4"
  temperature: 0.7
  max_tokens: 2048
```

### 3. Run the Agent

#### Gradio UI (Recommended)
```bash
python main.py --mode ui
```

#### CLI Mode
```bash
python main.py --mode cli
```

## 📚 Usage Examples

### Basic Chat with Knowledge Extraction

```python
from src.agent import AIAgent

agent = AIAgent("config.yaml")

# Chat - automatically extracts and stores knowledge
result = agent.process_message(
    "Python là một ngôn ngữ lập trình mạnh mẽ"
)

print(result["response"])
print(f"Stored {len(result['knowledge_extraction']['stored'])} entries")
```

### RAG Retrieval

```python
# Add knowledge
agent.add_knowledge("Việt Nam là quốc gia ở Đông Nam Á")
agent.add_knowledge("Thủ đô của Việt Nam là Hà Nội")

# Query with RAG
result = agent.process_message("Việt Nam ở đâu?")

# Retrieved context is automatically used in response
print(result["rag_context"])
```

### Update Agent Configuration

```python
# Change agent personality
agent.update_config(
    name="Sakura",
    personality="professional",
    language="en"
)

# Get current config
config = agent.get_config()
print(config)
```

### Knowledge Management

```python
# Add manual knowledge
success, entry_id = agent.add_knowledge(
    "Machine Learning là nhánh của AI",
    source="manual",
    confidence=0.95
)

# Get statistics
stats = agent.get_knowledge_stats()
print(f"Total entries: {stats['total_entries']}")

# Clear knowledge
agent.clear_knowledge()
```

## 🏗️ Project Structure

```
ai-agent/
├── config.yaml              # Configuration file
├── requirements.txt         # Dependencies
├── main.py                 # Entry point
├── example_usage.py        # Usage examples
├── README.md              # This file
└── src/
    ├── __init__.py
    ├── config.py           # Configuration management
    ├── agent.py            # Main AI Agent
    ├── embeddings.py       # Embedding generation
    ├── vector_db.py        # FAISS Vector Database
    ├── knowledge_extractor.py  # Knowledge extraction & judgment
    ├── rag_pipeline.py     # RAG pipeline
    ├── tools.py            # Tool implementations
    └── ui.py               # Gradio UI
```

## 🔧 Components

### 1. **Agent** (`agent.py`)
Main orchestrator that coordinates all components:
- Manages conversation history
- Processes messages through RAG pipeline
- Extracts knowledge from responses
- Handles configuration

### 2. **Vector Database** (`vector_db.py`)
FAISS-based storage for knowledge:
- Stores embeddings and metadata
- Supports similarity search
- Duplicate detection
- Persistent storage

### 3. **Knowledge Extractor** (`knowledge_extractor.py`)
Intelligent knowledge extraction:
- Extracts knowledge from text using LLM
- Judges if knowledge should be stored
- Evaluates confidence scores
- Filters noise

### 4. **RAG Pipeline** (`rag_pipeline.py`)
Complete RAG workflow:
- Extract → Judge → Embed → Store → Retrieve
- Builds context for responses
- Manages knowledge lifecycle

### 5. **Embeddings** (`embeddings.py`)
Text embedding management:
- Uses OpenAI embedding API
- Caches embeddings
- Computes similarity scores

### 6. **Tools** (`tools.py`)
Extensible tool system:
- Calculator tool
- Knowledge base query tool
- Easy to add new tools

### 7. **UI** (`ui.py`)
Gradio-based web interface:
- Chat interface
- Configuration management
- Knowledge management
- Tool execution

## 📊 RAG Pipeline Details

### Extract Phase
- LLM analyzes input text
- Identifies knowledge entries
- Extracts key information

### Judge Phase
- Evaluates knowledge quality
- Checks confidence scores
- Determines if worth storing

### Embed Phase
- Converts text to vectors
- Uses OpenAI embeddings
- Caches for efficiency

### Store Phase
- Adds to FAISS index
- Stores metadata
- Persists to disk

### Retrieve Phase
- Searches similar entries
- Ranks by relevance
- Builds context for responses

## 🎯 Configuration Options

### Agent Configuration
- `name`: Agent's name
- `age`: Agent's age
- `gender`: Agent's gender (male, female, other)
- `language`: Primary language (vi, en, ja, zh)
- `personality`: Personality type (friendly, professional, casual, humorous)
- `speaking_style`: Speaking style (natural, formal, casual, poetic)

### Knowledge Configuration
- `max_entries`: Maximum knowledge entries (default: 10000)
- `similarity_threshold`: Threshold for duplicate detection (default: 0.85)
- `min_confidence`: Minimum confidence to store (default: 0.6)
- `retention_days`: How long to keep entries (default: 365)

### RAG Configuration
- `top_k`: Number of entries to retrieve (default: 5)
- `chunk_size`: Text chunk size (default: 512)
- `chunk_overlap`: Overlap between chunks (default: 50)

## 🛠️ Extending the Agent

### Add Custom Tool

```python
from src.tools import Tool

class MyCustomTool(Tool):
    def execute(self, **kwargs):
        # Implement your tool logic
        return {"result": "..."}
    
    def get_description(self):
        return "My custom tool description"

# Register tool
agent.rag_pipeline.tool_manager.register_tool("my_tool", MyCustomTool())
```

### Custom Knowledge Extraction

```python
# Override knowledge extraction logic
from src.knowledge_extractor import KnowledgeExtractor

class CustomExtractor(KnowledgeExtractor):
    def extract_from_text(self, text, source="chat"):
        # Custom extraction logic
        pass
```

## 📈 Performance Tips

1. **Embedding Caching**: Embeddings are cached to reduce API calls
2. **Batch Processing**: Process multiple texts together
3. **Index Optimization**: Use appropriate FAISS index type
4. **Duplicate Detection**: Prevents redundant storage
5. **Confidence Filtering**: Only stores high-confidence knowledge

## 🔒 Security Considerations

1. **API Key Management**: Use environment variables, never hardcode
2. **Data Privacy**: Knowledge is stored locally
3. **Access Control**: Implement authentication for production
4. **Input Validation**: Sanitize user inputs

## 📝 Logging

Logs are saved to `logs/agent.log`:

```python
import logging
logger = logging.getLogger(__name__)
logger.info("Message")
logger.error("Error")
```

## [object Object]

### Issue: "OPENAI_API_KEY not set"
**Solution**: Create `.env` file and add your API key

### Issue: FAISS index errors
**Solution**: Delete `data/vector_db/` directory and restart

### Issue: Slow embedding generation
**Solution**: Check internet connection and OpenAI API status

## 📚 References

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Gradio Documentation](https://gradio.app)
- [RAG Papers](https://arxiv.org/search/?query=retrieval+augmented+generation)

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For issues and questions, please open an issue on GitHub.

---

**Made with ❤️ for AI enthusiasts**

