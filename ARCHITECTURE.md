# AI Agent Architecture Documentation

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface (Gradio)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    AI Agent (Main Orchestrator)          │   │
│  │  - Manages conversation history                         │   │
│  │  - Coordinates all components                           │   │
│  │  - Handles configuration                                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                    │
│         ┌────────────────────┼────────────────────┐              │
│         ▼                    ▼                    ▼              │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────┐    │
│  │  RAG Pipeline  │  │ Knowledge       │  │ Tool Manager │    │
│  │                │  │ Extractor       │  │              │    │
│  │ • Extract      │  │                 │  │ • Calculator │    │
│  │ • Judge        │  │ • Extract text  │  │ • Knowledge  │    │
│  │ • Embed        │  │ • Judge quality │  │   Base       │    │
│  │ • Store        │  │ • Score conf.   │  │ • Custom     │    │
│  │ • Retrieve     │  │                 │  │              │    │
│  └────────────────┘  └─────────────────┘  └──────────────┘    │
│         │                    │                    │              │
│         └────────────────────┼────────────────────┘              │
│                              ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Vector Database (FAISS)                     │   │
│  │  - Stores embeddings                                    │   │
│  │  - Metadata management                                  │   │
│  │  - Similarity search                                    │   │
│  │  - Duplicate detection                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         Persistent Storage (JSON + FAISS Index)         │   │
│  │  - ./data/vector_db/faiss_index.bin                     │   │
│  │  - ./data/vector_db/metadata.json                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   OpenAI API     │
                    │  • GPT-4         │
                    │  • Embeddings    │
                    └──────────────────┘
```

## Component Details

### 1. Agent (agent.py)

**Responsibility**: Main orchestrator and coordinator

**Key Methods**:
- `process_message()`: Full pipeline for user messages
- `_generate_response()`: Generate response using OpenAI
- `add_knowledge()`: Manually add knowledge
- `update_config()`: Update agent configuration
- `get_knowledge_stats()`: Get database statistics

**Flow**:
```
User Message
    ↓
Extract Knowledge
    ↓
Retrieve Context (RAG)
    ↓
Generate Response
    ↓
Extract Knowledge from Response
    ↓
Return Result
```

### 2. RAG Pipeline (rag_pipeline.py)

**Responsibility**: Manages the complete RAG workflow

**Pipeline Stages**:

#### Extract
- Uses LLM to identify knowledge entries in text
- Extracts key information and facts
- Returns list of ExtractedKnowledge objects

#### Judge
- Evaluates if knowledge should be stored
- Checks confidence scores
- Filters out low-quality information

#### Embed
- Converts text to vectors using OpenAI embeddings
- Caches embeddings for efficiency
- Returns 1536-dimensional vectors

#### Store
- Adds embeddings to FAISS index
- Stores metadata (source, confidence, etc.)
- Persists to disk

#### Retrieve
- Searches similar entries for queries
- Ranks by relevance (cosine similarity)
- Builds context for responses

**Key Methods**:
- `process_new_information()`: Full extract→judge→embed→store pipeline
- `retrieve_context()`: Retrieve relevant knowledge
- `build_rag_prompt()`: Build prompt with context
- `process_query_with_rag()`: Full query processing

### 3. Vector Database (vector_db.py)

**Responsibility**: Manages knowledge storage and retrieval

**Features**:
- FAISS index for efficient similarity search
- Metadata storage (JSON)
- Duplicate detection
- Entry management (add, delete, retrieve)

**Key Classes**:
- `KnowledgeEntry`: Represents single knowledge entry
- `VectorDB`: Main database class

**Key Methods**:
- `add_entry()`: Add new knowledge
- `find_similar()`: Search similar entries
- `is_duplicate()`: Check for duplicates
- `get_entry()`: Retrieve entry by ID
- `delete_entry()`: Mark entry as deleted
- `get_stats()`: Database statistics

**Storage Structure**:
```
data/vector_db/
├── faiss_index.bin      # FAISS index (binary)
└── metadata.json        # Metadata (JSON)
    {
      "metadata": [
        {
          "content": "...",
          "entry_id": "entry_0_1234567890",
          "source": "chat",
          "confidence": 0.95,
          "metadata": {...},
          "created_at": "2024-01-01T12:00:00"
        }
      ],
      "entry_map": {
        "entry_0_1234567890": 0
      }
    }
```

### 4. Knowledge Extractor (knowledge_extractor.py)

**Responsibility**: Extract and judge knowledge quality

**Key Classes**:
- `ExtractedKnowledge`: Dataclass for extracted knowledge
- `KnowledgeExtractor`: Main extractor class

**Key Methods**:
- `extract_from_text()`: Extract knowledge from text
- `judge_knowledge()`: Evaluate if knowledge should be stored
- `extract_from_tool_output()`: Extract from tool results

**Extraction Process**:
1. Build prompt with extraction instructions
2. Call LLM to identify knowledge entries
3. Parse JSON response
4. Return ExtractedKnowledge objects

**Judgment Process**:
1. Build judgment prompt
2. Call LLM to evaluate
3. Return (should_store, confidence, reason)

### 5. Embeddings (embeddings.py)

**Responsibility**: Text embedding generation and management

**Key Class**: `EmbeddingManager`

**Features**:
- OpenAI embedding API integration
- Embedding caching
- Cosine similarity calculation
- Batch embedding support

**Key Methods**:
- `embed_text()`: Single text embedding
- `embed_texts()`: Batch embedding
- `cosine_similarity()`: Calculate similarity
- `clear_cache()`: Clear embedding cache

**Embedding Model**: `text-embedding-3-small` (1536 dimensions)

### 6. Tools (tools.py)

**Responsibility**: Extensible tool system

**Base Class**: `Tool` (abstract)

**Built-in Tools**:
- `CalculatorTool`: Mathematical calculations
- `KnowledgeBaseTool`: Query knowledge base
- `WebSearchTool`: Web search (placeholder)

**Key Class**: `ToolManager`

**Methods**:
- `register_tool()`: Register new tool
- `execute_tool()`: Execute tool
- `list_tools()`: List available tools
- `get_tools_description()`: Get tool descriptions

**Adding Custom Tool**:
```python
class MyTool(Tool):
    def execute(self, **kwargs):
        return {"result": "..."}
    
    def get_description(self):
        return "Tool description"

manager.register_tool("my_tool", MyTool())
```

### 7. Configuration (config.py)

**Responsibility**: Configuration management

**Key Classes**:
- `AgentConfig`: Agent personality settings
- `OpenAIConfig`: OpenAI API settings
- `VectorDBConfig`: Database settings
- `KnowledgeConfig`: Knowledge management settings
- `RAGConfig`: RAG pipeline settings
- `Config`: Main configuration class

**Features**:
- YAML loading
- Environment variable substitution
- Pydantic validation
- Directory creation

### 8. UI (ui.py)

**Responsibility**: Gradio web interface

**Key Class**: `AgentUI`

**Tabs**:
1. **Chat**: Main conversation interface
2. **Configuration**: Update agent settings
3. **Knowledge Management**: Manage knowledge base
4. **Tools**: Execute tools
5. **About**: Information

**Features**:
- Real-time chat
- RAG context display
- Knowledge statistics
- Configuration management
- Tool execution

## Data Flow

### Message Processing Flow

```
User Input
    │
    ├─→ Knowledge Extraction
    │   ├─→ LLM analyzes text
    │   ├─→ Extracts knowledge entries
    │   └─→ Returns ExtractedKnowledge list
    │
    ├─→ Knowledge Judgment
    │   ├─→ Evaluate confidence
    │   ├─→ Check quality
    │   └─→ Decide to store or reject
    │
    ├─→ Embedding Generation
    │   ├─→ Convert text to vectors
    │   ├─→ Cache embeddings
    │   └─→ Return 1536-dim vectors
    │
    ├─→ Duplicate Detection
    │   ├─→ Search similar entries
    │   ├─→ Compare similarity scores
    │   └─→ Reject if duplicate
    │
    ├─→ Storage
    │   ├─→ Add to FAISS index
    │   ├─→ Store metadata
    │   └─→ Persist to disk
    │
    ├─→ RAG Retrieval
    │   ├─→ Embed user message
    │   ├─→ Search similar entries
    │   └─→ Build context
    │
    ├─→ Response Generation
    │   ├─→ Build system prompt
    │   ├─→ Include RAG context
    │   ├─→ Call GPT-4
    │   └─→ Return response
    │
    └─→ Response Knowledge Extraction
        ├─→ Extract knowledge from response
        ├─→ Judge and store
        └─→ Update knowledge base
```

## Configuration Flow

```
config.yaml
    │
    ├─→ YAML Parser
    │   └─→ Load configuration
    │
    ├─→ Environment Variable Substitution
    │   └─→ Replace ${VAR} with env vars
    │
    ├─→ Pydantic Validation
    │   └─→ Validate types and values
    │
    ├─→ Directory Creation
    │   ├─→ Create data/vector_db/
    │   └─→ Create logs/
    │
    └─→ Global Config Instance
        └─→ Available via get_config()
```

## Memory Management

### Long-term Memory (Vector DB)
- **Storage**: FAISS index + JSON metadata
- **Capacity**: Configurable (default 10,000 entries)
- **Retention**: Configurable (default 365 days)
- **Access**: O(log n) similarity search

### Short-term Memory (Conversation History)
- **Storage**: In-memory list
- **Capacity**: Last 4 exchanges (configurable)
- **Retention**: Session-based
- **Access**: O(1) append, O(n) search

### Embedding Cache
- **Storage**: In-memory dictionary
- **Capacity**: Unlimited (in practice)
- **Retention**: Session-based
- **Access**: O(1) lookup

## Duplicate Detection

**Algorithm**:
1. Embed new text
2. Search top-1 similar entry
3. Calculate similarity score
4. Compare with threshold (default 0.85)
5. If similarity >= threshold, mark as duplicate

**Similarity Calculation**:
```
similarity = cosine_similarity(vec1, vec2)
           = (vec1 · vec2) / (||vec1|| * ||vec2||)
```

## Error Handling

**Strategy**: Graceful degradation

1. **Embedding Errors**: Return zero vector
2. **LLM Errors**: Return default response
3. **Database Errors**: Log and continue
4. **Tool Errors**: Return error message

**Logging**: All errors logged to `logs/agent.log`

## Performance Characteristics

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Add entry | O(n) | O(1) |
| Search similar | O(log n) | O(k) |
| Embed text | O(1)* | O(d) |
| Judge knowledge | O(1)* | O(1) |
| Generate response | O(1)* | O(c) |

*: Depends on external API

## Scalability Considerations

1. **Vector DB**: FAISS supports millions of vectors
2. **Embeddings**: Cache prevents redundant API calls
3. **Metadata**: JSON file can be replaced with database
4. **Concurrency**: Add thread/async support for production
5. **Distributed**: Use Milvus or Pinecone for multi-node setup

## Security Considerations

1. **API Keys**: Use environment variables
2. **Input Validation**: Sanitize user inputs
3. **Output Filtering**: Filter sensitive information
4. **Access Control**: Implement authentication
5. **Data Privacy**: Encrypt sensitive data

## Future Enhancements

1. **Multi-agent**: Support multiple agents
2. **Knowledge Graph**: Build knowledge relationships
3. **Active Learning**: Learn from user feedback
4. **Semantic Search**: Improve search accuracy
5. **Distributed Storage**: Scale to multiple nodes
6. **Real-time Updates**: Stream responses
7. **Tool Chaining**: Chain multiple tools
8. **Fine-tuning**: Fine-tune models on domain data

