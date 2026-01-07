# Implementation Guide - AI Agent with Long-term Memory

## [object Object] Completion Summary

Bạn đã có một hệ thống AI Agent hoàn chỉnh với các tính năng:

### ✅ Hoàn thành
1. **Agent Core** - Tích hợp OpenAI GPT-4
2. **Long-term Memory** - FAISS Vector Database
3. **RAG Pipeline** - Extract → Judge → Embed → Store → Retrieve
4. **Knowledge Management** - Tự động trích xuất và lưu trữ
5. **Duplicate Detection** - Chống trùng lặp thông minh
6. **Configurable Personality** - Tùy chỉnh tên, tuổi, giới tính, ngôn ngữ, tính cách
7. **Tool System** - Mở rộng được với các công cụ tùy chỉnh
8. **Gradio UI** - Giao diện web thân thiện

## 🚀 Getting Started

### Step 1: Setup Environment

```bash
# Clone hoặc tạo project
mkdir ai-agent && cd ai-agent

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Step 2: Configure Agent

Edit `config.yaml`:

```yaml
agent:
  name: "Luna"              # Tên agent
  age: 25                  # Tuổi
  gender: "female"         # Giới tính
  language: "vi"           # Ngôn ngữ (vi, en, ja, zh)
  personality: "friendly"  # Tính cách
  speaking_style: "natural"  # Phong cách nói

openai:
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4"
  temperature: 0.7
  max_tokens: 2048

knowledge:
  max_entries: 10000
  similarity_threshold: 0.85  # Ngưỡng phát hiện trùng lặp
  min_confidence: 0.6         # Độ tin cậy tối thiểu
```

### Step 3: Run Agent

#### Option A: Gradio UI (Recommended)
```bash
python main.py --mode ui
# Truy cập http://localhost:7860
```

#### Option B: CLI Mode
```bash
python main.py --mode cli
```

#### Option C: Python Script
```python
from src.agent import AIAgent

agent = AIAgent("config.yaml")
result = agent.process_message("Xin chào!")
print(result["response"])
```

## 📚 Usage Examples

### Example 1: Basic Chat

```python
from src.agent import AIAgent

agent = AIAgent("config.yaml")

# Chat - tự động trích xuất kiến thức
result = agent.process_message(
    "Python là ngôn ngữ lập trình được tạo bởi Guido van Rossum"
)

print(f"Response: {result['response']}")
print(f"Stored: {len(result['knowledge_extraction']['stored'])} entries")
```

### Example 2: RAG Retrieval

```python
# Thêm kiến thức
agent.add_knowledge("Việt Nam nằm ở Đông Nam Á")
agent.add_knowledge("Thủ đô của Việt Nam là Hà Nội")

# Query - tự động lấy context từ knowledge base
result = agent.process_message("Việt Nam ở đâu?")

# Kiến thức liên quan được tự động sử dụng trong response
print(result["rag_context"])  # Xem kiến thức được lấy
```

### Example 3: Update Configuration

```python
# Thay đổi tính cách agent
agent.update_config(
    name="Sakura",
    personality="professional",
    language="en"
)

# Xem cấu hình hiện tại
config = agent.get_config()
print(config)
```

### Example 4: Knowledge Management

```python
# Thêm kiến thức thủ công
success, entry_id = agent.add_knowledge(
    "Machine Learning là nhánh của AI",
    source="manual",
    confidence=0.95
)

# Xem thống kê
stats = agent.get_knowledge_stats()
print(f"Total: {stats['total_entries']}")
print(f"Active: {stats['active_entries']}")

# Xóa tất cả kiến thức
agent.clear_knowledge()
```

### Example 5: Tool Usage

```python
from src.tools import ToolManager

tool_manager = ToolManager(agent.rag_pipeline)

# Dùng calculator
result = tool_manager.execute_tool("calculator", expression="2 + 2 * 3")
print(result["result"])  # 8

# Query knowledge base
result = tool_manager.execute_tool("knowledge_base", query="Python")
print(result["results"])
```

## 🔧 Advanced Configuration

### Knowledge Management Tuning

```yaml
knowledge:
  # Số lượng kiến thức tối đa
  max_entries: 10000
  
  # Ngưỡng phát hiện trùng lặp (0-1)
  # Cao = khắt khe hơn, ít lưu trùng
  # Thấp = dễ lưu trùng hơn
  similarity_threshold: 0.85
  
  # Độ tin cậy tối thiểu để lưu (0-1)
  # Cao = chỉ lưu kiến thức chắc chắn
  # Thấp = lưu nhiều kiến thức hơn
  min_confidence: 0.6
  
  # Thời gian lưu (ngày)
  retention_days: 365
```

### RAG Pipeline Tuning

```yaml
rag:
  # Số kiến thức lấy cho mỗi query
  top_k: 5
  
  # Kích thước chunk khi xử lý text
  chunk_size: 512
  
  # Overlap giữa các chunk
  chunk_overlap: 50
```

### OpenAI Tuning

```yaml
openai:
  model: "gpt-4"  # hoặc "gpt-3.5-turbo"
  
  # Sáng tạo (0-2): 0=xác định, 1=cân bằng, 2=sáng tạo
  temperature: 0.7
  
  # Số token tối đa
  max_tokens: 2048
  
  # Diversity (0-1)
  top_p: 0.9
```

## 🛠️ Extending the System

### Add Custom Tool

```python
from src.tools import Tool

class WeatherTool(Tool):
    def execute(self, city: str, **kwargs):
        # Implement weather API call
        return {
            "city": city,
            "temperature": 25,
            "condition": "sunny"
        }
    
    def get_description(self):
        return "Get weather information for a city"

# Register tool
from src.agent import AIAgent
agent = AIAgent("config.yaml")
agent.rag_pipeline.tool_manager.register_tool("weather", WeatherTool())

# Use tool
result = agent.rag_pipeline.tool_manager.execute_tool("weather", city="Hanoi")
```

### Custom Knowledge Extraction

```python
from src.knowledge_extractor import KnowledgeExtractor

class CustomExtractor(KnowledgeExtractor):
    def extract_from_text(self, text, source="chat"):
        # Custom extraction logic
        # Có thể sử dụng regex, NLP, v.v.
        extractions = []
        # ... implementation ...
        return extractions

# Use custom extractor
agent.rag_pipeline.extractor = CustomExtractor()
```

### Custom Vector DB

```python
# Thay thế FAISS bằng Milvus, Pinecone, v.v.
# Implement interface tương tự VectorDB

class MilvusDB:
    def add_entry(self, content, source, confidence, metadata):
        # Milvus implementation
        pass
    
    def find_similar(self, text, top_k):
        # Milvus implementation
        pass

# Use custom DB
agent.rag_pipeline.vector_db = MilvusDB()
```

## 📊 Monitoring & Debugging

### View Logs

```bash
# Real-time logs
tail -f logs/agent.log

# Search logs
grep "ERROR" logs/agent.log
grep "knowledge" logs/agent.log
```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

agent = AIAgent("config.yaml")
result = agent.process_message("Test")
```

### Performance Monitoring

```python
import time

# Measure response time
start = time.time()
result = agent.process_message("Test")
elapsed = time.time() - start

print(f"Response time: {elapsed:.2f}s")
print(f"Knowledge stored: {len(result['knowledge_extraction']['stored'])}")
print(f"Context retrieved: {result['rag_context']['retrieved_count']}")
```

## [object Object] Deployment

### Security Checklist

- [ ] Use environment variables for API keys
- [ ] Enable HTTPS for web interface
- [ ] Implement authentication/authorization
- [ ] Encrypt sensitive data
- [ ] Set up rate limiting
- [ ] Monitor API usage
- [ ] Regular backups of knowledge base
- [ ] Input validation and sanitization

### Performance Optimization

```python
# 1. Use embedding cache
embedding_manager.clear_cache()  # Periodic cleanup

# 2. Batch process multiple texts
embeddings = embedding_manager.embed_texts(texts)

# 3. Use appropriate FAISS index
# For 1M vectors: "IVF4096,Flat"
# For 100M vectors: "IVF65536,Flat"

# 4. Implement pagination for large results
results = vector_db.find_similar(query, top_k=100)
```

### Scaling Strategies

```
Single Agent (Current)
    ↓
Multiple Agents (Same Server)
    ↓
Distributed Agents (Multiple Servers)
    ↓
Multi-region Deployment
```

## 🧪 Testing

### Run Tests

```bash
python test_agent.py
```

### Write Custom Tests

```python
import unittest
from src.agent import AIAgent

class TestMyAgent(unittest.TestCase):
    def setUp(self):
        self.agent = AIAgent("config.yaml")
    
    def test_chat(self):
        result = self.agent.process_message("Test")
        self.assertIsNotNone(result["response"])
    
    def test_knowledge_extraction(self):
        result = self.agent.process_message("New knowledge")
        self.assertGreater(len(result["knowledge_extraction"]["extracted"]), 0)

if __name__ == "__main__":
    unittest.main()
```

## 📈 Performance Metrics

### Typical Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Embed text | 0.5-1s | API call |
| Search similar | 10-50ms | FAISS search |
| Judge knowledge | 1-2s | LLM call |
| Generate response | 2-5s | GPT-4 call |
| Full pipeline | 5-10s | All steps |

### Optimization Tips

1. **Cache embeddings** - Avoid re-embedding same text
2. **Batch operations** - Process multiple items together
3. **Use GPU** - For FAISS index operations
4. **Async processing** - Non-blocking operations
5. **Connection pooling** - Reuse API connections

## [object Object]

### Issue: "OPENAI_API_KEY not set"
```bash
# Solution: Set environment variable
export OPENAI_API_KEY="your-key-here"
# or create .env file
echo "OPENAI_API_KEY=your-key-here" > .env
```

### Issue: FAISS index errors
```bash
# Solution: Reset database
rm -rf data/vector_db/
# Restart agent
```

### Issue: Slow responses
```python
# Check performance
stats = agent.get_knowledge_stats()
print(f"Index size: {stats['index_size']}")

# If too large, implement cleanup
agent.clear_knowledge()
```

### Issue: Memory usage high
```python
# Clear embedding cache
agent.rag_pipeline.embedding_manager.clear_cache()

# Reduce max_entries in config
# Implement periodic cleanup
```

## 📚 Additional Resources

- [OpenAI Documentation](https://platform.openai.com/docs)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Gradio Documentation](https://gradio.app)
- [RAG Papers](https://arxiv.org/search/?query=retrieval+augmented+generation)
- [Vector Database Comparison](https://github.com/erikbern/ann-benchmarks)

## 🎓 Learning Path

1. **Beginner**: Run UI, chat with agent
2. **Intermediate**: Modify config, add custom tools
3. **Advanced**: Implement custom extractors, use different vector DB
4. **Expert**: Deploy to production, scale to multiple agents

## 📞 Support

For issues and questions:
1. Check logs: `logs/agent.log`
2. Review examples: `example_usage.py`
3. Read documentation: `ARCHITECTURE.md`
4. Check tests: `test_agent.py`

---

**Happy coding! 🚀**

