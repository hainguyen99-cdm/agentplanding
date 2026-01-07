"""Unit tests for AI Agent"""
import os
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TestConfig(unittest.TestCase):
    """Test configuration management"""
    
    def test_config_loading(self):
        """Test loading configuration from YAML"""
        from config import Config
        
        config = Config.from_yaml("config.yaml")
        
        self.assertEqual(config.agent.name, "Luna")
        self.assertEqual(config.agent.language, "vi")
        self.assertIsNotNone(config.openai.api_key)
    
    def test_config_defaults(self):
        """Test configuration defaults"""
        from config import Config
        
        config = Config()
        
        self.assertEqual(config.agent.name, "Luna")
        self.assertEqual(config.openai.model, "gpt-4")
        self.assertEqual(config.knowledge.max_entries, 10000)


class TestEmbeddings(unittest.TestCase):
    """Test embedding generation"""
    
    @patch('embedai.OpenAI')
    def test_embed_text(self, mock_openai):
        """Test text embedding"""
        from embeddings import EmbeddingManager
        import numpy as np
        
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        
        manager = EmbeddingManager()
        manager.client.embeddings.create = Mock(return_value=mock_response)
        
        embedding = manager.embed_text("test text")
        
        self.assertEqual(len(embedding), 1536)
        self.assertIsInstance(embedding, np.ndarray)
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        from embeddings import EmbeddingManager
        import numpy as np
        
        manager = EmbeddingManager()
        
        # Test identical vectors
        vec1 = np.array([1, 0, 0], dtype=np.float32)
        vec2 = np.array([1, 0, 0], dtype=np.float32)
        
        similarity = manager.cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, 1.0, places=5)
        
        # Test orthogonal vectors
        vec3 = np.array([0, 1, 0], dtype=np.float32)
        similarity = manager.cosine_similarity(vec1, vec3)
        self.assertAlmostEqual(similarity, 0.0, places=5)


class TestVectorDB(unittest.TestCase):
    """Test Vector Database"""
    
    def setUp(self):
        """Setup test database"""
        from vector_db import VectorDB
        import shutil
        
        # Use test database
        test_db_path = "./test_data/vector_db"
        if Path(test_db_path).exists():
            shutil.rmtree(test_db_path)
        
        # Mock config
        with patch('vector_db.get_config') as mock_config:
            config = Mock()
            config.vector_db.db_path = test_db_path
            config.vector_db.dimension = 1536
            config.vector_db.index_type = "IVF64,Flat"
            config.rag.top_k = 5
            config.knowledge.similarity_threshold = 0.85
            config.knowledge.min_confidence = 0.6
            config.knowledge.max_entries = 10000
            
            mock_config.return_value = config
            
            self.db = VectorDB()
    
    def tearDown(self):
        """Cleanup test database"""
        import shutil
        test_db_path = "./test_data"
        if Path(test_db_path).exists():
            shutil.rmtree(test_db_path)
    
    @patch('vector_db.EmbeddingManager')
    def test_add_entry(self, mock_embedding_manager):
        """Test adding knowledge entry"""
        import numpy as np
        
        # Mock embedding
        mock_embedding_manager.return_value.embed_text.return_value = np.zeros(1536, dtype=np.float32)
        
        success, entry_id = self.db.add_entry("Test knowledge")
        
        self.assertTrue(success)
        self.assertIn("entry_", entry_id)
    
    @patch('vector_db.EmbeddingManager')
    def test_duplicate_detection(self, mock_embedding_manager):
        """Test duplicate detection"""
        import numpy as np
        
        # Mock embedding
        mock_embedding_manager.return_value.embed_text.return_value = np.zeros(1536, dtype=np.float32)
        mock_embedding_manager.return_value.cosine_similarity.return_value = 0.9
        
        # Add first entry
        self.db.add_entry("Test knowledge")
        
        # Try to add duplicate
        success, result = self.db.add_entry("Test knowledge similar")
        
        # Should fail due to duplicate
        self.assertFalse(success)


class TestKnowledgeExtractor(unittest.TestCase):
    """Test knowledge extraction"""
    
    @patch('knowledge_extractor.OpenAI')
    def test_extract_from_text(self, mock_openai):
        """Test knowledge extraction from text"""
        from knowledge_extractor import KnowledgeExtractor
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='{"extractions": [{"content": "Test knowledge", "confidence": 0.9, "should_store": true, "reason": "Test"}]}'))]
        
        extractor = KnowledgeExtractor()
        extractor.client.chat.completions.create = Mock(return_value=mock_response)
        
        extractions = extractor.extract_from_text("Test text")
        
        self.assertGreater(len(extractions), 0)
        self.assertEqual(extractions[0].content, "Test knowledge")
    
    @patch('knowledge_extractor.OpenAI')
    def test_judge_knowledge(self, mock_openai):
        """Test knowledge judgment"""
        from knowledge_extractor import KnowledgeExtractor
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='{"should_store": true, "confidence": 0.9, "reason": "Valid knowledge"}'))]
        
        extractor = KnowledgeExtractor()
        extractor.client.chat.completions.create = Mock(return_value=mock_response)
        
        should_store, confidence, reason = extractor.judge_knowledge("Test knowledge")
        
        self.assertTrue(should_store)
        self.assertGreater(confidence, 0)


class TestRAGPipeline(unittest.TestCase):
    """Test RAG Pipeline"""
    
    @patch('rag_pipeline.VectorDB')
    @patch('rag_pipeline.KnowledgeExtractor')
    def test_process_new_information(self, mock_extractor, mock_db):
        """Test processing new information"""
        from rag_pipeline import RAGPipeline
        
        # Mock components
        mock_extractor.return_value.extract_from_text.return_value = []
        mock_db.return_value.add_entry.return_value = (True, "entry_1")
        
        pipeline = RAGPipeline()
        result = pipeline.process_new_information("Test information")
        
        self.assertIn("extracted", result)
        self.assertIn("stored", result)
    
    @patch('rag_pipeline.VectorDB')
    @patch('rag_pipeline.KnowledgeExtractor')
    def test_retrieve_context(self, mock_extractor, mock_db):
        """Test context retrieval"""
        from rag_pipeline import RAGPipeline
        
        # Mock components
        mock_db.return_value.find_similar.return_value = [
            ("Knowledge 1", 0.9),
            ("Knowledge 2", 0.8)
        ]
        
        pipeline = RAGPipeline()
        context = pipeline.retrieve_context("Test query")
        
        self.assertEqual(context.query, "Test query")
        self.assertEqual(len(context.retrieved_entries), 2)


class TestTools(unittest.TestCase):
    """Test tool system"""
    
    def test_calculator_tool(self):
        """Test calculator tool"""
        from tools import CalculatorTool
        
        tool = CalculatorTool()
        result = tool.execute(expression="2 + 2")
        
        self.assertEqual(result["result"], 4)
        self.assertEqual(result["status"], "success")
    
    def test_tool_manager(self):
        """Test tool manager"""
        from tools import ToolManager
        
        manager = ToolManager()
        
        # List tools
        tools = manager.list_tools()
        self.assertIn("calculator", tools)
        
        # Execute tool
        result = manager.execute_tool("calculator", expression="3 * 3")
        self.assertEqual(result["result"], 9)


class TestAgent(unittest.TestCase):
    """Test main Agent"""
    
    @patch('agent.OpenAI')
    @patch('agent.RAGPipeline')
    def test_agent_initialization(self, mock_rag, mock_openai):
        """Test agent initialization"""
        from agent import AIAgent
        
        agent = AIAgent("config.yaml")
        
        self.assertIsNotNone(agent.config)
        self.assertIsNotNone(agent.client)
        self.assertIsNotNone(agent.rag_pipeline)
    
    @patch('agent.OpenAI')
    @patch('agent.RAGPipeline')
    def test_update_config(self, mock_rag, mock_openai):
        """Test updating agent configuration"""
        from agent import AIAgent
        
        agent = AIAgent("config.yaml")
        original_name = agent.config.agent.name
        
        agent.update_config(name="TestAgent")
        
        self.assertEqual(agent.config.agent.name, "TestAgent")
        self.assertNotEqual(agent.config.agent.name, original_name)


def run_tests():
    """Run all tests"""
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Warning: OPENAI_API_KEY not set. Some tests will be skipped.")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddings))
    suite.addTests(loader.loadTestsFromTestCase(TestVectorDB))
    suite.addTests(loader.loadTestsFromTestCase(TestKnowledgeExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestRAGPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestTools))
    suite.addTests(loader.loadTestsFromTestCase(TestAgent))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())

