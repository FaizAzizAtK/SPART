import pytest
from unittest.mock import patch, MagicMock, call
import pandas as pd

import sys 
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from spart_prompt.optimizer import PromptOptimizer
from spart_prompt.llm_connector import LLMConnector

class MockLLM(LLMConnector):
    """Mock LLM for testing optimizer"""
    def __init__(self):
        super().__init__("test-model", token_limit=1000)
        self.responses = {}  # Map of prompts to responses
    
    def set_response(self, prompt, response):
        """Set a response for a given prompt"""
        self.responses[prompt] = response
        
    def __call__(self, prompt):
        """Return response for the closest matching key or a default"""
        # For simplicity, just return the first response if any
        if self.responses:
            if prompt in self.responses:
                return self.responses[prompt]
            return list(self.responses.values())[0]
        return "<prompt>Default response</prompt>"

class TestPromptOptimizer:
    """Tests for the PromptOptimizer class"""
    
    @pytest.fixture
    def mock_llm(self):
        """Fixture to provide a mock LLM"""
        with patch('transformers.AutoTokenizer.from_pretrained'):
            return MockLLM()
    
    @pytest.fixture
    def optimizer(self, mock_llm):
        """Fixture to provide a PromptOptimizer instance with mocked dependencies"""
        with patch('spart_prompt.optimizer.PromptEvaluator') as mock_evaluator_cls:
            mock_evaluator = MagicMock()
            mock_evaluator_cls.return_value = mock_evaluator
            optimizer = PromptOptimizer(llm=mock_llm, auto_confirm=True)
            optimizer.evaluator = mock_evaluator
            return optimizer
    
    @pytest.fixture
    def example_data(self):
        """Fixture providing example data for tests"""
        return pd.DataFrame({
            0: ["Input 1", "Input 2", "Input 3", "Input 4"],
            1: ["Output 1", "Output 2", "Output 3", "Output 4"]
        })
    
    def test_extract_prompt_from_xml(self, optimizer):
        """Test extracting prompt from XML tag"""
        # Test with valid XML
        xml_text = "Some text <prompt>Extracted prompt content</prompt> more text"
        result = optimizer.extract_prompt_from_xml(xml_text)
        assert result == "Extracted prompt content"
        
        # Test with invalid XML (should return original text)
        invalid_text = "No tags here"
        result = optimizer.extract_prompt_from_xml(invalid_text)
        assert result == invalid_text
    
    def test_optimize_prompt_success(self, optimizer, mock_llm, example_data):
        """Test prompt optimization with successful threshold achievement"""
        # Mock the LLM to return an optimized prompt
        mock_llm.set_response(
            "Original prompt",  # This argument is ignored in our mock
            "<prompt>Optimized system prompt</prompt>"
        )
        
        # Mock the evaluator to return high similarity scores on first attempt
        optimizer.evaluator.evaluate_similarity.side_effect = [
            # Original prompt evaluation
            (0.6, 0.7, ["original_output1", "original_output2"]),
            # Optimized prompt evaluation
            (0.9, 0.9, ["optimized_output1", "optimized_output2"])
        ]
        
        # Call optimize_prompt
        result = optimizer.optimize_prompt(
            meta_prompt="Original prompt",
            examples=example_data,
            num_examples=2,
            threshold=0.85,
            context="Context info",
            max_iterations=3
        )
        
        # Check result
        assert result["optimized_prompt"] == "Optimized system prompt"
        assert result["semantic_similarity"] == 0.9
        assert result["syntactic_similarity"] == 0.9
        assert result["evaluation_details"] == ["optimized_output1", "optimized_output2"]
        
        # Verify evaluator was called twice (once for original, once for optimized)
        assert optimizer.evaluator.evaluate_similarity.call_count == 2
    
    def test_optimize_prompt_iterations(self, optimizer, mock_llm, example_data):
        """Test prompt optimization requiring multiple iterations"""
        # Set up the LLM to return different prompts for each iteration
        optimizer.llm = MagicMock(side_effect=[
            "<prompt>Optimized prompt v1</prompt>",
            "<prompt>Optimized prompt v2</prompt>",
            "<prompt>Optimized prompt v3</prompt>"
        ])
        
        # Mock the evaluator to return gradually improving scores
        optimizer.evaluator.evaluate_similarity.side_effect = [
            # Original prompt evaluation
            (0.5, 0.6, ["original_output"]),
            # First optimization attempt
            (0.7, 0.75, ["v1_output"]),
            # Second optimization attempt
            (0.8, 0.8, ["v2_output"]),
            # Third optimization attempt
            (0.9, 0.9, ["v3_output"])
        ]
        
        # Call optimize_prompt
        result = optimizer.optimize_prompt(
            meta_prompt="Original prompt",
            examples=example_data,
            num_examples=2,
            threshold=0.85,
            max_iterations=3
        )
        
        # Check result from final iteration
        assert result["optimized_prompt"] == "Optimized prompt v3"
        assert result["semantic_similarity"] == 0.9
        assert result["syntactic_similarity"] == 0.9
        
        # Verify evaluator was called 4 times (once for original, 3 for optimized prompts)
        assert optimizer.evaluator.evaluate_similarity.call_count == 4
    
    def test_optimize_prompt_max_iterations(self, optimizer, mock_llm, example_data):
        """Test prompt optimization reaching max iterations without meeting threshold"""
        # Set up the LLM to return different prompts for each iteration
        optimizer.llm = MagicMock(side_effect=[
            "<prompt>Prompt v1</prompt>",
            "<prompt>Prompt v2</prompt>",
            "<prompt>Prompt v3</prompt>"
        ])
        
        # Mock the evaluator to return scores that never reach threshold
        optimizer.evaluator.evaluate_similarity.side_effect = [
            # Original prompt evaluation
            (0.5, 0.6, ["original_output"]),
            # First optimization attempt
            (0.6, 0.65, ["v1_output"]),
            # Second optimization attempt
            (0.7, 0.75, ["v2_output"]),
            # Third optimization attempt
            (0.8, 0.8, ["v3_output"])
        ]
        
        # Call optimize_prompt
        result = optimizer.optimize_prompt(
            meta_prompt="Original prompt",
            examples=example_data,
            num_examples=2,
            threshold=0.85,
            max_iterations=3
        )
        
        # Check result is best attempt
        assert result["optimized_prompt"] == "Prompt v3"
        assert result["semantic_similarity"] == 0.8
        assert result["syntactic_similarity"] == 0.8
        
        # Verify evaluator was called 4 times (once for original, 3 for optimized prompts)
        assert optimizer.evaluator.evaluate_similarity.call_count == 4