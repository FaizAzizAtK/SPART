import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

import sys 
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from spart_prompt.recommender import PromptRecommender
from spart_prompt.llm_connector import LLMConnector

class MockLLM(LLMConnector):
    """Mock LLM for testing recommender"""
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

class TestPromptRecommender:
    """Tests for the PromptRecommender class"""
    
    @pytest.fixture
    def mock_llm(self):
        """Fixture to provide a mock LLM"""
        with patch('transformers.AutoTokenizer.from_pretrained'):
            return MockLLM()
    
    @pytest.fixture
    def recommender(self, mock_llm):
        """Fixture to provide a PromptRecommender instance with mocked dependencies"""
        with patch('spart_prompt.recommender.PromptEvaluator') as mock_evaluator_class, \
            patch('spart_prompt.recommender.PromptOptimizer') as mock_optimizer_class:
            
            mock_evaluator = MagicMock()
            mock_evaluator_class.return_value = mock_evaluator
            
            mock_optimizer = MagicMock()
            mock_optimizer_class.return_value = mock_optimizer
            
            recommender = PromptRecommender(llm=mock_llm, auto_confirm=True)
            recommender.evaluator = mock_evaluator
            recommender.optimizer = mock_optimizer
            return recommender

    
    @pytest.fixture
    def example_data(self):
        """Fixture providing example data for tests"""
        return pd.DataFrame({
            0: ["Input 1", "Input 2", "Input 3", "Input 4"],
            1: ["Output 1", "Output 2", "Output 3", "Output 4"]
        })
    
    def test_extract_prompt_from_xml(self, recommender):
        """Test extracting prompt from XML tag"""
        xml_text = "Some text <prompt>Extracted prompt content</prompt> more text"
        result = recommender.extract_prompt_from_xml(xml_text)
        assert result == "Extracted prompt content"
    
    def test_confirm_with_user_auto(self, recommender):
        """Test auto-confirmation of token count"""
        assert recommender.confirm_with_user(500) is True
    
    def test_confirm_optimization_with_user_auto(self, recommender):
        """Test auto-confirmation of optimization"""
        assert recommender.confirm_optimization_with_user() is True
    
    @patch('builtins.input')
    def test_confirm_with_user_manual(self, mock_input):    
        """Test manual confirmation of token count"""
        # Set up recommender with auto_confirm=False
        with patch('transformers.AutoTokenizer.from_pretrained'):
            recommender = PromptRecommender(llm=MagicMock(), auto_confirm=False)
        
        # Test user confirms
        mock_input.return_value = "y"
        assert recommender.confirm_with_user(500) is True
        
        # Test user declines
        mock_input.return_value = "n"
        assert recommender.confirm_with_user(500) is False

    
    def test_recommend_success(self, recommender, mock_llm, example_data):
        """Test successful prompt recommendation"""
        # Mock LLM to return a prompt
        mock_llm.set_response(
            "test prompt",  # This argument is ignored in our mock
            "<prompt>Recommended system prompt</prompt>"
        )
        
        # Mock token counting to bypass confirmation
        recommender.llm._count_tokens = MagicMock(return_value=100)
        
        # Mock the evaluator to return scores above threshold
        recommender.evaluator.evaluate_similarity.return_value = (
            0.85, 0.9, ["generated_output1", "generated_output2"]
        )
        
        # Call recommend
        result = recommender.recommend(
            examples=example_data,
            num_examples=2,
            context="Test context",
            similarity_threshold=0.8
        )
        
        # Check result
        assert result["recommended_prompt"] == "Recommended system prompt"
        assert result["semantic_similarity"] == 0.85
        assert result["syntactic_similarity"] == 0.9
        assert result["prompt_outputs"] == ["generated_output1", "generated_output2"]
        assert result["recommendation"] == "Recommended"
        
        # Verify evaluator was called once
        recommender.evaluator.evaluate_similarity.assert_called_once()
        
        # Verify optimizer was not called
        recommender.optimizer.optimize_prompt.assert_not_called()

    
    def test_recommend_with_optimization(self, recommender, mock_llm, example_data):
        """Test recommendation with optimization"""
        # Mock LLM to return a prompt
        mock_llm.set_response(
            "test prompt",  # This argument is ignored in our mock
            "<prompt>Initial system prompt</prompt>"
        )
        
        # Mock token counting to bypass confirmation
        recommender.llm._count_tokens = MagicMock(return_value=100)
        
        # Mock the evaluator to return scores below threshold
        recommender.evaluator.evaluate_similarity.return_value = (
            0.6, 0.7, ["generated_output1", "generated_output2"]
        )
        
        # Mock the optimizer
        recommender.optimizer.optimize_prompt.return_value = {
            "optimized_prompt": "Optimized system prompt",
            "semantic_similarity": 0.85,
            "syntactic_similarity": 0.9,
            "evaluation_details": ["optimized_output1", "optimized_output2"]
        }
        
        # Call recommend
        result = recommender.recommend(
            examples=example_data,
            num_examples=2,
            context="Test context",
            similarity_threshold=0.8
        )
        
        # Check result
        assert result["recommended_prompt"] == "Initial system prompt"
        assert result["semantic_similarity"] == 0.6
        assert result["syntactic_similarity"] == 0.7
        assert result["recommendation"] == "Optimize"
        assert "optimization" in result
        assert result["optimization"]["optimized_prompt"] == "Optimized system prompt"
        
        # Verify evaluator was called once
        recommender.evaluator.evaluate_similarity.assert_called_once()
        
        # Verify optimizer was called once
        recommender.optimizer.optimize_prompt.assert_called_once()
