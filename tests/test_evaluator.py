import pytest
from unittest.mock import patch, MagicMock
import torch

import sys 
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from spart_prompt.evaluator import PromptEvaluator

class TestPromptEvaluator:
    """Tests for the PromptEvaluator class"""
    
    @pytest.fixture
    def evaluator(self, mock_llm):
        """Fixture to provide a PromptEvaluator instance with mocked dependencies"""
        with patch('spart.evaluator.SentenceTransformer') as mock_model:
            mock_model_instance = MagicMock()
            mock_model.return_value = mock_model_instance
            evaluator = PromptEvaluator(mock_llm)
            evaluator.model = mock_model_instance
            return evaluator
    
    def test_generate_prompt_output(self, evaluator, mock_llm):
        """Test generating output using the LLM"""
        # Set up the mock LLM response
        expected_prompt = "\n            test_prompt\n            {input_data} = test_input\n        "
        mock_llm.set_response(expected_prompt, "Generated output")
        
        # Call the method
        result = evaluator._generate_prompt_output("test_input", "test_prompt")
        
        assert result == "Generated output"
    
    def test_compute_semantic_similarity(self, evaluator):
        """Test computing semantic similarity between outputs"""
        # Set up mocks
        mock_tensor1 = torch.tensor([1.0, 2.0, 3.0])
        mock_tensor2 = torch.tensor([3.0, 2.0, 1.0])
        evaluator.model.encode.side_effect = [mock_tensor1, mock_tensor2]
        
        with patch('spart.evaluator.util.pytorch_cos_sim') as mock_cos_sim:
            mock_cos_sim.return_value = torch.tensor([[0.714]])
            
            # Call the method
            result = evaluator._compute_semantic_similarity("desired output", "actual output")
            
            # Check method calls
            assert evaluator.model.encode.call_count == 2
            mock_cos_sim.assert_called_once()
            
            # Check result
            assert round(result, 3) == 0.714
    
    def test_compute_syntactic_similarity(self, evaluator):
        """Test computing syntactic similarity using ROUGE-L"""
        with patch('spart.evaluator.rouge_scorer.RougeScorer') as mock_rouge:
            # Configure mock
            mock_rouge_instance = MagicMock()
            mock_rouge_score = {'rougeL': MagicMock(fmeasure=0.85)}
            mock_rouge_instance.score.return_value = mock_rouge_score
            mock_rouge.return_value = mock_rouge_instance
            
            # Call the method
            result = evaluator._compute_syntactic_similarity("desired output", "actual output")
            
            # Check method calls
            mock_rouge.assert_called_once_with(['rougeL'], use_stemmer=True)
            mock_rouge_instance.score.assert_called_once_with("desired output", "actual output")
            
            # Check result
            assert result == 0.85
    
    def test_evaluate_similarity(self, evaluator):
        """Test evaluating similarity across multiple input-output pairs"""
        # Set up test data
        input_column = ["input1", "input2"]
        output_column = ["output1", "output2"]
        generated_prompt = "test prompt"
        
        # Mock the single pair evaluation
        with patch.object(evaluator, '_evaluate_single_pair') as mock_evaluate:
            mock_evaluate.side_effect = [
                (0.9, 0.8, "generated1"),
                (0.7, 0.6, "generated2")
            ]
            
            # Call the method
            semantic_sim, syntactic_sim, outputs = evaluator.evaluate_similarity(
                input_column, output_column, generated_prompt
            )
            
            # Check calls
            assert mock_evaluate.call_count == 2
            
            # Check results
            assert semantic_sim == 0.8  # (0.9 + 0.7) / 2
            assert syntactic_sim == 0.7  # (0.8 + 0.6) / 2
            assert outputs == ["generated1", "generated2"]
    
    def test_evaluate_similarity_disabled_metrics(self, evaluator):
        """Test evaluation with disabled similarity metrics"""
        # Set up test data
        input_column = ["input1"]
        output_column = ["output1"]
        generated_prompt = "test prompt"
        
        # Mock the single pair evaluation
        with patch.object(evaluator, '_evaluate_single_pair') as mock_evaluate:
            mock_evaluate.return_value = (-1, -1, "generated1")
            
            # Call the method with disabled metrics
            semantic_sim, syntactic_sim, outputs = evaluator.evaluate_similarity(
                input_column, output_column, generated_prompt,
                use_semantic_similarity=False, 
                use_syntactic_similarity=False
            )
            
            # Check results
            assert semantic_sim == -1
            assert syntactic_sim == -1
            assert outputs == ["generated1"]
            
            # Verify disabled metrics were passed to _evaluate_single_pair
            mock_evaluate.assert_called_once_with(
                "input1", "output1", "test prompt", False, False
            )