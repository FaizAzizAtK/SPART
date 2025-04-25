import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from spart_prompt.llm_connector import LLMConnector
from spart_prompt.evaluator import PromptEvaluator
from spart_prompt.optimizer import PromptOptimizer
from spart_prompt.recommender import PromptRecommender


class MockLLMConnector(LLMConnector):
    """Mock LLM connector to simulate API calls with predefined responses."""
    
    def __init__(self):
        super().__init__("test-model", token_limit=1000)
        self.call_count = 0
        self.responses = {
            0: "Recommended prompt for test",  # First call (recommendation)
            1: "Optimized prompt for test"     # Second call (optimization)
        }

    def __call__(self, prompt):
        # Return different responses for recommendation and optimization
        response = self.responses.get(self.call_count, self.responses[1])  # Default to optimized
        self.call_count += 1
        print(f"CALL {self.call_count}: {response}")  # Added call count printing
        return response


class TestIntegration:
    """Integration tests for SPART components"""

    @pytest.fixture
    def mock_llm_connector(self):
        """Fixture providing a mock LLM connector"""
        return MockLLMConnector()

    @pytest.fixture
    def example_data(self):
        """Fixture providing example data for integration tests"""
        return pd.DataFrame({
            0: ["Raw input 1", "Raw input 2", "Raw input 3", "Raw input 4"],
            1: ["Processed output 1", "Processed output 2", "Processed output 3", "Processed output 4"]
        })

    @patch('sentence_transformers.SentenceTransformer')
    @patch('spart_prompt.evaluator.util.pytorch_cos_sim')
    @patch('rouge_score.rouge_scorer.RougeScorer')
    def test_evaluator_with_connector(self, mock_rouge, mock_cos_sim, mock_transformer, mock_llm_connector):
        """Test PromptEvaluator with LLMConnector integration"""
        
        # Set up mocks
        mock_transformer_instance = MagicMock()
        mock_transformer.return_value = mock_transformer_instance
        mock_transformer_instance.encode.return_value = MagicMock()
        
        mock_cos_sim.return_value = MagicMock()
        mock_cos_sim.return_value.item.return_value = 0.85
        
        mock_rouge_instance = MagicMock()
        mock_rouge.return_value = mock_rouge_instance
        mock_rouge_instance.score.return_value = {'rougeL': MagicMock(fmeasure=0.75)}

        # Create evaluator with mock connector
        evaluator = PromptEvaluator(llm=mock_llm_connector)

        # Test evaluation
        test_prompt = "Test prompt"
        test_inputs = ["Test input 1", "Test input 2"]
        test_expected = ["Expected output 1", "Expected output 2"]

        semantic_sim, syntactic_sim, outputs = evaluator.evaluate_similarity(
            test_inputs, test_expected, test_prompt
        )

        # Verify results
        assert isinstance(semantic_sim, float)
        assert isinstance(syntactic_sim, float)
        assert isinstance(outputs, list)

    def test_recommender_with_connector(self, mock_llm_connector, example_data):
        """Test PromptRecommender with LLMConnector integration"""
        
        recommender = PromptRecommender(llm=mock_llm_connector)

        # Patch evaluate_similarity to return fixed values
        with patch.object(recommender.evaluator, 'evaluate_similarity',
                         return_value=(0.85, 0.8, ["Output 1", "Output 2"])):

            # Run recommendation
            recommendation = recommender.recommend(
                examples=example_data,
                num_examples=2
            )

            # Verify recommendation
            assert recommendation is not None
            assert "recommended_prompt" in recommendation
            assert "Recommended prompt for test" in recommendation["recommended_prompt"]

    def test_optimizer_with_connector(self, mock_llm_connector, example_data):
        """Test PromptOptimizer with LLMConnector integration"""
        
        optimizer = PromptOptimizer(llm=mock_llm_connector)

        # bump the mock so its first call returns the "optimized" response
        mock_llm_connector.call_count = 1

        # Simulate different similarity results for each optimization attempt
        with patch.object(optimizer.evaluator, 'evaluate_similarity',
                            side_effect=[
                                (0.7, 0.7, ["Original output"]),
                                (0.9, 0.9, ["Optimized output"])
                            ]):

            test_prompt = "Original test prompt"

            optimized_result = optimizer.optimize_prompt(
                meta_prompt=test_prompt,
                examples=example_data,
                num_examples=2
            )

            # Verify the optimization result
            assert optimized_result is not None
            assert "optimized_prompt" in optimized_result
            assert "Optimized prompt for test" in optimized_result["optimized_prompt"]


    def test_full_pipeline_integration(self, mock_llm_connector, example_data):
        """Test full pipeline integration: Recommender -> Optimizer"""
        
        recommender = PromptRecommender(llm=mock_llm_connector)

        # Patch the optimizer's evaluator to always succeed, so optimization returns our mock LLM's "Optimized prompt for test"
        recommender.optimizer.evaluator.evaluate_similarity = MagicMock(
            return_value=(0.9, 0.9, ["Optimized output"])
        )

        # Mock the *first* evaluation (recommendation) to fall below threshold and trigger optimization
        with patch.object(recommender.evaluator, 'evaluate_similarity',
                        return_value=(0.7, 0.7, ["Original output"])):

            # Full pipeline: Recommend -> Optimize
            recommendation = recommender.recommend(
                examples=example_data,
                num_examples=2,
                similarity_threshold=0.8,  # Below threshold, triggers optimization
                max_iterations=1
            )

            # Verify the pipeline results
            assert recommendation is not None
            assert "recommended_prompt" in recommendation
            assert "Recommended prompt for test" in recommendation["recommended_prompt"]

            # Now optimization *should* have run and returned our optimized text
            assert "optimization" in recommendation
            assert "optimized_prompt" in recommendation["optimization"]
            assert "Optimized prompt for test" in recommendation["optimization"]["optimized_prompt"]


