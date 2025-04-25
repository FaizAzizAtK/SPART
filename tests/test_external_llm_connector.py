import pytest
from unittest.mock import patch, MagicMock
from langchain.schema import AIMessage

import sys 
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from spart_prompt.external_llm_connector import ExternalLLMConnector

class TestExternalLLMConnector:
    """Tests for the ExternalLLMConnector class"""
    
    @pytest.mark.parametrize("provider,model_name,expected_class", [
        ("openai", "gpt-4", "ChatOpenAI"),
        ("huggingface", "mistralai/Mistral-7B", "HuggingFaceHub"),
        ("cohere", "command", "Cohere")
    ])
    def test_initialization(self, provider, model_name, expected_class):
        """Test initialization with different providers"""
        with patch(f'spart.external_llm_connector.{expected_class}') as mock_class:
            ExternalLLMConnector(
                provider=provider,
                model_name=model_name,
                api_key="test-key",
                temperature=0.7
            )
            mock_class.assert_called_once()
    
    def test_invalid_provider(self):
        """Test initialization with invalid provider"""
        with pytest.raises(ValueError) as excinfo:
            ExternalLLMConnector(
                provider="invalid-provider",
                model_name="test-model",
                api_key="test-key"
            )
        assert "Unsupported provider" in str(excinfo.value)
    
    @patch('spart.external_llm_connector.ChatOpenAI')
    def test_openai_call(self, mock_openai):
        """Test successful OpenAI API call"""
        # Configure mock
        mock_instance = MagicMock()
        mock_response = MagicMock(spec=AIMessage)
        mock_response.content = "Generated response"
        mock_instance.invoke.return_value = mock_response
        mock_openai.return_value = mock_instance
        
        # Create connector with mock
        connector = ExternalLLMConnector(
            provider="openai",
            model_name="gpt-4",
            api_key="test-key"
        )
        
        # Mock token counting to avoid actual tokenizer
        with patch.object(connector, '_count_tokens', return_value=50):
            result = connector("Test prompt")
            assert result == "Generated response"
    
    @patch('spart.external_llm_connector.HuggingFaceHub')
    def test_huggingface_call(self, mock_hf):
        """Test successful Hugging Face API call"""
        # Configure mock
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = "Generated response"
        mock_hf.return_value = mock_instance
        
        # Create connector with mock
        connector = ExternalLLMConnector(
            provider="huggingface",
            model_name="mistralai/Mistral-7B",
            api_key="test-key"
        )
        
        # Mock token counting to avoid actual tokenizer
        with patch.object(connector, '_count_tokens', return_value=50):
            result = connector("Test prompt")
            assert result == "Generated response"
    
    @patch('spart.external_llm_connector.Cohere')
    def test_cohere_call(self, mock_cohere):
        """Test successful Cohere API call"""
        # Configure mock
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = "Generated response"
        mock_cohere.return_value = mock_instance
        
        # Create connector with mock
        connector = ExternalLLMConnector(
            provider="cohere",
            model_name="command",
            api_key="test-key"
        )
        
        # Mock token counting to avoid actual tokenizer
        with patch.object(connector, '_count_tokens', return_value=50):
            result = connector("Test prompt")
            assert result == "Generated response"
    
    @patch('spart.external_llm_connector.ChatOpenAI')
    def test_token_limit_exceeded(self, mock_openai):
        """Test behavior when token limit is exceeded"""
        connector = ExternalLLMConnector(
            provider="openai",
            model_name="gpt-4",
            api_key="test-key",
            token_limit=50
        )
        
        # Mock token counting to exceed limit
        with patch.object(connector, '_count_tokens', return_value=100):
            result = connector("Test prompt")
            assert result is None
            
    @patch('spart.external_llm_connector.ChatOpenAI')
    def test_api_error(self, mock_openai):
        """Test behavior when API call fails"""
        # Configure mock to raise exception
        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = Exception("API Error")
        mock_openai.return_value = mock_instance
        
        # Create connector with mock
        connector = ExternalLLMConnector(
            provider="openai",
            model_name="gpt-4",
            api_key="test-key"
        )
        
        # Mock token counting
        with patch.object(connector, '_count_tokens', return_value=50):
            result = connector("Test prompt")
            assert result is None