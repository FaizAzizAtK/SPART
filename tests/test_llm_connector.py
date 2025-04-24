import pytest
from unittest.mock import Mock, patch
from spart.llm_connector import LLMConnector


class TestLLMConnector:
    """Tests for the abstract LLMConnector base class."""

    def test_init(self):
        """Test initialization of abstract base class."""
        class ConcreteLLM(LLMConnector):
            def __call__(self, prompt):
                return "test response"

        connector = ConcreteLLM("test-model", 1000)
        assert connector.model_name == "test-model"
        assert connector.token_limit == 1000

    @patch('spart.llm_connector.AutoTokenizer')
    def test_token_counting(self, mock_tokenizer):
        """Test token counting functionality."""
        # Mock tokenizer
        mock_instance = Mock()
        mock_instance.encode.return_value = [101, 102, 103, 104]  # 4 tokens
        mock_tokenizer.from_pretrained.return_value = mock_instance

        class ConcreteLLM(LLMConnector):
            def __call__(self, prompt):
                return "test response"

        connector = ConcreteLLM("test-model")
        token_count = connector._count_tokens("This is a test.")
        assert token_count == 4
        mock_instance.encode.assert_called_once_with("This is a test.")

    @patch('spart.llm_connector.AutoTokenizer')
    def test_token_counting_fallback(self, mock_tokenizer):
        """Test token counting fallback when tokenizer fails."""
        # Mock tokenizer that raises an exception
        mock_tokenizer.from_pretrained.side_effect = Exception("Tokenizer not found")
        
        class ConcreteLLM(LLMConnector):
            def __call__(self, prompt):
                return "test response"
        
        # Create a second mock for the fallback tokenizer
        with patch('spart.llm_connector.AutoTokenizer.from_pretrained') as mock_fallback:
            mock_fallback_instance = Mock()
            mock_fallback_instance.encode.return_value = [101, 102, 103, 104, 105]  # 5 tokens
            mock_fallback.return_value = mock_fallback_instance
            
            connector = ConcreteLLM("test-model")
            # Token counting should work using the fallback tokenizer
            token_count = connector._count_tokens("This is a fallback test.")
            assert token_count == 5

    @patch('spart.llm_connector.AutoTokenizer')
    def test_token_counting_with_empty_string(self, mock_tokenizer):
        """Test token counting with an empty string."""
        mock_instance = Mock()
        mock_instance.encode.return_value = []  # No tokens
        mock_tokenizer.from_pretrained.return_value = mock_instance

        class ConcreteLLM(LLMConnector):
            def __call__(self, prompt):
                return "test response"

        connector = ConcreteLLM("test-model")
        token_count = connector._count_tokens("")
        assert token_count == 0
        mock_instance.encode.assert_called_once_with("")

    @patch('spart.llm_connector.AutoTokenizer')
    def test_token_counting_with_special_characters(self, mock_tokenizer):
        """Test token counting with special characters in the string."""
        mock_instance = Mock()
        mock_instance.encode.return_value = [101, 102, 103, 104]  # 4 tokens
        mock_tokenizer.from_pretrained.return_value = mock_instance

        class ConcreteLLM(LLMConnector):
            def __call__(self, prompt):
                return "test response"

        connector = ConcreteLLM("test-model")
        token_count = connector._count_tokens("This is a test! #special_chars")
        assert token_count == 4
        mock_instance.encode.assert_called_once_with("This is a test! #special_chars")

    @patch('spart.llm_connector.AutoTokenizer')
    def test_token_counting_with_long_text(self, mock_tokenizer):
        """Test token counting with a long string."""
        mock_instance = Mock()
        # Simulate a long string (say 100 tokens)
        mock_instance.encode.return_value = list(range(100))
        mock_tokenizer.from_pretrained.return_value = mock_instance

        class ConcreteLLM(LLMConnector):
            def __call__(self, prompt):
                return "test response"

        connector = ConcreteLLM("test-model")
        long_text = "This is a long test. " * 10  # Repeat the phrase to make it long
        token_count = connector._count_tokens(long_text)
        assert token_count == 100
        mock_instance.encode.assert_called_once_with(long_text)

    @patch('spart.llm_connector.AutoTokenizer')
    def test_token_counting_with_empty_string_fallback(self, mock_tokenizer):
        """Test token counting fallback when given an empty string."""
        # Mock tokenizer that raises an exception
        mock_tokenizer.from_pretrained.side_effect = Exception("Tokenizer not found")
        
        # Create a mock for the fallback tokenizer
        with patch('spart.llm_connector.AutoTokenizer.from_pretrained') as mock_fallback:
            mock_fallback_instance = Mock()
            mock_fallback_instance.encode.return_value = []  # No tokens for empty string
            mock_fallback.return_value = mock_fallback_instance
            
            class ConcreteLLM(LLMConnector):
                def __call__(self, prompt):
                    return "test response"
            
            # When the exception occurs, the fallback tokenizer should be used
            connector = ConcreteLLM("test-model")
            token_count = connector._count_tokens("")
            assert token_count == 0  # Expect 0 tokens
            mock_fallback_instance.encode.assert_called_once_with("")

    @patch('spart.llm_connector.AutoTokenizer')
    def test_token_counting_with_no_tokens(self, mock_tokenizer):
        """Test fallback to word count when there are no tokens."""
        # Mock tokenizer that raises an exception
        mock_tokenizer.from_pretrained.side_effect = Exception("Tokenizer not found")
        
        # Create a mock for the fallback tokenizer
        with patch('spart.llm_connector.AutoTokenizer.from_pretrained') as mock_fallback:
            mock_fallback_instance = Mock()
            mock_fallback_instance.encode.return_value = []  # No tokens
            mock_fallback.return_value = mock_fallback_instance
            
            class ConcreteLLM(LLMConnector):
                def __call__(self, prompt):
                    return "test response"
            
            # Fallback to word count (empty string should have no tokens)
            connector = ConcreteLLM("test-model")
            token_count = connector._count_tokens("    ")  # Only spaces
            assert token_count == 0  # Expect 0 tokens
            mock_fallback_instance.encode.assert_called_once_with("    ")



    @patch('spart.llm_connector.AutoTokenizer')
    def test_tokenizer_error_handling(self, mock_tokenizer):
        """Test tokenizer error handling."""
        # Simulate tokenizer error during initial load
        mock_tokenizer.from_pretrained.side_effect = Exception("Tokenizer error")
        
        # Create a mock for the fallback tokenizer
        mock_fallback = Mock()
        mock_fallback.encode.return_value = [101, 102, 103, 104]  # Simulating tokenization

        # Mock the fallback tokenizer to be used when the exception occurs
        with patch('spart.llm_connector.AutoTokenizer.from_pretrained', side_effect=mock_tokenizer.from_pretrained):
            mock_tokenizer.from_pretrained.side_effect = Exception("Tokenizer error")
            # Now mock the fallback tokenizer call
            with patch('spart.llm_connector.AutoTokenizer.from_pretrained', return_value=mock_fallback):
                
                class ConcreteLLM(LLMConnector):
                    def __call__(self, prompt):
                        return "test response"
                
                # When the exception occurs, the fallback tokenizer should be used
                connector = ConcreteLLM("test-model")
                token_count = connector._count_tokens("Fallback test")
                assert token_count == 4
                mock_fallback.encode.assert_called_once_with("Fallback test")
    
    def test_call_method_in_concrete_class(self):
        """Test if the __call__ method works in the concrete implementation."""
        class ConcreteLLM(LLMConnector):
            def __call__(self, prompt):
                return f"Response for: {prompt}"

        connector = ConcreteLLM("test-model")
        response = connector("Test prompt")
        assert response == "Response for: Test prompt"
