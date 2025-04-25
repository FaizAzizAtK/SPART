import pytest
from unittest.mock import patch, MagicMock
import subprocess
import logging

import sys 
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from spart_prompt.local_llm_connector import LocalLLMConnector

class TestLocalLLMConnector:
    """Tests for the LocalLLMConnector class"""
    
    @pytest.fixture
    def connector(self):
        """Fixture to provide a LocalLLMConnector instance"""
        with patch('transformers.AutoTokenizer.from_pretrained'):
            return LocalLLMConnector(
                model_name="llama2",
                timeout=30,
                token_limit=100
            )
    
    def test_initialization(self, connector):
        """Test initialization with parameters"""
        assert connector.model_name == "llama2"
        assert connector.timeout == 30
        assert connector.token_limit == 100
    
    @patch('subprocess.run')
    def test_successful_call(self, mock_run, connector):
        """Test successful subprocess call"""
        # Configure mock subprocess call
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Generated response"
        mock_run.return_value = mock_process
        
        # Mock token counting
        with patch.object(connector, '_count_tokens', return_value=50):
            result = connector("Test prompt")
            
            # Assert subprocess was called correctly
            mock_run.assert_called_once_with(
                ['ollama', 'run', 'llama2', 'Test prompt'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result == "Generated response"
    
    @patch('subprocess.run')
    @patch('logging.error')
    def test_subprocess_error(self, mock_log, mock_run, connector):
        """Test behavior when subprocess fails"""
        # Configure mock subprocess call
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stderr = "Command failed"
        mock_run.return_value = mock_process
        
        # Mock token counting
        with patch.object(connector, '_count_tokens', return_value=50):
            result = connector("Test prompt")
            assert result is None
            mock_log.assert_called_once()
    
    @patch('subprocess.run')
    @patch('logging.error')
    def test_timeout(self, mock_log, mock_run, connector):
        """Test behavior when subprocess times out"""
        # Configure mock subprocess call
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=['ollama', 'run', 'llama2', ''], timeout=30)
        
        # Mock token counting
        with patch.object(connector, '_count_tokens', return_value=50):
            result = connector("Test prompt")
            assert result is None
            mock_log.assert_called_once()
    
    @patch('logging.error')
    def test_token_limit_exceeded(self, mock_log, connector):
        """Test behavior when token limit is exceeded"""
        # Mock token counting to exceed limit
        with patch.object(connector, '_count_tokens', return_value=200):
            result = connector("Test prompt")
            assert result is None
            mock_log.assert_called_once()