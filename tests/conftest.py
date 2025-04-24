import pytest
from unittest.mock import MagicMock
import pandas as pd

import sys 
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from spart.llm_connector import LLMConnector


class MockLLMConnector(LLMConnector):
    """Mock implementation of LLMConnector for testing"""
    def __init__(self, *args, **kwargs):
        super().__init__("mock-model", token_limit=100)
        self.responses = {}
        self.default_response = "Default mock response"
        
    def set_response(self, prompt, response):
        self.responses[prompt] = response
        
    def set_default_response(self, response):
        self.default_response = response
        
    def __call__(self, prompt):
        return self.responses.get(prompt, self.default_response)

@pytest.fixture
def mock_llm():
    """Fixture to provide a mock LLM connector"""
    return MockLLMConnector()

@pytest.fixture
def example_data():
    """Fixture to provide example input-output pairs"""
    return pd.DataFrame({
        0: ["Input 1", "Input 2", "Input 3", "Input 4"],
        1: ["Output 1", "Output 2", "Output 3", "Output 4"]
    })