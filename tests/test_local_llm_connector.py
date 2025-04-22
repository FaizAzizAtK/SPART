import os
import sys
import pytest
from dotenv import load_dotenv

# Adjust the path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from SPART.local_llm_connector import LocalLLMConnector

@pytest.fixture(scope="module", autouse=True)
def load_env():
    """Automatically load environment variables before tests."""
    load_dotenv()

def test_local_llm_response_not_empty():
    """
    Test that the LocalLLMConnector successfully connects and returns a non-empty response
    for a given prompt.
    """
    # Initialize the connector for a local LLM (example: "llama3.1")
    llm = LocalLLMConnector("llama3.1")

    assert llm is not None, "Failed to connect to the local LLM."

    # Define a simple prompt
    prompt = "Explain the basics of artificial intelligence."

    # Get the response from the LLM
    response = llm(prompt)

    # Check if the response is not empty
    assert response, "LLM response is empty!"
