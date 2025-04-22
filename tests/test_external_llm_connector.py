import os
import sys
import pytest
from dotenv import load_dotenv

# Adjust the path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from spart.external_llm_connector import ExternalLLMConnector

@pytest.fixture(scope="module", autouse=True)
def load_env():
    """Automatically load environment variables before tests."""
    load_dotenv()

def test_openai_llm_response_not_empty():
    """
    Test that the ExternalLLMConnector successfully calls OpenAI's gpt-4o-mini
    and returns a non-empty response.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY is not set in the environment variables."

    llm = ExternalLLMConnector(provider="openai", model_name="gpt-4o-mini", api_key=api_key)

    prompt = "Explain quantum computing in simple terms."
    response = llm(prompt)

    assert response, "LLM response is empty!"
