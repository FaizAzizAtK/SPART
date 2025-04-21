import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from dotenv import load_dotenv
from SPART.external_llm_connector import LLMConnector

# Load environment variables from .env file
load_dotenv()

def test_openai_llm():
    """
    Tests the LLMConnector with OpenAI's gpt-4o-mini model.
    Ensures the API key is loaded from the .env file.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

    # Initialize the connector with OpenAI's GPT-4o-mini
    llm = LLMConnector(provider="openai", model_name="gpt-4o-mini", api_key=api_key)

    # Test prompt
    prompt = "Explain quantum computing in simple terms."
    response = llm.call_model(prompt)

    # Print response
    print("\nOpenAI GPT-4o-mini Response:\n", response)

    # Basic assertion to check if response is not empty
    assert response is not None and len(response) > 0, "LLM response is empty!"

# Run the test
if __name__ == "__main__":
    test_openai_llm()
