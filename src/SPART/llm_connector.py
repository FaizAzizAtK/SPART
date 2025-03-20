import logging
from langchain_community.llms import OpenAI, HuggingFaceHub, Cohere  # Updated import
from langchain_openai import ChatOpenAI  # Updated import for OpenAI model
from langchain.schema import AIMessage, HumanMessage
import os

class LLMConnector:
    """
    A connector to interact with various LLM providers via LangChain.
    Supports OpenAI, Hugging Face Hub, and Cohere.
    """

    def __init__(self, provider="openai", model_name=None, api_key=None, temperature=0.7):
        """
        Initializes the connector for the specified provider.
        
        Args:
            provider (str): The LLM provider to use ('openai', 'huggingface', 'cohere').
            model_name (str): The model name (e.g., 'gpt-4', 'tiiuae/falcon-7b').
            api_key (str, optional): API key for authentication (if required).
            temperature (float): Controls randomness (0.0 = deterministic, 1.0 = more random).
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key or os.getenv("LLM_API_KEY")  # Use env variable if no key is provided
        self.temperature = temperature  # Store temperature

        # Initialize the LLM model based on the provider
        if self.provider == "openai":
            self.llm = ChatOpenAI(model_name=self.model_name, openai_api_key=self.api_key, temperature=self.temperature)
        elif self.provider == "huggingface":
            self.llm = HuggingFaceHub(repo_id=self.model_name, huggingfacehub_api_token=self.api_key)
        elif self.provider == "cohere":
            self.llm = Cohere(model=self.model_name, cohere_api_key=self.api_key, temperature=self.temperature)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")


    def __call__(self, prompt):
        """
        Sends a prompt to the selected LLM and returns the response.

        Args:
            prompt (str): The input text to send to the model.

        Returns:
            str: The model's response.
        """
        try:
            if self.provider == "openai":
                # Ensure prompt doesn't exceed token limit
                if len(prompt.split()) > 512:  # OpenAI model token limit
                    prompt = " ".join(prompt.split()[:512])  # Truncate the prompt to 512 tokens

                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                return response.content if isinstance(response, AIMessage) else str(response)
            else:
                # For Hugging Face and Cohere, ensure we do not exceed token limits (model-dependent)
                if len(prompt.split()) > 512:  # Example limit, adjust as needed for the model
                    prompt = " ".join(prompt.split()[:512])  # Truncate the prompt to 512 tokens

                response = self.llm.invoke(prompt)  # Hugging Face and Cohere use direct text input
                return str(response)

        except Exception as e:
            logging.error(f"Error calling LLM: {str(e)}")
            return None
