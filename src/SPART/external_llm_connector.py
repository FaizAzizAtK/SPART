import logging
from typing import Optional

from langchain_community.llms import HuggingFaceHub, Cohere
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from transformers import AutoTokenizer

from SPART.llm_connector import LLMConnector


class ExternalLLMConnector(LLMConnector):
    """
    A connector to interact with various external LLM providers via LangChain.

    Supports OpenAI, Hugging Face Hub, and Cohere APIs.

    Attributes:
        provider (str): The chosen LLM provider ('openai', 'huggingface', 'cohere').
        api_key (str): The API key for authentication.
        temperature (float): Controls response randomness.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: str,
        temperature: float = 0.7,
        token_limit: int = 4096
    ) -> None:
        """
        Initialize the external LLM connector.

        Args:
            provider (str): The LLM provider to use ('openai', 'huggingface', 'cohere').
            model_name (str): The name of the specific model.
            api_key (str): API key for authentication.
            temperature (float, optional): Controls randomness of responses. Defaults to 0.7.
            token_limit (int, optional): Maximum token limit for input prompts. Defaults to 4096.
        """
        super().__init__(model_name, token_limit)
        self.provider = provider
        self.api_key = api_key
        self.temperature = temperature

        # Initialize the LLM model based on the provider
        if self.provider == "openai":
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                openai_api_key=self.api_key,
                temperature=self.temperature
            )
        elif self.provider == "huggingface":
            self.llm = HuggingFaceHub(
                repo_id=self.model_name,
                huggingfacehub_api_token=self.api_key
            )
        elif self.provider == "cohere":
            self.llm = Cohere(
                model=self.model_name,
                cohere_api_key=self.api_key,
                temperature=self.temperature
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        # Load tokenizer for models that support token limits
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name) if self.model_name else None
        except Exception as e:
            logging.warning(f"Could not load tokenizer for model {self.model_name}: {str(e)}")
            self.tokenizer = None

    def __call__(self, prompt: str) -> Optional[str]:
        """
        Sends a prompt to the selected external LLM and returns the response.

        Args:
            prompt (str): The input text to send to the model.

        Returns:
            Optional[str]: The model's response or None if an error occurs.
        """
        try:
            # Ensure the prompt does not exceed the token limit
            num_tokens = self._count_tokens(prompt)
            if num_tokens > self.token_limit:
                raise ValueError(f"Prompt exceeds token limit of {self.token_limit}. Token count: {num_tokens}")

            if self.provider == "openai":
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                return response.content if isinstance(
                    response, AIMessage) else str(response)
            else:
                # Direct text input for Hugging Face & Cohere
                response = self.llm.invoke(prompt)
                return str(response)

        except Exception as e:
            logging.error(f"Error calling external LLM: {str(e)}")
            return None
