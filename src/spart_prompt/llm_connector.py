import abc
from typing import Optional
from transformers import AutoTokenizer


class LLMConnector(abc.ABC):
    """
    An abstract base class for Language Model (LLM) connectors.

    This class defines the interface for interacting with different types of language models,
    including external API-based and local models. It provides a common structure for
    token counting, prompt handling, and model invocation.

    Attributes:
        model_name (str): The name or identifier of the specific model.
        token_limit (int): Maximum number of tokens allowed in input prompts.
    """

    def __init__(self, model_name: str, token_limit: int = 4096) -> None:
        """
        Initialize the base LLM connector.

        Args:
            model_name (str): The name or identifier of the specific model.
            token_limit (int, optional): Maximum number of tokens allowed in input prompts.
                Defaults to 4096.
        """
        self.model_name = model_name
        self.token_limit = token_limit

        # Try to load the tokenizer for the given model; fallback to 't5-small'
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(
                f"WARNING: Could not load tokenizer for model {model_name}: {e}")
            print("Falling back to 't5-small' tokenizer for rough estimates.")
            self.tokenizer = AutoTokenizer.from_pretrained(
                't5-small', legacy=False)

    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.

        Args:
            text (str): The input text to tokenize.

        Returns:
            int: The number of tokens in the text.
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text.split())  # Fallback: estimate by word count

    @abc.abstractmethod
    def __call__(self, prompt: str) -> Optional[str]:
        """
        Abstract method to invoke the language model with a given prompt.

        Args:
            prompt (str): The input text to send to the model.

        Returns:
            Optional[str]: The model's response or None if an error occurs.
        """
        pass
