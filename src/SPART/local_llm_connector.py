import logging
import subprocess
from typing import Optional

from spart.llm_connector import LLMConnector


class LocalLLMConnector(LLMConnector):
    """
    A class to interact with a local Ollama model via its command-line interface.

    Attributes:
        timeout (int): Maximum execution time before timeout occurs.
    """

    def __init__(
            self,
            model_name: str,
            timeout: int = 600,
            token_limit: int = 4096) -> None:
        """
        Initialize the local LLM connector.

        Args:
            model_name (str): The name of the Ollama model to interact with.
            timeout (int, optional): Maximum time to wait for execution. Defaults to 600 seconds.
            token_limit (int, optional): Maximum token limit for input prompts. Defaults to 4096.
        """
        super().__init__(model_name, token_limit)
        self.timeout = timeout

    def __call__(self, prompt: str) -> Optional[str]:
        """
        Sends a prompt to the specified local Ollama model and retrieves the response.

        Args:
            prompt (str): The input text to be processed by the model.

        Returns:
            Optional[str]: The model's generated output or None if an error occurs.

        Raises:
            RuntimeError: If there are issues with subprocess execution.
        """
        # Token limit check (using base class method)
        num_tokens = self._count_tokens(prompt)
        if num_tokens > self.token_limit:
            logging.error(f"Prompt exceeds token limit of {self.token_limit}. Token count: {num_tokens}")
            return None

        try:
            result = subprocess.run(
                ['ollama', 'run', self.model_name, prompt],
                capture_output=True,  # Capture standard output and error output
                text=True,  # Ensure output is in string format
                timeout=self.timeout  # Set a timeout for the execution
            )

            # Check for errors in execution
            if result.returncode != 0:
                error_message = f"Model execution failed for '{self.model_name}': {result.stderr}"
                logging.error(error_message)
                return None

            return result.stdout.strip()

        except subprocess.TimeoutExpired:
            error_message = f"Model execution timed out for '{self.model_name}'"
            logging.error(error_message)
            return None

        except Exception as e:
            logging.error(f"Unexpected error in model execution: {str(e)}")
            return None
