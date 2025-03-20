import subprocess
import logging

def prompt_llm(prompt, model_name):
    """
    Call the specified local Ollama model using its command-line tool with the provided prompt.
    Args:
        prompt (str): The prompt to feed into the model.
        model_name (str): The name of the model to use.
    Returns:
        str: The model's output after processing the prompt.
    Raises:
        RuntimeError: If there are issues with the model call or subprocess execution.
    """
    try:
        # Execute the Ollama model run command
        result = subprocess.run(
            ['ollama', 'run', model_name, prompt],
            capture_output=True,
            text=True,
            timeout=6000
        )
        # Check for any errors in model execution
        if result.returncode != 0:
            error_message = f"Model call error for {model_name}: {result.stderr}"
            logging.error(error_message)
            raise RuntimeError(error_message)
        # Return cleaned output
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        error_message = f"Model call timed out for {model_name}"
        logging.error(error_message)
        raise RuntimeError(error_message)
    except Exception as e:
        logging.error(f"Unexpected error in prompt_llm: {str(e)}")
        raise

def connect_local_llm(model_name):
    """
    Create a function that can be called with just a prompt for a specific model.
    Args:
        model_name (str): The name of the Ollama model to use.
    Returns:
        callable: A function that takes a prompt and calls the specified model.
    """
    return lambda prompt: prompt_llm(prompt, model_name=model_name)

