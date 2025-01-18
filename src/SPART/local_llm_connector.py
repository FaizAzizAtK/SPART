def prompt_llm(prompt, model_name):
    """
    Call the specified local Ollama model using its command-line tool with the provided prompt.
    
    :param prompt: The prompt to feed into the model.
    :param model_name: The name of the model to use.
    :return: The model's output as a string.
    """
    import subprocess
    result = subprocess.run(['ollama', 'run', model_name, prompt], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Error in model call: {result.stderr}")
    
    return result.stdout.strip()

# The function passed to the Recommender, allowing the user to specify the model
def connect_local_llm(model_name):
    # Return a function that only needs the prompt since model_name is fixed when the Recommender is instantiated
    return lambda prompt: prompt_llm(prompt, model_name)