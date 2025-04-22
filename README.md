# SPART: Streamlined Prompt Automation and Recommendation Tool (SOON TO BE PUBLISHED)

**SPART** is a Python package designed to simplify prompt engineering for Large Language Models (LLMs). It provides tools for generating, evaluating, and optimizing prompts automatically, making the process of creating effective prompts more efficient and less manual.

## Features

- **Connect to various LLM providers**: Use OpenAI, Cohere, HuggingFace, or local LLMs through Ollama.
- **Prompt recommendation**: Generate effective prompts based on input-output examples.
- **Prompt evaluation**: Test prompts against both semantic and syntactic similarity metrics.
- **Prompt optimization**: Improve existing prompts by applying good prompt engineering practices.
- **Multithreaded evaluation**: Process multiple inputs concurrently for faster testing.

## Installation

To install SPART, run the following command:

```bash
pip install spart
```

## Quick Start

Here's a quick example to get you started:

```python
from spart import ExternalLLMConnector, PromptRecommender
import pandas as pd

# Connect to your LLM provider
llm = ExternalLLMConnector(
    provider="openai",
    model_name="gpt-4o",
    api_key="your_api_key",
    temperature=0.7
)

# Create a recommender
recommender = PromptRecommender(llm)

# Example data 
# (inputs must be the first column and desired_outputs the second column)
examples = pd.DataFrame({
    'inputs': [
        "My name is John and I am 30 years old",
        "Hello I'm Emma, age 25"
    ],
    'desired_outputs': [
        "Name: John, Age: 30",
        "Name: Emma, Age: 25"
    ]
})

# Get a recommended prompt
results = recommender.recommend(
    examples=examples, # The input-output examples
    num_examples=1, # Use first example for training, rest for testing
    context="Extract name and age from text", # Extra context for the LLM
    similarity_threshold=0.85, # Threshold to reach before recommending
    max_iterations=3, # If threshold isn't reached then optimization will be attempted 3 times
    semantic_similarity=False, # Do not evaluate based on semantics
    syntactic_similarity=True # Do evaluate based on syntax
)

print(f"Recommended prompt: {results['recommended_prompt']}")
print(f"Semantic similarity: {results['semantic_similarity']}")
print(f"Syntactic similarity: {results['syntactic_similarity']}")
```

## Classes Overview

### LLMConnector (Abstract Base Class)
Base interface for connecting to LLMs with the following implementations:
- **ExternalLLMConnector**: Connect to OpenAI, Cohere, or HuggingFace
- **LocalLLMConnector**: Connect to local LLMs through Ollama

### PromptEvaluator
Evaluates prompts using:
- Semantic similarity (using vector embeddings)
- Syntactic similarity (using ROUGE-L score)

### PromptRecommender
Generates system prompts based on input-output examples, structured with:
- Role definition
- Guidelines
- Instructions
- Examples
- Context
- Goal

### PromptOptimizer
Improves existing prompts by applying prompt engineering best practices.

## Use Cases

- **Data transformation**: Generate prompts to convert data from one format to another.
- **Text summarization**: Build prompts that produce consistent summary formats.
- **Structured output generation**: Ensure LLM outputs follow specific formats.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
