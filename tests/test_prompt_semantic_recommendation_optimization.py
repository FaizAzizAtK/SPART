import os
import json
import pytest
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

# Add src to Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from spart.recommender import PromptRecommender
from spart.external_llm_connector import ExternalLLMConnector
from spart.optimizer import PromptOptimizer
import kagglehub

# Load environment variables
load_dotenv()

def format_gsm8k_for_prompt(dataset):
    """Format GSM8K dataset for prompt processing."""
    raw_texts = []
    answers = []
    for _, row in dataset.iterrows():
        question = row.get("question", "")
        answer = row.get("answer", "")
        raw_texts.append(question.strip())
        answers.append(answer.strip())
    return raw_texts, answers

@pytest.fixture(scope="module", autouse=True)
def setup_resources():
    """Set up resources needed for all tests."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

    # Initialize LLM connector and other resources
    llm = ExternalLLMConnector(provider="openai", model_name="gpt-4o-mini", api_key=api_key, token_limit=30000)
    recommender = PromptRecommender(llm)
    optimizer = PromptOptimizer(llm)

    # Load GSM8K dataset
    dataset_path = kagglehub.dataset_download("johnsonhk88/gsm8k-grade-school-math-8k-dataset-for-llm")
    file_path = os.path.join(dataset_path, "gsm8k", "main", "train-00000-of-00001.parquet")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"GSM8K train file not found at {file_path}")

    dataset = pd.read_parquet(file_path)
    raw_texts, formatted_results = format_gsm8k_for_prompt(dataset)

    # Prepare input-output data
    data = {
        "input_data": raw_texts[107],
        "desired_output": formatted_results[107]
    }

    return recommender, optimizer, data

def test_semantic_recommendation(setup_resources):
    """Test the semantic recommendation functionality."""
    recommender, _, data = setup_resources

    # Initialize parameter variables
    examples = [{"input_data": i, "desired_output": o} for i, o in zip(data["input_data"], data["desired_output"])]
    context = "You are solving grade school-level math word problems. Each input is a question, and the output is the correct answer with reasoning steps. Format should be natural language."
    num_examples = 7
    threshold = 0.79
    max_iterations = 3
    semantic_similarity = True
    syntactic_similarity = False

    # Perform the recommendation
    results = recommender.recommend(examples, num_examples, context, threshold, max_iterations, semantic_similarity, syntactic_similarity)

    # Save results to file
    file_path = f"test_results/semantic_recommend_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Semantic recommendation results saved to: {file_path}")

    # Check if similarity is above the threshold (0.79)
    assert len(results) > 0, "No results returned from recommend function."
    assert results[0].get('semantic_similarity', 0) > threshold, f"Semantic similarity is below the threshold of {threshold}"

def test_semantic_optimization(setup_resources):
    """Test the semantic optimization functionality."""
    _, optimizer, data = setup_resources

    # Initialize parameter variables
    initial_prompt = "Solve the grade school math word problem and explain your reasoning step by step. Provide the final answer at the end."
    examples = [{"input_data": i, "desired_output": o} for i, o in zip(data["input_data"], data["desired_output"])]
    context = "Each input is a math problem, and the output should be a natural language explanation ending with the answer."
    num_examples = 7
    threshold = 0.79
    max_iterations = 5
    semantic_similarity = True
    syntactic_similarity = False

    # Perform the optimization
    results_dict = optimizer.optimize_prompt(
        initial_prompt, examples, num_examples, threshold, context, max_iterations, semantic_similarity, syntactic_similarity
    )

    optimized_prompt = results_dict.get("optimized_prompt")
    assert optimized_prompt, "Optimization returned no prompt."

    # Save optimization results to file
    results = {
        "original_prompt": initial_prompt,
        "optimized_prompt": optimized_prompt,
        "similarity_metrics": results_dict.get("similarity_metrics", {}),
    }
    file_path = f"test_results/semantic_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Semantic optimization results saved to: {file_path}")

    # Check if similarity is above the threshold (0.79)
    similarity_metrics = results_dict.get("similarity_metrics", {})
    assert similarity_metrics.get("semantic_similarity", 0) > threshold, f"Semantic similarity is below the threshold of {threshold}"
