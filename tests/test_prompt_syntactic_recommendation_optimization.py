import os
import sys
import pytest
import json
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import kagglehub

# Adjust the path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from spart.recommender import PromptRecommender
from spart.external_llm_connector import ExternalLLMConnector
from spart.optimizer import PromptOptimizer

# Load environment variables automatically before tests
@pytest.fixture(scope="module", autouse=True)
def load_env():
    """Automatically load environment variables before tests."""
    load_dotenv()

# Function to process the dataset into raw text and formatted NER pairs
def generate_raw_text_and_format(dataset):
    """Processes the dataset into raw text and formatted token-NER pairs."""
    raw_texts = []
    formatted_results = []
    current_document = []
    current_formatted = []

    for _, row in dataset.iterrows():
        word = str(row["word"])
        ner_tag = row["ner"]

        if word == "-DOCSTART-":
            if current_document:
                raw_texts.append(" ".join(current_document))
                formatted_results.append(json.dumps(current_formatted))  # Store as string
            current_document = []
            current_formatted = []
            continue

        if word in [".", "\n"]:
            if current_document:
                current_document.append(word)
        else:
            current_document.append(word)
            current_formatted.append(f"Token: {word}, NER_tag: {ner_tag}")

    if current_document:
        raw_texts.append(" ".join(current_document))
        formatted_results.append(json.dumps(current_formatted))  # Store as string

    return raw_texts, formatted_results

@pytest.fixture(scope="module")
def setup_resources():
    """Set up resources needed for the test."""
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.fail("OPENAI_API_KEY is not set in the environment variables.")

    # Initialize the LLM connector and other components
    llm = ExternalLLMConnector(provider="openai", model_name="gpt-4o-mini", api_key=api_key, token_limit=30000)
    recommender = PromptRecommender(llm)
    optimizer = PromptOptimizer(llm)

    # Download and load the dataset
    path = kagglehub.dataset_download("alaakhaled/conll003-englishversion")
    train_file = os.path.join(path, 'train.txt')

    if not os.path.exists(train_file):
        pytest.fail(f"Dataset file 'train.txt' not found at path: {train_file}")

    dataset = pd.read_csv(train_file, sep=r"\s+", header=None, names=["word", "pos", "chunk", "ner", "other"])
    dataset['other'] = dataset['other'].fillna('O').astype(str)
    dataset = dataset.dropna(how="all")

    # Generate raw texts and formatted results
    raw_texts, formatted_results = generate_raw_text_and_format(dataset)

    data = {
        "input_data": raw_texts[:104],
        "desired_output": formatted_results[:104]
    }

    return recommender, optimizer, data, results_dir

def test_recommendation_process(setup_resources):
    """Test the recommendation process and check syntactic similarity."""
    recommender, _, data, results_dir = setup_resources

    examples = [{"input_data": i, "desired_output": o} for i, o in zip(data["input_data"], data["desired_output"])]
    num_examples = 4
    context = "Input data is raw tokenized text, and the desired output consists of tokens with their Named Entity Recognition label. The Tokens are labeled under one of the following labels [I-LOC, B-ORG, O, B-PER, I-PER, I-MISC, B-MISC, I-ORG, B-LOC]. The goal is to label all the tokens with its NER label"
    threshold = 0.95
    max_iterations = 3
    semantic_similarity = False
    syntactic_similarity = True

    # Run the recommendation
    results = recommender.recommend(examples, num_examples, context, threshold, max_iterations, semantic_similarity, syntactic_similarity)

    # Save results to JSON file
    file_path = os.path.join(results_dir, f"syntax_recommend_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(file_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Recommendation test results saved to: {file_path}")
    
    # Ensure results are valid
    assert len(results) > 0, "No results returned from recommend function."
    assert results[0]["syntactic_similarity"] >= 0.9, "Syntactic similarity below the required threshold (0.9)."

def test_optimization_process(setup_resources):
    """Test the optimization process and check syntactic similarity."""
    _, optimizer, data, results_dir = setup_resources

    generated_prompt = "Extract named entities and classify the tokens under one of the following labels [I-LOC, B-ORG, O, B-PER, I-PER, I-MISC, B-MISC, I-ORG, B-LOC]. The format should be [\"Token: India, NER_tag: B-LOC\", ..., ...]"
    examples = [{"input_data": i, "desired_output": o} for i, o in zip(data["input_data"], data["desired_output"])]
    num_examples = 4
    context = "Input data is raw tokenized text, and the desired output consists of tokens with their Named Entity Recognition label. The Tokens are labeled under one of the following labels [I-LOC, B-ORG, O, B-PER, I-PER, I-MISC, B-MISC, I-ORG, B-LOC]. The goal is to label all the tokens with its NER label"
    threshold = 0.95
    max_iterations = 5
    semantic_similarity = False
    syntactic_similarity = True

    # Run the optimization
    results_dict = optimizer.optimize_prompt(
        generated_prompt,
        examples,
        num_examples,
        threshold,
        context,
        max_iterations,
        semantic_similarity,
        syntactic_similarity
    )

    # Extract values from the returned dictionary
    optimized_prompt = results_dict["optimized_prompt"]
    similarity_metrics = results_dict["similarity_metrics"]

    # Save optimization results to a file
    results = {
        "original_prompt": generated_prompt,
        "optimized_prompt": optimized_prompt,
        "similarity_metrics": similarity_metrics,
    }

    file_path = os.path.join(results_dir, f"syntax_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(file_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Optimization test results saved to: {file_path}")
    
    # Ensure the syntactic similarity is above threshold
    assert optimized_prompt is not None, "Optimization returned no prompt."
    assert similarity_metrics["syntactic_similarity"] >= 0.9, "Syntactic similarity below the required threshold (0.9)."
    assert "semantic_similarity" in similarity_metrics, "Semantic similarity is missing."

