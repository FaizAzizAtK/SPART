import pandas as pd
import json
import os
import unittest
import sys
import kagglehub
from dotenv import load_dotenv
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from SPART.recommender import PromptRecommender
from SPART.external_llm_connector import ExternalLLMConnector
from SPART.optimizer import PromptOptimizer

os.environ.pop("MallocStackLogging", None)

load_dotenv()

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


class TestPromptOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up resources needed for all tests."""
        cls.results_dir = "test_results"
        os.makedirs(cls.results_dir, exist_ok=True)
        api_key = os.getenv("OPENAI_API_KEY")

        # Initialize a local LLM connector for testing
        cls.llm = ExternalLLMConnector(provider="openai", model_name="gpt-4o-mini", api_key=api_key, token_limit=30000)
        cls.recommender = PromptRecommender(cls.llm)
        cls.optimizer = PromptOptimizer(cls.llm)

        # Download and load the dataset
        path = kagglehub.dataset_download("alaakhaled/conll003-englishversion")
        train_file = os.path.join(path, 'train.txt')

        if not os.path.exists(train_file):
            raise FileNotFoundError("Dataset file 'train.txt' not found.")

        dataset = pd.read_csv(train_file, sep=r"\s+", header=None, names=["word", "pos", "chunk", "ner", "other"])
        dataset['other'] = dataset['other'].fillna('O').astype(str)
        dataset = dataset.dropna(how="all")

        # Generate raw texts and formatted results
        cls.raw_texts, cls.formatted_results = generate_raw_text_and_format(dataset)

        # Prepare input-output examples
        cls.data = {
            "input_data": cls.raw_texts[:104],
            "desired_output": cls.formatted_results[:104]
        }

    def test_recommend(self):
        """Test the recommendation process and save results to a JSON file."""
        print("\n### Running Recommendation Test ###")
        examples = [{"input_data": i, "desired_output": o} for i, o in zip(self.data["input_data"], self.data["desired_output"])]
        num_examples = 4
        context = "Input data is raw tokenized text, and the desired output consists of tokens with their Named Entity Recognition label. The Tokens are labeled under one of the following labels [I-LOC, B-ORG, O, B-PER, I-PER, I-MISC, B-MISC, I-ORG, B-LOC]. The goal is to label all the tokens with its NER label"
        threshold = 0.95
        max_iterations = 3
        semantic_similarity = False
        syntactic_similarity = True

        results = self.recommender.recommend(examples, num_examples, context, threshold, max_iterations, semantic_similarity, syntactic_similarity)

        file_path = os.path.join(self.results_dir, f"syntax_recommend_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(file_path, "w") as json_file:
            json.dump(results, json_file, indent=4)

        print(f"Recommendation test results saved to: {file_path}")
        self.assertTrue(len(results) > 0, "No results from recommend function.")

    def test_optimisation(self):
        """Test the optimization process for the NER task and save results."""
        print("\n### Running Optimisation Test ###")
        generated_prompt = "Extract named entities and classify the tokens a under one of the following labels [I-LOC, B-ORG, O, B-PER, I-PER, I-MISC, B-MISC, I-ORG, B-LOC]. The format should be [\"Token: India, NER_tag: B-LOC\", ..., ...]"
        examples = [{"input_data": i, "desired_output": o} for i, o in zip(self.data["input_data"], self.data["desired_output"])]
        num_examples = 4
        context = "Input data is raw tokenized text, and the desired output consists of tokens with their Named Entity Recognition label. The Tokens are labeled under one of the following labels [I-LOC, B-ORG, O, B-PER, I-PER, I-MISC, B-MISC, I-ORG, B-LOC]. The goal is to label all the tokens with its NER label"
        threshold = 0.95
        max_iterations = 5
        semantic_similarity = False
        syntactic_similarity = True

        results_dict = self.optimizer.optimize_prompt(
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
        similarity_metrics = {
            "semantic_similarity": results_dict["semantic_similarity"],
            "syntactic_similarity": results_dict["syntactic_similarity"],
            "evaluation_details": results_dict["evaluation_details"]
        }

        # Structure the final results dictionary for easy access
        results = {
            "original_prompt": generated_prompt,
            "optimized_prompt": optimized_prompt,
            "similarity_metrics": similarity_metrics,
        }

        file_path = os.path.join(self.results_dir, f"syntax_optimisation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(file_path, "w") as json_file:
            json.dump(results, json_file, indent=4)

        print(f"Optimisation test results saved to: {file_path}")
        self.assertIsNotNone(optimized_prompt, "Optimization returned no prompt.")
        self.assertTrue("semantic_similarity" in similarity_metrics and "syntactic_similarity" in similarity_metrics,
                        "Similarity metrics are incomplete.")


if __name__ == "__main__":
    unittest.main()
