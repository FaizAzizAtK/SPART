import os
import json
import unittest
import pandas as pd
import sys
import kagglehub
from dotenv import load_dotenv
from datetime import datetime

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from SPART.recommender import PromptRecommender
from SPART.external_llm_connector import ExternalLLMConnector

# Load environment variables
load_dotenv()

def format_gsm8k_for_prompt(dataset):
    raw_texts = []
    answers = []
    for _, row in dataset.iterrows():
        question = row.get("question", "")
        answer = row.get("answer", "")
        raw_texts.append(question.strip())
        answers.append(answer.strip())
    return raw_texts, answers

class TestOptimalNumExamplesSemanticGSM8K(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results_dir = "test_results"
        os.makedirs(cls.results_dir, exist_ok=True)
        api_key = os.getenv("OPENAI_API_KEY")

        cls.llm = ExternalLLMConnector(
            provider="openai",
            model_name="gpt-4o-mini",
            api_key=api_key,
            token_limit=30000
        )
        cls.recommender = PromptRecommender(cls.llm)

        dataset_path = kagglehub.dataset_download("johnsonhk88/gsm8k-grade-school-math-8k-dataset-for-llm")
        file_path = os.path.join(dataset_path, "gsm8k", "main", "train-00000-of-00001.parquet")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"GSM8K train file not found at {file_path}")

        dataset = pd.read_parquet(file_path)
        cls.raw_texts, cls.formatted_results = format_gsm8k_for_prompt(dataset)

    def test_optimal_num_examples_semantic(self):
        print("\n### Running Optimal Num Examples Semantic Test on GSM8K ###")

        context = ("You are solving grade school-level math word problems. Each input is a question, "
                   "and the output is a correct solution with explanation. Final answer must be at the end.")
        threshold = 0
        k = 3

        results = []

        for num_examples in range(0, 11):
            print(f"\n🔍 Testing with {num_examples} examples...")

            total_needed = num_examples + 20
            input_subset = self.raw_texts[:total_needed]
            output_subset = self.formatted_results[:total_needed]

            test_data = [{"input_data": i, "desired_output": o} for i, o in zip(input_subset, output_subset)]

            recommendation = self.recommender.recommend(
                test_data, num_examples, context, threshold, k,
                semantic_similarity=True, syntactic_similarity=False
            )

            semantic_similarity = recommendation.get("semantic_similarity", 0)
            print(f"🧠 Semantic Similarity for {num_examples} examples: {semantic_similarity:.4f}")

            results.append({"num_examples": num_examples, "semantic_similarity": semantic_similarity})

        file_path = os.path.join(self.results_dir, f"optimal_examples_semantic_gsm8k_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"\n📁 Results saved to: {file_path}")
        self.assertEqual(len(results), 11, "Should have 11 results for num_examples from 0 to 10")

if __name__ == "__main__":
    unittest.main()
