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
from SPART.optimizer import PromptOptimizer

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

class TestSemanticPromptOptimization(unittest.TestCase):
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
        cls.optimizer = PromptOptimizer(cls.llm)

        # Download and load GSM8K (public version)
        dataset_path = kagglehub.dataset_download("johnsonhk88/gsm8k-grade-school-math-8k-dataset-for-llm")
        file_path = os.path.join(dataset_path, "gsm8k", "main", "train-00000-of-00001.parquet")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"GSM8K train file not found at {file_path}")

        dataset = pd.read_parquet(file_path)
        cls.raw_texts, cls.formatted_results = format_gsm8k_for_prompt(dataset)

        cls.data = {
            "input_data": cls.raw_texts[7:11],
            "desired_output": cls.formatted_results[7:11]
        }

    def test_recommend(self):
        print("\n### Running Semantic Recommendation Test ###")
        examples = [{"input_data": i, "desired_output": o} for i, o in zip(self.data["input_data"], self.data["desired_output"])]
        
        context = "You are solving grade school-level math word problems. Each input is a question, and the output is the correct answer with reasoning steps. Format should be natural language."
        results = self.recommender.recommend(examples, 7, context, 0.85, 3, semantic_similarity=True, syntactic_similarity=False)

        file_path = os.path.join(self.results_dir, f"semantic_recommend_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Semantic recommendation results saved to: {file_path}")
        self.assertTrue(len(results) > 0)

    def test_optimization(self):
        print("\n### Running Semantic Optimization Test ###")
        initial_prompt = "Solve the grade school math word problem and explain your reasoning step by step. Provide the final answer at the end."
        examples = [{"input_data": i, "desired_output": o} for i, o in zip(self  .data["input_data"], self.data["desired_output"])]
        context = "Each input is a math problem, and the output should be a natural language explanation ending with the answer."
        print(examples)
        # results_dict = self.optimizer.optimize_prompt(
        #     initial_prompt, examples, 7, 0.85, context, 5,
        #     semantic_similarity=True, syntactic_similarity=False
        # )

        # results = {
        #     "original_prompt": initial_prompt,
        #     "optimized_prompt": results_dict["optimized_prompt"],
        #     "similarity_metrics": {
        #         "semantic_similarity": results_dict["semantic_similarity"],
        #         "syntactic_similarity": results_dict["syntactic_similarity"],
        #         "evaluation_details": results_dict["evaluation_details"]
        #     }
        # }

        # file_path = os.path.join(self.results_dir, f"semantic_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        # with open(file_path, "w") as f:
        #     json.dump(results, f, indent=4)

        # print(f"Semantic optimization results saved to: {file_path}")
        # self.assertIsNotNone(results["optimized_prompt"])

if __name__ == "__main__":
    unittest.main()
