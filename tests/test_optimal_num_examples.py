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
from SPART.llm_connector import LLMConnector

os.environ['MallocStackLogging'] = '0'  # Prevent memory issues

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


class TestOptimalNumExamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up resources needed for all tests."""
        cls.results_dir = "test_results"
        os.makedirs(cls.results_dir, exist_ok=True)
        api_key = os.getenv("OPENAI_API_KEY")

        # Initialize LLM connector and recommender
        cls.llm = LLMConnector(provider="openai", model_name="gpt-4o-mini", api_key=api_key)
        cls.recommender = PromptRecommender(cls.llm)

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

        # Ensure we have enough data for all tests
        min_data_required = 30  # Max `num_examples` (10) + 20 fixed test samples
        assert len(cls.raw_texts) >= min_data_required, "Not enough data samples for testing!"

    def test_optimal_num_examples(self):
        """Finds how changing `num_examples` from 0 to 10 affects the similarity, while testing against 20 fixed examples."""
        print("\n### Running Optimal Num Examples Test ###")

        context = ("Input data is raw tokenized text, and the desired output consists of tokens with their Named Entity Recognition label. "
                   "The Tokens are labeled under one of the following labels [I-LOC, B-ORG, O, B-PER, I-PER, I-MISC, B-MISC, I-ORG, B-LOC]. "
                   "The goal is to label all the tokens with its NER label.")
        threshold = 0

        results = []

        for num_examples in range(0, 11):  # Test from 0 to 10 examples
            print(f"\n🔍 Testing with {num_examples} examples...")

            # Dynamically calculate the total number of examples
            total_examples = num_examples + 20  # Training + 20 fixed test samples

            input_subset = self.raw_texts[:total_examples]
            output_subset = self.formatted_results[:total_examples]

            # Always test with the next 20 samples after training examples
            test_data = [{"input_data": i, "desired_output": o} for i, o in zip(input_subset[:num_examples+20], output_subset[:num_examples+20])]

            # Run recommendation with the examples and test data
            recommendation = self.recommender.recommend(test_data, num_examples, context, threshold)

            # Extract syntactic similarity
            syntactic_similarity = recommendation.get("syntax_similarity", 0)
            print(f"📊 Syntactic Similarity for {num_examples} examples: {syntactic_similarity}")

            results.append({"num_examples": num_examples, "syntactic_similarity": syntactic_similarity})

        # Save results for graphing later
        file_path = os.path.join(self.results_dir, f"optimal_examples_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(file_path, "w") as json_file:
            json.dump(results, json_file, indent=4)

        print(f"\n📁 Results saved to: {file_path}")
        self.assertTrue(len(results) == 11, "Expected results for 0-10 examples.")


if __name__ == "__main__":
    unittest.main()
