import sys
import os
import pandas as pd
import unittest
import json
from datetime import datetime

# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from SPART.recommender import PromptRecommender
from SPART.local_llm_connector import LocalLLMConnector
from SPART.optimizer import PromptOptimizer

class TestRecommender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up resources needed for all tests."""
        # Create test_results directory if it doesn't exist
        cls.results_dir = "test_results"
        if not os.path.exists(cls.results_dir):
            os.makedirs(cls.results_dir)

        # Initialise a local LLM connector for testing
        cls.llm = LocalLLMConnector("llama3.1", 2400)
        cls.recommender = PromptRecommender(cls.llm)
        cls.optimizer = PromptOptimizer(cls.llm)

        # Example data for testing
        cls.data = {
            "input_data": [
                "My name is Faiz and I am 21",
                "Hello my name is John and I am of the age 29",
                "Hey! I'm Sarah and I'm 24 years old",
                "Greetings, I'm Alice, aged 35",
                "They call me Robert, and I am currently 42 years old",
                "My name is Emily and I am 19",
                "My name is Alex and I am 30",
                "Hello there, I’m Sophia and I’m 27 years old",
                "Hey! My name is Liam, and I’m currently 22",
                "Greetings, I am Ethan and I’m 25",
                "Hi, I’m Olivia and I am 26",
                "Hello! My name is William, and I am 32 years old",
                "Hey there, I’m James and I’m 28",
                "My name is Ava, and I am 23",
                "They call me Noah, and I am 31 years old",
                "Hi, I’m Isabella and I’m 29",
                "Greetings! My name is Mason, and I am 33",
                "Hey, I’m Mia and I’m 20 years old",
                "Hello! I am Benjamin, aged 34",
                "My name is Charlotte, and I am 22",
                "Hey! I’m Elijah and I’m 40 years old",
                "Hello, my name is Amelia and I am 21",
                "Greetings! I’m Logan, aged 37",
                "They call me Lucas, and I am 38 years old",
                "Hey! My name is Harper and I am 25",
                "Hello, I am Henry and I am 27 years old",
                "Hi, I’m Evelyn and I’m 36",
                "My name is Daniel, and I am 39",
                "They call me Jack, and I am 41 years old",
                "Hello there, I am Scarlett and I am 26",
                "Hi! My name is Owen, and I am 35",
                "Hey! I am Stella and I’m 28 years old",
                "Greetings, I’m Samuel, and I’m 30",
                "My name is Violet and I’m 19",
                "They call me Matthew, and I am 24",
                "Hello! I am Elijah, aged 33",
                "Hey there, I’m Layla and I’m 20",
                "Greetings! My name is David, and I am 45",
                "Hi, I’m Hazel and I’m 32",
                "Hello! My name is Andrew and I am 42",
                "My name is Eleanor, and I am 23",
                "Hey! I’m Joseph and I’m 31 years old",
                "Greetings, I am Aria and I’m 29",
                "My name is Thomas and I am 37",
                "Hey! I am Penelope and I am 22",
                "They call me Sebastian, and I am 34",
                "Hi! My name is Zoey, and I am 27",
                "Hello, I am Jackson and I am 26",
                "Greetings, I am Lillian and I’m 28",
                "Hey! My name is Christopher and I am 44",
                "My name is Lily, and I am 25",
                "Hello! I am Anthony, aged 36",
                "Hey there, I’m Grace and I’m 30",
                "My name is Dylan, and I am 40",
                "They call me Madison, and I am 21",
                "Hey! I’m Wyatt and I’m 38 years old",
                "Greetings! My name is Willow and I’m 19",
                "Hello, I am Luke and I am 23 years old",
                "Hi, my name is Aubrey and I am 33",
                "Hey there! I’m Gabriel, and I am 41 years old",
                "My name is Addison, and I am 26",
                "They call me Carter, and I am 35",
                "Hello! I’m Natalie and I’m 32 years old",
                "Hi, I am Julian and I’m 39",
                "My name is Savannah, and I am 20",
                "Hey! I’m Landon and I’m 28",
                "Greetings, I am Brooklyn and I am 27",
                "Hello there, I’m Evan and I’m 31 years old",
                "My name is Paisley, and I am 24",
                "Hey! I am Nicholas and I am 29",
                "They call me Aiden, and I am 37 years old",
                "Hello, I am Bella and I’m 22",
                "Greetings! My name is Hunter, and I am 44",
                "Hey there, I’m Skylar and I’m 19",
                "Hi, I am Nathan and I am 36",
                "They call me Serenity, and I am 25",
                "Hello! I’m Dominic and I am 40",
                "Hey! My name is Aurora, and I’m 23",
                "My name is Isaiah, and I am 30",
                "Hello! I am Autumn, aged 21",
                "Greetings! I’m Charles and I am 42",
                "Hey there, I’m Leah and I am 34",
                "My name is Eli, and I am 27",
                "Hello! I am Genesis, aged 28",
                "They call me Aaron, and I am 39 years old",
                "Greetings! My name is Sadie and I am 24",
                "Hello, I’m Adrian and I am 41 years old",
                "My name is Derek and I am 23",
                "Hello, I’m Emily and I’m 28",
                "Hey! I’m Mason, aged 31",
                "Greetings! My name is Claire, and I am 34",
                "They call me Adam, and I’m 27 years old",
                "My name is Lily and I am 33",
                "Hi, I’m Samuel, and I am 25",
                "Hello! I am Victoria, aged 22",
                "Greetings, I’m Ian and I’m 29",
                "My name is Alice and I am 36",
                "Hey there! I’m Michael, and I’m 30",
                "My name is Brooke, and I am 24",
                "Hello, I’m Brian and I am 35",
                "Hey! I’m Natalie and I’m 32 years old"
            ],
            "desired_output": [
                "Name: Faiz, Age: 21",
                "Name: John, Age: 29",
                "Name: Sarah, Age: 24",
                "Name: Alice, Age: 35",
                "Name: Robert, Age: 42",
                "Name: Emily, Age: 19",
                "Name: Alex, Age: 30",
                "Name: Sophia, Age: 27",
                "Name: Liam, Age: 22",
                "Name: Ethan, Age: 25",
                "Name: Olivia, Age: 26",
                "Name: William, Age: 32",
                "Name: James, Age: 28",
                "Name: Ava, Age: 23",
                "Name: Noah, Age: 31",
                "Name: Isabella, Age: 29",
                "Name: Mason, Age: 33",
                "Name: Mia, Age: 20",
                "Name: Benjamin, Age: 34",
                "Name: Charlotte, Age: 22",
                "Name: Elijah, Age: 40",
                "Name: Amelia, Age: 21",
                "Name: Logan, Age: 37",
                "Name: Lucas, Age: 38",
                "Name: Harper, Age: 25",
                "Name: Henry, Age: 27",
                "Name: Evelyn, Age: 36",
                "Name: Daniel, Age: 39",
                "Name: Jack, Age: 41",
                "Name: Scarlett, Age: 26",
                "Name: Owen, Age: 35",
                "Name: Stella, Age: 28",
                "Name: Samuel, Age: 30",
                "Name: Violet, Age: 19",
                "Name: Matthew, Age: 24",
                "Name: Elijah, Age: 33",
                "Name: Layla, Age: 20",
                "Name: David, Age: 45",
                "Name: Hazel, Age: 32",
                "Name: Andrew, Age: 42",
                "Name: Eleanor, Age: 23",
                "Name: Joseph, Age: 31",
                "Name: Aria, Age: 29",
                "Name: Thomas, Age: 37",
                "Name: Penelope, Age: 22",
                "Name: Sebastian, Age: 34",
                "Name: Zoey, Age: 27",
                "Name: Jackson, Age: 26",
                "Name: Lillian, Age: 28",
                "Name: Christopher, Age: 44",
                "Name: Lily, Age: 25",
                "Name: Anthony, Age: 36",
                "Name: Grace, Age: 30",
                "Name: Dylan, Age: 40",
                "Name: Madison, Age: 21",
                "Name: Wyatt, Age: 38",
                "Name: Willow, Age: 19",
                "Name: Luke, Age: 23",
                "Name: Aubrey, Age: 33",
                "Name: Gabriel, Age: 41",
                "Name: Addison, Age: 26",
                "Name: Carter, Age: 35",
                "Name: Natalie, Age: 32",
                "Name: Julian, Age: 39",
                "Name: Savannah, Age: 20",
                "Name: Landon, Age: 28",
                "Name: Brooklyn, Age: 27",
                "Name: Evan, Age: 31",
                "Name: Paisley, Age: 24",
                "Name: Nicholas, Age: 29",
                "Name: Aiden, Age: 37",
                "Name: Bella, Age: 22",
                "Name: Hunter, Age: 44",
                "Name: Skylar, Age: 19",
                "Name: Nathan, Age: 36",
                "Name: Serenity, Age: 25",
                "Name: Dominic, Age: 40",
                "Name: Aurora, Age: 23",
                "Name: Isaiah, Age: 30",
                "Name: Autumn, Age: 21",
                "Name: Charles, Age: 42",
                "Name: Leah, Age: 34",
                "Name: Eli, Age: 27",
                "Name: Genesis, Age: 28",
                "Name: Aaron, Age: 39",
                "Name: Sadie, Age: 24",
                "Name: Adrian, Age: 41",
                "Name: Derek, Age: 23",
                "Name: Emily, Age: 28",
                "Name: Mason, Age: 31",
                "Name: Claire, Age: 34",
                "Name: Adam, Age: 27",
                "Name: Lily, Age: 33",
                "Name: Samuel, Age: 25",
                "Name: Victoria, Age: 22",
                "Name: Ian, Age: 29",
                "Name: Alice, Age: 36",
                "Name: Michael, Age: 30",
                "Name: Brooke, Age: 24",
                "Name: Brian, Age: 35",
                "Name: Natalie, Age: 32"
            ]
        }


    def test_recommend(self):
        print("RECCOMENDATION TEST")
        """Test the recommend function and save results to a JSON file."""
        examples = [{"input": input_data, "output": desired_output} for input_data, desired_output in zip(self.data["input_data"], self.data["desired_output"])]
        num_examples = 2
        context = "Input is a series of people stating their name and their age. Desired output is a structured classification of their Name and their Age"

        # Call the recommend function
        results = self.recommender.recommend(examples=examples, num_examples=2, context="The input is type int", threshold=0.85, max_iterations=3, semantic_similarity=False, syntactic_similarity=True)

        # Write the results to a JSON file
        file_path = os.path.join(self.results_dir, f"recommend_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(file_path, "w") as json_file:
            json.dump(results, json_file, indent=4)

        print(f"Recommend test results saved to: {file_path}")
        self.assertTrue(len(results) > 0, "The recommend function returned no results.")


    def test_optimisation(self):
        """Test the optimisation process and save results to a JSON file."""
        print("OPTIMIZATION TEST")
        optimizer = self.optimizer
        generated_prompt = "Extract the names and ages from the inputes in the format \"Name: {{name}}, Age: {{age}}\""  
        examples = [{"input": input_data, "output": desired_output} for input_data, desired_output in zip(self.data["input_data"], self.data["desired_output"])]
        context="Input is a series of people stating their name and their age. Desired output is a structured classification of their Name and their Age"
        threshold = 0.85
        num_examples = 2

        # optimize the prompt
        optimized_prompt, similarity_metrics = optimizer.optimize_prompt(
            generated_prompt,
            examples[:4],
            num_examples,
            threshold,
            context,
            semantic_similarity=False,
            syntactic_similarity=True,
        )

        # Prepare the results to save
        results = {
            "original_prompt": generated_prompt,
            "optimized_prompt": optimized_prompt,
            "similarity_metrics": similarity_metrics,
        }

        # Write the results to a JSON file
        file_path = os.path.join(self.results_dir, f"optimisation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(file_path, "w") as json_file:
            json.dump(results, json_file, indent=4)

        print(f"Optimisation test results saved to: {file_path}")
        self.assertIsNotNone(optimized_prompt, "The optimisation process did not return a prompt.")
        self.assertTrue("semantic_similarity" in similarity_metrics and "syntactic_similarity" in similarity_metrics,
                        "Similarity metrics are incomplete.")


if __name__ == "__main__":
    unittest.main()
