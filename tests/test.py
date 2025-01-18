import sys
import os
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from SPART.recommender import Recommender
from SPART.local_llm_connector import connect_local_llm

class TestRecommender(unittest.TestCase):
    def test_connect_llm(self):
        model_name = "llama3.1"
        llm_function = connect_local_llm(model_name)
        result = llm_function("Test prompt")  # Call the function with a prompt
        print(f"connect_local_llm result: {result}")  # Print the result
        self.assertIsInstance(result, str)

    def test_recommender(self):
        model_name = "llama3.1"
        llm_function = connect_local_llm(model_name)  # Connect to the actual LLM
        recommender = Recommender(llm_function)  # Pass the LLM function to the recommender

        data = {
            "input_data": [
                "My name is Faiz and I am 21",
                "Hello my name is John and I am of the age 29",
                "Hey! I'm Sarah and I'm 24 years old",
                "Greetings, I'm Alice, aged 35",
                "They call me Robert, and I am currently 42 years old",
                "My name is Emily and I am 19",
            ],
            "desired_outputs": [
                "Name: Faiz, Age: 21",
                "Name: John, Age: 29",
                "Name: Sarah, Age: 24",
                "Name: Alice, Age: 35",
                "Name: Robert, Age: 42",
                "Name: Emily, Age: 19",
            ]
        }

        # Call the recommender
        result = recommender.recommend(data, 3)  # 3 iterations
        print(f"recommend result: {result}")  # Print the result

        # Iterate through the list and check each result
        for res in result:
            self.assertIn('similarity', res)
            self.assertIn('recommended_prompt', res)
            self.assertGreaterEqual(res['similarity'], 0)
            self.assertLessEqual(res['similarity'], 1)


if __name__ == '__main__':
    unittest.main()
