from sentence_transformers import SentenceTransformer, util
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

class Recommender:
    def __init__(self, llm):
        self.llm = llm
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def _generate_similarity_score(self, input_data, desired_output, generalized_prompt):
        # Create a recommended prompt specific to this input data
        recommended_prompt = f'''
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful AI assistant.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            1. {generalized_prompt}
            2. {{input_data}} = {input_data}
            3. Only return the desired output. Do not include any introductory or conclusive text.
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        '''

        # Generate the output using the LLM for this specific input
        prompt_output = self.llm(recommended_prompt)

        # Calculate similarity between the desired output and generated output
        desired_embedding = self.model.encode(desired_output, convert_to_tensor=True)
        prompt_embedding = self.model.encode(prompt_output, convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(desired_embedding, prompt_embedding).item()

        return cosine_similarity, prompt_output

    def recommend(self, examples, no_of_iterations):
        dataframe = pd.DataFrame(examples)
        input_column = dataframe.iloc[:, 0]
        output_column = dataframe.iloc[:, 1]

        # Generate a generalized prompt
        generalized_prompt = f'''
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            Given input and output data, produce a detailed system prompt to guide a language model in completing the task effectively.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            1. Generate a generalized prompt that instructs an LLM to transform the given "Inputs" into the "Desired Outputs." Ensure the prompt is not overly specific to "Inputs," but is designed to reliably produce the exact "Desired Outputs". 
            2. The format of the generated output should exactly match the "Desired Outputs", ensuring the structure is specific and exact to the example including, explicitly mentioning same special characters.
            3. The recommended prompt should include a reference to the input variable {{input_data}}.
            4. Do not include anything but the prompt.
            5. Treat each row as a separate element in "Inputs" and "Desired Outputs"
            Inputs: {input_column} 
            Desired Outputs: {output_column}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        '''
        generated_prompt = self.llm(generalized_prompt)

        all_iterations_results = []

        def run_iteration():
            similarity_scores = []
            outputs = []
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self._generate_similarity_score, input_data, desired_output, generated_prompt)
                    for input_data, desired_output in zip(input_column, output_column)
                ]
                results = [future.result() for future in futures]
                similarity_scores = [result[0] for result in results]
                outputs = [result[1] for result in results]

            average_similarity = sum(similarity_scores) / len(similarity_scores)
            return {
                'similarity': average_similarity,
                'recommended_prompt': generated_prompt,
                'prompt_outputs': outputs
            }

        # Execute n iterations in parallel
        with ThreadPoolExecutor() as executor:
            all_iterations_results = list(executor.map(lambda _: run_iteration(), range(no_of_iterations)))

        return all_iterations_results
