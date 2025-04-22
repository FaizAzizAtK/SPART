import re
import pandas as pd
from spart.evaluator import PromptEvaluator
from spart.llm_connector import LLMConnector
from typing import Any, Optional


class PromptOptimizer:
    """
    A class for optimizing system prompts using iterative refinement based on similarity scores.

    Attributes:
        llm (LLMConnector): A function or API connector for generating responses from a language model (LLM).
        evaluator (PromptEvaluator): An instance for evaluating prompt effectiveness based on similarity.
        auto_confirm (bool): If True, optimization continues automatically without user confirmation.
    """

    def __init__(self, llm: LLMConnector, auto_confirm: bool = True):
        self.llm = llm
        self.evaluator = PromptEvaluator(llm)
        self.auto_confirm = auto_confirm

    def extract_prompt_from_xml(self, response_text: str) -> str:
        """
        Extracts the optimized prompt from the XML response.

        Args:
            response_text (str): The raw response containing the optimized prompt.

        Returns:
            str: The extracted prompt or the original response if extraction fails.
        """
        match = re.search(r"<prompt>(.*?)</prompt>", response_text, re.DOTALL)
        return match.group(1).strip() if match else response_text

    def optimize_prompt(
        self,
        meta_prompt: str,
        examples: pd.DataFrame,
        num_examples: int,
        threshold: float = 0.85,
        context: Optional[str] = None,
        max_iterations: int = 3,
        semantic_similarity: bool = True,
        syntactic_similarity: bool = True
    ) -> dict[str, Any]:
        """
        Optimizes a system prompt by iteratively refining it based on evaluation metrics.

        Args:
            meta_prompt (str): The initial system prompt.
            examples (list): Input-output pairs (as a DataFrame).
            num_examples (int): Number of examples used for prompt refinement.
            threshold (float, optional): Similarity threshold for stopping optimization. Defaults to 0.85.
            context (str, optional): Additional context for prompt optimization. Defaults to None.
            max_iterations (int, optional): Maximum optimization attempts. Defaults to 3.
            semantic_similarity (bool, optional): Whether to consider semantic similarity. Defaults to True.
            syntactic_similarity (bool, optional): Whether to consider syntactic similarity. Defaults to True.

        Returns:
            dict: Optimized prompt and evaluation results.
        """
        dataframe = pd.DataFrame(examples)

        input_column_for_prompt = dataframe.iloc[:num_examples, 0].tolist()
        output_column_for_prompt = dataframe.iloc[:num_examples, 1].tolist()
        input_column_for_evaluation = dataframe.iloc[num_examples:, 0].tolist()
        output_column_for_evaluation = dataframe.iloc[num_examples:, 1].tolist(
        )

        print("\n🔍 Evaluating Original Prompt...")
        orig_semantic_sim, orig_syntactic_sim, orig_evaluation = self.evaluator.evaluate_similarity(
            input_column_for_evaluation, output_column_for_evaluation, meta_prompt, semantic_similarity, syntactic_similarity)

        print(
            f"📊 Original Prompt Evaluation:\n🔹 Semantic Similarity: {orig_semantic_sim}\n🔹 Syntactic Similarity: {orig_syntactic_sim}")

        attempt = 0
        best_prompt = meta_prompt
        best_semantic_sim = orig_semantic_sim
        best_syntactic_sim = orig_syntactic_sim
        best_evaluation = orig_evaluation

        while attempt < max_iterations:
            attempt += 1
            print(f"\n🔄 Optimization Attempt {attempt}/{max_iterations}")

            context_prompt = f"Context: {context}" if context else ""

            meta_prompt = f'''
                <system>
                    <role_definition>
                        You are an AI specializing in refining system prompts to enhance the accuracy of input-to-output transformations. Your goal is to optimize a given system prompt using examples of input-output pairs.
                    </role_definition>

                    <guidelines>
                        - **Focus on Similarity Metrics**: Your only feedback comes from the provided similarity scores. Your refinements must increase **both semantic and syntactic similarity** while ensuring **strict format consistency**.
                        - **No Assumptions About Past Outputs**: You cannot see previous results; rely entirely on similarity scores to gauge performance.
                        - **Prioritize Structural Accuracy**: Ensure that outputs follow the structure of "Desired Outputs" exactly.
                        - **Minimize Variability**: If transformations are inconsistent, refine the prompt to enforce clearer constraints.
                        - **Improve Clarity & Constraints**: If similarity scores are low, the prompt likely lacks precision—make it more explicit.
                        - **Avoid Over-Specification**: Ensure the optimized prompt generalizes across **unseen inputs**, rather than just matching the provided examples.
                    </guidelines>

                    <instructions>
                        1. **Objective**: Modify the "Original Prompt" to increase **semantic similarity**, **syntactic similarity**, and **format adherence**.
                        2. **Identify Weaknesses**: Since you cannot see prior outputs, assume that **low similarity scores indicate deficiencies** in clarity, structure, or precision.
                        3. **Refine Step-by-Step**: Strengthen instructions **only** in areas that would logically improve similarity without over-constraining.
                        4. **Ensure Structural Consistency**: Outputs **must strictly follow** the structure of "Desired Outputs" (including formatting, special characters, etc.).
                        5. **Prompt Skeleton**: You MUST use these tags as a base to structure the optimized prompt:
                            <role_definition>(Describe the model's purpose, e.g., "You are an AI that specializes in...")</role_definition>,
                            <guidelines>(High-level rules or principles for completing the task.)</guidelines>,
                            <instructions>(Detailed steps or actions the model should follow to complete the task)</instructions>,
                            <examples>(Provide a few input-output examples)</examples>,
                            <context>(Relevant background or details about the task or input data)</context>,
                            <goal>(Define the intended outcome or objective of the task)</goal>
                        6. **Wrap the Prompt Properly**: The final system prompt should be enclosed in `<prompt><system>...</system></prompt>`.
                    </instructions>

                    <original_prompt>
                        {meta_prompt}
                    </original_prompt>

                    <good_examples>
                        Inputs: {input_column_for_prompt}
                        Desired Outputs: {output_column_for_prompt}
                    </good_examples>

                    <context>
                        {context_prompt}
                    </context>

                    <evaluation_results>
                        - **Original Semantic Similarity Score**: {orig_semantic_sim} (higher is better)
                        - **Original Syntactic Similarity Score**: {orig_syntactic_sim} (higher is better)
                        - Improve these scores by refining the prompt.
                    </evaluation_results>

                    <goal>
                        Generate a system prompt that improves on the "Original Prompt" to ensure accurate transformation into "Desired Outputs."
                    </goal>

                    <prompt>
                        <system>
                            (Generate the optimized system prompt here)
                        </system>
                    </prompt>
                </system>
            '''

            response = self.llm(meta_prompt)
            optimized_prompt = self.extract_prompt_from_xml(response)

            new_semantic_sim, new_syntactic_sim, new_evaluation = self.evaluator.evaluate_similarity(
                input_column_for_evaluation, output_column_for_evaluation, optimized_prompt, semantic_similarity, syntactic_similarity)

            print(
                f"📊 Optimized Prompt Evaluation:\n🔹 Semantic Similarity: {new_semantic_sim}\n🔹 Syntactic Similarity: {new_syntactic_sim}")

            if (
                (semantic_similarity and new_semantic_sim >= threshold and syntactic_similarity and new_syntactic_sim >= threshold) or
                (semantic_similarity and not syntactic_similarity and new_semantic_sim >= threshold) or
                (syntactic_similarity and not semantic_similarity and new_syntactic_sim >= threshold)
            ):
                print(
                    f"\n✅ Optimization successful!\n🔹 Final Semantic Similarity: {new_semantic_sim}\n🔹 Final Syntax Similarity: {new_syntactic_sim}")
                return {
                    "optimized_prompt": optimized_prompt,
                    "semantic_similarity": new_semantic_sim,
                    "syntactic_similarity": new_syntactic_sim,
                    "evaluation_details": new_evaluation
                }

            if new_semantic_sim > best_semantic_sim or new_syntactic_sim > best_syntactic_sim:
                best_prompt = optimized_prompt
                best_semantic_sim = new_semantic_sim
                best_syntactic_sim = new_syntactic_sim
                best_evaluation = new_evaluation

            if self.auto_confirm and attempt < max_iterations:
                print("\n⚡ Automatically optimizing further...")
            if self.auto_confirm is False:
                user_input = input(
                    "Would you like to optimize further? (y/n): ").strip().lower()
                if user_input != 'y':
                    print("❌ Stopping optimization. Returning best result so far.")
                    return {
                        "optimized_prompt": best_prompt,
                        "semantic_similarity": best_semantic_sim,
                        "syntactic_similarity": best_syntactic_sim,
                        "evaluation_details": best_evaluation
                    }

        print(
            f"\n⚠️ Max optimization attempts ({max_iterations}) reached. Returning best attempt.")
        return {
            "optimized_prompt": best_prompt,
            "semantic_similarity": best_semantic_sim,
            "syntactic_similarity": best_syntactic_sim,
            "evaluation_details": best_evaluation
        }
