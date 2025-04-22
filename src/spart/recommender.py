import re
from spart.llm_connector import LLMConnector
from spart.optimizer import PromptOptimizer
from spart.evaluator import PromptEvaluator
import pandas as pd
from typing import Any, Optional


class PromptRecommender:
    """
    A class for recommending and optimizing system prompts based on input-output examples.

    Attributes:
        llm (LLMConnector): A function or API connector for generating responses from a language model (LLM).
        evaluator (PromptEvaluator): An instance of PromptEvaluator for evaluating prompt effectiveness.
        optimizer (PromptOptimizer): An instance of PromptOptimizer for optimizing the system prompt.
        auto_confirm (bool): If True, optimization and confirmation proceed automatically without user intervention.
    """

    def __init__(self, llm: LLMConnector, auto_confirm: bool = True) -> None:
        """
        Initializes the PromptRecommender with a language model and an optional auto-confirmation flag.

        Args:
            llm (LLMConnector): The LLM function or API connector.
            auto_confirm (bool): Flag to enable or disable automatic confirmation. Defaults to True.
        """
        self.llm = llm
        self.evaluator = PromptEvaluator(self.llm)
        self.optimizer = PromptOptimizer(self.llm, auto_confirm=auto_confirm)
        self.auto_confirm = auto_confirm

    def extract_prompt_from_xml(self, response_text: str) -> str:
        """
        Extracts the optimized prompt from an XML response.

        Args:
            response_text (str): The raw response containing the optimized prompt.

        Returns:
            str: The extracted system prompt or the original response if extraction fails.
        """
        match = re.search(r"<prompt>(.*?)</prompt>", response_text, re.DOTALL)
        return match.group(1).strip() if match else response_text

    def confirm_with_user(self, token_count: int) -> bool:
        """
        Confirms with the user whether to proceed based on the token count.

        Args:
            token_count (int): The token count of the prompt.

        Returns:
            bool: True if the user confirms, False if the user declines.
        """
        if self.auto_confirm:
            print(
                f"Auto-confirm: Proceeding with the token count of {token_count}")
            return True
        else:
            print(
                f"The number of tokens being sent to the LLM is: {token_count}")
            response = input("Do you want to proceed? (y/n): ").strip().lower()
            return response == "y"

    def confirm_optimization_with_user(self) -> bool:
        """
        Confirms with the user whether to proceed with optimizing the prompt.

        Returns:
            bool: True if the user confirms, False if the user declines.
        """
        if self.auto_confirm:
            print("Auto-confirm: Proceeding with optimization.")
            return True
        else:
            response = input(
                "Optimization is recommended. Do you want to proceed with optimizing the prompt? (y/n): ").strip().lower()
            return response == "y"

    def recommend(
        self,
        examples: pd.DataFrame,
        num_examples: int,
        context: Optional[str] = None,
        similarity_threshold: float = 0.8,
        max_iterations: int = 3,
        semantic_similarity: bool = True,
        syntactic_similarity: bool = True
    ) -> dict[str, Any]:
        """
        Recommends an optimized system prompt based on provided examples.

        Args:
            examples (pd.DataFrame): A list of input-output example pairs.
            num_examples (int): The number of examples to use for generating the prompt.
            context (str, optional): Additional context for generating the prompt. Defaults to None.
            similarity_threshold (float, optional): The threshold for similarity scores. Defaults to 0.8.
            max_iterations (int, optional): The maximum number of optimization attempts. Defaults to 3.

        Returns:
            dict: A dictionary containing the recommendation and evaluation metrics.
        """
        dataframe = pd.DataFrame(examples)

        if num_examples > 0:
            input_column_for_prompt = dataframe.iloc[:num_examples, 0]
            output_column_for_prompt = dataframe.iloc[:num_examples, 1]
            input_column_for_evaluation = dataframe.iloc[num_examples:, 0]
            output_column_for_evaluation = dataframe.iloc[num_examples:, 1]
        else:
            input_column_for_prompt = []
            output_column_for_prompt = []
            input_column_for_evaluation = dataframe.iloc[:, 0]
            output_column_for_evaluation = dataframe.iloc[:, 1]

        context_prompt = f"\n**Context**: {context}" if context else ""

        meta_prompt = f'''
            <system>

                <role_definition>
                    You are an AI that specializes in generating system prompts for transforming raw inputs into structured outputs. Given example input-output pairs, your task is to derive a precise and functional system prompt that correctly guides a language model to perform the transformation.
                    Your goal is to infer the logic, rules, and structure that map inputs to outputs and construct a system prompt that accurately instructs an LLM to perform the same transformation on future data.
                </role_definition>

                <guidelines>
                    - **Extract Transformation Logic**: Identify the key transformations applied to the input to produce the output.
                    - **Generalize the Rule**: Ensure the generated system prompt captures the logic and structure of the transformation.
                    - **Be Domain-Specific**: The transformation might involve summarization, formatting, classification, rewriting, extraction, or another process—ensure the system prompt aligns with this purpose.
                    - **No Extra Explanation**: Do not describe the prompt-generation process; simply generate the system prompt that would perform the transformation.
                    - **Context**: If context is provided in <context> make sure to use that piece of information to make the LLM understand the task
                </guidelines>

                ---

                <instructions>
                    1. **Objective**: Generate a system prompt that enables an LLM to transform raw "Inputs" into structured "Desired Outputs" using inferred transformation rules.
                    2. **Derive Transformation Logic**: Analyze how the "Inputs" are being modified, formatted, or structured in the "Desired Outputs."
                    3. **Generalization**: Construct a system prompt that would allow an LLM to perform this transformation consistently on unseen data.
                    4. **Maintain Output Fidelity**: The generated system prompt should ensure outputs match the structure, format, and content of the provided "Desired Outputs" exactly.
                    5. **Structure**: Be careful with defining the structure of the output, make sure it follows the same special characters as "Desired Outputs", do not add or remove elements of the format. Double-check the exact structure.
                    6. **Prompt Skeleton**: You MUST use these tags as a base to build the transformation prompt:
                        <role_definition>(Describe the model's purpose (e.g., "You are an AI that specializes in...")</role_defintion>,
                        <guidelines>(High-level rules or principles for completing the task.)</guidelines>,
                        <instructions>(Detailed steps or actions the model should follow to complete the task)</instructions>,
                        <examples>(Show a short example inputs and expected outputs for the model)</examples>,
                        <context>(Provide relevant background or details about the task or input data)</context>,
                        <goal>(Define the intended outcome or objective of the task)</goal>
                    7. **Wrap in `<prompt>` and '<system> Tags**: The final system prompt should be enclosed in `<prompt><system>...</system></prompt>` tags.
                </instructions>

                <examples>
                    Inputs: `{input_column_for_prompt}`
                    Desired Outputs: `{output_column_for_prompt}`
                </examples>

                <context>
                    {context_prompt}
                </context>

                <goal>
                    Generate a system prompt that correctly transforms future instances of "Inputs" into the format of "Desired Outputs" using the inferred transformation logic. Make sure to follow the skeleton structure and enclose the entire prompt within the <prompt> tags.
                </goal>

                <prompt>
                    <system>
                        (Generate the transformation prompt here)
                    </system>
                </prompt>

            </system>
        '''

        token_count = self.llm._count_tokens(meta_prompt)
        if not self.confirm_with_user(token_count):
            print("Process aborted by user.")
            return None

        response = self.llm(meta_prompt)
        generated_prompt = self.extract_prompt_from_xml(response)

        avg_cosine_similarity, avg_rouge_score, outputs = self.evaluator.evaluate_similarity(
            input_column_for_evaluation,
            output_column_for_evaluation,
            generated_prompt,
            semantic_similarity,
            syntactic_similarity
        )

        recommendation = 'Recommended' if (
            (semantic_similarity and avg_cosine_similarity >= similarity_threshold) or
            (syntactic_similarity and avg_rouge_score >= similarity_threshold)
        ) else 'Optimize'

        print("\n📊 **Evaluation Results:**")
        print(f"🔹 Semantic Similarity: {avg_cosine_similarity}")
        print(f"🔹 Syntactic Similarity: {avg_rouge_score}")

        result = {
            'semantic_similarity': avg_cosine_similarity,
            'syntactic_similarity': avg_rouge_score,
            'recommended_prompt': generated_prompt,
            'prompt_outputs': outputs,
            'recommendation': recommendation
        }

        if result['recommendation'] == 'Optimize':
            print("\n⚠️ **Optimization Recommended**")
            if self.confirm_optimization_with_user():
                print(
                    f"🔄 Optimizing the prompt... (Max Attempts: {max_iterations})")
                optimization = self.optimizer.optimize_prompt(
                    generated_prompt,
                    examples,
                    num_examples,
                    similarity_threshold,
                    context,
                    max_iterations,
                    semantic_similarity,
                    syntactic_similarity,
                )

                optimization['semantic_similarity'] = optimization['semantic_similarity']
                optimization['syntactic_similarity'] = optimization['syntactic_similarity']

                print("\n✅ **Optimization Completed!**")
                print(
                    f"🔹 **Optimized Semantic Similarity**: {optimization['semantic_similarity']}")
                print(
                    f"🔹 **Optimized Syntactic Similarity**: {optimization['syntactic_similarity']}")

                result['optimization'] = optimization
            else:
                print(
                    "❌ **User declined optimization. Returning initial recommendation.**")

        return result
