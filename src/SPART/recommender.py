

import re
from SPART.optimizer import PromptOptimizer
from SPART.evaluator import PromptEvaluator
import pandas as pd

class PromptRecommender:
    def __init__(self, llm, auto_confirm=True):
        self.llm = llm
        self.evaluator = PromptEvaluator(self.llm)
        self.optimizer = PromptOptimizer(self.llm, auto_confirm=auto_confirm)
        self.auto_confirm = auto_confirm

    def extract_prompt_from_xml(self, response_text):
        match = re.search(r"<prompt>(.*?)</prompt>", response_text, re.DOTALL)
        return match.group(1).strip() if match else response_text  

    def confirm_with_user(self, token_count):
        if self.auto_confirm:
            print(f"Auto-confirm: Proceeding with the token count of {token_count}")
            return True
        else:
            print(f"The number of tokens being sent to the LLM is: {token_count}")
            response = input("Do you want to proceed? (y/n): ").strip().lower()
            return response == "y"

    def confirm_optimization_with_user(self):
        if self.auto_confirm:
            print("Auto-confirm: Proceeding with optimization.")
            return True
        else:
            response = input("Optimization is recommended. Do you want to proceed with optimizing the prompt? (y/n): ").strip().lower()
            return response == "y"

    def recommend(self, examples, num_examples, context=None, similarity_threshold=0.8, max_iterations=3):
        dataframe = pd.DataFrame(examples)
        

        if num_examples > 0:
            input_column_for_prompt = dataframe.iloc[:num_examples, 0]
            output_column_for_prompt = dataframe.iloc[:num_examples, 1]
            # Remaining data for evaluation
            input_column_for_evaluation = dataframe.iloc[num_examples:, 0]  
            output_column_for_evaluation = dataframe.iloc[num_examples:, 1]  
        else:
            input_column_for_prompt = []
            output_column_for_prompt = []
            input_column_for_evaluation = dataframe.iloc[:, 0]  
            output_column_for_evaluation = dataframe.iloc[:, 1]  


        context_prompt = f"\n**Context**: {context}" if context else ""

        # ✅ Your original meta-prompt remains **unchanged**
        generalized_prompt = f'''
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
                        <user>(Provide the user’s actual request or input for the task)</user>
                    7. **Wrap in `<prompt>` and '<system> Tags**: The final system prompt should be enclosed in `<prompt><system>...</system></prompt>` tags.
                </instructions>

                <examples>
                    Inputs: `{input_column_for_prompt}`
                    Desired Outputs: `{output_column_for_prompt}`
                </examples>

                <context>
                    {context_prompt}
                </context>

                <user>
                    Generate a system prompt that correctly transforms future instances of "Inputs" into the format of "Desired Outputs" using the inferred transformation logic. Make sure to follow the skeleton structure and enclose the entire prompt within the <prompt> tags.
                </user>

                <prompt>
                    <system>
                        (Generate the transformation prompt here)
                    </system>
                </prompt>

            </system>
        '''

        token_count = self.evaluator.count_tokens(generalized_prompt)
        if not self.confirm_with_user(token_count):
            print("Process aborted by user.")
            return None

        response = self.llm(generalized_prompt)
        generated_prompt = self.extract_prompt_from_xml(response)

        avg_cosine_similarity, avg_rouge_score, outputs = self.evaluator.evaluate_similarity(
            input_column_for_evaluation,  
            output_column_for_evaluation,  
            generated_prompt,
            use_semantic_similarity=self.evaluator.use_semantic_similarity,
            use_syntax_similarity=self.evaluator.use_syntax_similarity
        )

        recommendation = 'Recommended' if (
            (self.evaluator.use_semantic_similarity and avg_cosine_similarity >= similarity_threshold) or
            (self.evaluator.use_syntax_similarity and avg_rouge_score >= similarity_threshold)
        ) else 'Optimize'

        # ✅ Always print evaluation results before returning
        print("\n📊 **Evaluation Results:**")
        print(f"🔹 **Semantic Similarity**: {avg_cosine_similarity}")
        print(f"🔹 **Syntax Similarity**: {avg_rouge_score}")
        print(f"📝 **Recommendation**: {recommendation}")

        result = {
            'semantic_similarity': avg_cosine_similarity,
            'syntax_similarity': avg_rouge_score,
            'recommended_prompt': generated_prompt,
            'prompt_outputs': outputs,
            'recommendation': recommendation
        }

        if result['recommendation'] == 'Optimize':
            print("\n⚠️ **Optimization Recommended**")
            if self.confirm_optimization_with_user():
                print(f"🔄 Optimizing the prompt... (Max Attempts: {max_iterations})")
                optimized_prompt, optimization_metrics = self.optimizer.optimise_prompt(
                    generated_prompt, 
                    input_column_for_evaluation,  
                    output_column_for_evaluation,  
                    num_examples,
                    similarity_threshold,
                    context=context,
                    semantic_similarity=self.evaluator.use_semantic_similarity,
                    syntactic_similarity=self.evaluator.use_syntax_similarity,
                    max_iterations=max_iterations
                )

                print("\n✅ **Optimization Completed!**")
                print(f"🔹 **Optimized Semantic Similarity**: {optimization_metrics['semantic_similarity']}")
                print(f"🔹 **Optimized Syntax Similarity**: {optimization_metrics['syntactic_similarity']}")

                result['optimized_prompt'] = optimized_prompt
                result['optimization_metrics'] = optimization_metrics
            else:
                print("❌ **User declined optimization. Returning initial recommendation.**")

        return result
