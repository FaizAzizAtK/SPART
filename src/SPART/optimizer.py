import re
from SPART.evaluator import PromptEvaluator

class PromptOptimizer:
    def __init__(self, llm, auto_confirm=True):
        self.llm = llm  # Function used to connect to the language model (LLM)
        self.evaluator = PromptEvaluator(llm)  # Evaluator instance for similarity checks
        self.auto_confirm = auto_confirm

    def extract_prompt_from_xml(self, response_text):
        """Extracts the optimized prompt from the XML tags <prompt>...</prompt>."""
        match = re.search(r"<prompt>(.*?)</prompt>", response_text, re.DOTALL)
        return match.group(1).strip() if match else response_text  # Return cleaned prompt or raw response

    def optimise_prompt(self, generated_prompt, input_data, desired_output, num_examples, threshold=0.85, context=None, max_iterations=3, semantic_similarity=True, syntactic_similarity=True):
        """Iteratively optimizes the prompt based on performance of generated outputs."""

        if num_examples > 0:
            input_column_for_prompt = input_data[:num_examples]
            output_column_for_prompt = desired_output[:num_examples]
            # Remaining data for evaluation
            input_column_for_evaluation = input_data[num_examples:]  
            output_column_for_evaluation = desired_output[num_examples:]  
        else:
            input_column_for_prompt = []
            output_column_for_prompt = []
            input_column_for_evaluation = input_data
            output_column_for_evaluation = desired_output


        # **Initial Evaluation Before Optimization**
        print("\n🔍 **Evaluating Original Prompt...**")
        orig_semantic_sim, orig_syntactic_sim, orig_evaluation = self.evaluator.evaluate_similarity(
            input_column_for_evaluation, output_column_for_evaluation, generated_prompt
        )
        print(f"📊 **Original Prompt Evaluation:**\n🔹 **Semantic Similarity**: {orig_semantic_sim}\n🔹 **Syntax Similarity**: {orig_syntactic_sim}")

        attempt = 0
        best_prompt = generated_prompt
        best_semantic_sim = orig_semantic_sim
        best_syntactic_sim = orig_syntactic_sim
        best_evaluation = orig_evaluation

        while attempt < max_iterations:
            attempt += 1
            print(f"\n🔄 **Optimization Attempt {attempt}/{max_iterations}**")

            # **Pass similarity results to the meta-prompt**
            context_prompt = f"Context: {context}" if context else ""

            # Keep the original meta-prompt, only add similarity results
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
                            <user>(The user’s actual request for the task)</user>
                        6. **Wrap the Prompt Properly**: The final system prompt should be enclosed in `<prompt><system>...</system></prompt>`.
                    </instructions>

                    <original_prompt>
                        {generated_prompt}
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

                    <user>
                        Generate a system prompt that improves on the "Original Prompt" to ensure accurate transformation into "Desired Outputs."
                    </user>

                    <prompt>
                        <system>
                            (Generate the optimized system prompt here)
                        </system>
                    </prompt>
                </system>
            '''

            # Generate optimized prompt
            response = self.llm(meta_prompt)
            optimized_prompt = self.extract_prompt_from_xml(response)

            # **Evaluate Optimized Prompt**
            new_semantic_sim, new_syntactic_sim, new_evaluation = self.evaluator.evaluate_similarity(
                input_column_for_evaluation, output_column_for_evaluation, optimized_prompt
            )
            print(f"📊 **Optimized Prompt Evaluation:**\n🔹 **Semantic Similarity**: {new_semantic_sim}\n🔹 **Syntax Similarity**: {new_syntactic_sim}")

            # Check if the new prompt is better
            if (
                (semantic_similarity and new_semantic_sim >= threshold) and
                (syntactic_similarity and new_syntactic_sim >= threshold)
            ):
                print(f"\n✅ **Optimization successful!**\n🔹 **Final Semantic Similarity**: {new_semantic_sim}\n🔹 **Final Syntax Similarity**: {new_syntactic_sim}")
                return optimized_prompt, {
                    "semantic_similarity": new_semantic_sim,
                    "syntactic_similarity": new_syntactic_sim,
                    "evaluation_details": new_evaluation
                }

            # Keep the best-performing prompt
            if new_semantic_sim > best_semantic_sim or new_syntactic_sim > best_syntactic_sim:
                best_prompt = optimized_prompt
                best_semantic_sim = new_semantic_sim
                best_syntactic_sim = new_syntactic_sim
                best_evaluation = new_evaluation

            if self.auto_confirm:
                print("\n⚡ **Automatically optimizing further...**")
            else:
                user_input = input("Would you like to optimize further? (y/n): ").strip().lower()
                if user_input != 'y':
                    print("❌ **Stopping optimization. Returning best result so far.**")
                    return best_prompt, {
                        "semantic_similarity": best_semantic_sim,
                        "syntactic_similarity": best_syntactic_sim,
                        "evaluation_details": best_evaluation
                    }

        print(f"\n⚠️ **Max optimization attempts ({max_iterations}) reached. Returning best attempt.**")
        return best_prompt, {
            "semantic_similarity": best_semantic_sim,
            "syntactic_similarity": best_syntactic_sim,
            "evaluation_details": best_evaluation
        }
