from concurrent.futures import ThreadPoolExecutor
from transformers import T5Tokenizer
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PromptEvaluator:
    def __init__(self, llm, use_semantic_similarity=True, use_syntax_similarity=True, similarity_threshold=0.8):
        self.llm = llm
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.use_semantic_similarity = use_semantic_similarity
        self.use_syntax_similarity = use_syntax_similarity
        self.similarity_threshold = similarity_threshold

    def count_tokens(self, text):
        """Count the number of tokens in the given text."""
        return len(self.tokenizer.encode(text))

    def _generate_similarity_score(self, input_data, desired_output, generated_prompt, use_semantic_similarity, use_syntax_similarity):
        recommended_prompt = f'''
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful AI assistant.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            1. {generated_prompt}
            2. {{input_data}} = {input_data}
            3. Only return the desired output. Do not include any introductory or conclusive text.
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        '''

        prompt_output = self.llm(recommended_prompt)

        # Initialize similarity scores
        cosine_similarity = 0
        rouge_score = 0

        if use_semantic_similarity:
            # Calculate cosine similarity (semantic)
            desired_embedding = self.model.encode(desired_output, convert_to_tensor=True)
            prompt_embedding = self.model.encode(prompt_output, convert_to_tensor=True)
            cosine_similarity = util.pytorch_cos_sim(desired_embedding, prompt_embedding).item()
            cosine_similarity = min(max(cosine_similarity, 0), 1)

        if use_syntax_similarity:
            # Calculate ROUGE-L similarity (syntax)
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge_score = scorer.score(desired_output, prompt_output)['rougeL'].fmeasure

        return cosine_similarity, rouge_score, prompt_output


    def evaluate_similarity(self, input_column, output_column, generated_prompt, use_semantic_similarity=True, use_syntax_similarity=True):
        cosine_similarities = []
        rouge_scores = []
        outputs = []
        
        # Use multithreading to evaluate the similarity of each input/output pair
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._generate_similarity_score, input_data, desired_output, generated_prompt, use_semantic_similarity, use_syntax_similarity)
                for input_data, desired_output in zip(input_column, output_column)
            ]
            results = [future.result() for future in futures]
            cosine_similarities = [result[0] for result in results]
            rouge_scores = [result[1] for result in results]
            outputs = [result[2] for result in results]
        
        avg_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities) if use_semantic_similarity else 0
        avg_rouge_score = sum(rouge_scores) / len(rouge_scores) if use_syntax_similarity else 0
        
        return avg_cosine_similarity, avg_rouge_score, outputs

