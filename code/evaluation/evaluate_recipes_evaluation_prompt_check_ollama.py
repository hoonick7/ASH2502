import csv
import time
import random
import os
import logging
import concurrent.futures
import torch
import multiprocessing
from functools import partial
from datetime import timedelta
from langchain_ollama import ChatOllama
from loguru import logger
import re

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class RecipeEvaluator:
    # model_names = ["gemma2:2b", "gemma2:9b", "mistral:7b", "llama2:13b", "llama3.1:8b"]
    model_names = ["gemma2:9b", "mistral:7b", "llama3.1:8b", "llama3.2", "phi4"] # llama3.2 is 3b
    prompts = {
        # Prompt 1: Default
        1: """Evaluate the following recipe:

Original Dish: {original_dish}
Variation: {variation}
Generated Recipe:
{generated_recipe}

Please rate on a scale of 1-5 (integer values only) and provide a reason for each:
1. AUTHENTICITY: Rate how well the recipe preserves the essential characteristics of the original dish.
2. SENSITIVITY: Rate how well the recipe incorporates the target variation while maintaining relevance.
3. HARMONY: Rate how well the recipe balances authenticity and sensitivity to create a cohesive dish.
Format:
AUTHENTICITY: [rating]\nReason: [reason]\nSENSITIVITY: [rating]\nReason: [reason]\nHARMONY: [rating]\nReason: [reason]""",

        # Prompt 2: Assign role
        2: """Imagine you are a chef evaluating the quality of a recipe created by an AI assistant:

Original Dish: {original_dish}
Variation: {variation}
Generated Recipe:
{generated_recipe}

Evaluate this recipe based on the following criteria, providing a score (1-5) and detailed reasoning:
1. AUTHENTICITY: Does the generated recipe retain the cultural and culinary identity of the original dish? Consider factors such as ingredient choice and preparation method.
2. SENSITIVITY: Does the recipe align well with the requested variation? Assess the recipe's ability to incorporate new elements without losing coherence.
3. HARMONY: Is the overall dish harmonious and appealing? Evaluate whether the final recipe would work well in a real-world dining scenario.

Please answer in the following format:
AUTHENTICITY: [rating]\nReason: [reason]\nSENSITIVITY: [rating]\nReason: [reason]\nHARMONY: [rating]\nReason: [reason]""",

        # Prompt 3: Scoring scale
        3: """Recipe Assessment:

Original Dish: {original_dish}
Variation: {variation}
Generated Recipe:
{generated_recipe}

Using a scale of 1-5 (integer values only), evaluate the recipe on the following criteria:
1. AUTHENTICITY: How well does this recipe align with the original dish's core features?
    - Example: If the generated recipe is drastically different from the original dish, assign a score of 1.
      If it closely adheres to the original dish, assign a score of 5.
2. SENSITIVITY: How effectively does this recipe adapt to the specified variation?
    - Example: If the recipe simply lists relevant ingredients without proper integration, assign a score of 1.
      If it uses appropriate ingredients and techniques to meet the variation, assign a score of 5.
3. HARMONY: How balanced is the integration of authenticity and sensitivity in the recipe?
    - Example: If the recipe results in a logical and well-balanced dish, assign a score of 5.
      If the variation feels forced or incoherent, assign a score of 1.

For each score, provide detailed reasoning in this format:
AUTHENTICITY: [rating]\nReason: [reason]\nSENSITIVITY: [rating]\nReason: [reason]\nHARMONY: [rating]\nReason: [reason]""",

        # Prompt 4: Chain of Thought (CoT)
        4: """Evaluate the recipe step by step using logical reasoning:

Original Dish: {original_dish}
Variation: {variation}
Generated Recipe:
{generated_recipe}

Step 1: Identify the key characteristics of the original dish.
    - List the traditional ingredients and methods that define the dish.
    - Explain how these are preserved or altered in the generated recipe.

Step 2: Evaluate the adaptation to the requested variation.
    - Consider how effectively the recipe incorporates the variation's requirements.
    - Analyze the impact of these changes on the dish's integrity.

Step 3: Assess the overall harmony of the dish.
    - Reflect on how authenticity and adaptation interact to create a cohesive result.
    - Provide a balanced judgment of the final recipe's appeal.

For each score, consider the following criteria:
1. AUTHENTICITY: Rate how well the recipe preserves the essential characteristics of the original dish.
2. SENSITIVITY: Rate how well the recipe incorporates the target variation while maintaining relevance.
3. HARMONY: Rate how well the recipe balances authenticity and sensitivity to create a cohesive dish.

Summarize your findings with scores (1-5) and reasons for each:
AUTHENTICITY: [rating]\nReason: [reason]\nSENSITIVITY: [rating]\nReason: [reason]\nHARMONY: [rating]\nReason: [reason]""",

        # Prompt 5: Chain of Thought + Chef Simulation
        5: """Evaluate the recipe step by step with a professional chef's perspective:

Original Dish: {original_dish}
Variation: {variation}
Generated Recipe:
{generated_recipe}

Step 1: Analyze the original dish from a culinary standpoint.
    - Highlight the core techniques and flavors unique to the dish.
    - Examine how these are retained or modified in the recipe.

Step 2: Evaluate the recipe's response to the variation.
    - Assess the creativity and feasibility of the adaptation.
    - Consider whether the changes align with culinary principles.

Step 3: Judge the dish's overall success in a real-world context.
    - Reflect on its balance, presentation, and potential taste.
    - Determine its suitability for serving as intended.

For each score, consider the following criteria:
1. AUTHENTICITY: Rate how well the recipe preserves the essential characteristics of the original dish.
2. SENSITIVITY: Rate how well the recipe incorporates the target variation while maintaining relevance.
3. HARMONY: Rate how well the recipe balances authenticity and sensitivity to create a cohesive dish.

Provide scores (1-5) and detailed reasoning for each category:
AUTHENTICITY: [rating]\nReason: [reason]\nSENSITIVITY: [rating]\nReason: [reason]\nHARMONY: [rating]\nReason: [reason]""",


        # Prompt 6: Chain of Thought + Scoring Guidance
        6: """Evaluate the recipe step by step, guided by scoring criteria:

Original Dish: {original_dish}
Variation: {variation}
Generated Recipe:
{generated_recipe}

Step 1: Examine how the recipe aligns with the original dish.
    - Does it preserve the dish's essence? (Score 1-5)
    - Provide examples of alignment or deviation.

Step 2: Assess the recipe's adaptation to the variation.
    - Does it meet the variation's goals effectively? (Score 1-5)
    - Highlight strengths and weaknesses in execution.

Step 3: Evaluate the overall balance and coherence.
    - Does the recipe feel complete and harmonious? (Score 1-5)
    - Discuss how well authenticity and adaptation are integrated.

For each score, consider the following criteria:
1. AUTHENTICITY: Does the recipe preserve the original dish's key ingredients and techniques?
    - Score 1: Significant deviations from the original dish's characteristics.
    - Score 5: Strong adherence to the original dish's identity.
2. SENSITIVITY: How well does the recipe adapt to the requested variation?
    - Score 1: Poor integration of the variation.
    - Score 5: Creative and effective incorporation of the variation.
3. HARMONY: Is the recipe cohesive and appealing overall?
    - Score 1: The recipe lacks coherence or feels incomplete.
    - Score 5: The recipe is well-balanced and appealing.

Conclude with scores and reasons for each category:
AUTHENTICITY: [rating]\nReason: [reason]\nSENSITIVITY: [rating]\nReason: [reason]\nHARMONY: [rating]\nReason: [reason]""",

        # Prompt 7: Chain of Thought + Chef Simulation + Scoring Guidance
        7: """Evaluate the recipe comprehensively with a chef's perspective and scoring guidance:

Original Dish: {original_dish}
Variation: {variation}
Generated Recipe:
{generated_recipe}

Step 1: Analyze the original dish.
    - Identify its cultural and culinary significance.
    - Evaluate how well the recipe reflects these attributes (Score 1-5).

Step 2: Assess the variation's implementation.
    - Examine the creativity, relevance, and technical execution of the adaptation (Score 1-5).

Step 3: Judge the overall harmony.
    - Consider the dish's presentation, taste potential, and cohesion (Score 1-5).

For each score, consider the following criteria:
1. AUTHENTICITY: How well does the recipe reflect the core essence of the original dish?
    - Score 1: The essence of the original dish is poorly represented.
    - Score 5: The original dish's core essence is clearly preserved.
2. SENSITIVITY: Does the recipe effectively address the requested variation?
    - Score 1: The adaptation is irrelevant or poorly executed.
    - Score 5: The variation is creatively and effectively integrated.
3. HARMONY: Is the recipe cohesive, logical, and appealing?
    - Score 1: The recipe lacks balance or coherence.
    - Score 5: The recipe is harmonious and appealing.

Provide detailed reasoning for each score:
AUTHENTICITY: [rating]\nReason: [reason]\nSENSITIVITY: [rating]\nReason: [reason]\nHARMONY: [rating]\nReason: [reason]""",

        # Prompt 8: Chain of Thought + Chef Simulation + Scoring Guidance + Self-reflection
        8: """Evaluate the recipe comprehensively, including self-reflection on your evaluation:

Original Dish: {original_dish}
Variation: {variation}
Generated Recipe:
{generated_recipe}

Step 1: Analyze the original dish and its representation in the recipe.
    - Discuss the cultural and culinary essence and how it is preserved or altered (Score 1-5).

Step 2: Evaluate the recipe's adaptation to the variation.
    - Analyze the creativity and technical execution of the variation (Score 1-5).

Step 3: Assess the overall harmony.
    - Reflect on the balance and coherence between authenticity and adaptation (Score 1-5).

Step 4: Reflect on your evaluation process.
    - Did your reasoning align with the provided criteria?
    - Were there any assumptions or biases that influenced your judgment?
    - How could the evaluation process be improved?

For each score, consider the following criteria:
1. AUTHENTICITY: How well does the recipe preserve the cultural identity of the original dish?
    - Score 1: Poor preservation of cultural identity.
    - Score 5: Excellent preservation of cultural identity.
2. SENSITIVITY: Does the recipe creatively and appropriately adapt to the requested variation?
    - Score 1: The variation is poorly integrated.
    - Score 5: The variation is seamlessly and creatively incorporated.
3. HARMONY: How well does the recipe achieve balance and appeal?
    - Score 1: The recipe lacks coherence and balance.
    - Score 5: The recipe is cohesive, balanced, and appealing.

Summarize with scores and detailed reasoning:
AUTHENTICITY: [rating]\nReason: [reason]\nSENSITIVITY: [rating]\nReason: [reason]\nHARMONY: [rating]\nReason: [reason]\nREFLECTION: [reflection]"""
    }

    def __init__(self, output_filename='evaluated_recipes.csv'):
        self.output_filename = output_filename
        self.fieldnames = [
            'index', 'model', 'original_dish', 'variation', 'generated_recipe',
            'prompt_index', 'evaluator_model', 'evaluation', 'authenticity_score', 'authenticity_reason',
            'sensitivity_score', 'sensitivity_reason', 'harmony_score', 'harmony_reason', 'reflection'
        ]
        if not os.path.exists(self.output_filename):
            with open(self.output_filename, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                writer.writeheader()

        # Get available GPU count
        self.num_gpus = torch.cuda.device_count()
        logger.info(f"Found {self.num_gpus} GPUs")

    def evaluate_recipe(self, model_name, original_dish, variation, generated_recipe, prompt_index, gpu_id):
        # Configure Ollama to use specific GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        llm = ChatOllama(model=model_name)
        
        prompt = self.prompts[prompt_index].format(
            original_dish=original_dish, 
            variation=variation, 
            generated_recipe=generated_recipe
        )

        try:
            result = llm.invoke(prompt)
            result_text = result.content if isinstance(result, dict) else (result.text if hasattr(result, 'text') else str(result))
            logger.info(f"GPU {gpu_id}: Evaluated recipe for {original_dish} with {model_name}")
            return result_text
        except Exception as e:
            logger.error(f"GPU {gpu_id}: Error evaluating recipe for {original_dish} with {model_name}: {str(e)}")
            return f"Error: {str(e)}"
        
    def sort_results(self, filename):
        # Read the CSV file
        with open(filename, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            rows = list(reader)
        
        # Sort rows by index, prompt_index, and evaluator_model
        sorted_rows = sorted(rows, key=lambda x: (
            int(x['index']),
            int(x['prompt_index']), 
            self.model_names.index(x['evaluator_model'])
        ))
        
        # Write back to the file
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(sorted_rows)
        
        logger.info(f"Results sorted and saved to {filename}")

    def parse_evaluation(self, evaluation):
        if not evaluation or isinstance(evaluation, float):
            return {key: None for key in ['authenticity_score', 'authenticity_reason', 
                                        'sensitivity_score', 'sensitivity_reason', 
                                        'harmony_score', 'harmony_reason', 'reflection']}
        
        try:
            # Extract only the content part from LLM response
            if "content=" in evaluation:
                evaluation = evaluation.split("content=")[1].split(" additional_kwargs=")[0].strip('"')
            
            # Clean up markdown formatting and escaped quotes
            evaluation = evaluation.replace('\\"', '"').replace('\\n', '\n')
            
            patterns = [
                # Patterns for AUTHENTICITY
                (r'\*\*AUTHENTICITY:\*\*\s*(\d+)', 'authenticity_score'),  # Bold format
                (r'AUTHENTICITY:\s*(\d+)', 'authenticity_score'),          # Plain text
                (r'\*\*AUTHENTICITY:?\*\*.*?\*\*Reason:\*\*\s*(.*?)(?=\n*\*\*SENSITIVITY\*\*|\n*SENSITIVITY:|$)', 'authenticity_reason'),
                (r'AUTHENTICITY:.*?Reason:\s*(.*?)(?=\n*SENSITIVITY|$)', 'authenticity_reason'),
                
                # Patterns for SENSITIVITY
                (r'\*\*SENSITIVITY:\*\*\s*(\d+)', 'sensitivity_score'),    # Bold format
                (r'SENSITIVITY:\s*(\d+)', 'sensitivity_score'),            # Plain text
                (r'\*\*SENSITIVITY:?\*\*.*?\*\*Reason:\*\*\s*(.*?)(?=\n*\*\*HARMONY\*\*|\n*HARMONY:|$)', 'sensitivity_reason'),
                (r'SENSITIVITY:.*?Reason:\s*(.*?)(?=\n*HARMONY|$)', 'sensitivity_reason'),
                
                # Patterns for HARMONY
                (r'\*\*HARMONY:\*\*\s*(\d+)', 'harmony_score'),           # Bold format
                (r'HARMONY:\s*(\d+)', 'harmony_score'),                   # Plain text
                (r'\*\*HARMONY:?\*\*.*?\*\*Reason:\*\*\s*(.*?)(?=\n*\*\*REFLECTION\*\*|\n*REFLECTION:|$)', 'harmony_reason'),
                (r'HARMONY:.*?Reason:\s*(.*?)(?=\n*REFLECTION|\n\n|$)', 'harmony_reason'),
                
                # Pattern for REFLECTION
                (r'\*\*REFLECTION:?\*\*.*?\*\*Reason:\*\*\s*(.*?)(?=$)', 'reflection'),
                (r'REFLECTION:.*?Reason:\s*(.*?)(?=$)', 'reflection')
            ]

            result = {}
            for pattern, key in patterns:
                match = re.search(pattern, evaluation, re.DOTALL | re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    # Convert scores to integers
                    if '_score' in key:
                        try:
                            value = int(value)
                            if not (1 <= value <= 5):
                                value = None
                        except (ValueError, TypeError):
                            value = None
                    # Clean up text for reasons and reflection
                    elif '_reason' in key or key == 'reflection':
                        # Remove markdown formatting, multiple newlines, and extra whitespace
                        value = re.sub(r'\*\*|\n{2,}', ' ', value)
                        value = ' '.join(value.split())
                else:
                    value = None
                result[key] = value

            return result
            
        except Exception as e:
            logger.error(f"Error parsing evaluation: {str(e)}")
            # Return all fields as None if parsing fails
            return {key: None for key in ['authenticity_score', 'authenticity_reason', 
                                        'sensitivity_score', 'sensitivity_reason', 
                                        'harmony_score', 'harmony_reason', 'reflection']}

    # def evaluate_recipes(self, input_filename):
    #     start_time = time.time()
    #     with open(input_filename, 'r', newline='', encoding='utf-8', errors='replace') as file:
    #         reader = csv.DictReader(file)
    #         for index, row in enumerate(reader, start=1):
    #             elapsed_time = time.time() - start_time
    #             elapsed_str = str(timedelta(seconds=int(elapsed_time)))
    #             logger.info(f"Evaluating recipe {index}: {row['original_dish']} (Elapsed Time: {elapsed_str})")
    #             for prompt_index, prompt in self.prompts.items():
    #                 for model_name in self.model_names:
    #                     evaluation = self.evaluate_recipe(
    #                         model_name, 
    #                         row['original_dish'], 
    #                         row['variation'], 
    #                         row['generated_recipe'], 
    #                         prompt_index
    #                     )
    #                     parsed_evaluation = self.parse_evaluation(evaluation)

    #                     new_row = {
    #                         'index': row['index'],
    #                         'model': row['model'],
    #                         'original_dish': row['original_dish'],
    #                         'variation': row['variation'],
    #                         'generated_recipe': row['generated_recipe'],
    #                         'prompt_index': prompt_index,
    #                         'evaluator_model': model_name,
    #                         'evaluation': evaluation,
    #                         'authenticity_score': parsed_evaluation['authenticity_score'],
    #                         'authenticity_reason': parsed_evaluation['authenticity_reason'],
    #                         'sensitivity_score': parsed_evaluation['sensitivity_score'],
    #                         'sensitivity_reason': parsed_evaluation['sensitivity_reason'],
    #                         'harmony_score': parsed_evaluation['harmony_score'],
    #                         'harmony_reason': parsed_evaluation['harmony_reason'],
    #                         'reflection': parsed_evaluation.get('reflection', None)
    #                     }

    #                     self.save_partial_result(new_row)
    #                     time.sleep(1)  # Rate limit avoidance
    #     total_time = time.time() - start_time
    #     total_time_str = str(timedelta(seconds=int(total_time)))
    #     logger.info(f"Evaluation completed. Total time taken: {total_time_str}")

    def process_batch(self, batch, gpu_id):
        # Sort batch by index to maintain order
        sorted_batch = sorted(batch, key=lambda x: (x[0], x[2], self.model_names.index(x[3])))
        file_lock = multiprocessing.Lock()
        current_recipe_index = -1
        
        for index, row, prompt_index, model_name, total_recipes in sorted_batch:
            # Log when starting a new recipe
            if index != current_recipe_index:
                current_recipe_index = index
                elapsed_time = time.time() - time.time()  # Reset time for new recipe
                elapsed_str = str(timedelta(seconds=int(elapsed_time)))
                logger.info(f"GPU {gpu_id}: Processing Recipe {index}/{total_recipes}: {row['original_dish']} (Elapsed Time: {elapsed_str})")
            
            evaluation = self.evaluate_recipe(
                model_name,
                row['original_dish'],
                row['variation'],
                row['generated_recipe'],
                prompt_index,
                gpu_id
            )
            parsed_evaluation = self.parse_evaluation(evaluation)
            
            new_row = {
                'index': row['index'],
                'model': row['model'],
                'original_dish': row['original_dish'],
                'variation': row['variation'],
                'generated_recipe': row['generated_recipe'],
                'prompt_index': prompt_index,
                'evaluator_model': model_name,
                'evaluation': evaluation,
                'authenticity_score': parsed_evaluation['authenticity_score'],
                'authenticity_reason': parsed_evaluation['authenticity_reason'],
                'sensitivity_score': parsed_evaluation['sensitivity_score'],
                'sensitivity_reason': parsed_evaluation['sensitivity_reason'],
                'harmony_score': parsed_evaluation['harmony_score'],
                'harmony_reason': parsed_evaluation['harmony_reason'],
                'reflection': parsed_evaluation.get('reflection', None)
            }

            with file_lock:
                self.save_partial_result(new_row)

            # Log completion of evaluation
            logger.info(f"GPU {gpu_id}: Completed evaluation of {row['original_dish']} with {model_name} (Prompt {prompt_index})")
            time.sleep(1)  # Rate limit avoidance

    def evaluate_recipes(self, input_filename):
        start_time = time.time()
        
        # Read all recipes
        with open(input_filename, 'r', newline='', encoding='utf-8', errors='replace') as file:
            recipes = list(csv.DictReader(file))
        
        total_recipes = len(recipes)
        logger.info(f"Starting evaluation of {total_recipes} recipes")
        
        # Create tasks and distribute across GPUs\

        
        tasks = [(index, row, prompt_index, model_name, total_recipes) 
                for index, row in enumerate(recipes, start=1)
                for prompt_index in self.prompts.keys()
                for model_name in self.model_names]

        # Distribute tasks across GPUs
        batches = []
        batch_size = len(tasks) // self.num_gpus
        remaining = len(tasks) % self.num_gpus
        
        start_idx = 0
        for i in range(self.num_gpus):
            current_batch_size = batch_size + (1 if i < remaining else 0)
            batch = tasks[start_idx:start_idx + current_batch_size]
            batches.append(batch)
            start_idx += current_batch_size
        
        # Process batches in parallel
        with multiprocessing.Pool(processes=self.num_gpus) as pool:
            process_batch_with_gpu = [(batch, gpu_id) for gpu_id, batch in enumerate(batches)]
            pool.starmap(self.process_batch, process_batch_with_gpu)

        # Sort results after all evaluations are complete
        self.sort_results(self.output_filename)

        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        logger.info(f"Evaluation completed. Total time taken: {total_time_str}")

    def save_partial_result(self, row):
        with open(self.output_filename, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(row)
        logger.info(f"Partial result saved for recipe: {row['original_dish']} ({row['evaluator_model']})")

if __name__ == "__main__":
    input_csv = "../v0_recipes.csv"
    output_csv = "evaluated_recipes.csv"

    evaluator = RecipeEvaluator(output_csv)
    evaluator.evaluate_recipes(input_csv)
