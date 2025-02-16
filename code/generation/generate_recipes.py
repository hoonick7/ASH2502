import csv
import time
import argparse
from langchain_community.chat_models import ChatOllama
from loguru import logger
import re
import os

class RecipeGenerator:
    # model_names = ["gemma2:2b", "gemma2:9b", "mistral:7b", "llama2:13b", "llama3.1:8b", "gpt-4o-mini"]
    # gpt-4o-mini needs to be used via API
    model_names = ["gemma2:2b", "gemma2:9b", "mistral:7b", "llama2:13b", "llama3.1:8b"]
    dishes = ["Fried Rice", "Sandwich", "Soup Noodle", "Savoury Pie", "Fried Noodles", "Rolls", "Savory Waffle", "Fried Chicken", 
             "Barbecued Meat", "French Fries", "Burger", "Pasta", "Pancake", "Stew", "Pizza", "Burritos", "Crepes", "Lasagna", "Curry", "Salad"]
    variations = [
        'Japanese', 'Korean', 'Chinese', 'Thai', 'Vietnamese', 'Filipino', 'Indian', 'Russian',
        'Italian', 'French', 'Spanish', 'British', 'Irish', 'Greek', 'Scottish', 'Swedish',
        'Southern US', 'Brazilian', 'Mexican', 'Jamaican', 'Hawaiian', 'Costa Rican', 'Canadian', 'Peruvian',
        'Moroccan', 'Ethiopian', 'Algerian', 'Egyptian', 'Australian', 'Polynesian',
        'Buddhist', 'Hindu diet', 'Islamic diet', 'Jain diet', 'Kosher', 'Zoroastrian',
        'Aztec', 'Medieval', 'Byzantine', 'Ottoman'
    ]

    def __init__(self):
        self.index = 1

    def generate_recipe(self, model_name, dish, variation):
        llm = ChatOllama(model=model_name)
        prompt = f"""Can you apply the elements of {variation} cuisine to this dish and make it into a recipe?
Dish: {dish}
The response should be in the following form for ingredients and instructions each. For example:
ingredients: 
<<ingredient1>>,
<<ingredient2>>,
...

instructions: 
1. <<instruction1>>
2. <<instruction2>>
...
"""
        
        try:
            result = llm.invoke(prompt)
            result_text = result.text if hasattr(result, 'text') else str(result)
            result_text = re.sub(r'^content=', '', result_text)
            result_text = re.sub(r'response_metadata.*$', '', result_text, flags=re.DOTALL)
            result_text = result_text.strip()
            logger.info(f"Generated recipe for {dish} with {model_name} and variation: {variation}")
            return result_text
        except Exception as e:
            logger.error(f"Error generating recipe for {dish} with {model_name}: {str(e)}")
            return f"Error: {str(e)}"

    def extract_ingredients_instructions(self, result):
        ingredients_match = re.search(r'ingredients:\s*{(.*?)}', result, re.IGNORECASE | re.DOTALL)
        instructions_match = re.search(r'instructions:\s*{(.*?)}', result, re.IGNORECASE | re.DOTALL)
        
        ingredients = ingredients_match.group(1).strip() if ingredients_match else ""
        instructions = instructions_match.group(1).strip() if instructions_match else ""
        
        ingredients = ', '.join([line.strip().lstrip('0123456789. *') for line in ingredients.split('\n') if line.strip()])
        instructions = '\n'.join([line.strip() for line in instructions.split('\n') if line.strip()])
        
        return ingredients, instructions

    def generate_recipes(self):
        results = []
        for model in self.model_names:
            for dish in self.dishes:
                for variation in self.variations:
                    generated_recipe = self.generate_recipe(model, dish, variation)
                    ingredients, instructions = self.extract_ingredients_instructions(generated_recipe)
                    
                    results.append({
                        'index': self.index,
                        'model': model,
                        'original_dish': dish,
                        'variation': variation,
                        'generated_recipe': generated_recipe,
                        'ingredients': ingredients,
                        'instructions': instructions
                    })
                    self.index += 1
                    time.sleep(1)  # To avoid rate limiting
        return results

    def save_to_csv(self, results, filename='generated_recipes.csv'):
        fieldnames = ['index', 'model', 'original_dish', 'variation', 'generated_recipe', 'ingredients', 'instructions']
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        logger.info(f"Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate recipes with cultural and religious variations")
    args = parser.parse_args()

    start_time = time.time()

    generator = RecipeGenerator()
    results = generator.generate_recipes()
    generator.save_to_csv(results)

    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    print(f"Recipe generation completed. Total time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
