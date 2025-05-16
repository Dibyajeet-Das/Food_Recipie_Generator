import re
from django.http import JsonResponse
from django.shortcuts import render
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
from diffusers import StableDiffusionPipeline
import torch
import base64
from io import BytesIO
from PIL import Image

# Load the text-generation model
generator = pipeline("text-generation", model="gpt2-medium", device=-1)
generator.tokenizer.pad_token = generator.tokenizer.eos_token

# Load the image-generation model
image_generator = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", 
    torch_dtype=torch.float32,
    use_safetensors=True
).to("cpu")

# Load mbien/recipenlg for generating recipes
recipe_model_name = "mbien/recipenlg"
recipe_tokenizer = GPT2Tokenizer.from_pretrained(recipe_model_name)
recipe_tokenizer.pad_token = recipe_tokenizer.eos_token
recipe_model = GPT2LMHeadModel.from_pretrained(recipe_model_name)

def Home(request):
    return render(request, "Home.html")

def GenerateRecipe(dish_name):
    try:
        # Step 1: Generate ingredients
        ingredients_prompt = f"List the ingredients for {dish_name}:"
        ingredients_outputs = generator(
            ingredients_prompt,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7,
            truncation=True,
            eos_token_id=generator.tokenizer.eos_token_id
        )
        ingredients_text = ingredients_outputs[0]['generated_text'].replace(ingredients_prompt, "").strip()
        print("Ingredients:", ingredients_text)

        if not ingredients_text:
            return None

        # Step 2: Feed ingredients to recipe model
        recipe_input = f"items: {ingredients_text}"
        inputs = recipe_tokenizer(recipe_input, return_tensors="pt", padding=True, truncation=True)
        recipe_output = recipe_model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=500,
            num_return_sequences=1,
            temperature=0.7,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=60,
            top_p=0.95,
            eos_token_id=recipe_tokenizer.eos_token_id
        )
        generated_recipe = recipe_tokenizer.decode(recipe_output[0], skip_special_tokens=False)
        print("Generated recipe:", generated_recipe)

        # Extract ingredient and instruction sections using tokens
        ingredients = []
        instructions = []

        # Use special tokens to split text
        if "<INSTR_START>" in generated_recipe:
            parts = generated_recipe.split("<INSTR_START>")
            ingr_part = parts[0]
            instr_part = parts[1]

            # Clean ingredients by splitting on <NEXT_INGR>
            for line in ingr_part.split("<NEXT_INGR>"):
                line = line.replace("items:", "").replace("<INGR_END>", "").strip()
                if line:
                    ingredients.append(line)

            # Clean instructions by splitting on <NEXT_INSTR>
            for step in instr_part.split("<NEXT_INSTR>"):
                step = step.strip()
                if step:
                    instructions.append(step)
        else:
            # fallback in case token structure is missing
            return f"Dish Name: {dish_name}\n\n{generated_recipe}"

        # Format for output
        formatted_recipe = f"Dish Name: {dish_name}\n\n"

        if ingredients:
            formatted_recipe += "Ingredients:\n" + "\n".join(f"- {ing}" for ing in ingredients) + "\n\n"
        if instructions:
            formatted_recipe += "Instructions:\n" + "\n".join(f"{i+1}. {step}" for i, step in enumerate(instructions))

        return formatted_recipe


    except Exception as e:
        print(f"Recipe error: {e}")
        return None


def GenerateImage(dish_name):
    try:
        prompt = f"High quality, professional food photography of {dish_name}, realistic, delicious"
        image = image_generator(prompt, num_inference_steps=15, guidance_scale=7.5).images[0]
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    except Exception as e:
        print(f"Image generation error: {e}")
        return None

def GetRecipe(request):
    dish_name = request.GET.get("dish", "").strip()
    if not dish_name:
        return JsonResponse({"error": "Please enter a dish name."}, status=400)
    recipe = GenerateRecipe(dish_name)
    if not recipe:
        return JsonResponse({"error": f"Sorry, no recipe found for '{dish_name}'."}, status=404)
    image_base64 = GenerateImage(dish_name)
    if not image_base64:
        return JsonResponse({"error": "Failed to generate image."}, status=500)
    return JsonResponse({
        "dish": dish_name,
        "recipe": recipe,
        "image": image_base64
    })