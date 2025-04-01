from django.http import JsonResponse
from django.shortcuts import render
from transformers import pipeline

# Load the text-generation model
generator = pipeline("text-generation", model="gpt2", device=-1)

def Home(request):
    return render(request, "Home.html")  # ✅ Ensure Home.html exists in /templates

def GenerateRecipe(dish_name):
    """Generate a recipe for the given dish name using AI."""
    prompt = f"Recipe for {dish_name}:"

    try:
        generated_response = generator(prompt, max_length=200, num_return_sequences=1, temperature=0.7)
        recipe_text = generated_response[0]['generated_text'].replace(prompt, "").strip()

        if len(recipe_text) < 20 or "Recipe for" in recipe_text:
            return None  # AI failed to generate a proper recipe

        return recipe_text
    except Exception:
        return None  # Catch AI model errors

def GetRecipe(request):
    """Django view to generate a recipe."""
    dish_name = request.GET.get("dish", "").strip()  # ✅ Ensure consistency with frontend

    if not dish_name:
        return JsonResponse({"error": "Please enter a dish name."}, status=400)

    recipe = GenerateRecipe(dish_name)

    if not recipe:
        return JsonResponse({"error": f"Sorry, no recipe found for '{dish_name}'."}, status=404)

    return JsonResponse({"dish": dish_name, "recipe": recipe})
