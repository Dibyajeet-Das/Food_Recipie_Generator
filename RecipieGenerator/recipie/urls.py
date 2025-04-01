from django.urls import path
from .views import Home, GetRecipe  # Import views

urlpatterns = [
    path('', Home, name='Home'),  # ✅ Fix: Removed space before ''
    path('get-recipe/', GetRecipe, name='get-recipe'),  # API endpoint
]
