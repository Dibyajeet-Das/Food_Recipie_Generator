from django.urls import path
#importing the views from the views.py file
from .views import Home, GetRecipe  

urlpatterns = [
    path('', Home, name='Home'),  
    path('get-recipe/', GetRecipe, name='get-recipe'),# API endpoint
]
