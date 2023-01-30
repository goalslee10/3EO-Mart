from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('listpage',views.listpage),
]