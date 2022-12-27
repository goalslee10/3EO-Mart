from django.shortcuts import render
from .models import Product

def index(request):
    product_list = Product.objects.order_by('id')
    product = {'product_list': product_list}

    return render(request, 'listpage/product_list.html', product)