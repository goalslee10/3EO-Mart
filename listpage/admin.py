from django.contrib import admin
from .models import Product, Basket

class ProductAdmin(admin.ModelAdmin):
    list_display = ['id', 'product_name', 'product_amount', 'product_price', 'register_date']
    list_display_links = ['product_name']
    list_per_page = 25
    search_fields = ['product_name']
# Register your models here.

class BasketAdmin(admin.ModelAdmin):
    list_display = ['id', 'basket_name', 'basket_amount', 'basket_price']
    list_display_links = ['basket_name']
    list_per_page = 25
    search_fields = ['basket_name']
# Register your models here.

admin.site.register(Product, ProductAdmin)
admin.site.register(Basket, BasketAdmin)
