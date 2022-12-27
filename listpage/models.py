from django.db import models

# Create your models here.
class Product(models.Model):
    product_name = models.CharField(max_length=200)
    product_price = models.IntegerField()
    product_amount = models.IntegerField(default=0)
    register_date = models.DateTimeField()

    def __str__(self):
        return self.product_name