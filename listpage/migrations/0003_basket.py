# Generated by Django 4.0.3 on 2023-01-06 05:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('listpage', '0002_product_product_amount'),
    ]

    operations = [
        migrations.CreateModel(
            name='Basket',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('basket_name', models.CharField(max_length=200)),
                ('basket_price', models.IntegerField()),
                ('basket_amount', models.IntegerField(default=0)),
            ],
        ),
    ]