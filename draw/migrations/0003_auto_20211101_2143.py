# Generated by Django 3.2.8 on 2021-11-01 16:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('draw', '0002_auto_20211101_2017'),
    ]

    operations = [
        migrations.AlterField(
            model_name='parametrs',
            name='num_point',
            field=models.IntegerField(blank=True, default=2),
        ),
        migrations.AlterField(
            model_name='parametrs',
            name='speed_point',
            field=models.IntegerField(default=1, max_length=50),
        ),
    ]
