# Generated by Django 3.2.8 on 2021-11-01 15:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('draw', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='parametrs',
            name='graph',
        ),
        migrations.AddField(
            model_name='parametrs',
            name='num_point',
            field=models.IntegerField(blank=True, default='2'),
        ),
        migrations.AddField(
            model_name='parametrs',
            name='speed_point',
            field=models.CharField(default=1, max_length=50),
        ),
    ]