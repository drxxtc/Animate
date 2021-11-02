from django.db import models

# Create your models here.
class Parametrs(models.Model):
    num_point =models.IntegerField(blank=True, default=2)
    speed_point= models.IntegerField(max_length=50, default=1)
    def __str__(self):
        return str(self.num_point)
    