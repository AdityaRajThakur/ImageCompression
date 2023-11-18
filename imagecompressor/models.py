from django.db import models

# Create your models here.


class Image(models.Model):
    name = models.CharField(max_length = 20) 
    image = models.ImageField(upload_to = 'images')
    date = models.DateField(auto_now_add =True) 
    size = models.FloatField(max_length=20 , default=0)
    def __str__(self):
        return self.name



class CompressedImage(models.Model):
    name = models.CharField(max_length =20) 
    image = models.ImageField(upload_to = 'Compressed')
    date = models.DateField(auto_now_add =True) 
    # size = models.FloatField(max_length=20)
    size = models.FloatField(max_length=20 , default=0)
    def __str__(self):
        return self.name

