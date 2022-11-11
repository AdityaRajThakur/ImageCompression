from django.contrib import admin
from imagecompressor.models import Image ,CompressedImage
# Register your models here.
@admin.register(Image , CompressedImage)
class ImageAdmin(admin.ModelAdmin):
    list_display = ['id' ,'image',  'date']