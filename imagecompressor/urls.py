
from django.contrib import admin
from django.urls import path
from imagecompressor.views import Myview , SaveImage 
from imagecompressor.views import displayImage
 
urlpatterns = [
    path('' , Myview.as_view() , name =  'Homeview') , 
    path('sub' , SaveImage.as_view() ,name = 'MyImage') , 
    path('dis/' ,displayImage, name = "ImageDispaly" )
]
