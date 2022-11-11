import random
from django.shortcuts import redirect, render
from django.views.generic import TemplateView  ,CreateView
from imagecompressor.forms import MyImageForm
from imagecompressor.models import Image, CompressedImage
from django.conf import settings
from os.path import join 
#sklearn library 
from sklearn.cluster import KMeans 
from skimage import io 
import numpy as np 

# Create your views here.

class Myview(TemplateView):
    template_name = 'index.html' 
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = MyImageForm()
        return context 

class SaveImage(CreateView):
    mode = Image 
    success_url = '/dis' 
    form_class = MyImageForm 
    def form_valid(self, form):

        form.save() 

        self.compress(form.files['image']) 
        return super().form_valid(form)
    def compress(self , name):
        image = io.imread(join(settings.BASE_DIR ,f'media/images/{name}'))
        row , col , ch = image.shape 
        image = image.reshape(-1 , 3 ) 
        kmeans = KMeans(n_clusters=2) 
        kmeans.fit(image) 
        cimage = kmeans.cluster_centers_[kmeans.labels_]
        cimage = np.clip(cimage.astype('uint8') , 0 ,255) 
        cimage = cimage.reshape(row , col , 3 ) 
        idx = random.randint(0,255) 
        io.imsave(join(settings.BASE_DIR,f'media/compressed/{idx}.jpg') ,cimage)
        CompressedImage.objects.create(name = 'image',image = f'compressed/{idx}.jpg')

def displayImage(request):
    if request.method !='post':
        image = Image.objects.all() 
        cimage = CompressedImage.objects.all() 
        return render(request , 'image_display.html' , {'images':image  , 'cimage':cimage})
    return redirect('/')