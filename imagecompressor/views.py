from skimage import io 
import random
from django.shortcuts import redirect, render
from django.views.generic import TemplateView  ,CreateView
from imagecompressor.forms import MyImageForm
from imagecompressor.models import Image, CompressedImage
from django.conf import settings
from os.path import join , getsize
#sklearn library 
from sklearn.cluster import KMeans 
from skimage import io 
import numpy as np 

# Create your views here.
from sklearn.decomposition import PCA
from PIL import Image as imgPIL, ImageOps
import numpy as np 
import os 
import matplotlib.pyplot as plt 
# img_path = "D:\\ML_IMAGE_COMPRESSOR\\env\\src\\Image Used\\original_image\\person1.jpeg" 
class Myview(TemplateView):
    template_name = 'index.html' 
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = MyImageForm()
        # print(context['form']) 
        return context 


class SaveImage(CreateView): 
    model= Image 
    success_url = '/dis' 
    form_class = MyImageForm 
    def form_valid(self, form):
        form.save() 

        # print(form['size'].value())
    
        # Image.objects.create()
        self.compress(form.files['image']) 
        return super().form_valid(form)
    def img_data(self ,imgPath , dis = False):
        orig_img = imgPIL.open(imgPath)
        img_size_kb = os.stat(imgPath).st_size / 1024 
        data = orig_img.getdata()
        ori_pixels = np.array(data).reshape(*orig_img.size , -1) 
        #print(np.array(data))
        img_dim = ori_pixels.shape
        
        if dis:
            plt.imshow(orig_img)
            plt.show()
        data_dict = {} 
        data_dict['img_size_kb'] = img_size_kb
        data_dict['img_dim'] = img_dim 
        return data_dict
    def pca_compose(self , imgPath):
        orig_img  = imgPIL.open(imgPath) 
        orig_img = imgPIL.open(imgPath) 
        img = np.array(orig_img.getdata()) 
        img = img.reshape(*orig_img.size , -1) 
        pca_channel = {} 
        img_t = np.transpose(img) 
        for i in range(img.shape[-1]):
            per_channel = img_t[i] 
            channel = img_t[i].reshape(*img.shape[:-1]) 
            pca = PCA(random_state = 42) 
            fit_pca = pca.fit_transform(channel) 
            pca_channel[i] = (pca,  fit_pca) 
        return pca_channel 
    def pca_transform(self, pca_channel , n_components):
        temp_res = [] 
        for channel in range(len(pca_channel)):
            pca , fit_pca =pca_channel[channel] 
            pca_pixel = fit_pca[:,:n_components]
            pca_comp = pca.components_[:n_components, :] 
            compressed_pixel = np.dot(pca_pixel, pca_comp)  + pca.mean_ 
            temp_res.append(compressed_pixel) 
        compressed_img = np.transpose(temp_res) 
        compressed_img = np.array(compressed_img , dtype = np.uint8) 
        return compressed_img 
    def pca_compress(self , path , spath ):
        data = self.img_data(path)
        pca_channel = self.pca_compose(path)
        img = self.pca_transform(pca_channel , 100)
        # img = img.reshape(img.shape[1], img.shape[0] , -1)
        # idx = "compressed_using_pca3"
        # self.comp  = img 
        io.imsave(spath, img)
        print(self.mse(img, path),os.stat(spath).st_size / 1024 ,data['img_size_kb'] , data['img_dim'])
    def mse(self, predicted , img_path) :
        actual = io.imread(img_path)
        # print(actual.shape) 
        diff = (actual - predicted)**2
        return np.mean(diff) 
    def compress(self , name):
        imgpath = join(settings.BASE_DIR ,f'media/images/{name}')
        idx = random.randint(0,255) 
        savepath = join(settings.BASE_DIR,f'media/compressed/{idx}.jpg')
        savepath_kmeans = join(settings.BASE_DIR,f'media/kmeans/{idx}.jpg')
        
        image = io.imread(imgpath)
        row , col , ch = image.shape 
        image = image.reshape(-1 , 3 ) 
        kmeans = KMeans(n_clusters=80, max_iter= 100) 
        kmeans.fit(image) 
        cimage = kmeans.cluster_centers_[kmeans.labels_]
        cimage = np.clip(cimage.astype('uint8') , 0 ,255) 
        cimage = cimage.reshape(row , col , 3 ) 
        # print(name) ; 
        print("Original file - SIZE", getsize(f'media/images/{name}') / 1024 , "KB")
        io.imsave(savepath ,cimage) 
        self.pca_compress(savepath, savepath_kmeans)
        print("compressed  file - SIZE", getsize(f'media/kmeans/{idx}.jpg') / 1024 ,"KB")
        left = (getsize(f'media/images/{name}') / 1024) - getsize(f'media/kmeans/{idx}.jpg') / 1024 
        print('compression ratio', (left / (getsize(f'media/images/{name}') / 1024)) * 100 )
        CompressedImage.objects.create(name = 'image',image = f'kmeans/{idx}.jpg' , size =getsize(f'media/kmeans/{idx}.jpg') / 1024  )
        

def displayImage(request):
    if request.method !='post':
        image = Image.objects.all() 
        print(image[0]) 
        cimage = CompressedImage.objects.all() 
        return render(request , 'image_display.html' , {'images':image  , 'cimage':cimage })
    return redirect('/')