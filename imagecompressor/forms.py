from django import forms
from imagecompressor.models import Image 
from django import forms 


class MyImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields  = ['image']
    
    image = forms.ImageField(label='Image' , widget = forms.FileInput(
        attrs = {
            'class': 'form-control',
            'type':'file',
            'id':'formFile',
        }
    ))
    