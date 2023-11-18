from django import forms
from imagecompressor.models import Image

class MyImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ['image' , 'size']

    image = forms.ImageField(label='Image', widget=forms.FileInput(
        attrs={
            'class': 'form-control',
            'type': 'file',
            'id': 'formFile',
        }
    ))

    size = forms.FloatField(label='Size', widget=forms.NumberInput(
        attrs={
            'class': 'form-control',
            'id': 'formFile',
        }
    ))
