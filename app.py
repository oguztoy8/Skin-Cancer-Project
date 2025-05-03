import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model=load_model("my_cnn_model.keras")

def process_image(img):
    img=img.resize((170,170)) # boyutunu 170*170 yaptık
    img=np.array(img)
    img=img/255.0 #normalize
    img=np.expand_dims(img,axis=0)
    return img

st.title("Kanser resmi sınıflandırma :cancer:")
st.write("Resim seç ve model kanser olup olmadığını tahmin etsin")

file=st.file_uploader("Bir Resim sec", type=["jpg","jpeg","png"])

if file is not None:
    img=Image.open(file)
    st.image(img,caption='yuklenen resim')
    image= process_image(img)
    prediction=model.predict(image)
    predicted_class=np.argmax(prediction)
    
    class_names=['Kanser Degil','Kanser']
    st.write(class_names[predicted_class])