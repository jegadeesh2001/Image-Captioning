# Libraries
import streamlit as st
import pandas as pd

# Global Variables
theme_plotly = "streamlit"  # None or streamlit

from streamlit_player import st_player
from pickle import load
from numpy import argmax
import argparse
import os
from gtts.tts import gTTS
from keras.utils import pad_sequences
from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.xception import Xception
from keras.models import Model
from keras.models import load_model
import cv2
from io import BytesIO
from PIL import Image
import numpy as np
import base64
 
# def extract_features(filename):
# 	model = VGG16()
# 	model.layers.pop()
# 	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
# 	image = load_img(filename, target_size=(224, 224))
# 	image = img_to_array(image)
# 	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# 	image = preprocess_input(image)
# 	feature = model.predict(image, verbose=0)
#	return feature
@st.cache_resource
def load_model_all():
     tokenizer = load(open(r'models\tokenizer (1).p', 'rb'))
     model = load_model('models\model_9.h5')
     return tokenizer,model


def tts(text):
    tts = gTTS(text=text, lang='en')
    tts.save("tts.mp3")
    # return ipd.Audio("tts.mp3")
    return 'tts.mp3'


def extract_features(model,filename):
    
       
    # image = Image.open(filename)
    # image = image.resize((299,299))
    # image = np.expand_dims(image, axis=0)
    
    # image = image/127.5
    # image = image - 1.0
    # feature = model.predict(image)
    # return feature
        try:
            image = Image.open(filename)
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature
          

 
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
 

def generate_desc(model, tokenizer, image, maxlen):
    index_word = dict([(index,word) for word, index in tokenizer.word_index.items()]) 
    in_text = 'start'

    for iword in range(maxlen):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence],maxlen)
        yhat = model.predict([image,sequence],verbose=0)
        yhat = np.argmax(yhat)
        newword = index_word[yhat]
        in_text += " " + newword
        if newword == "end":
            break
    return(in_text)

    
def generate_captions(photo_path):
        
        max_length = 32
        xception_model = Xception(include_top=False, pooling="avg")
        tokenizer,model=load_model_all()
        photo = extract_features(xception_model,photo_path)
        description = generate_desc(model, tokenizer, photo, max_length)
        
        return description

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )

def text_to_speech(text):
     file = tts(text)
     #audio = open(file, 'rb').read()
     #st.audio(audio, format='audio/mp3')
     autoplay_audio(file)

def static_image():
	
    uploaded_file = st.file_uploader("Choose a JPEG Image file", type=['jpg','png','jpeg','jfif'])

    st.text(" ")
    col1, col2, col3 = st.columns(3)
    if uploaded_file is not None:
        with st.spinner('Captioning Image ...'):
            image1 = Image.open(uploaded_file)
            try:
                image = image1.save("saved_image.jpg")
                image_path = "saved_image.jpg"
            except:
                image = image1.save("saved_image.png")
                image_path = "saved_image.png"

            with col2:
                st.image(image1)
            
            captions=generate_captions(image_path)
            words=captions.split()
            new_sentence = ' '.join(words[1:-1])

        st.success("✅ Captioning Done!")
        
        st.header(new_sentence.capitalize())   
        
        
        text_to_speech(new_sentence)
        
        
        
		



def camera_image():
      
        img_file_buffer = st.camera_input("Take a picture")
        col1, col2, col3 = st.columns(3)
        if img_file_buffer is not None:
            # To read image file buffer as a PIL Image
            with st.spinner('Captioning Image ...'):
                 image1 = Image.open(img_file_buffer)
            
                 image = image1.save("saved_image.jpg")
                 image_path = "saved_image.jpg"
                
            
                 captions=generate_captions(image_path)
                 words=captions.split()
                 new_sentence = ' '.join(words[1:-1])
        
            st.success("✅ Captioning Done!")
            st.header(new_sentence.capitalize())   
            
            text_to_speech(new_sentence)





st.set_page_config(page_title='Image Captioning',
                   page_icon=':bar_chart:', layout='wide')
st.markdown("<h1 style='text-align: center; color: black;font-size:50px'>📸 IMAGE CAPTIONING</h1><hr>",
            unsafe_allow_html=True)

st.text("")
st.text("")
option = st.selectbox(
    'How would you like to be Select Image from?',
    ('Gallery','Camera'))

if option=='Gallery':
 
 static_image()

if option=='Camera':
 camera_image()

st.text(" ")
st.text(" ")




