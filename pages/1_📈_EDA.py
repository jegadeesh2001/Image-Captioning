
import streamlit as st
from PIL import Image


st.set_page_config(page_title='EDA',
                   page_icon=':bar_chart:', layout='wide')
st.markdown("<h1 style='text-align: center; color: black;font-size:50px'>📈 EXPLORATORY DATA ANALYSIS</h1><hr>",
            unsafe_allow_html=True)


with st.container():
        st.subheader("Images and their Captions")
        st.markdown('---')
        image1 = Image.open('images\sample (2).png')

        st.image(image1)


with st.container():
        st.text(' ')
        st.text(' ')
        
        st.subheader("Top 50 most frequent occuring words")
        st.markdown('---')
        image1 = Image.open('images\The top 50 most frequently appearing words (1).png')

        st.image(image1)
        st.markdown("""---""")


with st.container():
        st.text(' ')
        st.text(' ')
        
        st.subheader("Top 50 least frequent occuring words")
        st.markdown('---')
        image1 = Image.open('images\The least 50 most frequently appearing words (3).png')

        st.image(image1)
        st.markdown("""---""")

with st.container():
        st.text(' ')
        st.text(' ')
        
        st.subheader("Image captioning model architecture")
        st.markdown('---')
        image1 = Image.open('images\model.png')

        st.image(image1)
        st.markdown("""---""")

