import sys
print(sys.executable)
import tensorflow as tf
print(tf.__version__)

import streamlit as st
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

## Loading the tensorflow model for prediction

model=load_model('mode.h5')

with open ('Tokenizer.pkl','rb') as file:
    tokenizer=pickle.load(file)
    
st.title('Twitter Tweets Sentimental Analysis')
tweet=st.text_area('Enter the tweet: ')

if st.button('Predict Sentiment: ') and tweet.strip():
    sequences=tokenizer.texts_to_sequences([tweet])
    sequences=pad_sequences(sequences,padding='post',maxlen=99)
    
    prediction=model.predict(sequences)
    predicted_class=np.argmax(prediction,axis=1)[0]
    
    sentiment_map={0:'Negative',1:'Neutral',2:'Positve'}
    
    st.write("Sentiment",sentiment_map[predicted_class])
    
    
    
    
