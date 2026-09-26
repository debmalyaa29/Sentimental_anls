# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation (cached to load once)
@st.cache_resource
def load_imdb_model():
    return load_model('simple_rnn_imdb.keras')

model = load_imdb_model()

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


import streamlit as st

st.set_page_config(page_title='IMDB Sentiment Analysis', page_icon='🎬', layout='centered')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.block-container { max-width: 640px; padding-top: 3rem; }

h1 {
    font-family: 'Source Serif 4', Georgia, serif !important;
    font-weight: 600 !important;
    font-size: 30px !important;
    margin-bottom: 4px !important;
}

.stCaption, p { color: #6A7078; }

.stTextArea textarea {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 16px;
    border-radius: 10px;
    border: 1px solid #DAD9D2;
}

.stButton button {
    background-color: #2B5F5E;
    color: #F3F3EF;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
}
.stButton button:hover {
    background-color: #234c4b;
    color: #F3F3EF;
}

div[data-testid="stVerticalBlock"] > div:has(.stTextArea) {
    background: #FFFFFF;
    border: 1px solid #DAD9D2;
    border-radius: 10px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)

## streamlit app
# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):

    preprocessed_input=preprocess_text(user_input)

    ## MAke prediction
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')