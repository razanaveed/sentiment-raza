import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
import pickle
import time
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import requests
import os
"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""


# Define a global variable to store the loaded model (initially None)
loaded_model = None


@st.cache(allow_output_mutation=True)  # Cache the downloaded file and loaded model
def model_download_and_load_file(url, filename):
  """
  Downloads a file from a URL, saves it locally (if not already downloaded), and loads it.

  Args:
      url: The URL of the file to download.
      filename: The name to save the downloaded file as.

  Returns:
      The loaded data (e.g., model object, pandas DataFrame) or None on error.
  """

  # Check if the file already exists (assuming same directory)
  if not os.path.exists(filename):
    # Download the file if it doesn't exist
    response = requests.get(url, stream=True)

    if response.status_code == 200:
      with open(filename, 'wb') as f:
        for chunk in response.iter_content(1024):
          f.write(chunk)
      st.success("Model downloaded successfully!")
    else:
      st.error(f"Error downloading model: {response.status_code}")
      return None  # Indicate download error

  # Load the data from the downloaded file
  try:
    data = load_model(filename)
    return data
  except Exception as e:
    st.error(f"Error loading model: {e}")
    return None


#model = load_model('/data/model.h5')
model = model_download_and_load_file('https://drive.google.com/file/d/1-w8_fnXi5NOmK0usdvXfHr01kxw5io42','model.h5')


@st.cache(allow_output_mutation=True)  # Cache the downloaded file and loaded model
def token_download_and_load_file(url, filename):
  """
  Downloads a file from a URL, saves it locally (if not already downloaded), and loads it.

  Args:
      url: The URL of the file to download.
      filename: The name to save the downloaded file as.

  Returns:
      The loaded data (e.g., model object, pandas DataFrame) or None on error.
  """

  # Check if the file already exists (assuming same directory)
  if not os.path.exists(filename):
    # Download the file if it doesn't exist
    response = requests.get(url, stream=True)

    if response.status_code == 200:
      with open(filename, 'wb') as f:
        for chunk in response.iter_content(1024):
          f.write(chunk)
      st.success("Tokenizer downloaded successfully!")
    else:
      st.error(f"Error downloading tokenizer: {response.status_code}")
      return None  # Indicate download error

  # Load the data from the downloaded file
  try:
    with open(filename, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer
  except Exception as e:
    st.error(f"Error loading tokenizer: {e}")
    return None

tokenizer = token_download_and_load_file('https://drive.google.com/file/d/1-4WrF3Gzj5-nQRvi3i3t_USwrSnafgMQ','tokenizer.pkl')
#TOKENIZER_PATH = "/data/tokenizer.pkl"
#with open(TOKENIZER_PATH, 'rb') as handle:
#    tokenizer = pickle.load(handle)

SEQUENCE_LENGTH = 300

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)


def decode_sentiment(score, include_neutral=True):
    if include_neutral:
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE

def predict(text, include_neutral=True):
    start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label, "score": float(score),
       "elapsed_time": time.time()-start_at}


user_input = st.text_input("Enter your input here:")  # Replace with appropriate input element

if st.button("Process"):
    output = predict(user_input)
    st.write("**Output:**")
    st.write(output)
