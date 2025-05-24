import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
from tempfile import NamedTemporaryFile
from tensorflow.keras.models import load_model
import os

# Load model and encoder
model = load_model("ser_model_compatible.keras")
encoder = joblib.load("encoder.pkl")

# Feature extraction function
def extract_mfcc(file):
    y, sr = librosa.load(file, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# Prediction function
def predict_emotion(file_path):
    mfcc = extract_mfcc(file_path)
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.expand_dims(mfcc, -1)
    prediction = model.predict(mfcc)
    label = encoder.inverse_transform(prediction)
    return label[0][0]

# Streamlit UI
st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a `.wav` file to detect the emotion in speech.")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    # Save file temporarily
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    st.audio(temp_path, format="audio/wav")

    # Predict and show result
    with st.spinner('Analyzing...'):
        emotion = predict_emotion(temp_path)
        st.success(f"**Predicted Emotion:** {emotion.capitalize()}")

    # Option to download or clear
    if st.button("Clear"):
        os.remove(temp_path)
        st.experimental_rerun()
