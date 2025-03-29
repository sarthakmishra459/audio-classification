import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Load the trained model
MODEL_PATH = "model.h5"  # Ensure your model is saved in this path
model = load_model(MODEL_PATH)

# Function to convert audio to spectrogram
def audio_to_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Plot the spectrogram
    fig, ax = plt.subplots()
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    
    # Save to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    
    return Image.open(buf)

# Streamlit UI
st.title("Audio Classification using Neural Networks")
st.write("Upload an audio file to classify its category.")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    # Convert to spectrogram
    spectrogram_image = audio_to_spectrogram(uploaded_file)
    st.image(spectrogram_image, caption="Generated Spectrogram", use_column_width=True)
    
    # Preprocess image for model input
    spectrogram_image = spectrogram_image.resize((128, 128))  # Resize to match model input
    spectrogram_array = np.array(spectrogram_image) / 255.0  # Normalize
    spectrogram_array = np.expand_dims(spectrogram_array, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(spectrogram_array)
    predicted_class = np.argmax(prediction)
    
    st.write(f"Predicted Class: {predicted_class}")
