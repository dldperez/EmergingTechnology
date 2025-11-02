import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model('banana_ripeness_model.h5')
class_names = ['Overripe', 'Ripe', 'Rotten', 'Unripe']

# Set a banana-themed background using CSS
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://imgur.com/gallery/banana-background-BIRib"); /* Banana wallpaper */
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .main {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 15px;
        max-width: 700px;
        margin: auto;
        box-shadow: 0px 0px 25px rgba(0,0,0,0.2);
    }
    h1 {
        color: #fdd835;
        text-align: center;
        text-shadow: 2px 2px 4px #00000055;
    }
    </style>
""", unsafe_allow_html=True)

# Title and app description
st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("🍌 Banana Ripeness Detector")
st.write("Upload a banana image to see its predicted ripeness level!")

# File uploader
uploaded_file = st.file_uploader("Choose a banana image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx]

    st.subheader(f"Prediction: **{class_names[class_idx]}** 🍌")
    st.write(f"Confidence: {confidence*100:.2f}%")

st.markdown('</div>', unsafe_allow_html=True)
