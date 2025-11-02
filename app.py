import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model('banana_ripeness_model.h5')
class_names = ['Overripe', 'Ripe', 'Rotten', 'Unripe']

# Page layout
st.set_page_config(page_title="Banana Ripeness Detector", layout="wide")

# CSS styling
st.markdown("""
    <style>
    body {
        background-color: #fff3e0; /* optional banana-colored background */
    }
    h1, h2, h3, p {
        color: #3e2723; /* dark text for readability */
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    h1 {
        text-align: center;
        color: #fdd835;
    }
    .instructions {
        text-align: center;
        background-color:#1E1E1E;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .instructions h3 {
        color: #FFE135;
        font-size: 28px;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
        margin-bottom: 10px;
    }
    .instructions p {
        color: #FFD700;
        font-size: 14px;
        margin-top: 0;
    }
    .title-gif {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 150px;
        border-radius: 15px;
        box-shadow: 0px 0px 15px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Static GIF above the title
st.markdown(
    '<img class="title-gif" src="https://raw.githubusercontent.com/dldperez/EmergingTechnology/main/banana-gif-9.gif">',
    unsafe_allow_html=True
)

# Title
st.title("🍌 Banana Ripeness Detector 🍌")

# Clear and styled instructions
st.markdown("""
    <div class="instructions">
        <h3>Upload a banana image to see its predicted ripeness level!</h3>
        <p>Choose a banana image from your device below.</p>
    </div>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

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
