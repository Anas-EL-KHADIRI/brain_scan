import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.predict import predict_image

st.title("🧠 Brain Tumor Classification")
st.write("Upload a brain MRI scan to predict the tumor type.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", width=300)

    with st.spinner("Classifying..."):
        img_array = np.array(image.convert("RGB"))
        result = predict_image(img_array)

        predicted_class = result["predicted_class"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]

        if predicted_class == "notumor":
            st.success(f"**Result: No Tumor Detected ({confidence}%)**")
        else:
            st.error(f"**Result: {predicted_class.upper()} DETECTED ({confidence}%)**")

        st.write("### Class Probabilities:")
        for class_name, prob in probabilities.items():
            st.write(f"- **{class_name.capitalize()}**: {prob * 100:.2f}%")
            st.progress(float(prob))
