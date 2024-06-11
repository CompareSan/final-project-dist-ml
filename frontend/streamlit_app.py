import io
import os

import requests  # type: ignore
import streamlit as st
from PIL import Image

# Set the endpoint URL of your FastAPI backend
api_url = os.getenv("API_URL", "http://localhost:8000/predict")
st.title("Image Classification App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")

    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes_value = img_bytes.getvalue()

    response = requests.post(api_url, files={"file": img_bytes_value})
    prediction = response.text

    st.write("Prediction: ", prediction)
