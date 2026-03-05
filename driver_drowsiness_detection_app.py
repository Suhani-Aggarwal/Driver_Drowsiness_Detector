import streamlit as st
import cv2
import numpy as np
from skimage.feature import hog
import joblib

# Load model
model = joblib.load("saved_model.pkl")

st.set_page_config(page_title="Driver Drowsiness Detection", layout="centered")

st.title("😴 Driver Drowsiness Detection")

st.write("Choose Input Method")

option = st.radio("Select Option:", ["Upload Image", "Use Webcam"])

def extract_features(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64,64))
    return hog(img, pixels_per_cell=(8,8), cells_per_block=(2,2))

image = None

# 📂 Upload Option
if option == "Upload Image":
    uploaded = st.file_uploader("Upload Eye Image", type=["jpg","png","jpeg"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

# 📸 Webcam Option
if option == "Use Webcam":
    camera_photo = st.camera_input("Take a picture")
    if camera_photo:
        file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

# 🔍 Prediction
if image is not None:
    st.image(image, caption="Input Image", width=250)

    features = extract_features(image)
    pred = model.predict([features])[0]

    if pred == 1:
        st.error("🚨 DROWSY EYES DETECTED")
    else:
        st.success("✅ EYES OPEN - ALERT")