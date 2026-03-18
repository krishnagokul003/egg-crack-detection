import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Egg Crack Detection", layout="centered")

# ---------------- HIDE DEFAULT ----------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.phone {
    max-width: 380px;
    margin: auto;
    background: white;
    border-radius: 25px;
    padding: 20px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    text-align: center;
}

.title {
    font-size: 22px;
    font-weight: bold;
}

.subtitle {
    color: gray;
    margin-bottom: 10px;
}

.badge {
    padding: 10px;
    border-radius: 12px;
    font-weight: bold;
    margin-top: 10px;
}

.intact {
    background: #dcfce7;
    color: #166534;
}

.cracked {
    background: #fee2e2;
    color: #991b1b;
}

.stButton>button {
    width: 100%;
    background: #16a34a;
    color: white;
    padding: 12px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/egg_crack_model.h5")

model = load_model()

# ---------------- UI ----------------
st.markdown('<div class="phone">', unsafe_allow_html=True)

st.markdown('<div class="title">🥚 Egg Crack Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Capture or Upload Image</div>', unsafe_allow_html=True)

# ---------------- CAMERA INPUT ----------------
camera_image = st.camera_input("📸 Capture Egg Image")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Or Upload Image", type=["jpg", "jpeg", "png"])

# ---------------- IMAGE SELECTION LOGIC ----------------
if camera_image is not None:
    image = Image.open(camera_image).convert("RGB")
    st.session_state["image"] = image

elif uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.session_state["image"] = image

# ---------------- SHOW IMAGE ----------------
if "image" in st.session_state:
    st.image(st.session_state["image"], width=300)

# ---------------- DETECT BUTTON ----------------
detect = st.button("🔍 Detect")

# ---------------- PREDICTION ----------------
if detect and "image" in st.session_state:
    image = st.session_state["image"]

    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred >= 0.5:
        result = "CRACKED"
        confidence = pred * 100
        st.markdown(f'<div class="badge cracked">🔴 {result}</div>', unsafe_allow_html=True)
    else:
        result = "INTACT"
        confidence = (1 - pred) * 100
        st.markdown(f'<div class="badge intact">🟢 {result}</div>', unsafe_allow_html=True)

    st.write(f"Confidence: {confidence:.2f}%")
    st.progress(int(confidence))

st.markdown('</div>', unsafe_allow_html=True)