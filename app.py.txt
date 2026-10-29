
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Skin Cancer Image Classifier",
    page_icon="🔬",
    layout="centered"
)

st.title("Skin Cancer Image Classifier")
st.write("Upload a skin image to get an AI model prediction.")

MODEL_PATH = "improved_skin_cancer_cnn.keras"

CLASS_NAMES = [
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "pigmented benign keratosis",
    "seborrheic keratosis",
    "squamous cell carcinoma",
    "vascular lesion"
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error("Model could not be loaded.")
    st.exception(e)
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Classify Image"):
        resized_image = image.resize((224, 224))
        image_array = np.array(resized_image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_index]) * 100

        st.subheader("Model Prediction")
        st.write(f"Predicted class: **{CLASS_NAMES[predicted_index]}**")
        st.write(f"Model confidence: **{confidence:.2f}%**")

        st.warning(
            "This is an experimental educational model, "
            "not a medical diagnosis. Do not use this result "
            "to make medical decisions."
        )
