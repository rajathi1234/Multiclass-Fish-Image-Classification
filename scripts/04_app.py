import streamlit as st
import numpy as np

from keras.models import load_model
from keras.utils import load_img, img_to_array


# ==========================================
# 1. Page settings
# ==========================================

st.set_page_config(
    page_title="Fish Classification",
    page_icon="🐟"
)


# ==========================================
# 2. Title
# ==========================================

st.title("🐟 Multiclass Fish Image Classification")

st.write(
    "Upload a fish image to predict its class."
)


# ==========================================
# 3. Classes
# ==========================================

classes = [
    "Catla",
    "CommonCarp",
    "Mori",
    "Rohu",
    "SilverCarp"
]


# ==========================================
# 4. Load best model
# ==========================================

model = load_model(
    "models/mobilenet_fish_classifier.keras"
)


# ==========================================
# 5. Image upload
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a fish image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# 6. Prediction
# ==========================================

if uploaded_file is not None:

    image = load_img(
        uploaded_file,
        target_size=(128, 128)
    )

    st.image(
        image,
        caption="Uploaded Image",
        width=400
    )


    image_array = img_to_array(image)

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    prediction = model.predict(
        image_array,
        verbose=0
    )


    predicted_index = np.argmax(
        prediction
    )

    predicted_class = classes[
        predicted_index
    ]

    confidence = (
        prediction[0][predicted_index] * 100
    )


    st.success(
        f"Predicted Fish: {predicted_class}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )