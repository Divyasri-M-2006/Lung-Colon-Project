import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Cancer Detection AI", layout="centered")

# -------------------------------
# PREMIUM CSS (Poppins + UI)
# -------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

.title {
    font-size: 46px !important;
    font-weight: 700 !important;
    color: #38bdf8;
    text-align: center;
    line-height: 1.2;
}

.subtitle {
    text-align: center;
    color: #cbd5f5;
    margin-bottom: 25px;
}

.box {
    background: rgba(30, 41, 59, 0.85);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 0 25px rgba(56, 189, 248, 0.25);
    text-align: center;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.h5")

model = load_model()

# Class labels
class_names = ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']

# -------------------------------
# TITLE
# -------------------------------
st.markdown('<p class="title">🧠 AI-Powered Lung & Colon Cancer Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Histopathological Image Classification using Deep Learning</p>', unsafe_allow_html=True)

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload Histopathology Image", type=["jpg", "png", "jpeg"])

# -------------------------------
# GRAD-CAM FUNCTION (FINAL FIXED)
# -------------------------------
def get_gradcam(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        predictions = tf.squeeze(predictions)
        class_idx = tf.argmax(predictions)
        loss = predictions[class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

# -------------------------------
# MAIN LOGIC
# -------------------------------
if uploaded_file is not None:

    # Load and preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    img_resized = image.resize((224, 224))

    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -------------------------------
    # PREDICTION
    # -------------------------------
    pred = model.predict(img_array)
    pred_class = class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100

    # Display prediction
    st.markdown(f"""
    <div class="box">
        <h2 style='color:#22c55e;'>Prediction: {pred_class}</h2>
        <p style='color:#facc15;'>Confidence: {confidence:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------
    # GRAD-CAM
    # -------------------------------
    heatmap = get_gradcam(img_array, model, 'conv5_block3_out')

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img_original = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
    superimposed_img = cv2.addWeighted(img_original, 0.6, heatmap, 0.4, 0)

    # -------------------------------
    # DISPLAY IMAGES
    # -------------------------------
    st.markdown("### 🔥 Grad-CAM Visualization")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", width=320)

    with col2:
        st.image(
            cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB),
            caption="Grad-CAM",
            width=320
        )
