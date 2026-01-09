import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# -----------------------------
# Cargar modelo
# -----------------------------
@st.cache_resource
def load_cnn_model():
    # Cambia la ruta según tu modelo exportado
    # Si usaste model.export(), pon la carpeta
    MODEL_PATH = "skin_cancer_cnn_tf"  # Carpeta SavedModel
    # Si usaste .h5 moderno, sería: "skin_cancer_cnn_new.h5"
    model = load_model(MODEL_PATH)
    return model

model = load_cnn_model()

# -----------------------------
# Función de predicción
# -----------------------------
def predict_skin_cancer(uploaded_file, model):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    class_label = "Maligne" if prediction > 0.5 else "Benigne"
    return class_label, img

# -----------------------------
# Interfaz Streamlit
# -----------------------------
st.title("Detector de Melanoma")

st.markdown("""
Aquest és un detector de càncer de pell.
Adjunta una imatge i el model indicarà si el melanoma és benigne o maligne.
""")

uploaded_image = st.file_uploader(
    "Escull una imatge...", type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    class_label, img = predict_skin_cancer(uploaded_image, model)
    st.image(img, caption="Imatge analitzada", width=400)
    st.write(f"### Predicció: **{class_label}**")

st.markdown("""
### Informació sobre el model
Aquest model utilitza xarxes neuronals convolucionals entrenades amb imatges mèdiques.
No es garanteix una bona precisió amb fotografies fetes amb mòbil.
""")
