import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import pandas as pd
from PIL import Image
import io
import os

# Configurazione pagina
st.set_page_config(page_title="Dog Breed Classifier", layout="centered")

st.title("🐕 Classificatore di Razze Canine")
st.image("assets/dog.gif",  use_container_width=False)
st.write("Carica una foto del tuo cane e scopri di che razza è!")

# SIDEBAR
# About me per sidebar


def about_me():
    with st.sidebar:
        # Immagine del profilo
        st.image(
            "https://avatars.githubusercontent.com/u/72889405?v=4",
            width=120,
            caption="Veronica Schembri",
            output_format="auto",
        )

        # Nome e descrizione
        st.write("## Veronica Schembri")
        st.write("Front End Developer | Data Science & AI Enthusiast")

        # Sezione Social Media
        st.write("### Social Media")
        st.markdown(
            """
            - [🌐 Sito](https://www.veronicaschembri.com)
            - [🐙 GitHub](https://github.com/Pandagan-85)
            - [🔗 LinkedIn](https://www.linkedin.com/in/veronicaschembri/)
            - [📸 Instagram](https://www.instagram.com/schembriveronica/)
            """
        )


# 📌 Sidebar per selezionare il gioco
# Mostro gli stati per debuggare Indovina il numero

about_me()

# Istruzioni per caricare correttamente le immagini
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📸 Foto Chiare")
    st.write("Usa immagini ben illuminate per una migliore previsione.")

with col2:
    st.subheader("🐶 Un Solo Cane")
    st.write(
        "Assicurati che nell'immagine ci sia un solo cane per risultati accurati.")

with col3:
    st.subheader("🖼️ Buona Qualità")
    st.write("Evita immagini sfocate o parzialmente coperte.")

# Funzione per preparare le immagini caricate


def prepare_image(image, img_size=224):
    """
    Prepara un'immagine per il modello:
    1. Ridimensiona a img_size x img_size
    2. Normalizza
    """
    image = tf.image.resize(image, [img_size, img_size])
    return tf.cast(image, tf.float32) / 255.0

# Funzione per ottenere l'etichetta dalla previsione


def get_pred_label(prediction_probabilities, unique_breeds):
    """
    Converte le probabilità di previsione in un'etichetta.
    """
    return unique_breeds[np.argmax(prediction_probabilities)]

# Funzione per creare il grafico a barre delle top N razze


def plot_top_breeds(prediction_probabilities, unique_breeds, n=10):
    """
    Crea un grafico a barre delle top N razze più probabili utilizzando Altair per una resa estetica migliore.
    """
    # Ottieni gli indici delle top N probabilità
    top_idxs = np.argsort(prediction_probabilities)[-n:][::-1]

    # Ottieni le etichette e i valori corrispondenti
    top_breeds = [unique_breeds[i] for i in top_idxs]
    top_values = [prediction_probabilities[i] * 100 for i in top_idxs]

    # Crea un DataFrame per la visualizzazione
    df = pd.DataFrame({'Breed': top_breeds, 'Confidence': top_values})

    # Crea il grafico con Altair
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X("Confidence:Q", title="Confidence (%)"),
            y=alt.Y("Breed:N", sort="-x", title="Razza"),
            color=alt.condition(
                alt.datum.Confidence == max(df["Confidence"]),
                # Evidenzia la razza con il valore più alto
                alt.value("green"),
                alt.value("skyblue"),
            ),
            tooltip=["Breed", "Confidence"]
        )
        .properties(title="Top 10 Razze Più Probabili", width=600, height=400)
    )

    st.altair_chart(chart, use_container_width=True)

# Caricamento del modello per la classificazione delle razze canine


@st.cache_resource
def load_dog_breed_model():
    """
    Carica il modello salvato con supporto per KerasLayer.
    """
    import tensorflow_hub as hub
    from tensorflow.keras.models import load_model

    # Definisci il custom object scope per KerasLayer
    custom_objects = {'KerasLayer': hub.KerasLayer}

    # Carica il modello con il custom object scope
    model = load_model('modello.h5', custom_objects=custom_objects)
    return model

# Caricamento del modello per il rilevamento dei cani


@st.cache_resource
def load_dog_detector_model():
    """
    Carica il modello per rilevare se un'immagine contiene un cane.
    Utilizziamo un modello MobileNetV2 pre-addestrato su ImageNet.
    """
    # Carica MobileNetV2 pre-addestrato
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet', include_top=True)
    return base_model

# Funzione per verificare se un'immagine contiene un cane


def is_dog(image, model):
    """
    Verifica se l'immagine contiene un cane utilizzando un modello pre-addestrato.
    Utilizziamo le classi ImageNet dove le classi 151-268 sono razze di cani.

    Args:
        image: immagine in formato PIL
        model: modello pre-addestrato per la classificazione

    Returns:
        bool: True se è un cane, False altrimenti
        float: confidenza nella rilevazione del cane
    """
    # Converti l'immagine nel formato richiesto da MobileNetV2
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_resized = tf.image.resize(img_array, (224, 224))
    img_expanded = tf.expand_dims(img_resized, 0)

    # Preprocessa l'immagine per MobileNetV2
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_expanded)

    # Ottieni le previsioni
    predictions = model.predict(img_preprocessed)

    # In ImageNet, le classi 151-268 (indici 150-267) sono razze di cani
    dog_class_indices = range(150, 268)

    # Somma le probabilità di tutte le classi di cani
    dog_probability = np.sum(predictions[0][dog_class_indices])

    # Se la probabilità è superiore a una soglia, diciamo che è un cane
    threshold = 0.5  # Puoi modificare questa soglia in base alle tue esigenze
    return dog_probability > threshold, dog_probability

# Caricamento delle razze uniche


@st.cache_data
def load_unique_breeds():
    import json
    try:
        with open('unique_breeds.json', 'r') as f:
            unique_breeds = json.load(f)
        return np.array(unique_breeds)
    except FileNotFoundError:
        st.error("File delle razze non trovato.")
        return None

# Funzione per fare previsioni sulle immagini caricate


def predict_breed(image, model, unique_breeds):
    """
    Fa una previsione su un'immagine.
    """
    # Converti l'immagine PIL in un tensore
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    # Prepara l'immagine
    img_processed = prepare_image(img_array)
    # Aggiungi la dimensione del batch
    img_batch = tf.expand_dims(img_processed, axis=0)
    # Fai la previsione
    prediction = model.predict(img_batch)
    # Ottieni l'etichetta
    breed = get_pred_label(prediction[0], unique_breeds)
    # Ottieni il punteggio di confidenza
    confidence = np.max(prediction[0]) * 100
    return breed, confidence, prediction[0], img_processed.numpy()


# Carica i modelli e le razze
try:
    dog_breed_model = load_dog_breed_model()
    dog_detector_model = load_dog_detector_model()
    unique_breeds = load_unique_breeds()

    if unique_breeds is None:
        st.warning(
            "Per favore carica il file unique_breeds.npy prima di continuare.")
        # Opzione per caricare il file unique_breeds
        uploaded_breeds = st.file_uploader(
            "Carica il file unique_breeds.npy", type="npy")
        if uploaded_breeds is not None:
            unique_breeds = np.load(uploaded_breeds, allow_pickle=True)
            st.success("Razze caricate con successo!")
except Exception as e:
    st.error(f"Errore nel caricamento dei modelli o delle razze: {e}")
    st.stop()

# Interfaccia per il caricamento delle immagini
uploaded_files = st.file_uploader("Carica una o più immagini di cani", type=[
                                  "jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.write(f"Hai caricato {len(uploaded_files)} immagini.")

    # Per ogni immagine caricata
    for i, uploaded_file in enumerate(uploaded_files):
        # Leggi l'immagine caricata
        image = Image.open(uploaded_file)

        # Crea due colonne: una per l'immagine, una per il grafico
        col1, col2 = st.columns([1, 1.5])

        with col1:
            st.image(
                image, caption=f"Immagine {i+1}", use_container_width=True)

            # Verifica se l'immagine contiene un cane
            is_dog_image, dog_confidence = is_dog(image, dog_detector_model)

            if is_dog_image:
                st.success(
                    f"✅ L'immagine contiene un cane (confidenza: {dog_confidence*100:.1f}%)")

                # Fai la previsione della razza solo se l'immagine contiene un cane
                breed, confidence, prediction_probs, processed_img = predict_breed(
                    image, dog_breed_model, unique_breeds)

                st.success(f"Razza predetta: {breed}")
                st.info(f"Confidenza: {confidence:.2f}%")

                # Pulsante per vedere l'immagine preprocessata
                if st.button(f"Mostra immagine preprocessata {i+1}"):
                    st.image(
                        processed_img, caption="Immagine preprocessata", use_container_width=True)

                with col2:
                    # Crea e mostra il grafico delle top 10 razze
                    plot_top_breeds(prediction_probs, unique_breeds, n=10)
            else:
                st.error(
                    f"❌ L'immagine non sembra contenere un cane (confidenza cane: {dog_confidence*100:.1f}%)")
                st.warning(
                    "Per favore carica un'immagine con un cane per vedere la classificazione delle razze.")

        # Aggiungi un separatore tra le immagini
        if i < len(uploaded_files) - 1:
            st.markdown("---")
else:
    # Mostra alcune istruzioni quando non ci sono immagini caricate
    st.info("Carica le foto dei cani per vedere le previsioni e i grafici delle razze più probabili.")

    # Opzionale: mostra esempi di razze che il modello può identificare
    if unique_breeds is not None and len(unique_breeds) > 0:
        st.write("Alcune delle razze che il modello può identificare:")
        sample_breeds = unique_breeds[:10] if len(
            unique_breeds) > 10 else unique_breeds
        st.write(", ".join(sample_breeds))

# Aggiungi una nota informativa
st.markdown("---")
st.write("Questo classificatore utilizza un modello di deep learning addestrato su diverse razze di cani.")
st.write("Nota: l'accuratezza delle previsioni può variare in base alla qualità dell'immagine.")
# Aggiungi il link alla repo
st.markdown(
    "### Vuoi scoprire il processo dietro la costruzione del modello? Dai un’occhiata al notebook su GitHub! Il modello è stato realizzato con TensorFlow utilizzando il transfer learning su MobileNet 👇")
st.markdown(
    "🔗 [Codice Notebook per il lavoro sul modello su GitHub](https://github.com/Pandagan-85/ZTM-Machine-learning/blob/main/end_to_end_dog_vision.ipynb)")
