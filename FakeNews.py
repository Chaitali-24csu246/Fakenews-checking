# app_hf_nb.py
import streamlit as st
import pandas as pd
import nltk
import requests
import os
import joblib

# Streamlit page setup
st.set_page_config(page_title="Fake News Detector (Hybrid)", page_icon="📰")
st.title("📰 Fake News Detector (Hybrid: HF + NB)")
st.write("Paste news text or upload a file (TXT/CSV, max 100 words) to check if it’s likely real or fake.")

# --- Hugging Face API setup ---
HF_API_URL = "https://api-inference.huggingface.co/models/Pulk17/Fake-News-Detection"
HF_API_TOKEN = "ADD YOUR HUGGING FACE API TOKEN READ ONLY HERE"  # Replace with your HF token
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# --- NLTK setup for file processing ---
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)
nltk.download('punkt', quiet=True)

# --- Preprocess uploaded file ---
def preprocess_file(file):
    text = ""
    if file.type == "text/csv":
        df_file = pd.read_csv(file)
        text = " ".join(df_file.iloc[0].astype(str).values)
    else:
        text = file.read().decode("utf-8")
    words = nltk.word_tokenize(text)
    return " ".join(words[:100])

# --- Hugging Face prediction ---
def predict_bert(text):
    payload = {"inputs": text}
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    try:
        hf_output = response.json()       # Outer list
        hf_output = hf_output[0]          # Inner list
        best = max(hf_output, key=lambda x: x['score'])  # Pick the one with highest score
        raw_label = best['label']
        confidence = best['score']
        label_map = {"LABEL_0": "Fake", "LABEL_1": "Real"}
        label = label_map.get(raw_label, raw_label)
        return label, confidence
    except Exception as e:
        st.error(f"Hugging Face API error: {response.text}")
        return None, 0

# --- Load NB model ---
nb_model = joblib.load("fake_news_model.joblib")

# --- NB prediction function ---
def predict_nb(text):
    probs = nb_model.predict_proba([text])[0]
    label = nb_model.classes_[probs.argmax()]
    confidence = max(probs)
    return label, confidence

# --- Ensemble function ---
def ensemble_predict(text):
    nb_label, nb_conf = predict_nb(text)
    bert_label, bert_conf = predict_bert(text)

    # If both agree, return that
    if nb_label == bert_label:
        return nb_label, (nb_conf + bert_conf)/2
    # Otherwise, return the one with higher confidence
    if nb_conf > bert_conf:
        return nb_label, nb_conf
    else:
        return bert_label, bert_conf

# --- User input ---
news_text = st.text_area("Paste news text here:")
uploaded_file = st.file_uploader("Or upload a file (txt or csv, max 100 words)", type=["txt","csv"])

if st.button("Check News"):
    if news_text.strip() != "":
        text_to_check = news_text
    elif uploaded_file is not None:
        text_to_check = preprocess_file(uploaded_file)
        st.write(f"Using first 100 words of the file for analysis:\n{text_to_check}")
    else:
        st.warning("Please enter news text or upload a file.")
        st.stop()
    
    # --- Prediction ---
    label, confidence = ensemble_predict(text_to_check)
    if label:
        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Confidence:** {confidence*100:.1f}%")


