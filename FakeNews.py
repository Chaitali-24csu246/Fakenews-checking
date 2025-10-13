#Fake news text detector using kaggle datasets
# fake_news_detector_real.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk

nltk.download('punkt')

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.write("Paste news text or upload a file to check if it’s likely real or fake.")

# --- Load dataset ---
@st.cache_data(show_spinner=True)
def load_dataset(fake_path="archive/Fake.csv", real_path="archive/True.csv"):
    fake_df = pd.read_csv(fake_path)
    fake_df['label'] = "Fake"
    real_df = pd.read_csv(real_path)
    real_df['label'] = "Real"
    combined = pd.concat([fake_df, real_df], ignore_index=True)
    combined = combined[['text','label']]  # Keep only relevant columns
    combined = combined.dropna(subset=['text'])
    return combined

df = load_dataset()

# --- Train classifier ---
@st.cache_resource(show_spinner=True)
def train_model(data):
    X = data['text']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = make_pipeline(TfidfVectorizer(stop_words='english', max_df=0.9), MultinomialNB())
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

model, accuracy = train_model(df)
st.write(f"Model accuracy on test data: {accuracy*100:.1f}%")

# --- File preprocessing ---
def preprocess_file(file):
    if file.type == "text/csv":
        df_file = pd.read_csv(file)
        text = " ".join(df_file.iloc[0].astype(str).values)  # take first row
    else:
        text = file.read().decode("utf-8")
    words = nltk.word_tokenize(text)
    text_limited = " ".join(words[:100])  # limit to 100 words
    return text_limited

# --- User input ---
st.subheader("Check a single news article")
news_text = st.text_area("Paste news text here:")

st.subheader("Or upload a file (txt or csv, max 100 words)")
uploaded_file = st.file_uploader("Choose a file", type=["txt","csv"])

# --- Prediction ---
if st.button("Check News"):
    if news_text.strip() != "":
        text_to_check = news_text
    elif uploaded_file is not None:
        text_to_check = preprocess_file(uploaded_file)
        st.write(f"Using first 100 words of the file for analysis:\n{text_to_check}")
    else:
        st.warning("Please enter news text or upload a file.")
        st.stop()
    
    prediction = model.predict([text_to_check])[0]
    prediction_proba = model.predict_proba([text_to_check])[0]
    confidence = max(prediction_proba) * 100
    
    st.markdown(f"**Prediction:** {prediction}")
    st.markdown(f"**Confidence:** {confidence:.1f}%")

