# train_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
fake_df = pd.read_csv("archive/Fake.csv")
fake_df['label'] = "Fake"
real_df = pd.read_csv("archive/True.csv")
real_df['label'] = "Real"
df = pd.concat([fake_df, real_df], ignore_index=True)
df = df[['text','label']].dropna()

# Train model
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = make_pipeline(TfidfVectorizer(stop_words='english', max_df=0.9),
                      MultinomialNB())
model.fit(X_train, y_train)

# Evaluate
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Test Accuracy: {acc*100:.1f}%")

# Save model
joblib.dump(model, "fake_news_model.joblib")
print("Model saved as fake_news_model.joblib")
