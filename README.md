# Hybrid Fake News Detector

This project is a **hybrid Fake News Detector** built with **Streamlit**, combining:

1. **Naive Bayes (NB) model** trained locally on kaggle dataset(https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. **BERT-based Hugging Face (HF) model** for more nuanced and recent news detection.

It allows users to **paste news text or upload a TXT/CSV file** (max 100 words) and predicts whether the news is likely **Real** or **Fake**.
It relies on hugging face model for primary detection, but in case of failure, NB model kicks in.
---
NOTE: This repository also includes local model training file and related datasets. THEY ARE NOT REQD. FOR CODE EXECUTION
## **Features**

-  Uses both a local NB model and HF BERT model.
-  Ensemble prediction:
  - If both models agree, returns the averaged confidence.
  - If they disagree, returns the model with higher confidence.
- File upload support (TXT / CSV).
- Handles up to 100 words for analysis.( Not more to handle overload and maintain simplicity)
- Streamlit-based UI for easy web interaction.

---

## **Setup & Installation**

### 1. Clone the repository


git clone <your-repo-url>
cd <your-repo-folder>

#create venv in terminal
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
#install dependencies
pip install -r requirements.txt
#optional mac os acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
Set up Hugging Face API token

Go to Hugging Face
 and login/signup.

Generate an API token.read only

Replace the HF_API_TOKEN in app_hf_nb.py with your token.
RUN FILE FROM TERMINAL
streamlit run app_hf_nb.py
