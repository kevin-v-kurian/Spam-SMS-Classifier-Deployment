import streamlit as st
import joblib
import string
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer

# Load once (download stopwords if not present)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ---------- Preprocessing function ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    words = [w for w in text.split() if w not in stopwords.words('english')]
    return " ".join(words)

# ---------- Load model + vectorizer ----------
# (Export these from your notebook as spam_model.pkl and vectorizer.pkl)
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Spam Detector", page_icon="✉️", layout="centered")

st.title("📩 Spam SMS Detector")
st.write("Enter a message below and check if it's spam or not:")

user_input = st.text_area("Your Message", height=100)

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        cleaned = clean_text(user_input)
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]
        
        if prediction == 1:  # assuming 1 = spam
            st.error("🚨 This message looks like **SPAM**.")
        else:
            st.success("✅ This message looks **NOT SPAM**.")
