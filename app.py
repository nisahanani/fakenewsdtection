import streamlit as st
import pickle
import re
import string

# 1. Load model & vectorizer menggunakan pickle
@st.cache_resource
def load_assets():
    model = pickle.load(open("model_news.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, tfidf = load_assets()

# 2. Fungsi pembersihan teks (Sama seperti fasa training)
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

st.title("📰 Fake News Detection System")
st.write("This application predicts whether a news article is fake or real using NLP.")

# Input text
news_text = st.text_area("Enter news article text:", height=200)

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Proses pembersihan -> transform -> predict
        cleaned_text = clean_text(news_text)
        text_vector = tfidf.transform([cleaned_text])
        prediction = model.predict(text_vector)

        # Paparan hasil berdasarkan label anda (0=Fake, 1=Real)
        if prediction[0] == 0:
            st.error("🔴 Result: FAKE NEWS")
        else:
            st.success("🟢 Result: REAL NEWS")
