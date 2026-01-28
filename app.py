import streamlit as st
import joblib

# Load model & vectorizer
model = joblib.load("model_lr.pkl")
tfidf = joblib.load("tfidf.pkl")

st.title("📰 Fake News Detection System")
st.write("This application predicts whether a news article is fake or real using NLP.")

# Input text
news_text = st.text_area("Enter news article text:")

# Prediction
if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        text_vector = tfidf.transform([news_text])
        prediction = model.predict(text_vector)

        if prediction[0] == 1:
            st.error("🔴 Fake News")
        else:
            st.success("🟢 Real News")
