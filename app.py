import streamlit as st
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 1. Page Config
st.set_page_config(page_title="NLP Fake News Detector", layout="wide")

# 2. Functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text

@st.cache_resource
def load_assets():
    with open('model_fake_news.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# --- UI SIDEBAR (Requirement: Measurement on Performance) ---
st.sidebar.title("Model Metrics")
st.sidebar.info("Model: Passive Aggressive Classifier")
st.sidebar.write("✅ Accuracy: 94.5%") # Ganti dengan nilai sebenar anda
st.sidebar.write("✅ Language: English")

# --- MAIN UI ---
st.title("📰 AI-Powered Fake News Detection")
st.markdown("Enter news content below to analyze its authenticity.")

user_input = st.text_area("News Content:", height=200)

if st.button("Analyze Now"):
    if user_input:
        model, vectorizer = load_assets()
        
        # Preprocessing
        cleaned = clean_text(user_input)
        vec_input = vectorizer.transform([cleaned])
        
        # Prediction
        prediction = model.predict(vec_input)[0]
        
        # UI Columns for Results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Result")
            if prediction == 0: # 0 = Fake
                st.error("🚨 THIS IS FAKE NEWS")
            else:
                st.success("✅ THIS IS REAL NEWS")
            
            # Feature: Confidence Score (Using decision_function for PAC model)
            score = model.decision_function(vec_input)
            confidence = abs(score[0]) # Nilai lebih tinggi = lebih yakin
            st.metric("Model Confidence Score", f"{confidence:.2f}")

        with col2:
            st.subheader("Word Cloud Visualization")
            # Feature: Word Cloud (Requirement: Visual Aids)
            wc = WordCloud(background_color='white', width=400, height=300).generate(cleaned)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            
    else:
        st.warning("Please paste some text first.")
