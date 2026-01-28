import streamlit as st
import pickle
import re

# 1. Page Setup
st.set_page_config(page_title="Multi-Model NLP Detector", layout="wide")

# 2. Load Models (Pastikan anda sudah save kedua-dua model .pkl)
@st.cache_resource
def load_all_models():
    # Load Vectorizer (Biasanya dikongsi)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    # Load LR Model
    with open('model_lr.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    # Load Naive Bayes Model
    with open('model_nb.pkl', 'rb') as f:
        nb_model = pickle.load(f)
    return vectorizer, lr_model, nb_model

def clean_text(text):
    return re.sub(r'\W', ' ', text.lower())

# --- UI ---
st.title("📰 Fake News Detection: LR vs Naive Bayes")
st.write("Compare how different algorithms classify the same news article.")

user_input = st.text_area("Paste News Content:", height=200)

if st.button("Compare Models"):
    if user_input:
        vec, lr, nb = load_all_models()
        cleaned = clean_text(user_input)
        vec_text = vec.transform([cleaned])
        
        # Predictions
        pred_lr = lr.predict(vec_text)[0]
        pred_nb = nb.predict(vec_text)[0]
        
        # Display Side-by-Side
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Logistic Regression")
            if pred_lr == 1:
                st.success("Result: REAL")
            else:
                st.error("Result: FAKE")
            # Show Probability
            prob_lr = lr.predict_proba(vec_text)[0]
            st.write(f"Confidence: {max(prob_lr)*100:.2f}%")

        with col2:
            st.header("Naive Bayes")
            if pred_nb == 1:
                st.success("Result: REAL")
            else:
                st.error("Result: FAKE")
            # Show Probability
            prob_nb = nb.predict_proba(vec_text)[0]
            st.write(f"Confidence: {max(prob_nb)*100:.2f}%")
            
    else:
        st.warning("Please enter text first.")
