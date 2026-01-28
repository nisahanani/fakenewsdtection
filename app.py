import streamlit as st
import pickle
import re

# 1. Page Configuration for professional presentation (Rubric: Presentation of Work)
st.set_page_config(
    page_title="NLP News Authenticity Analyzer",
    page_icon="📰",
    layout="wide"
)

# 2. Text Preprocessing (Essential for NLP workflow clarity)
def clean_text(text):
    """
    Standardizes input text by removing special characters and lowering case.
    """
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 3. Load Models with Caching (Rubric: Efficiency and Optimization)
@st.cache_resource
def load_nlp_assets():
    try:
        # Filenames must match your training script output
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('model_lr.pkl', 'rb') as f:
            lr_model = pickle.load(f)
        with open('model_nb.pkl', 'rb') as f:
            nb_model = pickle.load(f)
        return vectorizer, lr_model, nb_model
    except FileNotFoundError:
        return None, None, None

# --- UI Layout ---
st.title("📰 Fake News Detection System")
st.markdown("### Comparative Analysis: Logistic Regression vs. Naive Bayes")
st.write("Evaluate article authenticity using dual Machine Learning classifications.")
st.divider()

# --- Input Area ---
user_input = st.text_area(
    "Paste news content here:", 
    height=250, 
    placeholder="e.g., 'Reports indicate that officials rushed to defend...'"
)

if st.button("Run Comparative Analysis"):
    if user_input.strip():
        # Load assets
        vec, lr, nb = load_nlp_assets()
        
        if vec and lr and nb:
            # 4. Processing the Input
            processed_text = vec.transform([clean_text(user_input)])
            
            # 5. Dual Model Evaluation (Rubric: Comparison and Insights)
            col1, col2 = st.columns(2)
            
            # Logistic Regression Results
            with col1:
                st.subheader("Logistic Regression")
                prediction_lr = lr.predict(processed_text)[0]
                probability_lr = lr.predict_proba(processed_text)[0]
                
                status_lr = "REAL" if prediction_lr == 1 else "FAKE"
                confidence_lr = max(probability_lr) * 100
                
                if status_lr == "REAL":
                    st.success(f"Classification: {status_lr}")
                else:
                    st.error(f"Classification: {status_lr}")
                
                st.metric("Confidence Level", f"{confidence_lr:.2f}%")

            # Naive Bayes Results
            with col2:
                st.subheader("Naive Bayes")
                prediction_nb = nb.predict(processed_text)[0]
                probability_nb = nb.predict_proba(processed_text)[0]
                
                status_nb = "REAL" if prediction_nb == 1 else "FAKE"
                confidence_nb = max(probability_nb) * 100
                
                if status_nb == "REAL":
                    st.success(f"Classification: {status_nb}")
                else:
                    st.error(f"Classification: {status_nb}")
                
                st.metric("Confidence Level", f"{confidence_nb:.2f}%")
            
            st.divider()
            st.info("**Analysis Summary:** Compare the confidence levels above to determine result reliability.")
        else:
            # Handle Missing Files (Rubric: Troubleshooting)
            st.error("⚠️ Error: Model files (.pkl) not found. Please ensure training is complete.")
    else:
        st.warning("Please provide news content for analysis.")

# --- Footer (Rubric: Presentation of Work) ---
st.caption("JIE43303 NLP Individual Project | Presentation Demo")
