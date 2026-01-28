import streamlit as st
import pickle
import re
import string

# 1. Load the "Intelligence" (The two .pkl files)
@st.cache_resource
def load_assets():
    # Loading the brain and the translator saved in Step 1
    model = pickle.load(open('model_news.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    return model, vectorizer

model, vectorizer = load_assets()

# 2. Preprocessing Function (The logic used in your Methodology)
def clean_text(text):
    text = text.lower() # Normalization
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) # Special character removal
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# 3. User Interface (The Demo Website)
st.title("🔍 Fake News Detection System")
st.write("Using Logistic Regression to verify news authenticity.")

# User pastes news content here
news_input = st.text_area("Enter News Content:", height=200)

if st.button("Predict"):
    if news_input:
        # Clean the input text
        cleaned = clean_text(news_input)
        # Transform using the 5,000 feature vectorizer
        vectorized_text = vectorizer.transform([cleaned])
        # Model makes the decision
        prediction = model.predict(vectorized_text)
        
        # Display the result
        if prediction[0] == 1:
            st.success("Analysis: REAL NEWS")
        else:
            st.error("Analysis: FAKE NEWS")
    else:
        st.warning("Please enter some text.")
