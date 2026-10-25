import streamlit as st
import joblib
import pandas as pd

st.title("🛡️ AI-Based Fake Review Detection")

@st.cache_resource
def load_model():
    return joblib.load("model.joblib"), joblib.load("vectorizer.joblib")

try:
    model, vectorizer = load_model()
    review = st.text_area("Paste review here:")
    if st.button("Detect"):
        vec = vectorizer.transform([review])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        if pred == 1:
            st.error(f"🚨 FAKE REVIEW - {prob[1]*100:.1f}%")
        else:
            st.success(f"✅ GENUINE REVIEW - {prob[0]*100:.1f}%")
except:
    st.error("Run python train_model.py first!")