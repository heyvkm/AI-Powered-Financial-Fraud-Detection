import joblib
import streamlit as st

from config import MODEL_PATH, ENCODER_PATH


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_encoder():
    return joblib.load(ENCODER_PATH)
