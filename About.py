import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os.path
import gdown


st.set_page_config(
    page_title="PD Curves model",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Modeling the Probability of Survival in a loan portfolio using XGBoost"
    }
)
st.title('📉 Modeling the Probability of Survival in a loan portfolio using XGBoost')
st.header('👦 Author: Alexey Rodionov, HSE FCS, BSE203')
st.subheader('💻 Github link: https://github.com/AlexiRod')

DATA_PATH = "./data/lending_club_dataset_for_xgbse.csv"
MODEL_PATH = "./data/estimator.pkl"

@st.cache_data
def load_data(): 
    with st.spinner("Loading data..."):  

        if os.path.isfile(DATA_PATH) is False:
            with st.spinner("Downloading data from drive..."):
                url = "https://drive.google.com/uc?id=1j76gcHyVH3_Mwizlo6zzRXTIYXdTXAwF"
                output = DATA_PATH
                gdown.download(url, output, quiet = False)
                st.success("Data downloaded!")
                st.balloons()

        data = pd.read_csv(DATA_PATH, low_memory = False)
        data.drop('Unnamed: 0', axis = 1, inplace = True)
        return data
    
@st.cache_resource
def load_model():
    with st.spinner("Loading model..."):  

        if os.path.isfile(MODEL_PATH) is False:
            with st.spinner("Downloading model from drive..."):
                url = "https://drive.google.com/uc?id=1ZcnAxXa-lXoa5TpQpadWgSQFxTbV33Tj"
                output = MODEL_PATH
                gdown.download(url, output, quiet = False)
                st.success("Model downloaded!")
                st.balloons()

        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    