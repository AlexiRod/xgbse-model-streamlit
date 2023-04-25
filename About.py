import streamlit as st
import numpy as np
import pandas as pd

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
