import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path
import gdown

st.set_page_config(
    page_title="PD Curves from dataset",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Modeling the Survival Embeddings and Default Probability with data from dataset"
    }
)

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


def plot_LSC(mean, upper_ci, lower_ci, row_number):
    fig, ax = plt.subplots()

    ax.plot(mean.columns, mean.iloc[0])
    ax.fill_between(mean.columns, lower_ci.iloc[0], upper_ci.iloc[0], alpha = 0.2)

    ax.set_title(f'Lifetime survival curve for client №{row_number}')
    ax.set_ylabel('Survival probability')
    ax.set_xlabel('Month')
    return fig

def plot_LPD(mean, upper_ci, lower_ci, row_number):
    fig, ax = plt.subplots()

    mean = 1 - mean
    upper_ci = 1 - upper_ci
    lower_ci = 1 - lower_ci

    ax.plot(mean.columns, mean.iloc[0])
    ax.fill_between(mean.columns, lower_ci.iloc[0], upper_ci.iloc[0], alpha = 0.2)

    ax.set_title(f'Lifetime probability of default for client №{row_number}')
    ax.set_ylabel('Probability of default')
    ax.set_xlabel('Month')
    return fig


def plot_LSC_with_streamlit(mean):
    df = pd.DataFrame([list(mean.iloc[0]), list(mean.columns)])
    df.reset_index()
    df_display = df.T.rename(columns={0:'Survival probability', 1:'Month'})
    st.line_chart(df_display, x = 'Month', y = 'Survival probability')

def plot_LPD_with_streamlit(mean):
    mean = 1 - mean
    df = pd.DataFrame([list(mean.iloc[0]), list(mean.columns)])
    df.reset_index()
    df_display = df.T.rename(columns={0:'Probability of default', 1:'Month'})
    st.line_chart(df_display, x = 'Month', y = 'Probability of default')



data = load_data()
model = load_model()
N = data.shape[0]


st.title('Plot survival embeddings with data from dataset')
st.divider()


st.header('How would you like to enter amount of rows to display?')
option = st.selectbox(
    'Select display type',
    ('Number of rows', 'Percent of dataset'))

rows_amount = 5
if 'rows_amount' not in st.session_state:
    st.session_state.rows_amount = 5

if option == 'Number of rows': 
    rows_amount = st.slider('Number of rows to display', 0, N, 5)
else:
    percent = st.slider('Percent of rows from dataset to display', 0.0, 100.0, 0.54, 0.01, format = "%f")
    rows_amount = int(N * percent / 100)

data_button = st.button("Show")
data_header = st.header(f'First {st.session_state.rows_amount} rows of dataset:')
data_table = st.dataframe(data.head(st.session_state.rows_amount))
if data_button:
    st.session_state.rows_amount = rows_amount
    data_header.header(f'First {rows_amount} rows of dataset:')
    data_table.dataframe(data.head(rows_amount))


st.divider()

with st.form("plot_form"):
    st.header('Survival embedding for client from dataset')
    st.subheader('Enter client number')
    row_number = st.number_input('Client (row) number', min_value = 0, max_value = data.shape[0])
    plot_button = st.form_submit_button("Plot survival curve")

    if plot_button:
        df = data.loc[[row_number]]
        st.caption('Selected client:')
        st.dataframe(df)
        with st.spinner("Modeling survival curve..."):  
            X = df.drop(['duration', 'loan_result'], axis = 1)
            mean, upper_ci, lower_ci = model.predict(X, return_ci = True)

            col1, col2 = st.columns(2)
            with col1:
                st.write(plot_LSC(mean, upper_ci, lower_ci, row_number))
                plot_LSC_with_streamlit(mean)
            with col2:
                st.write(plot_LPD(mean, upper_ci, lower_ci, row_number))
                plot_LPD_with_streamlit(mean)
