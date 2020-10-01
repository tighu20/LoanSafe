import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model
from sklearn.model_selection import train_test_split

from embeddings import One_Hot_Encoder

categorical_columns = [
    "Country",
    "Sector",
    "Activity",
    "Local Currency",
]
st.title('Loan Safe : ML powered Risk Assesment ')

ordinal_columns = ["Loan Amount"]
st.header("Loading Big ML Dataset...")

raw_df=pd.read_csv("BigML_Dataset1.csv")

st.subheader("Dataset load Done!")

st.header('Raw Dataset')

st.dataframe(raw_df.head(8))



models=["Random Forrest", "Logistic Regression", "XGBoost", "Neural Network", "Deep Neural Network w/ pretrained Embeddings"]
max_length = 1
option = st.sidebar.selectbox(
    'Which ML Model would you like to run?', models)

country_list=raw_df['Country'].value_counts().index.tolist()
activity_list=raw_df['Activity'].value_counts().index.tolist()
sector_list=raw_df['Sector'].value_counts().index.tolist()
currency_list=raw_df['Local Currency'].value_counts().index.tolist()

arg1=st.sidebar.slider("Please Select the Loan Amount", 100, 2000)

arg2=st.sidebar.selectbox("Please Select the Applicant Country",country_list)

arg5=st.sidebar.selectbox("Please Select the Local Currency",currency_list)

arg3=st.sidebar.selectbox("Please Select the Applicant Activity",activity_list)

arg4=st.sidebar.selectbox("Please Select the Applicant Sector",sector_list)
b1=st.sidebar.button("Get Predictions", key=1)
if b1 and option=="Neural Network":
    st.subheader("Fetching Neural Network Model")
    model = load_model('plain_nn')

    print("model loaded")

    df_demo = pd.read_csv("file1.csv")

    df_demo.loc[0, 'Country'], df_demo.loc[0, 'Local Currency'], df_demo.loc[0, 'Loan Amount'], df_demo.loc[0, 'Activity'], df_demo.loc[0, 'Sector'] = arg2, arg5, arg1, arg3, arg4


    categorical_encoder_demo = One_Hot_Encoder(df_demo, categorical_columns)
    categorical_data_demo = categorical_encoder_demo.encode(
        df_demo, categorical_columns, max_length
    )
    ordinal_data_demo = [
        df_demo[c]
        for c in ordinal_columns
    ]
    imput_data_demo = categorical_data_demo + ordinal_data_demo
    input_data_demo_test = [data for data in imput_data_demo]

    y_pred = []

    y_pred.append(model.predict(input_data_demo_test))
    st.header("Based on Data Provided...")
    if y_pred[0] == float(0):
        st.subheader('Given Applicant has low prob of risk')
    else:


        st.subheader("Given Applicant has high Default Risk")


if b1 and option=="Deep Neural Network w/ pretrained Embeddings":
    st.subheader("Fetching Deep Neural Network Model")
    model1 = load_model('deep_nn')
    print("model loaded")
    df_demo = pd.read_csv("file1.csv")

    df_demo.loc[0, 'Country'], df_demo.loc[0, 'Local Currency'], df_demo.loc[0, 'Loan Amount'], df_demo.loc[
        0, 'Activity'], df_demo.loc[0, 'Sector'] = arg2, arg5, arg1, arg3, arg4

    categorical_encoder_demo = One_Hot_Encoder(df_demo, categorical_columns)
    categorical_data_demo = categorical_encoder_demo.encode(
        df_demo, categorical_columns, max_length
    )
    ordinal_data_demo = [
        df_demo[c]
        for c in ordinal_columns
    ]
    imput_data_demo = categorical_data_demo + ordinal_data_demo
    input_data_demo_test = [data for data in imput_data_demo]

    y_pred = []

    y_pred.append(model1.predict(input_data_demo_test))
    st.header("Based on Data Provided...")
    if y_pred[0] == float(0):
        st.subheader('Given Applicant has low prob of risk')
    else:

        st.subheader("Given Applicant has high Default Risk")










