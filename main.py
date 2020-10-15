import pandas as pd
import streamlit as st
from keras.models import load_model
from embeddings import One_Hot_Encoder
from PIL import Image
import pickle
import xgboost as xgb
categorical_columns = [
    "Country",
    "Sector",
    "Activity",
    "Local Currency",
]

low_risk_image = Image.open('images/lowrisk.JPG')
high_risk_image = Image.open('images/highrisk.JPG')
model_infrence = Image.open('images/shap.JPG')
ordinal_columns = ["Loan Amount"]


@st.cache
#streamlit cached funtion to load ds into web app
def load_ds():
    rdf = pd.read_csv("data/BigML_Dataset.csv")
    st.header('Raw Dataset')
    return rdf

#funtion to load sidebar which contains all the features that the lending institution will input to get risk assesment
def load_sidebar(raw_df):
    arg0 = st.sidebar.selectbox('Which ML Model would you like to run?', models)
    arg1 = st.sidebar.slider("Please Select the Loan Amount", 100, 2000)
    arg2 = st.sidebar.selectbox("Please Select the Applicant Country", country_list)
    arg5 = st.sidebar.selectbox("Please Select the Local Currency", currency_list)
    arg3 = st.sidebar.selectbox("Please Select the Applicant Activity", activity_list)
    arg4 = st.sidebar.selectbox("Please Select the Applicant Sector", sector_list)
    b1 = st.sidebar.button("Get Predictions", key=1)
    b2 = st.button("XGB Model Infrence", key=1)

    return [arg0, arg1, arg2, arg3, arg4, arg5], [b1, b2]

# to load saved machine learning models using pickle from saved models directory
def fetch_model(model_name):
    loaded_model = pickle.load(open('saved_models/'+model_name+'.sav', 'rb'))
    return loaded_model

# get a empty sample_df which can be fitted with all the input values  
def get_sampledf(args):
    input_df = pd.read_csv("data/sample_df.csv")
    ColNameList = ["Country_"+args[2],"Local Currency_"+args[5],"Activity_"+args[3],"Sector_"+args[4]]
    input_df.loc[0, ColNameList] = 1
    input_df.loc[0, "Loan Amount"] = arguments[1]

    return input_df

# load images on the application based on the result of the selected machine learning model
def show_risk_image(result):
    if int(result) == 0:
        st.image(low_risk_image, caption='Default Risk', use_column_width=True)
    else:
        st.image(high_risk_image, caption='Default Risk', use_column_width=True)

st.title('Loan Safe : ML powered Risk Assesment ')
st.header("Loading Big ML Dataset...")
raw_df = load_ds()

country_list = raw_df['Country'].value_counts().index.tolist()
activity_list = raw_df['Activity'].value_counts().index.tolist()
sector_list = raw_df['Sector'].value_counts().index.tolist()
currency_list = raw_df['Local Currency'].value_counts().index.tolist()
models = ["Random Forrest", "Logistic Regression", "XGBoost", "Neural Network",
          "Deep Neural Network w/ pretrained Embeddings"]

st.subheader("Dataset Retrieved!")
st.dataframe(raw_df.head(10))

arguments, buttons = load_sidebar(raw_df)

max_length = 1

# if conditional tree to drive code based on buttons clicked on the streamlit web application which calls the above defined funtions to make predictions 
# using the machine learning model

if buttons[0] and arguments[0] == "Neural Network":
    st.subheader("Fetching Neural Network Model")
    model = load_model('saved_models/plain_nn')

    print("model loaded")

    df_sample = pd.read_csv("data/sample_dataset.csv")

    df_sample.loc[0, 'Country'], df_sample.loc[0, 'Local Currency'], df_sample.loc[0, 'Loan Amount'], df_sample.loc[
        0, 'Activity'], df_sample.loc[0, 'Sector'] = arguments[2], arguments[5], arguments[1], arguments[3], arguments[4]

    categorical_encoder_demo = One_Hot_Encoder(df_sample, categorical_columns)
    categorical_data_demo = categorical_encoder_demo.encode(
        df_sample, categorical_columns, max_length
    )
    ordinal_data_demo = [
        df_sample[c]
        for c in ordinal_columns
    ]
    imput_data_demo = categorical_data_demo + ordinal_data_demo
    input_data_demo_test = [data for data in imput_data_demo]

    y_pred = []

    y_pred.append(model.predict(input_data_demo_test))
    st.header("Based on Data Provided...")

    show_risk_image(y_pred[0])

elif buttons[0] and arguments[0] == "Deep Neural Network w/ pretrained Embeddings":
    st.subheader("Fetching Deep Neural Network Model")
    model = load_model('saved_models/deep_nn')
    print("model loaded")
    df_sample = pd.read_csv("data/sample_dataset.csv")

    df_sample.loc[0, 'Country'], df_sample.loc[0, 'Local Currency'], df_sample.loc[0, 'Loan Amount'], df_sample.loc[
        0, 'Activity'], df_sample.loc[0, 'Sector'] = arguments[2], arguments[5], arguments[1], arguments[3], arguments[4]

    categorical_encoder_demo = One_Hot_Encoder(df_sample, categorical_columns)
    categorical_data_demo = categorical_encoder_demo.encode(
        df_sample, categorical_columns, max_length
    )
    ordinal_data_demo = [
        df_sample[c]
        for c in ordinal_columns
    ]
    imput_data_demo = categorical_data_demo + ordinal_data_demo
    input_data_demo_test = [data for data in imput_data_demo]

    y_pred = []

    y_pred.append(model.predict(input_data_demo_test))
    st.header("Based on Data Provided...")
    show_risk_image(y_pred[0])

elif buttons[0] and arguments[0] == "Random Forrest":
    st.subheader("Fetching Random Forrest Model")
    rand_forrest = fetch_model("rand_forrest")
    x = get_sampledf(arguments)
    res=rand_forrest.predict(x)
    st.header("Based on Data Provided...")
    show_risk_image(res[0])

elif buttons[0] and arguments[0] == "Logistic Regression":
    st.subheader("Fetching Logistic Regression Model")
    log_reg = fetch_model("log_reg")
    x = get_sampledf(arguments)
    res=log_reg.predict(x)
    st.header("Based on Data Provided...")
    show_risk_image(res[0])

elif buttons[0] and arguments[0] == "XGBoost":
    st.subheader("Fetching XGBoost Model")
    xgbm = fetch_model("xgboost")
    x = get_sampledf(arguments)
    dmat = xgb.DMatrix(data=x)
    res=xgbm.predict(dmat)[0]
    st.header("Based on Data Provided...")
    show_risk_image(1 if res>0.5 else 0)

if buttons[1]:
    st.image(model_infrence, caption='SHAP Summary Plot', use_column_width=True)
