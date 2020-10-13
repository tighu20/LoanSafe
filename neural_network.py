import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import streamlit as st
from keras.models import load_model
from tensorflow import keras
from embeddings import One_Hot_Encoder, embeddings_models
def model_performance(y_test,y_pred,y_prob):
  precision=metrics.precision_score(y_test,y_pred[0],average="binary")
  recall=metrics.recall_score(y_test,y_pred[0],average="binary")
  f1=metrics.f1_score(y_test,y_pred[0])

  print(precision,recall,f1)



def dataframe_to_numpy(df, categorical_columns, ordinal_columns):
    """Converts data in dataframe to numpy array that includes:
    1) one-hot encoded categorical columnhs
    2) normalized ordinal columns"""

    le = preprocessing.LabelEncoder()
    Xtmp = (df[categorical_columns].copy()).apply(lambda col: le.fit_transform(col))

    ohe = preprocessing.OneHotEncoder(handle_unknown="ignore", sparse=False)
    X = np.transpose(ohe.fit_transform(Xtmp))

    for c in ordinal_columns:
        X = np.vstack([X, normalize_column(df[c])])

    return np.transpose(X)

def normalize_column(df_column, center_at_zero=False):
    """Converts an unnormalized dataframe column to a normalized
    1D numpy array
    Default: normalizes between [0,1]
    (center_at_zero == True): normalizes between [-1,1] """

    normalized_array = np.array(df_column, dtype="float64")
    amax, amin = np.max(normalized_array), np.min(normalized_array)
    normalized_array -= amin
    if center_at_zero:
        normalized_array *= 2.0 / (amax - amin)
        normalized_array -= 1.0
    else:
        normalized_array *= 1.0 / (amax - amin)
    return normalized_array

def save_embeddings(
    _output_path, _trained_model, _categorical_columns, _categorical_encoder
):
    """Save the embeddings in an output directory,
    return a dictionary with the embeddings information"""
    embs = {}
    for c in _categorical_columns:
        embs[c] = _trained_model.extract_weights(c.replace(" ", "_") + "_embedding")
        column_names = _categorical_encoder.retrieve_names(
            c, range(0, embs[c].shape[0])
        )
        df_embs = pd.DataFrame(data=embs[c])
        df_embs[c] = column_names
        df_embs.to_csv(_output_path + c + "_embedding.csv", mode="w", index=False)
    return embs

raw_df=pd.read_csv("BigML_Dataset.csv")
st.title('This is the Raw Dataset')
st.write(raw_df.head(5))
raw_df=raw_df.loc[(raw_df['Status'] == "paid") | (raw_df['Status'] == "defaulted")]

raw_df=raw_df.drop(["Funded Amount","Delinquent","id","Funded Date","Country Code","Paid Date","Paid Amount","Name","Use","Funded Date.year","Funded Date.month","Funded Date.month","Funded Date.day-of-month","Funded Date.day-of-week","Funded Date.hour","Funded Date.minute","Funded Date.second","Paid Date.year","Paid Date.month","Paid Date.day-of-month","Paid Date.day-of-week","Paid Date.hour","Paid Date.minute","Paid Date.second"],axis=1)

raw_df["Local Currency"].fillna("USD", inplace=True)


useful_columns = [
    "Loan Amount",
    "Country",
    "Sector",
    "Activity",
    "Status",
    "Local Currency",
]

categorical_columns = [
    "Country",
    "Sector",
    "Activity",
    "Local Currency",
]

ordinal_columns = ["Loan Amount"]
df_train, df_test = train_test_split(raw_df, test_size=0.3,   random_state=55)

count_class_0, count_class_1 = df_train["Status"].value_counts()
df_class_0 = df_train[df_train["Status"] == "paid"]
df_class_1 = df_train[df_train["Status"] == "defaulted"]
df_class_1_over = df_class_1.sample(int(1 * count_class_0), replace=True, random_state=3)
df_train_over = pd.concat([df_class_0, df_class_1_over], axis=0)

train_length = len(df_train_over)


df_concatenated = pd.concat([df_train_over, df_test])


df_concatenated.tail(1).to_csv('file1.csv')

X_concatenated = dataframe_to_numpy(
        df_concatenated, categorical_columns, ordinal_columns
    )
y_concatenated = np.array(
    (
        pd.get_dummies(df_concatenated["Status"], columns=["Status"])["defaulted"]
    ).tolist()
)

# Split the train from the test arrays
X_train, X_test = X_concatenated[:train_length, :], X_concatenated[train_length:, :]
y_train, y_test = y_concatenated[:train_length], y_concatenated[train_length:]


vocabulary_sizes = [df_concatenated[c].nunique() for c in categorical_columns]
max_length = 1

categorical_encoder = One_Hot_Encoder(df_concatenated, categorical_columns)
categorical_data = categorical_encoder.encode(
    df_concatenated, categorical_columns, max_length
)

ordinal_data = [
    (normalize_column(df_concatenated[c])).reshape(-1, 1)
    for c in ordinal_columns
]

input_data = categorical_data + ordinal_data
input_data_train = [data[:train_length, :] for data in input_data]
input_data_test = [data[train_length:, :] for data in input_data]

emb_model = embeddings_models(
    vocabulary_sizes,
    max_length,
    categorical_columns,
    len(ordinal_columns),

)

train_embeddings_model = emb_model(
        input_data_train,
        y_train,
        input_data_test,
        y_test,
        50,
        500,
        [32,32,32,8],
    )

y_pred, y_prob = [], []

y_pred.append(train_embeddings_model.predict(input_data_test))
y_prob.append(train_embeddings_model.predict_prob(input_data_test))

model_performance(y_test,y_pred,y_prob)



output_embeddings = "C:/Users/Tigmanshu/PycharmProjects/insight_demo/"
embs = save_embeddings(
          output_embeddings,
          train_embeddings_model,
          categorical_columns,
          categorical_encoder,
      )

train_embeddings_model.save("plain_nn")

embs_list = list(embs.values())
deploy_embeddings_model = emb_model(
    input_data_train,
    y_train,
    input_data_test,
    y_test,
    50,
    500,
    [64,64,64,64,64,8],
    embs_list,
)
deploy_embeddings_model.save("deep_nn")

y_pred, y_prob = [], []

y_pred.append(deploy_embeddings_model.predict(input_data_test))
y_prob.append(deploy_embeddings_model.predict_prob(input_data_test))

model_performance(y_test,y_pred,y_prob)

