# -*- coding: utf-8 -*-
"""Insightdemo_tradml.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IEcE728UzEdwmkKkgg9I2xLS1Dsl7Xgq
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
raw_df=pd.read_csv("data/BigML_Dataset.csv")
raw_df=raw_df.loc[(raw_df['Status'] == "paid") | (raw_df['Status'] == "defaulted")]
raw_df=raw_df.drop(["Funded Amount","Delinquent","id","Funded Date","Country Code","Paid Date","Paid Amount","Name","Use","Funded Date.year","Funded Date.month","Funded Date.month","Funded Date.day-of-month","Funded Date.day-of-week","Funded Date.hour","Funded Date.minute","Funded Date.second","Paid Date.year","Paid Date.month","Paid Date.day-of-month","Paid Date.day-of-week","Paid Date.hour","Paid Date.minute","Paid Date.second"],axis=1)
raw_df["Local Currency"].fillna("USD", inplace=True)
raw_df['Status'] = raw_df['Status'].apply(lambda x: 1 if  x == "paid"  else 0)


category_dummies=pd.get_dummies(raw_df[['Country']],prefix=['Country'])
raw_df=pd.concat([raw_df,category_dummies],axis=1).drop(['Country'],axis=1)

category_dummies=pd.get_dummies(raw_df[['Local Currency']],prefix=['Local Currency'])
raw_df=pd.concat([raw_df,category_dummies],axis=1).drop(['Local Currency'],axis=1)

category_dummies=pd.get_dummies(raw_df[['Activity']],prefix=['Activity'])
raw_df=pd.concat([raw_df,category_dummies],axis=1).drop(['Activity'],axis=1)

category_dummies=pd.get_dummies(raw_df[['Sector']],prefix=['Sector'])
raw_df=pd.concat([raw_df,category_dummies],axis=1).drop(['Sector'],axis=1)


# count_class_0, count_class_1 = raw_df["Status"].value_counts()
# df_class_0 = raw_df[raw_df["Status"] == 1]
# df_class_1 = raw_df[raw_df["Status"] == 0]
# df_class_1_over = df_class_1.sample(int(1 * count_class_0), replace=True, random_state=3)
# df_train_over = pd.concat([df_class_0, df_class_1_over], axis=0)


x_train, x_val, y_train, y_val = train_test_split(raw_df.drop(columns = ['Status']), 
                                                    raw_df['Status'], 
                                                    test_size=0.2, 
                                                    random_state=1)

import pickle
import xgboost as xgb
dtrain = xgb.DMatrix(data = x_train, label = y_train)
dval = xgb.DMatrix(data = x_val, label = y_val)

param = {'max_depth':6,
         'eta': 0.37,
         'enable_experimental_json_serialization': True,
         'silent':1,
         'objective':'binary:logistic',
         'eval_metric': 'auc',
          'gamma': 10,
          'lambda': 10,
          'alpha': 20,
          'subsample' : 0.8,
          'min_child_weight': 5,
          'colsample_bytree' :0.79,
          ,'colsample_bynode' : 0.6
          ,'colsample_bylevel':0.7
          #,'scale_pos_weight' : 0.276,
          'maximize' : 'FALSE'
        ,'n_jobs' : -1
         #,'base_score' : ???
         #,'max_delta_step' : 6.0
        }

watchlist = [(dtrain, 'train'), (dval, 'eval')]
num_round = 50 
bst = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds = 20)

filename = 'xgboost.sav'
pickle.dump(bst, open(filename, 'wb'))

