"""
@file: Salary_Predict.py
@project: pythonProject
@author: JennyChou
@date_created: 9/1/2020
@brief: Task is to predict salary given job description
"""

import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


def rank_feature(feature):
    group_salary = feature.describe()['salary']
    group_salary_mean = group_salary['mean'].apply(np.floor)
    group_salary_rank = group_salary_mean.rank(ascending=False).astype(int)
    return dict(group_salary_rank)


# Load data
# Extract zip file
# filename = "SalaryPredictions.zip"
# zipRef = zipfile.ZipFile(filename, 'r')
# zipRef.extractall()
# zipRef.close()

# Load csv files into pandas DataFrame
train = pd.read_csv(os.path.join("data", "train_features.csv"), header=0, index_col=0)
train['salary'] = pd.read_csv(os.path.join("data", "train_salaries.csv"), header=0, index_col=0)
test_features = pd.read_csv(os.path.join("data", "test_features.csv"), header=0, index_col=0)


# Data cleaning
# Check for NaN
print("Any NaN in train dataframe:", train.isnull().values.any())
print("Any NaN in test dataframe:", test_features.isnull().values.any())

# Check for salary <= 0 and remove
print("Any invalid salary in df:", train['salary'].le(0).values.any())
invalid_salaries = train.index[train['salary'].le(0)].tolist()
print("Remove following jobs with invalid salary:", invalid_salaries)
train = train.drop(invalid_salaries)

# Drop duplicated rows
train.drop_duplicates(inplace=True)


# Explore data
# Data shape
print("train features: ", train.shape)
print("test features: ", test_features.shape)

print(train.head(3))
print(train.dtypes)
print(train.columns)
print(train.groupby(['jobType']).size())
print(train.groupby(['degree']).size())
print(train.groupby(['major']).size())
print(train.groupby(['industry']).size())
print(train['yearsExperience'].describe())
print(train['milesFromMetropolis'].describe())

train_data = train.copy()
train_data['companyId'] = train_data['companyId'].apply(lambda x: int(x.replace("COMP", "")))
test_features['companyId'] = test_features['companyId'].apply(lambda x: int(x.replace("COMP", "")))

# # One hot encoding all non-numerical features
# train_data = train.copy()
# for feature in ['jobType', 'degree', 'major', 'industry']:
#     for item in train_data[feature].unique():
#         col_name = feature + "_" + item
#         train_data[col_name] = (train_data[feature]==item).astype(int)
# train_data = train_data.drop(['jobType', 'degree', 'major', 'industry'], axis=1)
# print(train_data.corr())

# # convert to numerical representation
# train_data = train.copy()
# encoder = sklearn.preprocessing.LabelEncoder()
# for feature in ['jobType', 'degree', 'major', 'industry']:
#     train_data[feature]=encoder.fit_transform(train_data[feature])
# tmp1=train_data.corr()


# convert to numerical representation according to their rank in their feature
for feature in ['jobType', 'degree', 'major', 'industry']:
    feature_rank = rank_feature(train_data.groupby([feature]))
    train_data[feature]=train_data[feature].apply(lambda x: feature_rank[x])
    test_features[feature]=test_features[feature].apply(lambda x: feature_rank[x])

# tmp=train_data.corr()

train_data = train_data.drop(columns=['companyId'])
test_features = test_features.drop(columns=['companyId'])


# plt.figure()
# train_data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False, fontsize=8)
# # plt.boxplot(housing.values, labels=housing.columns)

# plt.figure()
# scatter = train_data[['milesFromMetropolis', 'salary']]
# pd.plotting.scatter_matrix(train_data)
# plt.show()

train_salary = train_data.pop('salary')


models, names, results = [], [], []
# models.append(('LR', LinearRegression()))
# models.append(('LASSO', Lasso()))
# models.append(('EN', ElasticNet()))
# models.append(('KNN', KNeighborsRegressor(n_neighbors=100)))
# models.append(('CART', DecisionTreeRegressor()))
# # models.append(('SVR', SVR()))
#
# actual = train_salary.values.astype(float)#.reshape(len(train_salary),-1)
# for name, model in models:
#     model.fit(train_data, train_salary)
#     predict = model.predict(train_data)
#     mse = ((predict-actual)**2).mean()
#     print("%s: MSE=%.3f" % (name, mse))
# """
# LR: mean=-392.373, std=2.064
# LASSO: mean=-393.193, std=2.080
# EN: mean=-399.804, std=2.150
# KNN: mean=-438.713, std=1.737
# CART: mean=-658.190, std=3.125
# SVR: takes forever
# """


# pipelines = []
# pipelines.append(("scaledLR",
#                   Pipeline([("Scaler", StandardScaler()),
#                             ("LR", LinearRegression())])))
# pipelines.append(("scaledLASSO",
#                   Pipeline([("Scaler", StandardScaler()),
#                             ("LASSO", Lasso())])))
# pipelines.append(("scaledEN",
#                   Pipeline([("Scaler", StandardScaler()),
#                             ("EN", ElasticNet())])))
# pipelines.append(("scaledKNN",
#                   Pipeline([("Scaler", StandardScaler()),
#                             ("KNN", KNeighborsRegressor(n_neighbors=100))])))
# pipelines.append(("scaledCART",
#                   Pipeline([("Scaler", StandardScaler()),
#                             ("CART", ExtraTreesRegressor())])))
# pipelines.append(("scaledSVR",
#                   Pipeline([("Scaler", StandardScaler()),
#                             ("SVR", SVR())])))
# actual = train_salary.values.astype(float)#.reshape(len(train_salary),-1)
# for name, model in pipelines:
#     model.fit(train_data, train_salary)
#     predict = model.predict(train_data)
#     mse = ((predict-actual)**2).mean()
#     print("%s: MSE=%.3f" % (name, mse))
# """
# scaledLR: mean=-392.373, std=2.064
# scaledLASSO: mean=-397.218, std=2.243
# scaledEN: mean=-509.142, std=3.354
# scaledKNN:
# scaledCART: mean=-519.673, std=2.268
# scaledSVR
# """


ensembles = []
## Boosting
# ensembles.append(('scaledAB',
#                   Pipeline([('Scaler', StandardScaler()),
#                             ('AB', AdaBoostRegressor())])))
# ensembles.append(('scaledGBR',
#                   Pipeline([('Scaler', StandardScaler()),
#                             ('GBR', GradientBoostingRegressor())])))
# ## Bagging
# ensembles.append(('scaledRF',
#                   Pipeline([('Scaler', StandardScaler()),
#                             ('RF', RandomForestRegressor())])))
# ensembles.append(('scaledET',
#                   Pipeline([('Scaler', StandardScaler()),
#                             ('ET', ExtraTreesRegressor())])))
# actual = train_salary.values.astype(float)#.reshape(len(train_salary),-1)
# for name, model in ensembles:
#     model.fit(train_data, train_salary)
#     predict = model.predict(train_data)
#     mse = ((predict-actual)**2).mean()
#     mae = abs(predict-actual).mean()
#     print("%s: MSE=%.3f" % (name, mse, mae))
# """
# scaledAB: mean=-547.508, std=8.046
# scaledGBR: mean=-366.483, std=2.171
# scaledRF: mean=-446.668, std=1.408
# scaledET: mean=-519.577, std=2.215

# scaledAB: MSE=540.137
# scaledGBR: MSE=366.223
# scaledRF: MSE=141.635
# scaledRF: MSE=141.607
# scaledET: MSE=109.461, MAE=5.999165538538077
# """


ensembles = []
# ## Boosting
# ensembles.append(('AB', AdaBoostRegressor()))
# ensembles.append(('GBR', GradientBoostingRegressor()))
## Bagging
# ensembles.append(('RF', RandomForestRegressor()))
ensembles.append(('ET', ExtraTreesRegressor()))
results.clear()
names.clear()
actual = train_salary.values.astype(float)#.reshape(len(train_salary),-1)
for name, model in ensembles:
    model.fit(train_data, train_salary)
    predict = model.predict(train_data)
    mse = ((predict-actual)**2).mean()
    mae = abs(predict-actual).mean()
    print("%s: MSE=%.3f, MAE=%.3f" % (name, mse, mae))
# """
# AB: MSE=540.320
# GBR: MSE=366.223
# RF: MSE=141.631
# ET: MSE=109.461, MAE=5.999
# """


predictions = model.predict(test_features).round(3)
test_salary = pd.DataFrame(predictions, index=test_features.index)
test_salary.to_csv(os.path.join("data", "test_salary_prediction.csv"))
