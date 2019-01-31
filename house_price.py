# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 23:36:09 2019

@author: vlade
"""

### ToDo: Find way to fill in missing data, sort out variables - i.e. which ones are categorical and which ones are dummies (this means
### going through the dataset in excel and labelling which ones are categorical and which variables should be dummies), feature engineering,
### looking at which variables are going to be the most important in ML (examine correlations and graph), do regression etc.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

# Import data
os.chdir("C:\\Users\\vlade\\Documents\\Python Scripts\\Kaggle\\Housing Prediction")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Create dataframes and copies
df = train.copy()
train.drop(["Id"], inplace = True, axis = 1)
df.drop(["Id"], inplace = True, axis = 1)

# Examine missing data
def missing_dat(df):
    plt.figure(figsize = (16, 8))
    sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
    plt.show()

missing_dat(train)

# Take log of sale price to resolve skewness
sns.distplot(train["SalePrice"])
plt.show()
df['SalePrice'] = np.log1p(train['SalePrice']) 
sns.distplot(train["SalePrice"])
plt.show()
print("Improvement in skeness")

# Examine relationships of numerical data
num = train.select_dtypes(exclude = ["object"])

def k_larg_corr(df, k, graph):
    corr_mat = df.corr()
    n = k
    cols = corr_mat.nlargest(n, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(train[cols].values.T)
    return cols
    
    if graph == True:
        plt.figure(figsize = (16, 12))
        sns.set(font_scale=1.00)
        hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                         annot_kws={'size': 10}, yticklabels = cols.values, 
                         xticklabels = cols.values)
        plt.show()

dat = k_larg_corr(num, 20, False) # Look at top 20 correlated variables

for i in range(len(dat)):
    sns.scatterplot(x = num[dat[i]], y = num["SalePrice"])
    plt.show()
    

# Transform categorical data into dummy variables
cat = train.select_dtypes(include = ["object"]).columns
train = pd.get_dummies(train, columns = cat, drop_first = True)

# Looking at most relevant features for target
corr_mat = train.corr()
f, ax = plt.subplots(figsize = (16, 12))
sns.heatmap(corr_mat, vmax = .8, square = True)

# Zoomed heatmap - Fix this
cols = k_larg_corr(num, 11, False)

### Regression analysis

# Create train test split & drop nans
df = train[cols].dropna()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[cols].drop(labels = "SalePrice", axis = 1), df['SalePrice'],
                                                    test_size = 0.30, random_state = 101)

# Model 1) Simple linear regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
coeff_df = pd.DataFrame(reg.coef_, df.drop(labels = "SalePrice", axis = 1).columns, columns = ['Coefficient'])

reg.score(X_train, y_train)
reg_pred = reg.predict(X_test)

mse = mean_squared_error(y_test, reg_pred)

sns.distplot((y_test - reg_pred),bins=50)
plt.scatter(y_test, reg_pred)

# Model 2) Gradient boosting






