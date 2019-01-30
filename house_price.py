# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 23:36:09 2019

@author: vlade
"""

### ToDo: Find way to fill in missing data, sort out variables - i.e. which ones are categorical and which ones are dummies (this means
### going through the dataset in excel and labelling which ones are categorical and which variables should be dummies), feature engineering,
### looking at which variables are going to be the most important in ML (examine correlations and graph), do regression etc.

### Note on variable interactions - could there be non-linear effects with variables such as quality variables which go from 1-10? Should graph this.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import data
os.chdir("C:\\Users\\vlade\\Documents\\Python Scripts\\Kaggle\\Housing Prediction")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Create dataframes and copies
df = train.copy()
train.drop(["Id"], inplace = True, axis = 1)
df.drop(["Id"], inplace = True, axis = 1)

# Examine missing data
plt.figure(figsize = (16, 8))
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()

# Take log of sale price to resolve skewness
sns.distplot(df["SalePrice"])
plt.show()
df['SalePrice'] = np.log1p(df['SalePrice']) 
sns.distplot(df["SalePrice"])
plt.show()
print("Improvement in skeness")

# Looking at most relevant features for target
corr_mat = df.corr()
f, ax = plt.subplots(figsize = (12, 9))
sns.heatmap(corr_mat, vmax = .8, square = True)


# Zoomed heatmap
k = 10 # number of variables to graph
cols = corr_mat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.00)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

sns.pairplot(df[cols], size = 2.5)
plt.show()

