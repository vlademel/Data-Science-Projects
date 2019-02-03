# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 22:33:14 2019

@author: vlade
"""

### ---                                  ---
### --- Kaggle PUBG Data Science Project ---
### ---                                  ---

### Step by step procees
### 0) Graph data, check data labels and think about the data
### 1) Clean data - Take care of outliers
### 2) Take care of missing values
### 3) Feature engineering

### --- Import modules

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

### --- Import data

os.chdir("C:\\Users\\vlade\\Documents\\Python Scripts\\Kaggle\\PUBG")
train = pd.read_csv("train_V2.csv")
test = pd.read_csv("test_V2.csv")

### --- Some data cleaning - dropping ID
train.drop(["Id"], axis = 1, inplace = True)

### - Data visualisation

# Plotting nans
def missing_dat(df):
    plt.figure(figsize = (16, 8))
    sns.heatmap(df.isnull(),yticklabels=True,cbar=False,cmap='viridis')
    plt.show()
#sns.distplot(train["winPlacePerc"])

# Check to see in which columns nans exist
for i in train.columns:
    x = train[i].isnull().values.any()
    print("in column " + str(i) + " : " + str(x))
    
train["winPlacePerc"].isnull().sum()
# Only target column has 1 NaN, so remove
train.dropna(inplace = True)

# Look at variable correlations - allows us to see which variables we can look at and use in feature engineering
corr_mat = train.corr()
n = 28
cols = corr_mat.nlargest(n, 'winPlacePerc')['winPlacePerc'].index
cm = np.corrcoef(train[cols].values.T)

plt.figure(figsize = (16, 12))
sns.set(font_scale=1.00)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                 annot_kws={'size': 10}, yticklabels = cols.values,
                 xticklabels = cols.values)
plt.show()
# Take features with correlation of over 0.3 with target variable and view correlations with rest of vars

cor_of_cor = []
for x in -np.sort(-cm[0]):
    cor_of_cor.append(x)
        
b = []
c = []    
for x, y in zip(cor_of_cor, list(cols)):
    if abs(x) > 0.3:
        b.append(x)
        c.append(y)
cor_of_cor = np.array(b)
cor_of_cor = pd.DataFrame({"corr": cor_of_cor}, index = c)
cm2 = np.corrcoef(train[c].values.T)

# Show correlations amongst highly correlated variables
plt.figure(figsize = (16, 12))
sns.heatmap(cm3, cbar=True, annot=True, square=True, fmt='.2f',
                 annot_kws={'size': 10}, yticklabels = c,
                 xticklabels = c)
plt.show()
# Plot histograms of all variables
counter = 0
fig, ax = plt.subplots(nrows = 6, ncols = 4, figsize = (6,6))
for i in range(ax.shape[0]):
    for j in range(ax.shape[1]):
        sns.distplot(train.iloc[:, counter])
    counter +=1

# Maybe plot some countplots of some variable here
# Kills by bucket
data = train.copy()
data["kills"].loc[data['kills'] > data['kills'].quantile(0.99)] = '8+'
plt.figure(figsize = (15, 10))
sns.countplot(data['kills'].astype('str').sort_values())
plt.title("Kill Count", fontsize = 15)
plt.show()

# Avg damage done by kill bucket
avg_dmg = data.groupby(["kills"]).mean()["damageDealt"]
sns.boxplot(x = avg_dmg.index, y = avg_dmg.index, data = avg_dmg)

# Avg distance walked by kill bucket
avg_dist = data.groupby(["kills"]).mean()["walkDistance"]
sns.barplot(x = avg_dist.index, y = avg_dist.values, data = avg_dist)

# Avg distance walked per heal

data.loc[data['heals'] > data['heals'].quantile(0.99)] = '12+'
plt.figure(figsize = (15, 10))
sns.countplot(data['heals'].astype('str').sort_values())
plt.title("Heal Count", fontsize = 15)
plt.show()

avg_heal = data.groupby(["heals"]).mean()["walkDistance"]
sns.barplot(x = avg_heal.index, y = avg_heal.iloc[:, 0], data = avg_heal)

# Avg heals per kill - the more you get into a gunfight, the more likely you are to need to heal



