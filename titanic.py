# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 13:45:49 2019

@author: vlade
"""
### This is a script that takes a look at the famous Titanic dataset (downloaded from Kaggle)
### The idea here is to predict whether the passenger would survive

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

os.chdir("C:\\Users\\vlade\\Documents\\Python Scripts\\Kaggle\\Titanic") # Change directory to your own
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
#train = pd.concat([train, test])

# Taking a brief look at the data
train.info()
train.describe()
train.head()

# Look for missing data
plt.figure(figsize = (10, 6))
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

### Age needs to be filled in, cabin column should be dropped. 
### To fill in age we could build a simple regression model that takes age
### as a function of sex and pclass
sns.boxplot(data = train, x = "Pclass", y = "Age", hue = "Sex")

### Some feature engineering
###
### 1) Turn the title of passenger name into dummy variable

ser = pd.Series(train["Name"])
names = ser.str.extract(r'([A-Z]{1}[a-z]*\.)') 
train["Name"] = names 

### From the boxplot we can see that sex and pclass seem to be good predictors
### of age. Now we fill in with average for different classes and sex.
cols = ["Sex", "Embarked", "Name"]
train = pd.get_dummies(train, columns = cols, drop_first = True)

# Drop unused series
train.drop(["Cabin", "PassengerId", "Ticket"], inplace = True, axis = 1)

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    Sex = cols[2]
    
    if pd.isnull(Age):
        return train.loc[(train["Pclass"] == Pclass) 
                         & (train["Sex_male"] == Sex)]["Age"].mean() 
    else:
        return Age

train["Age"] = train[["Age", "Pclass", "Sex_male"]].apply(impute_age, axis = 1)

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.40, 
                                                    random_state=101)

### Begin looking at different machine learning methods
###
### 1) Logistic Regression

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
log_pred = logmodel.predict(X_test)

print(classification_report(y_test, log_pred))
print(confusion_matrix(y_test, log_pred))

### 2) K Nearest Neighbours

from sklearn.neighbors import KNeighborsClassifier

error_rate = []

# Use elbow method to select n_neighbours - may take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# Graph shows somewhere between 5 and 10. I will choose 7.

knmodel = KNeighborsClassifier(n_neighbors = 7) 
knmodel.fit(X_train,y_train)
kn_pred = knmodel.predict(X_test)

print(classification_report(y_test, kn_pred))
print(confusion_matrix(y_test, kn_pred))

### 3) Random Forests    

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

print(classification_report(y_test, rfc_pred))
print(confusion_matrix(y_test, rfc_pred))