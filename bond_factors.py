# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 13:30:39 2019

@author: vlade
"""

import quandl
import numpy as np
import pandas as pd
import seaborn as sns

#data = quandl.get("USTREASURY/YIELD", returns="pandas", trim_start="2009-01-02")
#data.drop("2 MO", inplace = True, axis = 1)
data.columns = ["1", "3", "6", "12", "24", "36", "60", "84", "120", "240", "360"]

### PCA methodology
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaled_dat = StandardScaler().fit_transform(data)
pca = PCA(n_components = 3)
principalComponents = pca.fit_transform(scaled_dat)
pca_df = pd.DataFrame(principalComponents, columns = ["PC 1", "PC 2", "PC 3"])
pca_df.index = data.index

sns.lineplot(data=pca_df)

### Nelson Siegal method
from scipy.optimize import fmin

l = 0.0609 # Lambda value - can I optimise this value?
fp = lambda c, x: c[0] + (c[1]*((1-np.exp(-l*x))/(l*x))) + (c[2]*(((1-np.exp(-l*x))/(l*x))-np.exp(-l*x))) # Define polynomial function
e = lambda p, x, y: ((fp(p,x)-y)**2).sum()  # Error function to minimize
x = np.array([float(i) for i in data.columns]) # The periods for which data are available

factors = pd.DataFrame(columns = ["Factor 1", "Factor 2", "Factor 3"])

for index, row in data.iterrows():
    
    p0 = np.array([5,-3,1])  # initial parameter values
    p = fmin(e, p0, args=(x,row)).reshape(1,3) # fitting the data with fmin
    temp = pd.DataFrame(p, columns = ["Factor 1", "Factor 2", "Factor 3"])
    factors = factors.append(temp)

factors.index = data.index
sns.lineplot(data=factors)

def curve_builder(factors):
    
    yldDf = pd.DataFrame(columns = x)
    
    for i in x:
        
        yldList = []
    
        for index, row in factors.iterrows():
        
            yldList.append(row[0] + (row[1]*((1-np.exp(-l*i))/(l*i))) + (row[2]*(((1-np.exp(-l*i))/(l*i))-np.exp(-l*i))))
               
        yldDf[i] = pd.Series(yldList)
    
    return yldDf

### Some graphing
NS_curve = curve_builder(factors)
PCA_curve = curve_builder(pca_df)
sns.lineplot(data=NS_curve.iloc[1])
sns.lineplot(data=PCA_curve.iloc[1])
