# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 17:44:39 2020

@author: Patricia
"""

import pandas as pd

base = pd.read_csv('plano_saude2.csv')

x = base.iloc[:,0:1].values
y = base.iloc[:,1].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10) # n_estimators = numero de arvores de decisao
regressor.fit(x,y)
score = regressor.score(x,y)


import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.plot(x,regressor.predict(x),color='red')

import numpy as np
x_teste = np.arange(min(x),max(x),0.1)
x_teste = x_teste.reshape(-1,1)
plt.scatter(x,y)
plt.plot(x_teste,regressor.predict(x_teste),color='red')

regressor.predict(np.array(40).reshape(-1,1))
