# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 17:44:39 2020

@author: Patricia
"""

import pandas as pd

base = pd.read_csv('plano_saude2.csv')

x = base.iloc[:,0:1].values
y = base.iloc[:,1:2].values

#kernel linear
from sklearn.svm import SVR
regressor_linear = SVR(kernel = 'linear')
regressor_linear.fit(x,y)

import matplotlib.pyplot as plt

plt.scatter(x,y)
plt.plot(x,regressor_linear.predict(x),color = 'red')

regressor_linear.score(x,y)


#kernel polinomial
regressor_poly = SVR(kernel='poly',degree=3)
regressor_poly.fit(x,y)
plt.scatter(x,y)
plt.plot(x,regressor_poly.predict(x),color = 'red')
regressor_poly.score(x,y)

#kernel rbf -> mais indicado; precisa de pre-processamento
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

regressor_rbf = SVR(kernel='rbf')
regressor_rbf.fit(x,y)


plt.scatter(x,y)
plt.plot(x,regressor_rbf.predict(x),color = 'red')
regressor_rbf.score(x,y)

import numpy as np
# o inverse_transform vai fzr o escalonamento reverso, recolocando os dados na escala original
previsao1 = scaler_y.inverse_transform(regressor_linear.predict(scaler_x.transform(np.array(40).reshape(1, -1))))
previsao2 = scaler_y.inverse_transform(regressor_poly.predict(scaler_x.transform(np.array(40).reshape(1, -1))))
previsao3 = scaler_y.inverse_transform(regressor_rbf.predict(scaler_x.transform(np.array(40).reshape(1, -1))))
