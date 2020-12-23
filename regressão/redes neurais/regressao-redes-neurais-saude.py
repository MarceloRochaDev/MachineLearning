# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 17:44:39 2020

@author: Patricia
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

base = pd.read_csv('plano_saude2.csv')

x = base.iloc[:,0:1].values
y = base.iloc[:,1:2].values

scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)


from sklearn.neural_network import MLPRegressor
regressor = MLPRegressor()
regressor.fit(x,y)
regressor.score(x,y)


import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.plot(x,regressor.predict(x),color='red')


import numpy as np
# o inverse_transform vai fzr o escalonamento reverso, recolocando os dados na escala original
previsao = scaler_y.inverse_transform(regressor.predict(scaler_x.transform(np.array(40).reshape(1, -1))))
