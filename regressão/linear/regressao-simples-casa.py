# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 13:10:52 2020

@author: Patricia
"""

import pandas as pd

base = pd.read_csv('house_prices.csv')
x = base.iloc[:,5:6].values # colocando 5:6 ja transforma em np array matriz, ao inves de fzr reshape; pega só a area
y = base.iloc[:,2].values #preço da casa

from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x,y,
                                                                  test_size=0.3,
                                                                  random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_treinamento,y_treinamento)
score = regressor.score(x_treinamento,y_treinamento)

import matplotlib.pyplot as plt
plt.scatter(x_treinamento,y_treinamento)
plt.plot(x_treinamento,regressor.predict(x_treinamento),color='red')

previsoes = regressor.predict(x_teste)

resultado = abs(y_teste - previsoes)
resultado.mean()

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_teste,previsoes)
mse = mean_squared_error(y_teste,previsoes)

plt.scatter(x_teste,y_teste)
plt.plot(x_teste,regressor.predict(x_teste),color='red')
regressor.score(x_teste,y_teste)
