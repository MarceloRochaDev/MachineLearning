# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 13:10:52 2020

@author: Patricia
"""

import pandas as pd

base = pd.read_csv('house_prices.csv')
x = base.iloc[:,3:19].values #
y = base.iloc[:,2].values #preço da casa

from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x,y,
                                                                  test_size=0.3,
                                                                  random_state=0)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4)
x_treinamento_poly = poly.fit_transform(x_treinamento)
x_teste_poly = poly.transform(x_teste)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_treinamento_poly,y_treinamento)
score = regressor.score(x_treinamento_poly,y_treinamento)

previsoes = regressor.predict(x_teste_poly)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste,previsoes)