# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 15:39:47 2020

@author: Patricia
"""

import pandas as pd
import numpy as np

base = pd.read_csv('plano_saude2.csv')

x = base.iloc[:,0:1].values #0:1 pega só o 0 e coloca ja no formato np array em amtriz sem precisar fzro reshape
y = base.iloc[:,1].values

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly = PolynomialFeatures(degree=2) #degree= grau, no caso, 2 (ao quadrado)
x_poly=poly.fit_transform(x)

regressor = LinearRegression()
regressor.fit(x_poly,y)
score = regressor.score(x_poly,y)

regressor.predict(poly.transform(np.array(40).reshape(1, -1)))

import matplotlib.pyplot as plt

plt.scatter(x,y)
plt.plot(x,regressor.predict(poly.fit_transform(x)),color='red')
