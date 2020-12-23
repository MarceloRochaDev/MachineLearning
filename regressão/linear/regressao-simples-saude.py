# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 08:25:13 2020

@author: Patricia
"""

import pandas as pd

base = pd.read_csv('plano_saude.csv')

x = base.iloc[:,0].values # pra transformar em numpy array
y = base.iloc[:,1].values

import numpy as np

correlacao = np.corrcoef(x,y)

x = x.reshape(-1,1) # precisa ta em formato de matriz pra passar pro prox passo

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y) #treinamento

#y=b0+b1*x
regressor.intercept_ #b0
regressor.coef_ #b1

import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.plot(x,regressor.predict(x),color= 'red')
plt.title('Regressão simples')
plt.xlabel('idade')
plt.ylabel('custo') 

# previsão pessoa com 40 anos
previsao1 = regressor.intercept_ + regressor.coef_ * 40
previsao2 = regressor.predict(np.array(40).reshape(1, -1))

score = regressor.score(x,y)

from yellowbrick.regressor import ResidualsPlot
visualizador = ResidualsPlot(regressor)
visualizador.fit(x,y)
visualizador.poof()