# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 21:56:57 2020

@author: Patricia
"""

import pandas as pd
import numpy as np

base = pd.read_csv('credit_data.csv')
base = base.dropna() # apaga registros não preenchidos

from pyod.models.knn import KNN
detector = KNN()
detector.fit(np.array(base.iloc[:,4]).reshape(-1,1))

previsoes = detector.labels_ # se o valor for 0, n é outliner. Se for 1, é outlier
confiança_previsoes = detector.decision_scores_ # quanto maior o valor, maior a chance de ser outlier

outliers=[]
for i in range(len(previsoes)): # vai colocar o indice dos outliers na variavel outliers
    if previsoes[i] == 1:
        outliers.append(i)
        
lista_outliers = base.iloc[outliers,:] #passa o indice dos outliers para o iloc e pega todas colunas, pra pega todos os registros de outliers