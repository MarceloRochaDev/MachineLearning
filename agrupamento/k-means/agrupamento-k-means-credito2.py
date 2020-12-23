# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 15:28:02 2020

@author: Patricia
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

base = pd.read_csv('credit_card_clients.csv',header=1)
base['BILL_TOTAL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3'] + base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6']

x = base.iloc[:,[1,2,3,4,5,25]].values # captura os atributos 1 e 25
scaler = StandardScaler()
x = scaler.fit_transform(x)

# descobrir o número de clusters (pelo Elbow method)
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.xlabel('n clusters')
plt.ylabel('wcss')

# dps de achado o numero de clusters (4), inciamos o treinamento
 
kmeans = KMeans(n_clusters = 4, random_state = 0)
previsoes = kmeans.fit_predict(x) # faz o treinamento e faz a previsão de cada um dos valores


lista_clientes = np.column_stack((base,previsoes)) # concate previsoes (coloca na ultima coluna) à base 
lista_clientes = lista_clientes[lista_clientes[:,26].argsort()] #ordena pelas previsoes de 0 à 4

