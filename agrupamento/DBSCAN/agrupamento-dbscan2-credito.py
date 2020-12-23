# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 15:28:02 2020

@author: Patricia
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

base = pd.read_csv('credit_card_clients.csv',header=1)
base['BILL_TOTAL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3'] + base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6']

x = base.iloc[:,[1,25]].values # captura os atributos 1 e 25
scaler = StandardScaler()
x = scaler.fit_transform(x)

dbscan = DBSCAN(eps=0.37,min_samples=4)
previsoes = dbscan.fit_predict()
unicos, quantidade = np.unique(previsoes, return_counts = True) # cada indice de quantidade corresponde ao indice de unico -> retorna quantos valores tem de cada


plt.scatter(x[previsoes == 0, 0],x[previsoes == 0, 1], s = 100, c ='red',label='cluster1')
plt.scatter(x[previsoes == 1, 0],x[previsoes == 1, 1], s = 100, c ='orange',label='cluster2')
plt.scatter(x[previsoes == 2, 0],x[previsoes == 2, 1], s = 100, c ='green',label='cluster3')

plt.xlabel('Limite')
plt.ylabel('Gastos')
plt.legend()


lista_clientes = np.column_stack((base,previsoes)) # concate previsoes (coloca na ultima coluna) à base 
lista_clientes = lista_clientes[lista_clientes[:,26].argsort()] #ordena pelas previsoes de 0 à 4

