# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 15:28:02 2020

@author: Patricia
"""

import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import numpy as np

base = pd.read_csv('credit_card_clients.csv',header=1)
base['BILL_TOTAL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3'] + base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6']

x = base.iloc[:,[1,25]].values # captura os atributos 1 e 25
scaler = StandardScaler()
x = scaler.fit_transform(x)

# descobrir o número de clusters
dendrograma = dendrogram(linkage(x,method='ward'))

#dps de achado pelo metodo acima o numero 3 de clusters
hc = AgglomerativeClustering(n_cluster = 3, affinity = 'euclidean', linkage='ward')
previsoes = hc.fit_predict(base)

plt.scatter(x[previsoes==0,0],x[previsoes==0,1], s=100, c='red',label='cluster 1')
plt.scatter(x[previsoes==1,0],x[previsoes==1,1], s=100, c='blue',label='cluster 2')
plt.scatter(x[previsoes==2,0],x[previsoes==2,1], s=100, c='green',label='cluster 3')

plt.xlabel('Limite')
plt.ylabel('Gastos')
plt.legend()

