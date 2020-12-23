# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 15:21:47 2020

@author: Patricia
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

x,y = make_blobs(n_samples=200, centers = 4) #gera amostras, no caso pra testarmos o kmeans
plt.scatter(x[:,0],x[:,1])

kmeans = KMeans(n_clusters = 4)
kmeans.fit(x)

previsoes = kmeans.predict(x)
plt.scatter(x[:,0],x[:,1],c = previsoes)