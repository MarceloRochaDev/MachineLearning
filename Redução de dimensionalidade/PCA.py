# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 18:54:28 2020

@author: Patricia
"""

import pandas as pd

base = pd.read_csv('census.csv')

previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values
                
from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelencoder_previsores.fit_transform(previsores[:, 13])


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)


#redução da dimensionalidade
from sklearn.decomposition import PCA
pca = PCA(n_components= None) #n_components = número de atributos que quer q ele gera
previsores_treinamento = pca.fit_transform(previsores_treinamento)
previsores_teste=pca.transform(previsores_teste)
componentes = pca.explained_variance_ratio_ # somando os valores até o 6 da uma boa representabilidade da base de dados

# usando entao o 6 achado no passo anterior
pca = PCA(n_components= 6) #gera 6 atributos que serão usados ao inves dos 15 originais
previsores_treinamento = pca.fit_transform(previsores_treinamento)
previsores_teste=pca.transform(previsores_teste)
componentes = pca.explained_variance_ratio_ #


# importação da biblioteca
from sklearn.ensemble import RandomForestClassifier
classificador = RandomForestClassifier(n_estimators = 40, criterion='entropy',random_state=0)
# criação do classificador
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
