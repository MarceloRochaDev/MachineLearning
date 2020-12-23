# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 08:32:08 2020

@author: Patricia
"""

import pandas as pd
import numpy as np

# pré-processamento (feito no arquivo pre-tratamento-de-dados)
base = pd.read_csv('credit_data.csv')
base.loc[base.age < 0,'age'] = 40.92

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 0:3]) 
previsores[:,0:3] = imputer.transform(previsores[:,0:3]) 

from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)


# divisao das bases em teste e treino
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste=train_test_split(previsores,classe,test_size=0.25, random_state=0)


#Naive bayes
from sklearn.neural_network import MLPClassifier
classificador=MLPClassifier(verbose = True, #verbose mostra no console o aprendizado
                            max_iter=1000, #numero maximo de iterações
                            tol=0.0000001,# tolerancia; qnd duas iterações seguidas tiver um incremento menor q tol ele parará
                            solver='adam',# metodo de solução coloca o descida estocastica de gradiente  
                            hidden_layer_sizes=(100), # numero de camadas ocultas e neurônios, ex(100,200) 100 neuronios na 1° camada e 200 na segunda
                            activation='relu')
classificador.fit(previsores_treinamento,classe_treinamento) # treinamento do classificador
previsoes = classificador.predict(previsores_teste) #testa o classificador

#Comparar as previsoes com os valores verdadeiros da classe teste
from sklearn.metrics import confusion_matrix, accuracy_score
precisao=accuracy_score(classe_teste,previsoes)
matriz = confusion_matrix(classe_teste,previsoes) # matriz que mostra os falsos-verdadeiro,... linha~valor real, coluna~valor suposto

import collections # pra fzr a contagem dos valores
collections.Counter(classe_teste) # se a classficação der menor que a contagem(majority learner), não usar a classifcação