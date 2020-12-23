# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 21:13:46 2020

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


from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import StratifiedKFold # método de analise crusada recomendado
from sklearn.metrics import accuracy_score, confusion_matrix
kfold=StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0) #n_split -> numero de divisoes q vai fzr
resultados = []
matrizes=[]
for indice_treinamento, indice_teste in kfold.split(previsores,
                                                    np.zeros(shape=(previsores.shape[0],1))):
    classificador = GaussianNB()
    classificador.fit(previsores[indice_treinamento],classe[indice_treinamento])
    previsoes = classificador.predict(previsores[indice_teste])
    precisao = accuracy_score(classe[indice_teste],previsoes)
    matrizes.append(confusion_matrix(classe[indice_teste],previsoes))
    resultados.append(precisao)
    
matriz_final=np.mean(matrizes, axis=0)
resultados = np.asarray(resultados)
resultados.mean()
    