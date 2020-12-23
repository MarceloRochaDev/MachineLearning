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

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 0:3]) 
previsores[:,0:3] = imputer.transform(previsores[:,0:3]) 

from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)


# divisao das bases em teste e treino
from sklearn.cross_validation import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste=train_test_split(previsores,classe,test_size=0.25, random_state=0)

#classificador
from sklearn.ensemble import RandomForestClassifier
classificador=RandomForestClassifier(n_estimators=40, criterion='entropy',random_state=0)
classificador.fit(previsores_treinamento,classe_treinamento)# aplica o treinamento
previsoes = classificador.predict(previsores_teste) # resultado da previsao sobre os previsores teste

#Comparar as previsoes com os valores verdadeiros da classe teste
from sklearn.metrics import confusion_matrix, accuracy_score
precisao=accuracy_score(classe_teste,previsoes)
matriz = confusion_matrix(classe_teste,previsoes) 