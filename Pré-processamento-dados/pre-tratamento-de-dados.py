# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 15:31:52 2020

@author: Patricia
"""

import pandas as pd
import numpy as np
# base.age= base['age']
# base[base.age<0] filtro 

base=pd.read_csv('credit_data.csv') # Carrega a base dados
base.describe() # exibe algumas infos estatisticas (media,desvio,...)
base.loc[base['age']<0] # localiza uma coluna com uma condição (idade menor que 0)

#apagar a coluna
base.drop('age',1,inplace=True)

#apagar as linhas dos registros com problema
base.drop(base[base.age<0].index,inplace=True)

# preencher os valores manualmente
# preencher os valores com a média
base.mean()
base['age'].mean()
base['age'][base.age>0].mean()
base.loc[base.age < 0,'age'] = 40.92 # na tabela base, onde base.age <0, atualiza o campo age pra 40.92


# valores não preenchidos
pd.isnull(base['age']) # true ou false de todos registros condição se e nulo ou n
base.loc[pd.isnull(base['age'])] # traz os registros nulos

previsores = base.iloc[:, 1:4].values # pega todas as linhas das colunas 1 à 3
classe = base.iloc[:, 4].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values=np.nan, strategy='mean') # usa a stragery media pra preencher os valores faltantes
imputer = imputer.fit(previsores[:, 0:3]) # aplica nos previsores
previsores[:,0:3] = imputer.transform(previsores[:,0:3]) # recebe a transformação feita pelo imputer nele mesmo




## escalonação - padronização (com base no desvio padrão) e normalização de valores muito distantes entre colunas (ex:60mil de renda 20 anos normaliza pra fica algo do tipo 2:4 pra 0.9 algo assim,proximo)
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)