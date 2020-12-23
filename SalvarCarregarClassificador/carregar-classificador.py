# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 08:32:08 2020

@author: Patricia
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler 
# pré-processamento (feito no arquivo pre-tratamento-de-dados)
base = pd.read_csv('credit_data.csv')
base.loc[base.age < 0,'age'] = 40.92
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 0:3]) 
previsores[:,0:3] = imputer.transform(previsores[:,0:3]) 
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

svm = pickle.load(open('svm_finalizado.sav','rb'))
random_forest = pickle.load(open('random_forest_finalizado.sav','rb'))
mlp = pickle.load(open('mlp_finalizado.sav','rb'))

resultado_svm = svm.score(previsores,classe)
resultado_random_forest = random_forest.score(previsores,classe)
resultado_mlp = mlp.score(previsores,classe)

novo_registro = [[50000,40,5000]]
novo_registro = np.asarray(novo_registro)
novo_registro = novo_registro.reshape(-1,1) #tem q fzr pra escalonar os dados (se n eles vao zerar)
novo_registro = scaler.fit_transform(novo_registro) #escalonar os novos dados pq o classificador ta escalonado
novo_registro = novo_registro.reshape(-1,3) #volta ao formato orignal, 1 x 3 no caso

resposta_svm = svm.predict(novo_registro)
resposta_random_forest = random_forest.predict(novo_registro)
resposta_mlp = mlp.predict(novo_registro)




