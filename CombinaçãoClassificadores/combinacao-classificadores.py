# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 08:32:08 2020

@author: Patricia
"""

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler()


svm = pickle.load(open('svm_finalizado.sav','rb'))
random_forest = pickle.load(open('random_forest_finalizado.sav','rb'))
mlp = pickle.load(open('mlp_finalizado.sav','rb'))

novo_registro = [[50000,40,5000]]
novo_registro = np.asarray(novo_registro)
novo_registro = novo_registro.reshape(-1,1) #tem q fzr pra escalonar os dados (se n eles vao zerar)
novo_registro = scaler.fit_transform(novo_registro) #escalonar os novos dados pq o classificador ta escalonado
novo_registro = novo_registro.reshape(-1,3) #volta ao formato orignal, 1 x 3 no caso

resposta_svm = svm.predict(novo_registro)
resposta_random_forest = random_forest.predict(novo_registro)
resposta_mlp = mlp.predict(novo_registro)

paga=0
nao_paga=0

if resposta_svm[0] == 1:
    paga+=1
else:
    nao_paga+=1
if resposta_random_forest[0] == 1:
    paga+=1
else:
    nao_paga+=1
if resposta_mlp[0] == 1:
    paga+=1
else:
    nao_paga+=1




