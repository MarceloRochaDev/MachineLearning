# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 09:15:28 2020

@author: Patricia
"""

import pandas as pd

base = pd.read_csv('risco_credito.csv')
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values
                  
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])
                 
from sklearn.tree import DecisionTreeClassifier, export # n aceita valores categoricos
classificador =DecisionTreeClassifier(criterion='entropy') 
classificador.fit(previsores, classe) # aplica o treinamento do modelo e gera a árvore
print(classificador.feature_importances_)

export.export_graphviz(classificador,out_file='arvore.dot', # é gerado um arquivo arvore.dot pra visualizar a árvore com o app graphviz (tem que baixar))
                       feature_names=['historia','divida','garantia','renda'],
                       class_names=['alto','moderado','baixo'],
                       filled=True,
                       leaves_parallel=True)
#historia boa, divida alta, garantinas nenhuma, renda>35
#historia boa, divida alta, garatias adequadas, renda <15
resultado = classificador.predict([[0,0,1,2],[3,0,0,0]])
print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_)
