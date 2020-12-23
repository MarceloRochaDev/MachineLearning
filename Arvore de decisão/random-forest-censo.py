# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:31:42 2020

@author: Patricia
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:55:44 2020

@author: Patricia
"""
import pandas as pd

base = pd.read_csv('census.csv') 
previsores=base.iloc[:,0:14].values
classe=base.iloc[:,14].values

#labelencoder - > transforma os categoricos em dados numericos -> tem uma certa odernação -> pra dados ordinais (tipo tamanho de roupa,ordenado,..)
# onehotencoder -> pega os dados do labelencoder e transforma em variáveis dummy -> pra dados não ordinais, sem ordenação (tipo cor de pele)


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_previsores=LabelEncoder()
#labels = labelencoder_previsores.fit_transform(previsores[:,1])
previsores[:,1]=labelencoder_previsores.fit_transform(previsores[:,1])
previsores[:,3]=labelencoder_previsores.fit_transform(previsores[:,3])
previsores[:,5]=labelencoder_previsores.fit_transform(previsores[:,5])
previsores[:,6]=labelencoder_previsores.fit_transform(previsores[:,6])
previsores[:,7]=labelencoder_previsores.fit_transform(previsores[:,7])
previsores[:,8]=labelencoder_previsores.fit_transform(previsores[:,8])
previsores[:,9]=labelencoder_previsores.fit_transform(previsores[:,9])
previsores[:,13]=labelencoder_previsores.fit_transform(previsores[:,13])

onehotencoder= OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13]) # colunade indice 8 da tabela de base pra ser transformada
previsores = onehotencoder.fit_transform(previsores).toarray()

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

#escalonamento
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)


#treinamento e test
from sklearn.cross_validation import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste=train_test_split(previsores,classe,test_size=0.25, random_state=0)

#Classificador
from sklearn.ensemble import RandomForestClassifier
classificador=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classificador.fit(previsores_treinamento,classe_treinamento)# aplica o treinamento
previsoes = classificador.predict(previsores_teste) # resultado da previsao sobre os previsores teste

#Comparar as previsoes com os valores verdadeiros da classe teste
from sklearn.metrics import confusion_matrix, accuracy_score
precisao=accuracy_score(classe_teste,previsoes)
matriz = confusion_matrix(classe_teste,previsoes)