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



#Naive bayes
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

classificadorSVM = SVC(kernel = 'rbf', C= 2.0)
classificadorSVM.fit(previsores,classe)

classificadorRandomForest=RandomForestClassifier(n_estimators = 40, criterion = 'entropy')
classificadorRandomForest.fit(previsores,classe)

classificadorMLP = MLPClassifier(verbose=True,
                                 max_iter=1000,
                                 tol=0.00001,
                                 solver='adam',
                                 hidden_layer_sizes=(100),
                                 activation='relu',
                                 batch_size = 200,
                                 learning_rate_init = 0.001)

classificadorMLP.fit(previsores,classe)

import pickle
pickle.dump(classificadorSVM,open('svm_finalizado.sav','wb'))
pickle.dump(classificadorRandomForest,open('random_forest_finalizado.sav','wb'))
pickle.dump(classificadorMLP,open('mlp_finalizado.sav','wb'))

