# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 21:09:18 2020

@author: Patricia
"""

import Orange
base=Orange.data.Table('credit_data.csv')

base_dividida=Orange.evaluation.testing.sample(base,n=0.25)
base_treinamento=base_dividida[1]
base_teste=base_dividida[0]
len(base_treinamento)
len(base_teste)

#base line classifier
classificador = Orange.classification.MajorityLearner() #classifica pelo maioria
resultado = Orange.evaluation.testing.TestOnTestData(base_treinamento,base_teste, [classificador])

print(Orange.evaluation.CA(resultado))