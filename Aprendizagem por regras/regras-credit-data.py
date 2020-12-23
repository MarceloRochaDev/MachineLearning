# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:02:24 2020

@author: Patricia
"""

import Orange

base = Orange.data.Table('credit_data.csv')
base.domain #colocando i# antes do id o Orange ignora a coluna do id

base_dividida = Orange.evaluation.testing.sample(base,n=0.25)
base_treinamento=base_dividida[1]
base_teste=base_dividida[0]
len(base_treinamento)
len(base_teste)

cn2_learner = Orange.classification.rules.CN2Learner() #instancia cn2
classificador = cn2_learner(base_treinamento) # pega a base de dados de treino e gera as regras

for regras in classificador.rule_list:
    print(regras)

resultado = Orange.evaluation.testing.TestOnTestData(base_treinamento,base_teste,[classificador]) #pega o resultado da classificação da base teste e verifica com a base_treinada
print(Orange.evaluation.CA(resultado))