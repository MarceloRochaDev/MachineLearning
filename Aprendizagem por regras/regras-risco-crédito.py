# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 15:45:54 2020

@author: Patricia
"""

import Orange #aceita categoricos

base = Orange.data.Table('risco_credito.csv')
base.domain
  #obs: colocar c# antes do nome da coluna da clase (c#riscos) pra especifica quem é a classe
cn2_learner= Orange.classification.rules.CN2Learner() #instancia o cn2
classificador = cn2_learner(base)  # pega a base de dados e gera as regras
for regras in classificador.rule_list:
    print(regras)
resultado = classificador([['boa','alta','nenhuma','acima_35'],['ruim','alta','adequada','0_15']])
for i in resultado:
    print(base.domain.class_var.values[i])