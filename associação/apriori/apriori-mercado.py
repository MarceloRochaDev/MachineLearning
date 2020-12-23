# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 10:54:41 2020

@author: Patricia
"""
import pandas as pd

dados = pd.read_csv('mercado.csv',header = None)

transacoes = []
for i in range(0,10):
    transacoes.append([str(dados.values[i, j]) for j in range(0,4)])

from apyori import apriori
regras = apriori(transacoes, min_support = 0.3, min_confidence = 0.8, min_lift = 2, min_lenght =2) #suporte e confiança igual da aula teorica;min_length = numero de atrbutos minimos para gerar regra

resultados = list(regras)

resultados2 = [list(x) for x in resultados]

resultadoFormatado = []
for j in range(0,3):
    resultadoFormatado.append([list(x) for x in resultados2[j][2]])