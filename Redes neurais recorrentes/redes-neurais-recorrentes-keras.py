# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 15:36:59 2020

@author: Patricia
"""

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, CuDNNLSTM
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

############################ Colocar normalizador dos dados, igual no redes convolucionais
previsores =[algo]
classes=[algo]

#normalização dos dados
previsores = previsores/255
classes = classes/255

previsores_teste, previsores_treinamento, classes_teste, classes_treinamento = train_test_split(previsores,classes,test_size=0.2,random_state=4)

# RNN
classificador= Sequential()

#classificador.add(CuDNNLSTM(128,input_shape=(5,3),activation='relu',return_sequences = True)) #forma alternativa e mais rápida
classificador.add(LSTM((128),input_shape=(5,3),activation='relu',return_sequences = True)) # primeiro parâmetro é o número de neuronios
    # (5,1) é o entradas e saídas de treinamento
classificador.add(Dropout(0.2))

#classificador.add(CuDNNLSTM(128) #forma alternativa e mais rápida
classificador.add(LSTM(128),activation='relu')
classificador.add(Dropout(0.2))

classificador.add(Dense(32,activation='relu'))
classificador.add(Dropout(0.2))

classificador.add(Dense(3),activation='softmax') # 3 classes na saída

classificador.compile(loss='mean_absolute_error',optimizer='adam',metrics=['accuracy'])
classificador.fit(previsores_treinamento,classes_treinamento,epochs=50,validation_data=(previsores_teste,classes_teste))


