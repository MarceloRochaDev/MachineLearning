# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 14:34:15 2020

@author: Patricia
"""

from sklearn.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers.normalization import BatchNormalization



classificador = Sequential()
#Convolução
classificador.add(Conv2D(32,(3,3), input_shape=(64, 64, 3), activation = 'relu')) # Vai ser feita 32 convoluções
                                                        # operador de convolução 3x3
                                                        # input_shape : dimensão de entrada dos dados
classificador.add(BatchNormalization)

#Pooling
classificador.add(MaxPooling2D(pool_size=(2,2))) # pool_size: tamanho do operador


# repetindo a convolução e o pooling pra melhorar o resultado
classificador.add(Conv2D(32,(3,3), input_shape=(64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization)
classificador.add(MaxPooling2D(pool_size=(2,2)))

#Flatterning
classificador.add(Flatten())

#Rede densa
classificador.add(Dense(units=100, activation = 'relu'))
classificador.add(Dense(units=100, activation = 'relu'))
classificador.add(Dense(units=16,activation = 'softmax')) #softmax -> usada quando a saida tem mais de 2 categorias
                                                # foi usado 16 neuronios na saida pq tem 16 classes de saída
classificador.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# dps faz um fit pra treinar o classificador
#ex:
classificador.compile(loss='mean_absolute_error',optimizer='adam',metrics=['accuracy'])
classificador.fit(previsores_treinamento,classes_treinamento,epochs=50,validation_data=(previsores_teste,classes_teste))