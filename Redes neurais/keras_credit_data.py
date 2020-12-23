import pandas as pd
import numpy as np

base = pd.read_csv('credit_data.csv')
base.loc[base.age < 0, 'age'] = 40.92
               
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

from keras.models import Sequential
from keras.layers import Dense
classificador = Sequential()						#dense significa que todos neurônios vao se ligar à todos da próxima camada
classificador.add(Dense(units = 2, activation = 'relu', input_dim = 3)) #units é o numero de neurônios que vai ter nessa camada oculta(entrada+saida)/2, input_dim é o número de atributos previsores e só vai na primeira chamada
classificador.add(Dense(units = 2, activation = 'relu'))
classificador.add(Dense(units = 1, activation = 'sigmoid')) # 1 neuronio na camada de saída por ser binário, colocar uma sigmoide; se tivesse mais saídas, colocaria softmax
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 100) # adam é descida do gradiente estocastica, colocado bath size=10; epochs é o número de vez que ele vai ajustar o peso
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5) #Converte os resultados pra True, se maior q 0.5, e false, caso contrário

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)
