# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 23:26:43 2022

@author: Juanjo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

def visualizar(real,prediccion):
    plt.plot(real[0:len(prediccion)], color='red',label='Precio real')
    plt.plot(prediccion, color='blue', label='Precio predicho')
    plt.xlabel('Tiempo')
    plt.legend()
    plt.show()

'''Lectura de datos'''
df = pd.read_csv('price-btc.csv', index_col='Date', parse_dates=['Date'])

'''Train set / test set'''
train = df[:'2021'].iloc[:,[False,True,False,False,False,False]]
test = df[2021:].loc[:,[False,True,False,False,False,False]]

'''Normalizacion del set de entrenamiento'''
sc = MinMaxScaler(feature_range=(0,1))
train_sc = sc.fit_transform(train)

'''Entrenamiento de la red'''
timeSteps = 100
xTrain = []
yTrain = []
for i in range(0, len(train_sc)- timeSteps):
    xTrain.append(train_sc[i:i+timeSteps, 0])
    yTrain.append(train_sc[i+timeSteps, 0])


'''Hay que usar numpy array por optimización y reshape'''
xTrain, yTrain = np.array(xTrain), np.array(yTrain)

'''Se anade una dimension a xTrain ya que lo precisa Keras'''
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1],1))

'''Parametros para proporcionar a keras'''
dim_entrada = (xTrain.shape[1],1)
dim_salida = 1
na = 50

'''units = neuronas de capa | return_sequences = TRUE(si habrá más capas) | input_shape = dimensión entrada |
    dROPOUT(%) = número de neuronas que queremos ignorar en la capa de regularización (normalmente 20%)'''
regresor = Sequential() #Inicializa el modelo

''' capa 1 '''
regresor.add(LSTM(units=na, input_shape=dim_entrada))
''' capa 2 
regresor.add(LSTM(units=na))'''

''' capa output '''
regresor.add(Dense(units=dim_salida))

regresor.compile(optimizer='rmsprop', loss='mse') #mse = mean_squared_error


'''Encajar red reuronal en set de entrenamiento
    epochs = iteraciones para entrenar tu modelo
    batch_size = numero ejemplos entrenamiento ( cuanto más alto, más memoria necesitarás)'''
regresor.fit(xTrain, yTrain, epochs = 20, batch_size = 32)

'''Normalizar el conjunto de Test y relizamos las mismas operaciones'''
auxTest = sc.transform(test.values)
xTest = []
for i in range(0, len(auxTest)-timeSteps):
    xTest.append(auxTest[i:i+timeSteps,0])
    
xTest = np.array(xTest)
xTest = np.reshape(xTest, (xTest.shape[0],xTest.shape[1],1))

'''Realizamos predición'''
prediccion = regresor.predict(xTest)

'''Desescalamos la predicción para que se encuentre entre valores normales '''
prediccion = sc.inverse_transform(prediccion)

'''Graficamos'''
visualizar(test.values, prediccion)















