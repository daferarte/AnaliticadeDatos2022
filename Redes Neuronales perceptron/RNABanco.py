# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 15:15:19 2020

@author: dafer
"""
#librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importar datos
dataset = pd.read_csv('Churn_Modelling.csv')

#variables independientes en matriz x 
X=dataset.iloc[:, 3:12].values
#variable dependiente en matriz
Y=dataset.iloc[:,13].values

#convirtiendo datos categoricos
#Importamos paquetes de sklearn
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import LabelEncoder
X[:,2]=LabelEncoder().fit_transform(X[:,2])

#cambio de los datos categoricos en la matriz x columna 0
ct=ColumnTransformer([('encoder',OneHotEncoder(),[1])],remainder='passthrough')
#reemplazar los datos categoricos
X=np.array(ct.fit_transform(X), dtype=np.float)



#separando modelo de entrenamiento y pruebas
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Escalado de categorias
from sklearn.preprocessing import StandardScaler
#objeto que realiza el escalado en x
sc_X = StandardScaler()
#cambio los datos de x test y x train
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)

#parte red neuronal - crear red

#libreria theano computacion numerica permite utilizar la tarjeta grafica
#libreria tensorflow 
#libreria keras

import keras
from keras.models import Sequential 
from keras.layers import Dense

#inicializar la red neuronal(secuencial significa que la red sera creada en secuencias, capa por capa, manualmente)
classifier = Sequential()

#agregar la capa input y primera capa oculta
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=12))

#agregar segunda capa capa oculta
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))

#agregando capa de salida

classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

#compilar la red neuronal
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#agregando el set de entrenamiento y epoch
classifier.fit(X_train, Y_train, batch_size = 100, epochs=3000)

#prediccion 

Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

#crear matriz de confusion
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test, Y_pred)

#guardar el archivo
import os
#ruta para guardar
target_dir = './modelo/'
#si la ruta no existe creamos la carpeta
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
    
#guardar el modelo entrenado
classifier.save('./modelo/modelo.h5')
classifier.save_weights('./modelo/pesos.h5')


#predecir un cliente

#cliente con las siguientes caracteristicas
#españa
#puntaje data credito 400
#genero femenino
#edad 35
#ternure 3
#balance 11111
#numero de productos 0 
#tarjeta de credito 1
#esta activo 0
#salario 150345
#creamos la fila con el nuevo cliente
nuevocli = np.array(([[0,0,1,400,0,35,3,11111,0,1,0,150345]]))
#escalado de caracteristicas
nuevocli=sc_X.transform(nuevocli)
#prediccion
nuevocli=classifier.predict(nuevocli)

#a una linea
nuevoclien=classifier.predict(sc_X.transform(np.array([[0,0,1,400,0,35,3,11111,0,1,0,150345]])))

nuevocli=(nuevocli>0.5)







