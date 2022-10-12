# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:37:46 2022

@author: dafer
"""

import numpy as np
import pandas as pd

#importar datos
dataset = pd.read_csv('Churn_Modelling.csv')

#variables independientes en matriz x 
X=dataset.iloc[:, 3:12].values
#variable dependiente en matriz
Y=dataset.iloc[:,13].values

#convirtiendo datos categoricos
#importando sklearn
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

X[:,2]=LabelEncoder().fit_transform(X[:,2])

#cambio de los datos categoricos en la matriz x columna 1
ct=ColumnTransformer([('encoder',OneHotEncoder(),[1])],remainder='passthrough')
#reemplazar los datos categoricos
X=np.array(ct.fit_transform(X), dtype=np.float)


######### aqui empiezan las reedes neuronales #########

# separando modelo de entrenamiento y de pruebas
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 0)
                                                    
#escalando categorias
from sklearn.preprocessing import StandardScaler
#objeto que realiza el escalado en x
sc_X = StandardScaler()
#cambio de los datos de x test y x train
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)

#parte 1 red neuronal - crear red

#theano
#tensorflow
#keras

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








