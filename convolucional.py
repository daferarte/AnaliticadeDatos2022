# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:39:50 2022

@author: dafer
"""

#creando red neuronal convolucional
#utilizamos keras
from keras.models import Sequential
from keras.layers import Dense

#las siguientes librerias sirven para qie la red realice la convolucion,
# agrupamiento y aplanado

from keras.layers import Conv2D #convolucion
from keras.layers import MaxPooling2D #agrupamiento
from keras.layers import Flatten #aplanamiento

# Parte numero 1

#inicializamos la red convolucional
classifier=Sequential();

# paso 1 agregar una convolucion
classifier.add(Conv2D(32,(3,3), input_shape=(64,64,3), activation='relu'))

# paso 2 pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# agregando otra capa convolucional
classifier.add(Conv2D(64,(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# agregando otra capa convolucional
classifier.add(Conv2D(128,(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# agregando otra capa convolucional
classifier.add(Conv2D(32,(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# paso 3 aplanamiento
classifier.add(Flatten())

#paso 4 crear la capa completamente conectada y capa de salida 
classifier.add(Dense(units=128, activation='relu'))
#capa de salida
classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

# parte 2 adaptar las imagenes

from keras.preprocessing.image import ImageDataGenerator

# enriquerismiento del data set evitar sobre encajamiento

train_datagen= ImageDataGenerator(
            rescale=1./255, # reescalar las imagenes
            shear_range=0.2, # agarra diferentes partes de la imagen y crea su propio arreglo
            zoom_range=0.2, # hacer zoom a la imagen
            horizontal_flip=True # mueve horizontalmente la imagen
        )

test_datagen=ImageDataGenerator(
            rescale=1./255, # reescalar las imagenes
        )

training_set = train_datagen.flow_from_directory(
            'dataset/training_set', #directorio de las imagenes
            target_size=(64,64), #define el tamaño de las imagenes
            batch_size=32,
            class_mode='binary'
        )

test_set= test_datagen.flow_from_directory(
            'dataset/test_set', #directorio de las imagenes
            target_size=(64,64), #define el tamaño de las imagenes
            batch_size=32,
            class_mode='binary'
        )

datos=classifier.fit(
            training_set,
            steps_per_epoch=600,
            epochs=2,
            validation_data=test_set,
            validation_steps=2000
        )

datos2=classifier.fit(
            training_set,
            steps_per_epoch=600,
            epochs=2,
            validation_data=test_set,
            validation_steps=2000
        )


import matplotlib.pyplot as plt

print(datos.history.keys())
print(datos.history['loss'],'r',datos2.history['loss'],'g')
plt.plot(datos.history['loss'],'r',datos2.history['loss'],'g')
plt.ylabel('Perdida de datos')
plt.xlabel('epocas')
plt.show()
# parte 3 realizar las predicciones

import numpy as np
from keras.preprocessing import image

test_image=image.load_img(
        'dataset/single_prediction/cat_or_dog_5.jpg',
        target_size=(64,64)
        )
test_image=image.img_to_array(test_image) #crea imagen en un arreglo de 3 dimensiones

# agregamos otra dimension
test_image=np.expand_dims(test_image,axis=0)

result=classifier.predict(test_image)

print(result)

training_set.class_indices

if result[0][0]==1:
    prediction='Perro'
else:
    prediction='Gato'




















