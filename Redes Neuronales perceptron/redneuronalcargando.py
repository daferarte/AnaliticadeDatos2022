# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 17:28:12 2020

@author: dafer
"""


import os
import numpy as np
import keras
from keras.models import Sequential
from keras.models import load_model

modelo='./modelo/modelo.h5'
pesos_modelo='./modelo/pesos.h5'
classifier = load_model(modelo)
classifier.load_weights(pesos_modelo)

nuevocliguar=np.array([[-0.956927,-0.614337,1.74134,-2.63028,-1.11339,-0.394017,-0.739106,-1.08735,-2.54812,0.660114,-1.0243,0.881275]])
nuevocliguar=classifier.predict(nuevocliguar)