import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras import Input
from keras.layers import Dense, Dropout, Activation, Flatten
from model.metrics import *

def UNET (input_shape = (256,256,1)):
    input = keras.layers.Input(shape=input_shape)
    
    conv1 = Conv2D(32, kernel_size=(7,7), activation='relu', padding='same')(input)
    conv1 = Dropout(0.5)(conv1)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = Conv2D(32, kernel_size=(7,7), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(64, kernel_size=(5,5), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.5)(conv2)
    conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = Conv2D(64, kernel_size=(5,5), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.5)(conv3)
    conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

    conv4 = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.5)(conv4)
    conv4 = BatchNormalization(axis=-1)(conv4)
    conv4 = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(conv4)
    
    up5 = concatenate([UpSampling2D(size=(2,2))(conv4), conv3], axis=-1)
    conv5 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(up5)
    conv5 = Dropout(0.5)(conv5)
    conv5 = BatchNormalization(axis=-1)(conv5)
    conv5 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2,2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(64, kernel_size=(5,5), activation='relu', padding='same')(up6)
    conv6 = Dropout(0.5)(conv6)
    conv6 = BatchNormalization(axis=-1)(conv6)
    conv6 = Conv2D(64, kernel_size=(5,5), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2,2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(32, kernel_size=(7,7), activation='relu', padding='same')(up7)
    conv7 = Dropout(0.5)(conv7)
    conv7 = BatchNormalization(axis=-1)(conv7)
    conv7 = Conv2D(32, kernel_size=(7,7), activation='relu', padding='same')(conv7)

    conv8 = Conv2D(1, kernel_size=(1,1), activation='sigmoid')(conv7)

    model = keras.models.Model(inputs=[input], outputs=[conv8])

    model.compile(optimizer='Adam', loss= dice_coef_loss, metrics=[dice_coef,'binary_crossentropy','accuracy'])

    return model



