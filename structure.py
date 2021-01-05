# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 17:22:30 2020

@author: User
"""


import numpy as np
import cv2
import re
import glob
import tensorflow as tf
from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Add, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import pickle
import os
import re
from scipy import ndimage, misc
from skimage.transform import resize, rescale
from matplotlib import pyplot
from tqdm import tqdm

def autoencoder():
    Input_img = Input(shape=(256, 256, 3))  

    #encoding architecture
    x1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(Input_img)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x1)
    x3 = MaxPool2D(padding='same')(x2)
    x4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x3)
    x5 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x4)
    x6 = MaxPool2D(padding='same')(x5)
    x7 = Conv2D(342, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x6)
    x8 =  Conv2D(342, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x7)
    x9 = MaxPool2D(padding='same')(x8)
    x10 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x9)



    #encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
    # decoding architecture
    x11 = UpSampling2D()(x10)
    x12 = Conv2D(342, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x11)
    x13 = Conv2D(342, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x12)
    x14 = Add()([x8, x13])
    x15 = UpSampling2D()(x14)
    x16 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x15)
    x17 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x16)
    x18 = Add()([x5, x17])
    x19 = UpSampling2D()(x18)
    x20 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x19)
    x21 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x20)
    x22 = Add()([x2, x21])
    # x3 = UpSampling2D((2, 2))(x3)
    # x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x3)
    # x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(x2)
    decoded = Conv2D(3, (3, 3), padding='same',activation='relu', kernel_regularizer=regularizers.l1(10e-10))(x22)
    autoencoder = Model(Input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])
    
    return autoencoder

def pixleated_structre():
    Input_img = Input(shape=(256, 256, 3))  
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(Input_img)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x1)
    x3 = MaxPool2D(padding='same')(x2)
    x4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x3)
    x5 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x4)
    x6 = MaxPool2D(padding='same')(x5)
    encoded = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x6)
    x7 = UpSampling2D()(encoded)
    x8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x7)
    x9 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x8)
    x10 = Add()([x5, x9])
    x11 = UpSampling2D()(x10)
    x12 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x11)
    x13 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x12)
    x14 = Add()([x2, x13])
    decoded = Conv2D(3, (3, 3), padding='same',activation='relu', kernel_regularizer=regularizers.l1(10e-10))(x14)
    autoencoder = Model(Input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])
    
    return autoencoder