#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:19:50 2018

@author: shivam
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from keras.callbacks import Callback
from keras.utils import np_utils
from keras.models import Model, Sequential, load_model, model_from_json
from keras.optimizers import RMSprop,Adam
from keras import backend as K
from keras.layers import (Input, Dense, Lambda, Dropout, Activation, LSTM, TimeDistributed, Convolution1D,
                          Convolution2D, MaxPooling1D, MaxPooling2D, Flatten)
# import collection
# import tensorboard
import numpy as np
from optparse import OptionParser
from scipy.io.wavfile import read
from sys import stderr, argv
import librosa
from scipy.io import wavfile
from scipy import signal
import wave

import tensorflow as tf
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

from keras.utils import multi_gpu_model

import os
from datapro import DataProcess
from os import listdir
import sys
import sklearn
from sklearn.preprocessing import normalize
import h5py

path = "/raid/dataset/sorted/"
f = os.path.join(path, "testfile.hdf5")
p = os.path.join(path, "process.hdf5")


def main():

    #b_obj = DataProcess()
    #b_obj.one_hotY_genre_strings(10000)

    #one_hot(1000)

    with h5py.File(p, 'r') as svr:
        train = svr['shuff_mat'][:]
        labels = svr['shuff_one'][:]
    train = np.abs(train)
    print('train max & min',train.max(),train.min())
    """
    train_update = (train - train_min)
    train_update_max = train_update.max()
    train = (train_update / train_update_max)
    """
    #train = np.expand_dims(train, axis=3)
    print(labels, train, train.max(), train.min())
    modaal(train, labels)


def modaal(train, labels):
    from keras.optimizers import RMSprop
    from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
    from keras.utils import np_utils
    input_shape = (130,40)# (train.shape[0],train.shape[1])
    #model = Sequential()

    """
    model.add(Convolution2D(filters=12, kernel_size=(15, 5), strides=(2,2), padding='valid',
                            input_shape=(40, 130, 1), data_format='channels_last', activation='relu',
                            use_bias='True'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(filters=8, kernel_size=(5, 5), padding='Same', activation='relu', use_bias='True'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(filters=4, kernel_size=(5, 5), padding='Same', activation='relu', use_bias='True'))
    model.add(Dropout(0.5))
    """
    model = Sequential()
    model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
    model.add(Dense(units=labels.shape[1], activation='softmax'))
    """
    # model.add(Convolution2D(filters = 15, kernel_size = (5,5), padding = 'Same', activation ='relu',use_bias= 'True'))
    # model.add(Dropout(0.5))

    # model.add(MaxPooling2D(2))

    # model.add(Convolution2D(filters = 12, kernel_size = (4,4), padding = 'Same', activation ='relu',use_bias= 'True'))
    # model.add(Dropout(0.5))
    
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())
    model.add(Dense(64, activation="relu", use_bias='True'))
    model.add(Dropout(0.5))
    model.add(Dense(labels.shape[1], activation="sigmoid"))
    """


    optimizer = RMSprop(lr=1e-4)
    #optimizer = Adam(lr=1e-4)
    model = multi_gpu_model(model, gpus=4)
    ##model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    ##model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.summary()
    model.save = '/raid/dataset/'

    nb_epoch = 6000
    batch_size = 4096*4

    # Callback for loss logging per epoch
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.val_losses = []

        def on_epoch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=5, min_lr=0.000001)

    history = LossHistory()

    # checkpoint
    filepath = "/raid/dataset/2l9fbmodel.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    # model.fit(train,labels, batch_size=batch_size, epochs=nb_epoch, validation_split=0.25, verbose=1, shuffle=True, callbacks=reduce_lr)
    model.fit(train, labels, batch_size=batch_size, epochs=nb_epoch, validation_split=0.25, verbose=1, shuffle=True,
              callbacks=[history])

    # serialize model to JSON
    model_json = model.to_json()
    with open("/raid/dataset/2l9fbmodel.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("/raid/dataset/2l9fbmodel.h5")
    print("Saved model to disk")
    # later...
    return model

if __name__ == "__main__":
    main()



testfolder = '/raid/dataset/fma_medium/sorted/test/'


def modelpred(test, maxi=16000):
    # load json and create model
    json_file = open('/raid/dataset/2l9fbmodel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("/raid/dataset/2l9fbmodel.h5")
    print("Loaded model from disk")
    for i in listdir(testfolder):

        for j in listdir(testfolder + i):

            test = str(testfolder + i + '/' + j)

            testina = np.vstack([np.reshape(
                librosa.load(test)[0][0:int(len(librosa.load(test)[0]) / (22050 * 3)) * (22050 * 3):],
                (int(len(librosa.load(test)[0]) / (22050 * 3)), (22050 * 3)))])
            # print(testina.shape)

            tempina = np.ones([40, 130])
            for ij in testina:
                y1 = ij
                sr1 = 22050
                # ste = librosa.feature.melspectrogram(y=y1,sr=sr1)

                S1 = librosa.feature.mfcc(y=y1, sr=sr1)

                # S1 = librosa.power_to_db(ste,ref=np.max)
                # print(tempa.shape)
                # print(S1.shape)
                if tempina.all() == 1:
                    tempina = tempina * S1
                else:
                    tempina = np.dstack((tempina, S1))
            # print(tempina.shape)
            # print('hi')
            tsh = np.rot90(tempina, k=1, axes=(2, 0))
            # print(tsh.shape)
            tda = np.rot90(tsh, k=1, axes=(1, 2))
            # print(tda.shape)

            tda = (tda / -80)
            test = np.expand_dims(tda, axis=3)
            # print(test[2].shape)

            pred = loaded_model.predict(test)
            # print(pred)
            # print('Normalized prediction')

            Row_Normalized = normalize(pred, norm='l1', axis=1)
            # print(np.around(Row_Normalized,decimals=3))
            print('below are the predictions for 3 second window')

            a = np.argmax(Row_Normalized, axis=1)
            # a = Row_Normalized.index(np.amax((np.around(Row_Normalized,decimals=3)),axis=1))
            print(a.T)

            unique, counts = np.unique(a, return_counts=True)
            print(i + ' ' + j)
            print(dict(zip(unique,
                           counts)))  # print('15 repeted:' + a.count(0) )  # print('17 repeted:' + a.count(1) )  # print('21 repeted:' + a.count(2) )
