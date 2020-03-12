# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 21:49:43 2020

@author: IDALAB HCM - II
"""
import csv
import numpy as np
import random
mt=[]
ct=0
dataf=open("C:/Users/IDALAB HCM - II/Downloads/Vehicle-Trajectory-Research-master/enc-sequences.csv", 'r')
rd=csv.reader(dataf, delimiter=',', quoting=csv.QUOTE_NONE)
for row in rd:
    if ct%2==0:
        xt=[]
        i=0
        while i<len(row) and row[i]!="":
            xt.append(float(row[i]))
            i+=2
        mt.append(xt)
    ct+=1

import numpy as np
from keras.layers import BatchNormalization, Add
from keras.layers import Bidirectional
from keras.models import Model, Sequential, Input
from keras.layers.recurrent import GRU, LSTM
from keras.layers.core import Dense, Masking
from sklearn.metrics import classification_report
from keras.optimizers import Nadam
from keras import regularizers
from keras.layers import TimeDistributed, Embedding
from keras.utils import to_categorical

x_train=[]
y_train=[]

from keras.preprocessing.sequence import pad_sequences
padded = pad_sequences(mt)
for i in range(len(padded)):
    x_train.append(padded[i,:-1])
    #y_train.append(padded[i,-1])
    y_train.append(padded[i,1:])
    
x_train=np.reshape(np.array(x_train),(len(x_train),len(x_train[0]),1))
y_train=np.reshape(np.array(y_train),(len(x_train),len(x_train[0]),1))

def data_gen(mt):
    i=0
    while True:
    #for i in range(len(mt)):
        x_train=np.reshape(np.array(mt[i][:-1]),(1,-1,1))
        y_train=np.array(mt[i][-1])
        #y_train=to_categorical(y_train, num_classes=1503)
        y_train=np.reshape(y_train,(1,-1))
        #print (x_train.shape)
        yield x_train, y_train
        i=(i+1)%len(mt)

def get_model():
    #density
    model=Sequential()
    #model.add(Embedding(len(lab), 16, input_length=ts, batch_input_shape=(1,ts)))
    #model.add(Masking(mask_value=0., batch_input_shape=(1,ts,1)))
    model.add(Bidirectional(LSTM(20, input_shape=(482, 1), return_sequences=True)))
    #model.add(GRU(20, implementation=1, activity_regularizer=None, return_sequences=True))
    model.add(TimeDistributed(Dense(1503, activation='softmax')))
    #model.add(Dense(1503, activation='softmax'))
    #model.add(Flatten(batch_input_shape=(1,)))
    #model.add(Activation('softmax'))
    #print(model.summary())

    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='Adamax', metrics = ['accuracy'])
    return model

model=get_model()
model.load_weights("C:/Users/IDALAB HCM - II/Downloads/Vehicle-Trajectory-Research-master/best-weights-10-1.1996.hdf5")

def split_sequences(trlist, n_steps_in, n_steps_out):
    X, y = list(), list()
    for sequences in trlist:
        for i in range(len(sequences)):
            sequences=np.array(sequences)
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the dataset
            if out_end_ix > len(sequences):
                break
        	# gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
            X.append(seq_x)
            y.append(seq_y)
    return np.array(X), np.array(y)

#x_train, y_train=split_sequences(mt, 3, 1)
#n_features = x_train.shape[2]

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
#filepath='best-weights-{epoch:02d}-{loss:.4f}.hdf5'
#mc = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

#callbacks_list=[es,mc]
history = model.fit(x_train,y_train, epochs=50, verbose=1, 
          validation_split=0.3, 
          batch_size=4, 
          shuffle=True)

model.save('RNN-mod.h5')
model.save_weights('RNN-mod_Weights.h5')

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()