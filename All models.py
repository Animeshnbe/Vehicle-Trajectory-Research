# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:30:51 2020
MODELS HERE!
@author: Animesh
"""

from keras.layers import BatchNormalization, Add, Flatten, Activation
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
    #y_train.append(padded[i,1:])
    y_train.append(padded[i,1:])
    
x_train=np.reshape(np.array(x_train),(len(x_train),len(x_train[0]),1))
y_train=np.reshape(np.array(y_train),(len(x_train),len(x_train[0]),1))
#converted to 3d tensor of (batch_size,time_steps,feature_dim)

'''
for i in range(len(y_train)):
    print(len(lab))
    y_train[i]=to_categorical(y_train[i], num_classes=1503)

#y_train=np.array(y_train, dtype='int32')
#y_train=y_train[np.newaxis,:]

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
'''
def get_model():
    #density
    model=Sequential()
    #model.add(Embedding(len(lab), 16, input_length=ts, batch_input_shape=(1,ts)))
    #model.add(Masking(mask_value=0., batch_input_shape=(1,ts,1)))
    model.add(Bidirectional(LSTM(482, input_shape=(482, 1), return_sequences=True)))
    #model.add(GRU(20, implementation=1, activity_regularizer=None, return_sequences=True))
    model.add(TimeDistributed(Dense(1503, activation='softmax')))
    #model.add(Dense(1503, activation='softmax'))
    #model.add(Flatten(batch_input_shape=(1,)))
    #model.add(Activation('softmax'))
    print(model.summary())

    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='Adamax', metrics = ['accuracy'])
    return model

model=get_model()
#model.fit_generator(data_gen(mt), steps_per_epoch=len(mt), epochs=10, verbose=1)
'''
np.random.seed(7)
from keras.wrappers.scikit_learn import KerasClassifier
mmodel=KerasClassifier(build_fn=get_model, epochs=10, batch_size=4, verbose=1)
from sklearn.model_selection import GridSearchCV
#batch_size=[1, 4, 6, 8, 12]
#epochs=[10,20,30]
#optimizer=['SGD','RMSprop','Adagrad','Adadelta','Adam','Adamax','Nadam'] #, optimizer=optimizer
learn_rate = [0.001, 0.005, 0.01, 0.1, 0.2]
param_grid=dict(optimizer=optimizer)
grid=GridSearchCV(estimator=mmodel,param_grid=param_grid, n_jobs=1, cv=3)
grid_result=grid.fit(x_train,y_train)

#grid_result.best_score_, best_params_
means=grid_result.cv_results_['mean_test_score']
std=grid_result.cv_results_['std_test_score']
params=grid_result.cv_results_['params']
print(grid_result.best_score_, grid_result.best_params_)


import random

for epoch in range(10):
	# generate new random sequence
	k=random.randint(0, 96)
	# fit model for one epoch on this sequence
	model.fit(x_train[k], y_train[k], epochs=1, batch_size=1, verbose=1)
'''
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
filepath='best-weights-{epoch:02d}-{loss:.4f}.hdf5'
mc = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

callbacks_list=[es,mc]
model.fit(x_train,y_train, epochs=10, verbose=1, 
          validation_split=0.3, 
          batch_size=4, 
          shuffle=True,
          callbacks=callbacks_list)

k=67+random.randint(0, 28)
y=model.predict_classes(np.reshape(x_train[k],(1,-1,1)))
y=np.reshape(y,(-1,1))
from nltk.translate.bleu_score import sentence_bleu
score = sentence_bleu(y_train[k], y[k])
model.save('RNN-mod.h5')
model.save_weights('RNN-mod_Weights.h5')
'''
mx=0
for m in y_train:
    if m in y[k]:
        mx+=y_train[k].count(m)
P=[]
for m in y:
    P.append(min(y[k].count(m),mx)/len(y[k]))
R=/wr)
Fm=10*P*R/(R+9P)
p=0.5*pow((c/u),3)
Meteor=Fm*(1-p)
print(P,)
'''
###############################MODEL 2#########################################
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

x_train, y_train=split_sequences(modtrlist, 3, 1)
n_features = x_train.shape[2]

import keras
#from keras.layers import BatchNormalization, Add, Flatten, Activation
from keras.models import Model, Sequential, Input
from keras.layers.recurrent import GRU, LSTM
from keras.layers.core import Dense, Masking
#from sklearn.metrics import classification_report
#from keras.optimizers import Nadam
#from keras import regularizers
from keras.layers import TimeDistributed, Embedding, Bidirectional, RepeatVector
#from keras.utils import to_categorical
# define model
model = Sequential()
model.add(LSTM(20, activation='relu', input_shape=(3, n_features)))
model.add(RepeatVector(1))
model.add(Bidirectional(LSTM(20, activation='relu', return_sequences=True)))
model.add(TimeDistributed(Dense(2)))
model.compile(optimizer='Adamax', loss='mse', metrics=['accuracy'])
# fit model
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
model.fit(x_train,y_train, epochs=10, verbose=1, 
          validation_split=0.3, 
          batch_size=500, 
          shuffle=True,
          callbacks=[es])

x_input = np.array([[39.93366,116.4557],[39.9425,116.45912],[39.94288,116.45537]])
x_input = x_input.reshape((1, 3, n_features))
yhat = model.predict(x_input, verbose=0)

def get_model():
    #density
    model=Sequential()
    #model.add(Embedding(len(lab), 16, input_length=ts, batch_input_shape=(1,ts)))
    #model.add(Masking(mask_value=0., batch_input_shape=(1,ts,1)))
    model.add(Bidirectional(LSTM(482, input_shape=(482, 1), return_sequences=True)))
    #model.add(GRU(20, implementation=1, activity_regularizer=None, return_sequences=True))
    model.add(TimeDistributed(Dense(1503, activation='softmax')))
    #model.add(Dense(1503, activation='softmax'))
    #model.add(Flatten(batch_input_shape=(1,)))
    #model.add(Activation('softmax'))
    print(model.summary())

    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='Adamax', metrics = ['accuracy'])
    return model

model=get_model()

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
model.fit(x_train,y_train, epochs=10, verbose=1, 
          validation_split=0.3, 
          batch_size=4, 
          shuffle=True,
          callbacks=[es])

##########################MODEL 3###############################################
import numpy as np
import csv
ct=0
mt=[]
with open('graphrep.csv', mode='r') as dtfile:
    rd=csv.reader(dtfile, delimiter=',')
    for row in rd:
        if ct%2==0:
            trajectory=[]
            for j in range(len(row)):
                trajectory.append(float(row[j]))
            mt.append(trajectory)
        ct+=1
    
from keras.layers import BatchNormalization, Add, Flatten
from keras.models import Model, Sequential, Input
from keras.layers.recurrent import GRU
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
    y_train.append(padded[i,1:])
    
x_train=np.reshape(np.array(x_train),(len(x_train),len(x_train[0]),1))
#y_train=np.reshape(np.array(x_train),(len(x_train),len(x_train[0]),1))

#converted to 3d tensor of (batch_size,time_steps,feature_dim)

for i in range(len(y_train)):
    #print(len(lab))
    y_train[i]=to_categorical(y_train[i], num_classes=1503)

y_train=np.array(y_train, dtype='int32')
#y_train=y_train[np.newaxis,:]
'''
def data_gen(mt):
    i=0
    while True:
    #for i in range(len(mt)):
        x_train=np.reshape(np.array(mt[i][:-1]),(1,-1,1))
        y_train=np.reshape(np.array(mt[i][1:]),(1,-1,1))
        #y_train=np.reshape(to_categorical(y_train, num_classes=len(lab)),(len(y_train),len(lab)))
        #print (x_train.shape)
        yield x_train, y_train
        i=(i+1)%len(mt)
'''
def get_model():
    model=Sequential()
    #model.add(Embedding(len(lab), 16, input_length=ts, batch_input_shape=(1,ts)))
    #model.add(Masking(mask_value=0., batch_input_shape=(1,ts,1)))
    model.add(GRU(16, implementation=1, activity_regularizer=regularizers.l1(0.01), return_sequences=True, input_shape=(None, 1)))
    model.add(TimeDistributed(Dense(1503, activation='softmax')))
    #model.add(Flatten(input_shape=(1, 1, 1)))
    #model.add(Dense(1503, activation='softmax'))
    print(model.summary())

    model.compile(loss='categorical_crossentropy',
              optimizer='Nadam', metrics = ['accuracy'])
    return model

model=get_model()
'''
#model.fit_generator(data_gen(mt), steps_per_epoch=len(mt), epochs=10, verbose=1)
np.random.seed(7)
from keras.wrappers.scikit_learn import KerasClassifier
mmodel=KerasClassifier(build_fn=get_model)
from sklearn.model_selection import GridSearchCV
batch_size=[1, 4, 6, 8, 12]
epochs=[10,20,30]
#optimizer=['SGD','RMSprop','Adagrad','Adadelta','Adam','Adamax','Nadam'] #, optimizer=optimizer
param_grid=dict(batch_size=batch_size, epochs=epochs)
grid=GridSearchCV(estimator=mmodel,param_grid=param_grid, n_jobs=1, cv=3)
grid_result=grid.fit(x_train,y_train)

#grid_result.best_score_, best_params_
means=grid_result.cv_results_['mean_test_score']
std=grid_result.cv_results_['std_test_score']
params=grid_result.cv_results_['params']
'''
model.fit(x_train,y_train, epochs=10, verbose=1, 
          validation_split=False, 
          batch_size=16, 
          shuffle=True,
          callbacks=False)

features=[mt[i][:-1] for i in range(len(mt))]
labels=[mt[i][-1] for i in range(len(mt))]
features=np.array(features)
labels=np.array(labels)
#features=features.reshape([-1,])
i=round(len(mt)*0.75)
feature_train, feature_test=features[:i],features[i:]
label_train, label_test=labels[:i],labels[i:]
#feature_train, feature_test, label_train, label_test = train_test_split(features,labels,test_size=0.25)
model.fit_generator(train_generator(feature_train,label_train), steps_per_epoch=30, epochs=10, verbose=1)

from Bio import pairwise2
from Bio.pairwise2 import format_alignment

alignments = pairwise2.align.localxx(y_train,label_train, 2, -1, -0.5, -0.1)

# Use format_alignment method to format the alignments in the list
for a in alignments:
    print(format_alignment(*a))