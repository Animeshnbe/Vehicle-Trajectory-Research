ct=0
'''
trlist=[]

with open('tr2.csv', mode='r') as dtfile:
    rd=csv.reader(dtfile,delimiter=',')
    for row in rd:
        if ct%2==0:
            k=row[0][2:-2].split('], [')
            trajectory=[]
            for j in range(len(k)):
                lat,lon=k[j].split(', ')
                trajectory.append([float(lat),float(lon)])
            trlist.append(trajectory)
        ct+=1

modtrlist=[]
import requests
for i in trlist:
    if len(i)>2:
        d={}
        pointslist=[] 
        for j in i:
            #print(j[0])
            d["latitude"]= float(j[0])
            d["longitude"]= float(j[1])
            pointslist.append(d.copy())
            
        response=requests.post('http://dev.virtualearth.net/REST/v1/Routes/SnapToRoad?interpolate=true&key=Arc88WQ28IRnvtfPwzKcNaZb_z40jjwR6ETFLYfIeXPRqmnl4DH8CX9RSnXvrDlm',json={"points":pointslist, "interpolate": "true"})
        if response.status_code==200:
            k=response.json()['resourceSets'][0]['resources'][0]['snappedPoints']
            #if len(k)<=3:
             #   print(k)
            trajectory=[]
            for i in k:
                trajectory.append([float(i['coordinate']['latitude']),float(i['coordinate']['longitude'])])
            modtrlist.append(trajectory)
   
b = list()
for sublist in modtrlist:
    if (sublist not in b) and len(sublist)>2:
        b.append(sublist)

import re
xt=[]
with open('trajectories1.csv', mode='r') as dtfile:
    rd=csv.reader(dtfile,delimiter=',')
    for row in rd:
        if ct%2==0:
            xt.append(row)
        ct+=1

for i in range(len(xt)):
    trajectory=[]
    for j in range(len(xt[i])):
        y=re.findall(r'\d+\.*\d*',xt[i][j])
        trajectory.append([float(y[0]),float(y[1])])
        xall.append([float(y[0]),float(y[1])])
    trlist.append(np.array(trajectory,dtype=float))

xall=[]
for i in b:
    for j in i:
        xall.append(j)
xall=np.array(xall)

from math import radians, sin, cos, asin, sqrt
def distfind(src,dest):
    slon,slat = src[0],src[1]
    elon,elat = dest[0],dest[1]
    #print(slon)
    slat=radians(float(slat))
    slon=radians(float(slon))
    elat=radians(float(elat))
    elon=radians(float(elon))
    dlat=elat-slat
    dlon=elon-slon
    a = sin(dlat/2)**2 + cos(slat)*cos(elat) * sin(dlon/2)**2
    #c = 2 * atan2(sqrt(a), sqrt(1-a))
    dist=2*6371010*asin(sqrt(a))
    return dist
    #gmaps = googlemaps.Client(key='AIzaSyANek40q60bomgvZAYTs5sOSrMZTh9AFn8')
    #print(gmaps.distance_matrix((40.714224, -73.961452),(116.43003,39.93209))['rows'][0]['elements'][0]['distance']['value'])
    
from sklearn.metrics import pairwise_distances
def diff_affinity(X):
    return pairwise_distances(X, metric=distfind)

from sklearn.cluster import AgglomerativeClustering
clusters = AgglomerativeClustering(n_clusters=None, affinity=diff_affinity, compute_full_tree='auto', connectivity=None, distance_threshold=100.0, linkage='complete')
clusters.fit(xall) 
y=clusters.labels_
from collections import Counter
lab=Counter(y).keys()

ctr=0
s=0
mt=[]
for i in range(len(b)):
    modtraj=[]
    modtraj2=[]
    s=s+len(b[i])
    while ctr<s:
        modtraj.append(y[ctr])
        modtraj2.append([y[ctr],xall[ctr]])
        #remove no change cell sequences
        while ctr+1<s and y[ctr]==y[ctr+1]:
            ctr+=1
        ctr+=1
    mt.append(modtraj)
     
import networkx as nx
import matplotlib.pyplot as plt
g=nx.MultiGraph()

with open('graphrep.csv', mode='w') as dtfile:
    wr=csv.writer(dtfile, delimiter=',')
    for i in range(len(mt)):
        wr.writerow(mt[i])
        for j in range(len(mt[i])-1):
            g.add_edge(mt[i][j],mt[i][j+1], weight=distfind(b[i][j],b[i][j+1]))
options = {'node_color': 'green', 'node_size': 25, 'width': 0.2}
nx.draw(g, with_labels=True, cmap=plt.cm.Reds_r, k=0.25, iterations=20, **options)
g.number_of_nodes()
'''
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