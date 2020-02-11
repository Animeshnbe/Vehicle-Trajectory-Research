'''
from datetime import datetime
import os
path="C:/Users/mayank/Downloads/ANIMESH/datasets/dataset_tsmc2014/"
trlist=[]
for file in os.scandir(path):
    with open(file, 'r') as c_file:
        cread=csv.reader(c_file,delimiter='\t')
        lines=0
        for row in cread:
            stx=row[7]
            st=stx.replace('+0000 ','')
            dt=datetime.strptime(st,'%a %b %d %H:%M:%S %Y')
            #coord=row[4].split('\t')
            #print(coord)
            trlist.append([float(row[4]),float(row[5]),dt])

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
clusters.fit(trlist[:2]) 
y=clusters.labels_
'''

import csv
trajlist=[]
ct=0
dataf=open("C:/Users/Animesh/Documents/vta research project/matched-trajectories.csv", 'r')
rd=csv.reader(dataf, delimiter=',', quoting=csv.QUOTE_NONE)
for row in rd:
    if ct%2==0:
        xt=[]
        i=0
        while i<len(row) and row[i]!="":
        #for i in range(0,len(row),2):
            try:
                lon=float(row[i][3:-1])
                lat=float(row[i+1][2:-3])
            except ValueError as e:
                print(ct, i)
                break
            xt.append([lat,lon])
            i+=2
        trajlist.append(xt)
    ct+=1
'''
import folium
import random
def createmap(trajlist,i):
    my_map = folium.Map(location = [39.907388,116.397013], zoom_start = 11.5)  
    for xt in trajlist:
        if len(xt)!=0:
            loc=[tuple(l) for l in xt]
            random_number = random.randint(0,16777215)
            hex_number = str(hex(random_number))
            hex_number ='#'+ hex_number[2:]
            folium.PolyLine(locations = loc, color=hex_number, weight=1.5, line_opacity = 0.5).add_to(my_map)
     
    my_map.save("C:/Users/Animesh/Documents/vta research project/my_map"+str(i)+".html")

createmap(trajlist,0)
'''
#MAP_MATCHING
modtrlist=[]
import requests
for i in trajlist:
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
        if len(trajectory)>0:
            modtrlist.append(trajectory)
        #else:
         #   modtrlist.append(i)
        
#createmap(modtrlist,1)
from mapbox import MapMatcher
def mapmatch(coords):
    b=coords.tolist()
    if len(b)>100:
        matchlist=[]
        lim=0
        while 100*(lim+1)<len(b):
            matchlist.append(trajprocess(b[100*lim:100*(lim+1)]))
            lim+=1
        matchlist.append(trajprocess(b[100*lim:]))
        return matchlist
    else:
        return trajprocess(b)
            
def trajprocess(b):
    #b=[[39.9155,116.50661],[39.9156,116.49625]]
    '''
    b=[[13.418946862220764, 52.50055852688439],
       [13.419011235237122, 52.50113000479732],
       [13.419756889343262, 52.50171780290061],
       [13.419885635375975, 52.50237416816131],
       [13.420631289482117, 52.50294888790448]]
    '''
    line={
    "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": b}
    }
    service = MapMatcher(access_token="sk.eyJ1Ijoib3ZlcndhdGNoLTEyIiwiYSI6ImNrMjI5ZnFjdzBqZm4zcHBnN3N4NXgyYXgifQ.KZkzHpsZJPpax5dnCJPf6Q")
    response = service.match(line, profile='mapbox.driving')
    corrected = response.geojson()
    if corrected['code']=='Ok':
        print (corrected['features'][0]['properties']['confidence'])
        return corrected['features'][0]['properties']['matchedPoints']
    else:
        print (corrected['code'])
        return b
  
import numpy as np
def floatify(coordinates):
    coords=np.zeros(shape=(len(coordinates),2), dtype='float')
    k=0
    for i in coordinates:
        coords[k]=[float(i[0]),float(i[1])]
        k=k+1
    return coords

modtrlist1=[]
for i in range(len(trajlist)):
    modtrlist1.append(mapmatch(floatify(trajlist[i])))
with open('trajmod.csv', mode='w') as dtfile:
    wr=csv.writer(dtfile, delimiter=',')
    for i in range(len(modtrlist)):
        wr.writerow(modtrlist[i])

#np.savetxt('trajectories.csv',np.array(trlist),delimiter=',')

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