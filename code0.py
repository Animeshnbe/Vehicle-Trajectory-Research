# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 11:44:01 2019

@author: Animesh
"""

import numpy as np
import os
import pandas as pd
import csv
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import Delaunay
from math import radians, sin, cos, asin, sqrt
from datetime import datetime
from mapbox import MapMatcher
from collections import Counter

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
 
def lcs(X, Y): 
    # find the length of the strings 
    m = len(X) 
    n = len(Y) 
  
    # declaring the array for storing the dp values 
    L = [[None]*(n + 1) for i in range(m + 1)] 
  
    for i in range(m + 1): 
        for j in range(n + 1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j], L[i][j-1]) 
  
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1] 
    return L[m][n] 
    
def vclustering(coords):
    #clustering nodes by maximum decrease in variance
    #clusters = AgglomerativeClustering(n_clusters=None, affinity='precomputed', compute_full_tree='auto', connectivity=None, distance_threshold=100.0, linkage='average')
    clusters = AgglomerativeClustering(n_clusters=None, affinity=diff_affinity, compute_full_tree='auto', connectivity=None, distance_threshold=100.0, linkage='average')
    clusters.fit(coords) 
    labels=clusters.labels_
    clab=[[] for i in range(len(Counter(labels).keys()))]
    n=len(clab)
    clusters = AgglomerativeClustering(n_clusters=n, affinity='euclidean', compute_full_tree='auto', connectivity=None, linkage='ward')
    clusters.fit(coords) 
    labels=clusters.labels_
    for i in range(n):
        clab[labels[i]].append(coords[i])
    centroid=[np.mean(np.array(clab[labels[i]]), axis=0) for i in range(len(Counter(labels).keys()))]
    #print(centroid.shape)
    for i in range(len(coords)):
        coords[i]=centroid[labels[i]]
    return coords,centroid
    
def timeint(start, end):
    sdt = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    edt = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
    timel=(edt-sdt).total_seconds()
    return timel

def floatify(coordinates):
    coords=np.zeros(shape=(len(coordinates),2), dtype='float')
    k=0
    for i in coordinates:
        coords[k]=[float(i[0]),float(i[1])]
        k=k+1
    return coords

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
    

path = "C:/Users/Animesh/Documents/vta research project/DATASETS/t-drive Beijing/taxi_log_2008_by_id/"
#path = "C:/Users/Animesh/Documents/vta research project/test/"
traject=[] #list of all points for extracting representatives
trlist=[] #list of all trajectories by vehicle id
for file in os.scandir(path):
    temp=[]
    with open(file,'r') as gps:
        reader = csv.reader(gps, delimiter=',')
        prev=''
        for row in reader:
            traject.append(row)
            temp.append(row[2:])
            if prev!='' and timeint(prev,row[1])>600.0:
                trlist.append(temp)
                temp=[]
                prev=''
            else:
                prev=row[1]       
    trlist.append(temp)

'''
coordinates=[]
for i in traject:
    coordinates.append(i[2:])
    
coord=floatify(coordinates)
coord,cent=vclustering(coord) #all coordinates representative 
k=0
modtrlist=trlist[:]
for i in range(len(trlist)):
    for j in range(len(trlist[i])):
        modtrlist[i][j]=coord[k]
        k+=1
#print(coord)

p=0
#countdistinct=list()
for i in range(len(modtrlist)):
    modtrlist[i]=mapmatch(floatify(modtrlist[i]))
    #trlist[i]=floatify(trlist[i])
    q=p+len(modtrlist[i])
    modtrlist[i]=[[j,traject[k][1]] for j,k in zip(modtrlist[i],range(p,q))]
    p=q
        

def trans_affinity(X):
    return pairwise_distances(X, metric=timeint)
def edgecluster(mt):
    clusters = AgglomerativeClustering(n_clusters=None, affinity=trans_affinity, compute_full_tree='auto', connectivity=None, distance_threshold=600.0, linkage='average')
    mt=np.array(mt)
    ar=np.array(mt[:,-1]).reshape((-1,1))
    #clusters.linkage(ar, method='average')
    clusters.fit(ar)
'''
#edgecluster(modtrlist)
'''
with open('trajmod.csv', mode='w') as dtfile:
    wr=csv.writer(dtfile, delimiter=',')
    for i in range(len(modtrlist)):
        wr.writerow(modtrlist[i])
'''
np.savetxt('trajectories.csv',np.array(trlist),delimiter=',')

#cg=CandidateGraph()