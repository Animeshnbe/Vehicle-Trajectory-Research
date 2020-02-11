# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:52:36 2019

@author: mayank
"""

import csv
from datetime import datetime
import os
path="C:/Users/Animesh/Documents/vta research project/DATASETS/Foursquare_NYC_and_Tokyo/"
path2="C:/Users/mayank/Downloads/ANIMESH/datasets/GPS Trajectory/"
trlist=[]
userlist=[]
for file in os.scandir(path):
    with open(file, 'r') as c_file:
        cread=csv.reader(c_file,delimiter='\t')
        lines=0
        for row in cread:
            userlist.append(float(row[0]))
            stx=row[7]
            st=stx.replace('+0000 ','')
            dt=datetime.strptime(st,'%a %b %d %H:%M:%S %Y')
            #coord=row[4].split('\t')
            #print(coord)
            trlist.append([float(row[4]),float(row[5]),dt])

from collections import Counter
lab=Counter(userlist).keys()
print(len(lab))
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
ctr=0
s=0
mt=[]
for i in range(len(x)):
    modtraj=[]
    s=s+len(x[i])
    while ctr<s:
        modtraj.append(y[ctr])
        #remove no change cell sequences
        while ctr+1<s and y[ctr]==y[ctr+1]:
            ctr+=1
        ctr+=1
    if len(modtraj)>1:
        mt.append(modtraj)
def timeint(sdt, edt):
    timel=(edt-sdt).total_seconds()
    return timel

def edgeclustering(tl):  
import networkx as nx
'''