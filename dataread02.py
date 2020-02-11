# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:52:36 2019

@author: mayank
"""
import numpy as np
import os
from pathlib import Path
import csv
path="C:/Users/mayank/Downloads/ANIMESH/datasets/dataset_tsmc2014/"
path2="C:/Users/mayank/Downloads/ANIMESH/datasets/GPS Trajectory/"
'''
path3="C:/Users/Animesh/Documents/vta research project/DATASETS/Geolife Trajectories 1.3/Data/"
entry=Path(path3)
ctr=0
trajectories=[]
coords=[]
for direc in entry.iterdir():
    #print(path3+direc.name)
    for x in os.scandir(path3+direc.name+"/Trajectory/"):
        trlist=[]
        with open(x, mode='r') as dtfile:
            rd=csv.reader(dtfile,delimiter=',')
            ct=0
            for row in rd:
                ct+=1
                if ct<7:
                    continue
                trlist.append([row[0],row[1],row[4]])
                coords.append([float(row[0]),float(row[1])])
        if len(trlist)>3:
            trajectories.append(trlist)

times=[]
for direc in entry.iterdir():
    for x in os.scandir(path3+direc.name+"/Trajectory/"):
        with open(x, mode='r') as dtfile:
            rd=csv.reader(dtfile,delimiter=',')
            ct=0
            for row in rd:
                ct+=1
                if ct<7:
                    continue
                times.append(int(row[6][:2]))
                
time_ranges = np.zeros(24, dtype='uint8')
for t in times:
    time_ranges[t]+=1

import matplotlib.pyplot as plt

y_pos = np.arange(24)
plt.bar(y_pos, time_ranges, align='center', alpha=0.5)
plt.xticks(np.arange(0, 24, 2))
plt.ylabel('Distribution')
plt.title('Cars by time of day')

plt.show()
    
town_center=np.mean(c, axis=0)
bounds=[[min(coords[:,0]),min(coords[:,1])],[max(coords[:,0]),max(coords[:,1])]]
'''
import gmplot
latitude_list = [ 30.3358376, 30.307977, 30.3216419 ] 
longitude_list = [ 77.8701919, 78.048457, 78.0413095 ] 
  
gmap3 = gmplot.GoogleMapPlotter(30.3164945, 
                                78.03219179999999, 13) 
  
# scatter method of map object  
# scatter points on the google map 
gmap3.scatter( latitude_list, longitude_list, '# FF0000', size = 40, marker = False ) 
gmap3.heatmap( latitude_list, longitude_list ) 
#gmap3.plot(latitude_list, longitude_list, 'cornflowerblue', edge_width = 2.5) 
  
gmap3.draw( "C:\\Users\\Animesh\\Desktop\\Final Year R\\map1.html" ) 
import geopandas

import pandas as pd
df=pd.read_csv(path2+'go_track_trackspoints.csv')
trl=df.groupby('track_id').apply(lambda x: x[['latitude','longitude']].values.astype('float32').tolist())

trlist=[]
for i in trl:
    g=[]
    for j in range(len(i)):
        g.append([float(i[j][0]),float(i[j][1])])
    trlist.append(g)
    
import csv
with open("tr2.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(trlist)
####################################################################    
    
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
    return float(dist)
    #gmaps = googlemaps.Client(key='AIzaSyANek40q60bomgvZAYTs5sOSrMZTh9AFn8')
    #print(gmaps.distance_matrix((40.714224, -73.961452),(116.43003,39.93209))['rows'][0]['elements'][0]['distance']['value'])
    
from sklearn.metrics import pairwise_distances
def diff_affinity(X):
    return pairwise_distances(X, metric=distfind)

from sklearn.cluster import AgglomerativeClustering
clusters = AgglomerativeClustering(n_clusters=None, affinity=diff_affinity, compute_full_tree=True, connectivity=None, linkage='complete', distance_threshold=100.0)
clusters.fit(trlist[:2]) 
y=clusters.labels_

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
'''