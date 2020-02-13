# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 11:44:01 2019

@author: Animesh
"""
from dataread0 import timeint, mapmatch, floatify, diff_affinity
import numpy as np
import os
import csv
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
 
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

#path = "C:/Users/Animesh/Documents/vta research project/DATASETS/t-drive Beijing/taxi_log_2008_by_id/"
path = "C:/Users/Animesh/Documents/vta research project/test/"

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

p=0
for i in range(len(modtrlist)):
    modtrlist[i]=mapmatch(floatify(modtrlist[i]))
    #trlist[i]=floatify(trlist[i])
    q=p+len(modtrlist[i])
    modtrlist[i]=[[j,traject[k][1]] for j,k in zip(modtrlist[i],range(p,q))]
    p=q
    
###############################METHOD 2######################################
import re

ct=0
xt=[]
with open('trajectories.csv', mode='r') as dtfile:
    rd=csv.reader(dtfile,delimiter=',')
    for row in rd:
        if ct%2==0:
            xt.append(row)
        ct+=1

xall=[]
trlist=[]
for i in range(len(xt)):
    trajectory=[]
    for j in range(len(xt[i])):
        y=re.findall(r'\d+\.*\d*',xt[i][j])
        trajectory.append([float(y[0]),float(y[1])])
        xall.append([float(y[0]),float(y[1])])
    trlist.append(np.array(trajectory,dtype=float))

clusters = AgglomerativeClustering(n_clusters=None, affinity=diff_affinity, compute_full_tree='auto', connectivity=None, distance_threshold=100.0, linkage='complete')
clusters.fit(xall) 
y=clusters.labels_
lab=Counter(y).keys()

ctr=0
s=0
mt=[]
for i in range(len(modtrlist)):
    modtraj=[]
    modtraj2=[]
    s=s+len(modtrlist[i])
    while ctr<s:
        modtraj.append(y[ctr])
        modtraj2.append([y[ctr],xall[ctr]])
        #remove no change cell sequences
        while ctr+1<s and y[ctr]==y[ctr+1]:
            ctr+=1
        ctr+=1
    mt.append(modtraj)