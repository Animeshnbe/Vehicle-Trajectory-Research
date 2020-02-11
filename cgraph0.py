# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:00:17 2019

@author: Animesh
"""
'''
import networkx as nx
import numpy as np
G = nx.Graph()

def reprpts(coords,poi):
    #coords=Delaunay(coords)
    thres=3000 #metres
    print(np.zeros((coords.shape[0],1),dtype=coords.dtype))
    np.concatenate([coords, np.zeros((coords.shape[0],1),dtype=coords.dtype)], axis=1)
    unrep=[]
    for i in coords:
        if distfind(poi,i[:2])<thres and i[2]==0:
            unrep.append(i)
    centroid=np.mean(unrep,axis=0)
    unrep=[]
    for i in coords:
        if distfind(centroid,i)<thres:
            i[2]=centroid
            unrep.append(i)
        
    for i in unrep:
        if coords[i.simplices] and coords[i.simplices][2]==0:
            poi=i
        else:
            poi=coords[math.random.randint]

class CandidateGraph:
    
    def __init__(self, cdtpoints):
        for i in cdtpoints:
            if i not in self.vertices:
                self.vertices=1
            else:
                self.vertices[i]=self.vertices.get[i]+1
        self.vertices = cdtpoints
    def roadSegment(self, trlist):
        #to do edge clustering by number of transitions
        rseg=[]
        transitioncount=[]
        for i in range(len(trlist)-1):
            if ([trlist[i],trlist[i+1]] not in rseg) and ([trlist[i+1],trlist[i]] not in rseg):
                rseg.append([trlist[i],trlist[i+1]]) #unique road segments only
                transitioncount.append(1)
            else:
                if ([trlist[i],trlist[i+1]] in rseg):
                    transitioncount[rseg.index([trlist[i],trlist[i+1]])]+=1
                else:
                    transitioncount[rseg.index([trlist[i+1],trlist[i]])]+=1
        roads=[]
        for i in range(len(rseg)):
            roads.append([rseg[i],transitioncount[i]])
'''
from numpy import array
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
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
	return array(X), array(y)
 
modtrlist=array([[12.34,5.12],[1.,1.95],[13.2,32.42],[21.4,98.12],[57.01,8.00],[4.6,12.95]])
x_train, y_train=split_sequences(modtrlist, 3, 1)

print (x_train)
print(y_train)