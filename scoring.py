# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 01:59:46 2020

@author: Animesh
"""

#Score based feature vector generation
import numpy as np
from dataread0 import timeint
import math

dbscan_cent=[]
with open('C:/Users/Animesh/Documents/vta research project/DBScan.csv','r') as gps:
    reader = csv.reader(gps, delimiter=' ')
    dbscan_cent=[]
    for row in reader:
        dbscan_cent.append(row[0])
    
with open('/content/drive/My Drive/raw_traj_tdrive.csv','r') as gps:
  trlist=[]
  reader = csv.reader(gps, delimiter=',')
  ctr=0
  for row in reader:
    if (ctr%2)==0:
      trlist.append(row)    
    ctr+=1 
    
def road_segs():
    
def check_diag(pt,src,dest):
    m=(dest[1]-src[1])/(dest[0]-src[0])
    md=-1/m
    if (md*(pt[0]-src[0])+src[1]-pt[1])*(md*(dest[0]-src[0])+src[1]-dest[1])<0 or (md*(pt[0]-dest[0])+dest[1]-pt[1])*(md*(src[0]-dest[0])+dest[1]-src[1])<0:
        return False
    else:
        return True
    
def cell_score(traj):
    src=traj[0]
    dest=traj[-1]
    grid=tuple(np.subtract(dest,src)/(len(traj)-2))
    acc=src
    cells=[]
    while (acc[0]<dest[0] and acc[1]<dest[1]):
        cells.append([acc,tuple(np.add(acc,grid))])
        
    for pts in range(len(traj)):
        
    #calc inflows    
    for cell in cells:
        ss=count_inflow(cell,)
        
    cells=[(49,24),(23,13),(11,0),(40,17),(12,5),(34,14)]
    ss=[100,45,78,12,23,9]
    ci=np.argsort(ss)
    sorted_cells=cells[ci[]]
    cells=cells[]
    rank_match=0
    traj_score=0
    cell_score=[]
    for pts in range(len(traj)):
        k=timeint(traj[pts-1][2],traj[pts][2])/timeint(src[2],dest[2])
        numb=[]
        for i in dbscan_cent:
            if(i[2]>src[2] and i[2]<dest[2] and i[0]>src[0] and i[0]<dest[0] and i[1]>src[1] and i[1]<dest[1]):
                numb.append(i)
        supp=len(numb)
        flag=False
        for sg in cells:
            if (traj[pts][0]>sg[0][0] and traj[pts][0]<sg[1][0] and traj[pts][1]>sg[0][1] and traj[pts][1]<sg[1][1]):
                flag=True
                break
        if flag==False:
            if checkdiag(traj[pts],src,dest):
                rank_match=-2
            if (traj[pts][0]>sg[0][0] and traj[pts][0]<sg[1][0] and traj[pts][1]>sg[0][1] and traj[pts][1]<sg[1][1]):
                rank=ss[inflow(sg)]
                if (rank[traj[pts-1]<rank[traj[pts]):
                    rank_match+=1
                else:
                    rank_match-=1
                if (rank[traj[pts+1]>rank[traj[pts]):
                    rank_match+=1
                else:
                    rank_match-=1
        m1=(seed(dest)[1]-seed(src)[1])/(seed(dest)[0]-seed(src)[0])
        m2=(traj[pts+1][1]-traj[pts][1])/(traj[pts+1][0]-traj[pts][0])
        theta=math.atan((m1-m2)/(1+m1*m2))
        cell_score.append(rank_match*math.exp(-k*supp)*math.cos(theta))
        w=0
        for roads in road_segs(cell):
            m2=(roads[-1][1]-roads[0][1])/(roads[-1][0]-roads[0][0])
            w+=math.cos(math.atan((m1-m2)/(1+m1*m2)))
        traj_score+=w*cell_score
    return traj_score