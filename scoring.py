# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 01:59:46 2020

@author: Animesh
"""

#Score based feature vector generation
import numpy as np
from dataread0 import timeint,distfind
import math
import csv
    
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
 
def seed(pt):
    gps = open('C:/Users/Animesh/Documents/vta research project/actual_cells.csv','r')
    reader = csv.reader(gps, delimiter=' ')
    mindist=np.inf
    for row in reader:
        coord=row[0].split(', ')
        lon=float(coord[0][1:])
        lat=float(coord[1][:-1])
        if distfind(pt,(lon,lat))<mindist:
            mindist=distfind(pt,(lon,lat))
            seedpt=(lon,lat)
    return seedpt            
                
def check_diag(pt,src,dest):
    m=(dest[1]-src[1])/(dest[0]-src[0])
    md=-1/m
    if (md*(pt[0]-src[0])+src[1]-pt[1])*(md*(dest[0]-src[0])+src[1]-dest[1])<0 or (md*(pt[0]-dest[0])+dest[1]-pt[1])*(md*(src[0]-dest[0])+dest[1]-src[1])<0:
        return True
    else:
        return False
     
def inflow(cell,ts):
    inflo=0
    outflo=0
    for pt in traject:
        if (pt[0]>cell[0][0] and pt[1]>cell[0][1]) and (pt[0]<cell[1][0] and pt[1]<cell[1][1]):
            if ts>pt[3]:
                inflo+=1
            else:
                outflo+=1
    return inflo, outflo

def cell_score(traj):
    src=traj[0]
    dest=traj[-1]
    ind1=((datetime.strptime(src[2], '%Y-%m-%d %H:%M:%S')-datetime.strptime('2008-02-02 13:30:00', '%Y-%m-%d %H:%M:%S')).total_seconds())//600
    ind2=((datetime.strptime(dest[2], '%Y-%m-%d %H:%M:%S')-datetime.strptime('2008-02-02 13:30:00', '%Y-%m-%d %H:%M:%S')).total_seconds())//600
    minlat=min(src[1],dest[1])
    minlon=min(src[1],dest[1])
    maxlat=max(src[0],dest[0])
    maxlon=max(src[0],dest[0])
    blc=[minlon,minlat]
    trc=[maxlon,maxlat]
    
    roi=np.subtract(trc,blc)
    if ind2!=ind1:
        dbscan_cent
    
    grid=tuple(np.subtract(dest,src)/(len(traj)-2))
    acc=src
    cells=[]
    while (acc[0]<dest[0] and acc[1]<dest[1]):
        cells.append([acc,tuple(np.add(acc,grid))])
        
    #calc inflows    
    for cell in cells:
        inflow(cell,ts)    
        
    cells=[(49,24),(23,13),(11,0),(40,17),(12,5),(34,14)]
    ss=[100,45,78,12,23,9]
    cells=np.argsort(cells,axis=1)
    
    rank_match=0
    traj_score=0
    cell_score=[]
    rank=[]
    for pts in range(len(traj)):
        k=timeint(traj[pts-1][2],traj[pts][2])/timeint(src[2],dest[2])
        numb=[]
        for i in dbscan_cent:
            if(i[2]>src[2] and i[2]<dest[2] and i[0]>src[0] and i[0]<dest[0] and i[1]>src[1] and i[1]<dest[1]):
                numb.append(i)
        supp=len(numb)
        
        if check_diag(traj[pts],src,dest):
            rank_match=-2
        else:
            if (rank[pts-1]<rank[pts]):
                rank_match+=1
            else:
                rank_match-=1
            if (rank[pts+1]>rank[pts]):
                rank_match+=1
            else:
                rank_match-=1
            
        flag=False
        for sg in cells:
            if (traj[pts][0]>sg[0][0] and traj[pts][0]<sg[1][0] and traj[pts][1]>sg[0][1] and traj[pts][1]<sg[1][1]):
                flag=True
                rank[traj[pts]]=timer[traj[pts][time]]
                break
            
        if flag==False:
            
            if (traj[pts][0]>sg[0][0] and traj[pts][0]<sg[1][0] and traj[pts][1]>sg[0][1] and traj[pts][1]<sg[1][1]):
                rank[traj[pts]]=ss[inflow(sg)]
        
        
        rank = [x for _,x in sorted(zip(ss,cells))]
            
                
        m1=(seed(dest)[1]-seed(src)[1])/(seed(dest)[0]-seed(src)[0])
        m2=(traj[pts+1][1]-traj[pts][1])/(traj[pts+1][0]-traj[pts][0])
        theta=math.atan((m1-m2)/(1+m1*m2))
        cell_score.append(rank_match*math.exp(-k*supp)*math.cos(theta))
        w=0
        for roads in road_segs(cell):
            m2=(roads[-1][1]-roads[0][1])/(roads[-1][0]-roads[0][0])
            w+=math.cos(math.atan((m1-m2)/(1+m1*m2)))
        traj_score+=w*cell_score
    return cell_score,traj_score