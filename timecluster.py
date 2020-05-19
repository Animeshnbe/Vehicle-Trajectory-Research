# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 06:42:29 2020

@author: IDALAB HCM - II
"""

import csv
import numpy as np
import os
path="C:/Users/IDALAB HCM - II/Downloads/Vehicle-Trajectory-Research-master/T-drive Taxi Trajectories/release/taxi_log_2008_by_id/"

from datetime import datetime
def timeint(start, end):
    sdt = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    edt = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
    timel=(edt-sdt).total_seconds()
    return timel

traject=[] #list of all points for extracting representatives
trlist=[] #list of all trajectories by vehicle id
for file in os.scandir(path):
    temp=[]
    with open(file,'r') as gps:
        reader = csv.reader(gps, delimiter=',')
        prev=''
        for row in reader:
            #k=row[2:].split('\t')
            
            if 116.10<=float(row[2])<=116.71 and 39.69<=float(row[3])<=40.18:
                traject.append(row)
                temp.append(row[1:])
                if prev!='' and timeint(prev,row[1])>600.0:
                    trlist.append(temp)
                    temp=[]
                    prev=''
                else:
                    prev=row[1]       
    trlist.append(temp)

#%%
coordinates=[[] for i in range(889)]
for i in traject:
    tm = datetime.strptime(i[1], '%Y-%m-%d %H:%M:%S')
    index=(tm-datetime.strptime('2008-02-02 13:30:00', '%Y-%m-%d %H:%M:%S')).total_seconds()
    index=int(index//600)
    h=i[2:]
    fi=[float(h[0]),float(h[1])]
    coordinates[index].append(fi)

#%%    
from math import radians, sin, cos, asin, sqrt
def distfind(src,dest):
    slon,slat = src[0],src[1]
    elon,elat = dest[0],dest[1]
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

from sklearn.cluster import DBSCAN 
from collections import Counter
#%%
f=open("representative_pts2.csv", "w", newline="")
writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
clusters = DBSCAN(eps=500, min_samples=10, metric=distfind)
master_sheet=[]
for i in coordinates:
    if len(i)>1:
        clusters.fit(i)
        labels=clusters.labels_
        clab=[[] for i in range(len(Counter(labels).keys()))]
        n=len(clab)
        for j in range(len(i)):
            clab[labels[j]].append(i[j])
        centroid=[[np.mean([clab[j][k] for k in range(len(clab[j]))], axis=0),len(clab[j])] for j in range(len(clab)-1)]
        #centroid=list(centroid).insert(0,len(clab))
        master_sheet.append(centroid)
        writer.writerow(centroid)
        
#%%
        
from scipy.spatial import Voronoi, voronoi_plot_2d

import matplotlib.pyplot as plt
pts=[]

for i in centroid:
    pts.append(i[0])
vor = Voronoi(pts)

def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:

        raise ValueError("Requires 2D input")

    new_regions = []

    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)

    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point

    all_ridges = {}

    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue
        # reconstruct a non-finite region

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue
            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())
    return new_regions, np.asarray(new_vertices)

regions,vertices=voronoi_finite_polygons_2d(vor)
polygons=[]
for reg in regions:
    polygon=vertices[reg]
    polygons.append(polygon)
voronoi_plot_2d(vor)
plt.show()

#%%
print(list(polygons[0]))
#%%
import folium
my_map1 = folium.Map(location = [39.907388,116.397013], zoom_start = 11)

for i in polygons:
    l=[]
    for j in i:
        l.append([j[1],j[0]])
    folium.Polygon(locations = l).add_to(my_map1)
    
my_map1.save("C:/Users/IDALAB HCM - II/Downloads/Vehicle-Trajectory-Research-master/map_voronoi.html")
#%%
#normalising to 24hr period
daynormalized=[[] for i in range(144)]
for i in range(144):
    for j in range(5):
        daynormalized[i].append(coordinates[j*144+i+63])
        
for i in range(63):
    daynormalized[81+i].append(coordinates[i])
    
for i in range(783,889):
    daynormalized[(i-783)].append(coordinates[i])
    
#%%
for i in range(889):
    if len(coordinates[i])<=1:
        print(i)
#%%
import geopandas as gpd
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
area = world[world.name == 'Italy']

area = area.to_crs(epsg=3395)    # convert to World Mercator CRS
area_shape = area.iloc[0].geometry   # get the Polygon
#%%
dataf=open("C:/Users/IDALAB HCM - II/Downloads/Vehicle-Trajectory-Research-master/actual_cells.csv", 'r')
rd=csv.reader(dataf, delimiter=',', quoting=csv.QUOTE_NONE)
seeds=[]
for row in rd:
    seeds.append([float(row[0]),float(row[1])])
#%%
def cleanindex(ind):
    if ind>798:
        ind-=3
    elif ind>796:
        ind-=2
    elif ind>364:
        ind-=1
        
    return int(ind)

def seed(s):
    pt=[float(s[1]),float(s[2])]
    mindist=np.inf
    for j in range(len(seeds)):
        if distfind(pt,seeds[j])<mindist:
            mindist=distfind(pt,seeds[j])
            seedpt=seeds[j]
    return seedpt            
                
def check_diag(pt,s,d):
    sr=[float(s[1]),float(s[2])]
    des=[float(d[1]),float(d[2])]
    if des[0]!=sr[0] and des[1]!=sr[1]:
        m=(des[1]-sr[1])/(des[0]-sr[0])
        md=-1/m
    elif des[0]==sr[0]:
        md=0
    else:
        if des[0]>sr[0]:
            md=np.inf
        else:
            md=0-np.inf
    if (md*(pt[0]-sr[0])+sr[1]-pt[1])*(md*(des[0]-sr[0])+sr[1]-des[1])<0 or (md*(pt[0]-des[0])+des[1]-pt[1])*(md*(sr[0]-des[0])+des[1]-sr[1])<0:
        return True
    else:
        return False
#%%    
import math
score_trlist=[]
ctr=0
faltu=[]
for i in trlist:
    if len(i)>2:
        src=i[0]
        dest=i[-1]
        ind1=((datetime.strptime(src[0], '%Y-%m-%d %H:%M:%S')-datetime.strptime('2008-02-02 13:30:00', '%Y-%m-%d %H:%M:%S')).total_seconds())//600
        ind2=((datetime.strptime(dest[0], '%Y-%m-%d %H:%M:%S')-datetime.strptime('2008-02-02 13:30:00', '%Y-%m-%d %H:%M:%S')).total_seconds())//600
        minlat=min(float(src[2]),float(dest[2]))
        minlon=min(float(src[1]),float(dest[1]))
        maxlat=max(float(src[2]),float(dest[2]))
        maxlon=max(float(src[1]),float(dest[1]))
        blc=[minlon,minlat]
        trc=[maxlon,maxlat]
        
        roi=np.subtract(trc,blc)
        
        ind1=cleanindex(ind1)
        ind2=cleanindex(ind2)
        if ind2!=ind1:
            dbs=master_sheet[ind1:ind2+1]
            coords=[]
            for m in range(len(dbs)):
                for j in dbs[m]:
                    coords.append(j[0])
            
            clusters = DBSCAN(eps=500, min_samples=1, metric=distfind)
            clusters.fit(coords)
            labels=clusters.labels_
            clab=[[] for i in range(len(Counter(labels).keys()))]
            n=len(clab)
            for j in range(len(coords)):
                clab[labels[j]].append(coords[j])
            dbscan_cent=[[np.mean([clab[j][k] for k in range(len(clab[j]))], axis=0)] for j in range(len(clab))]
            
        else:
            dbscan_cent=[]
            for j in master_sheet[ind1]:
                dbscan_cent.append(j[0])
         
        rank=[]
        hsize=0.61/136
        vsize=0.49/144
        for pts in range(len(i)):
            r=int((float(i[pts][1])-116.10)/hsize)+int((float(i[pts][2])-39.69)/vsize)
            rank.append(r)
        cell_score=[]
        for pts in range(len(i)-1):
            tf=timeint(i[pts-1][0],i[pts][0])/timeint(src[0],dest[0])
            
            numb=[]
            for k in dbscan_cent:
                #print(k[0][1])
                #break
                if(k[0][1]>minlat and k[0][1]<maxlat and k[0][0]>minlon and k[0][0]<maxlon):
                    numb.append(k)
            supp=len(numb)
            
            #break
            
            rank_match=0.00001
            if check_diag([float(i[pts][1]),float(i[pts][2])],src,dest):
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
            
            denom0=(seed(dest)[0]-seed(src)[0])
            if denom0!=0:
                m1=(seed(dest)[1]-seed(src)[1])/denom0
            else:
                if seed(dest)[1]>seed(src)[1]:
                    m1=np.inf
                else:
                    m1=0-np.inf
            denom=distfind([float(i[pts+1][1]),float(i[pts][2])],[float(i[pts][1]),float(i[pts][2])])
            if denom!=0:
                m2=distfind([float(i[pts][1]),float(i[pts+1][2])],[float(i[pts][1]),float(i[pts][2])])/denom
            else:
                if float(i[pts+1][2])>float(i[pts][2]):
                    m2=np.inf
                else:
                    m2=0-np.inf
            if (m1==np.inf or m1==(0-np.inf)) and (m2==np.inf or m2==(0-np.inf)):
                sc=0.000000001
                
            else:
                if m1==np.inf:
                    theta=np.pi/2-math.atan(m2)
                elif m1==(0-np.inf):
                    theta=0-np.pi/2-math.atan(m2)
                elif m2==np.inf:
                    theta=math.atan(m1)-np.pi/2
                elif m2==(0-np.inf):
                    theta=math.atan(m1)+np.pi/2
                else:
                    theta=math.atan((m1-m2)/(1+m1*m2))                  
                sc=float(rank_match*math.exp((0-tf)*supp)*math.cos(theta))
                
            cell_score.append(sc)
        score_trlist.append(cell_score)
    else:
        #print(ctr)
        faltu.append(ctr)
    ctr+=1
    #%%
print(abs(0-np.inf))