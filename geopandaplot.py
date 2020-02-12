# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 22:06:57 2020

@author: Animesh
"""

import geopandas as gp
df1=gp.read_file("C:/Users/Animesh/Documents/vta research project/DATASETS/t-drive Beijing/Beijing/Beijing_Links.shp")
df2=gp.read_file("C:/Users/Animesh/Documents/vta research project/DATASETS/t-drive Beijing/Beijing/Beijing_Nodes.shp")
df3=gp.read_file("C:/Users/Animesh/Documents/vta research project/DATASETS/t-drive Beijing/Beijing/2008.shp")
#print(df2.head)
df2.iloc[1]['geometry'].coords[0][0]
df1.iloc[0].plot()
df1=df1.to_crs({'init': 'epsg:4326'})
df2=df2.to_crs({'init': 'epsg:4326'})

import folium
locations = [[df2.iloc[i]['geometry'].coords[0][1],df2.iloc[i]['geometry'].coords[0][0]] for i in range(len(df2))]
my_map = folium.Map(location = [39.907388,116.397013], zoom_start = 11.5)
for point in range(0, len(locations)):
    folium.Marker(locations[point], popup=df2['type'][point]).add_to(my_map)

my_map.save("C:/Users/Animesh/Documents/vta research project/my_map_nodes.html")

from datetime import datetime
def timeint(start, end):
    sdt = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    edt = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
    timel=(edt-sdt).total_seconds()
    return timel
import os
import csv
path = "C:/Users/Animesh/Documents/vta research project/DATASETS/t-drive Beijing/taxi_log_2008_by_id/"
traject=[] #list of all points for extracting representatives
trlist=[] #list of all trajectories by vehicle id
from pyproj import Proj
from shapely.geometry import Point, LineString
#inProj = Proj({'init': 'epsg:32650'}) 
#outProj = Proj({'init': 'epsg:4326'})
for file in os.scandir(path):
    with open(file,'r') as gps:
        temp=[]
        reader = csv.reader(gps, delimiter=',')
        prev=''
        for row in reader:
            traject.append(row)
            temp.append(tuple((float(row[2]),float(row[3]))))
            if prev!='' and timeint(prev,row[1])>600.0:
                trlist.append(temp)
                temp=[]
                prev=''
            else:
                prev=row[1]       
        trlist.append(temp) 
  
def slot (dt):
    h=(int(dt.strftime("%H")))*6 + int(dt.strftime("%M"))//10
    return int(dt.strftime("%d"))-2, h

t_win=np.zeros(shape=(7,144))

for t in traject:
    t_win[slot(datetime.strptime(t[1], '%Y-%m-%d %H:%M:%S'))] += 1
    
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.,24.,(1/6))

plt.bar(x, t_win[3,:],width=0.5, align='center', alpha=0.5)
#plt.xticks(,)
plt.ylabel('Traffic Count')
plt.title('Distribution of traffic')
plt.show()

lengths=[]
temp=0
while temp<len(trlist):
    if len(trlist[temp])<=1:
        trlist.pop(temp)
        
    else:
        lengths.append(len(trlist[temp]))
        temp+=1
        
import numpy as np
with open('raw_traj_tdrive.csv', mode='w') as dtfile:
    wr=csv.writer(dtfile, delimiter=',')
    for i in range(len(trlist)):
        wr.writerow(trlist[i])
        
seed_points_src=[xt[0] for xt in trlist]
seed_points_dest=[xt[-1] for xt in trlist]

from math import radians, sin, cos, asin
def dist(src,dest):
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

def closest(cells,s,gamma):
    min_dist=np.inf
    flag=False
    for c in cells:
        if dist(np.mean(c,axis=0),s)<gamma and dist(np.mean(c,axis=0),s)<min_dist:
            min_dist=dist(np.mean(c,axis=0),s)
            c.append(s)
            flag=True
    if flag==False:
        c=[s]
        cells.append(c)
    return cells
            
cells=[]
for s in seed_points_src:
    cells = closest(cells,s,1000)

for c in cells:
    c=[np.mean(c,axis=0)]
    
import pandas as pd
obj=1;
for temp in trlist:
    df = pd.DataFrame({'Object Id':obj})
    gdf = gp.GeoDataFrame(df, geometry=LineString(temp))
    obj+=1
    break
    
gdf.plot()

for row in trajectories:
    for gps in row:
        x,y=transform(outProj, inProj, float(gps[0]), float(gps[1]))
        lat.append(x)
        lon.append(y)

import matplotlib.pyplot as plt
import pandas as pd

data = {'Latitude':lat, 'Longitude':lon}

# Create DataFrame
df = pd.DataFrame(data)
gdf = gp.GeoDataFrame(df, crs=outProj, geometry=gp.points_from_xy(df.Latitude, df.Longitude))

#oneway, bridge, tunnel, type

#gdf = gp.GeoDataFrame(df, crs=inProj, geometry=geometry)
fig, ax = plt.subplots(figsize=(15,15))
# set aspect to equal. This is done automatically
# when using *geopandas* plot on it's own, but not when
# working with pyplot directly.
#ax.set_aspect('equal')
gdf.plot(ax=ax, color='red', markersize=0.7)
df1.plot(ax=ax, color='grey', linewidth=0.5)
#ax = df2.plot()
#gdf.plot(ax=ax, color='red')
plt.show()
fig.savefig('traj-overlay.png', quality=100, transparent=True)

#line.distance(point)
#create your in and out projections with pyroj:


#Iterate through your column of points and convert:
new_geo=[]
#oneway, bridge, tunnel, type
'''
for pt in df2.geometry:
    #need to pull out data form Shapely object
    coords_obj = list(pt.coords)
    # transform points us pyroj
    x,y = transform(inProj,outProj, coords_obj[0][0], coords_obj[0][1])
    # put in your empty list as the desired Shapely object
    new_geo.append([x,y])
    # convert to a pandas series

#empty_list =pandas.Series(empty_list)
# replace your geopandas column
#geometry = new_geo