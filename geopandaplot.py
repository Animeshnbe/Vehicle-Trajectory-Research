# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 22:06:57 2020

@author: Animesh
"""

import geopandas as gp
df1=gp.read_file("C:/Users/Animesh/Documents/vta research project/DATASETS/t-drive Beijing/Beijing/Beijing_Links.shp")
df2=gp.read_file("C:/Users/Animesh/Documents/vta research project/DATASETS/t-drive Beijing/Beijing/Beijing_Nodes.shp")
df3=gp.read_file("C:/Users/Animesh/Documents/vta research project/DATASETS/t-drive Beijing/Beijing/2008.shp")
#df1.plot()
#print(df2.head)
#df2.iloc[1]['geometry'].coords[0][0]
#df1.iloc[0].plot()

df1=df1.to_crs({'init': 'epsg:4326'})
df2=df2.to_crs({'init': 'epsg:4326'})
rwy_tun = df1.set_index('id')

list(df1.iloc[0]['geometry'].coords)
len(df1)
import folium
locations = [[(y,x) for x,y in df1.iloc[i]['geometry'].coords] for i in range(len(df1))]

my_map = folium.Map(location = [39.907388,116.397013], zoom_start = 11.5)
for l in locations:
    folium.PolyLine(locations = l, line_opacity = 0.5).add_to(my_map)

#for point in range(0, len(locations)):
#    folium.Marker(locations[point], popup=df2['type'][point]).add_to(my_map)

my_map.save("C:/Users/Animesh/Documents/vta research project/my_map_nodes.html")

    
import pandas as pd
obj=1;
for temp in trlist:
    df = pd.DataFrame({'Object Id':obj})
    gdf = gp.GeoDataFrame(df, geometry=LineString(temp))
    obj+=1
    break
gdf.plot()

for row in trlist:
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
from pyproj import Proj
from shapely.geometry import Point, LineString
inProj = Proj({'init': 'epsg:32650'}) 
outProj = Proj({'init': 'epsg:4326'})
for pt in df2.geometry:
    #need to pull out data form Shapely object
    coords_obj = list(pt.coords)
    # transform points us pyroj
    x,y = transform(inProj,outProj, coords_obj[0][0], coords_obj[0][1])
    # put in your empty list as the desired Shapely object
    new_geo.append([x,y])
    # convert to a pandas series
