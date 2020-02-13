# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:52:36 2019

@author: animesh
"""

#####################################PLOTTING####################################
#####################################FOLIUM######################################
import folium
import random
def createmap(trajlist,i):
    #i is used for file naming for multiple plots
    my_map = folium.Map(location = [39.907388,116.397013], zoom_start = 11.5)  
    for xt in trajlist:
        if len(xt)!=0:
            loc=[tuple(l) for l in xt]
            random_number = random.randint(0,16777215)
            hex_number = str(hex(random_number))
            hex_number ='#'+ hex_number[2:]
            folium.PolyLine(locations = loc, color=hex_number, weight=1.5, line_opacity = 0.5).add_to(my_map)
     
    my_map.save("C:/Users/Animesh/Documents/vta research project/my_map"+str(i)+".html")

#####################################GMaps PLOT######################################
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

#############################Geopandas Plot of City##################################
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
